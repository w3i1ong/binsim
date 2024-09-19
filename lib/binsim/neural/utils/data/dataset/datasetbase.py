import dgl
import torch
import random
import rocksdb
import numpy as np
from itertools import chain
from abc import abstractmethod
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
from binsim.disassembly.core import BinsimFunction
from binsim.neural.nn.globals.siamese import SiameseSampleFormat, EmbeddingDistanceMetric

__all__ = ['SampleDatasetBase', 'RandomSamplePairDatasetBase']


class SampleDatasetBase(Dataset):
    def __init__(self, samples: List[BinsimFunction],
                 sample_id: List[int] = None,
                 tags: List[Any] = None,
                 *neural_input_args,
                 neural_input_cache_rocks_file=None,
                 with_name=False,
                 **neural_input_kwargs):
        """
        This is the base class for sample datasets. The purpose of SampleDataset is to
        :param samples: A list of CFG samples.
        :param sample_id: Ids for CFG samples, if two CFGs are in the same classes, they should have same ids.
        :param tags: tags for CFG samples.
        :param neural_input_args: Extra arguments for as_neural_input.
        :param with_name: whether return sample_id and sample_name in __getitem__ method.
        :param neural_input_kwargs: Extra keyword arguments for as_neural_input.
        """
        super().__init__()
        self._data = samples
        self._neural_input_args = neural_input_args
        self._neural_input_kwargs = neural_input_kwargs
        self._tags = tags
        self._with_name = with_name
        self._sample_id = sample_id
        self._neural_input_cache_rocks_file = neural_input_cache_rocks_file
        self._neural_input_cache_rocks_db = None

        if with_name:
            assert sample_id is not None, "If with_name is True, sample_id must be provided."
            assert len(sample_id) == len(samples), "The length of sample_id must be equal to the length of data."
            assert tags is not None, "If with_name is True, tags must be provided."
            assert len(tags) == len(samples), "The length of tags must be equal to the length of data."

    @property
    def neural_input_cache_file(self):
        if self._neural_input_cache_rocks_db is None:
            self._neural_input_cache_rocks_db = rocksdb.DB(self._neural_input_cache_rocks_file,
                                                           rocksdb.Options(), read_only=True)
        return self._neural_input_cache_rocks_db

    @staticmethod
    def batch_graph(graphs: List[dgl.DGLGraph]):
        if 'nodeId' in graphs[0].ndata:
            base = 0
            for graph in graphs:
                delta = graph.ndata['nodeId'].max() + 1
                graph.ndata['nodeId'] += base
                base += delta
        return dgl.batch(graphs)

    def __len__(self):
        return len(self._data)

    def transform_sample(self, sample):
        if self._neural_input_cache_rocks_file is None:
            return sample.as_neural_input(*self._neural_input_args, **self._neural_input_kwargs)
        else:
            data = self.neural_input_cache_file.get(sample.content_hash.encode())
            return sample.preprocess_neural_input_raw(data, **self._neural_input_kwargs)


    def __getitem__(self, item):
        cfg = self._data[item]
        sample = self.transform_sample(cfg)
        if self._with_name:
            return sample, self._sample_id[item], cfg.name, self._tags[item]
        return sample

    def collate_fn(self, data) -> Any:
        if self._neural_input_cache_rocks_file is None:
            return self.collate_fn_py(data, **self._neural_input_kwargs)
        else:
            return self.collate_fn_raw(data, **self._neural_input_kwargs)

    @staticmethod
    @abstractmethod
    def collate_fn_py(data, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def collate_fn_raw(data, **kwargs):
        pass

    def collate_fn_with_name(self, data) -> Any:
        data, ids, names, tags = zip(*data)
        samples = self.collate_fn(data)
        ids = torch.from_numpy(np.array(ids, dtype=np.int32))
        return samples, ids, names, tags


class RandomSamplePairDatasetBase(Dataset):
    def __init__(self, data: Dict[str, List[BinsimFunction]],
                 *args, sample_format: SiameseSampleFormat = None, neural_input_cache_rocks_file=None,  **kwargs):
        super().__init__()
        self._data = data
        self._keys = None
        self._name2id = None
        self._neural_input_args = args
        self._neural_input_kwargs = kwargs
        self.sample_format = sample_format
        self._neural_input_cache_rocks_file = neural_input_cache_rocks_file
        self._neural_input_cache_rocks_db = None

    @property
    def neural_input_cache_file(self):
        if self._neural_input_cache_rocks_db is None:
            self._neural_input_cache_rocks_db = rocksdb.DB(self._neural_input_cache_rocks_file, rocksdb.Options(), read_only=True)
        return self._neural_input_cache_rocks_db

    @property
    def sample_format(self):
        return self._sample_format

    @sample_format.setter
    def sample_format(self, value):
        if value is None:
            value = SiameseSampleFormat.Pair

        assert isinstance(value, (SiameseSampleFormat, str)), \
            "sample_format must be an instance of SiameseSampleFormat or a string."

        if isinstance(value, str):
            value = SiameseSampleFormat(value)

        self._keys = list()
        for func_name, cfg_list in self._data.items():
            self._keys.extend([(func_name, idx) for idx in range(len(cfg_list))])
        self._name2id = {key: idx for idx, key in enumerate(self._data.keys())}
        self._sample_format = value

    @property
    @abstractmethod
    def SingleSampleClass(self):
        raise NotImplementedError

    def transform_sample(self, sample):
        if self._neural_input_cache_rocks_file is None:
            return sample.as_neural_input(*self._neural_input_args, **self._neural_input_kwargs)
        else:
            data =  self.neural_input_cache_file.get(sample.content_hash.encode())
            return sample.preprocess_neural_input_raw(data, **self._neural_input_kwargs)

    def __len__(self):
        if self.sample_format == SiameseSampleFormat.Pair:
            return len(self._keys) * 2
        else:
            return len(self._keys)

    def __getitem__(self, item):
        if self.sample_format in (
                SiameseSampleFormat.Pair, SiameseSampleFormat.PositivePair, SiameseSampleFormat.PositivePairSplit):
            return self._generate_sample_paris(item)
        elif self.sample_format == SiameseSampleFormat.Triplet:
            return self._generate_sample_triplet(item)
        else:
            raise NotImplementedError("Unknown sample format: {}".format(self.sample_format))

    def _generate_sample_triplet(self, item):
        # choose anchor
        anchor_name, anchor_idx = self._keys[item]
        sample_list = self._data[anchor_name]
        anchor = sample_list[anchor_idx]
        # choose positive
        if len(sample_list) == 1:
            positive = sample_list[0]
        else:
            while True:
                positive_idx = random.randrange(0, len(sample_list))
                if positive_idx != anchor_idx:
                    break
            positive = sample_list[positive_idx]

        # choose negative
        while True:
            negative_name, negative_idx = random.choice(self._keys)
            if negative_name != anchor_name:
                negative = self._data[negative_name][negative_idx]
                break

        return self.transform_sample(anchor), self._name2id[anchor_name], \
            self.transform_sample(positive), self._name2id[anchor_name], \
            self.transform_sample(negative), self._name2id[negative_name]

    def _generate_sample_paris(self, item):
        if self.sample_format == SiameseSampleFormat.Pair:
            index, pair_type = item // 2, item % 2
            anchor_name, anchor_idx = self._keys[index]
            anchor = self._data[anchor_name][anchor_idx]
            anchor_id = self._name2id[anchor_name]

            if pair_type == 0:  # similar pair
                sample_list = self._data[anchor_name]
                if len(sample_list) == 1:
                    positive = anchor
                else:
                    positive_idx = random.randrange(0, len(sample_list))
                    while positive_idx == anchor_idx:
                        positive_idx = random.randrange(0, len(sample_list))
                    positive = sample_list[positive_idx]

                return self.transform_sample(anchor), anchor_id, \
                    self.transform_sample(positive), anchor_id, \
                    1
            else:  # dissimilar pair
                while True:
                    negative_name, negative_idx = random.choice(self._keys)
                    if negative_name != anchor_name:
                        break
                negative = self._data[negative_name][negative_idx]
                negative_id = self._name2id[negative_name]
                return self.transform_sample(anchor), anchor_id, \
                    self.transform_sample(negative), negative_id, \
                    0
        elif self.sample_format in (SiameseSampleFormat.PositivePair, SiameseSampleFormat.PositivePairSplit):
            anchor_name, anchor_idx = self._keys[item]
            anchor = self._data[anchor_name][anchor_idx]
            anchor_id = self._name2id[anchor_name]
            sample_list = self._data[anchor_name]
            if len(sample_list) == 1:
                positive = sample_list[0]
            else:
                positive_idx = random.randrange(0, len(sample_list))
                while positive_idx == anchor_idx:
                    positive_idx = random.randrange(0, len(sample_list))
                positive = sample_list[positive_idx]

            return self.transform_sample(anchor), anchor_id, \
                self.transform_sample(positive), anchor_id, \
                1
    def collate_fn(self, data) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        data = list(zip(*data))
        if self._neural_input_cache_rocks_file is None:
            samplers_collate_fn = self.SingleSampleClass.collate_fn_py
        else:
            samplers_collate_fn = self.SingleSampleClass.collate_fn_raw
        match self.sample_format:
            case SiameseSampleFormat.Pair:
                anchor, anchor_ids, another, another_ids, labels = data
                samples = samplers_collate_fn(list(chain.from_iterable(zip(anchor, another))),
                                              **self._neural_input_kwargs)
                labels = torch.from_numpy(np.array(labels, dtype=np.float32))
                anchor_ids, another_ids = torch.tensor(anchor_ids), torch.tensor(another_ids)
                sample_ids = torch.stack([anchor_ids, another_ids], dim=1)
            case SiameseSampleFormat.Triplet:
                anchors, anchor_ids, positives, positive_ids, negative, negative_ids = data
                samples = samplers_collate_fn(list(chain.from_iterable(zip(anchors, positives, negative))),
                                                            **self._neural_input_kwargs)
                labels = None
                sample_ids = torch.stack([anchor_ids, positive_ids, negative_ids], dim=1)
            case SiameseSampleFormat.PositivePair:
                anchors, anchor_ids, positives, positive_ids, labels = data
                samples = samplers_collate_fn(list(chain.from_iterable(zip(anchors, positives))), **self._neural_input_kwargs)
                anchor_ids, positive_ids = torch.tensor(anchor_ids), torch.tensor(positive_ids)
                sample_ids = torch.stack([anchor_ids, positive_ids], dim=1)
                labels = torch.tensor(labels)
            case SiameseSampleFormat.PositivePairSplit:
                anchors, anchor_ids, positives, positive_ids, labels = data
                anchors = samplers_collate_fn(anchors, **self._neural_input_kwargs)
                positives = samplers_collate_fn(positives, **self._neural_input_kwargs)
                samples = (anchors, positives)
                anchor_ids, positive_ids = torch.tensor(anchor_ids), torch.tensor(positive_ids)
                sample_ids = torch.stack([anchor_ids, positive_ids], dim=1)
                labels = torch.tensor(labels)

            case _:
                raise NotImplementedError
        return samples, sample_ids.reshape([-1]), labels

    def __repr__(self):
        return f"{self.__class__.__name__}(sample_format={self.sample_format})"
