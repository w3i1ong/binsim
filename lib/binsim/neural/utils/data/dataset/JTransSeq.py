import torch
from dgl import DGLGraph
from binsim.disassembly.backend.ida import JTransSeq
from typing import List, Tuple, Dict, Any, Iterable
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase
from binsim.neural.nn.globals.siamese import SiameseSampleFormat


class JTransSeqSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[JTransSeq],
                 samples_id: torch.Tensor=None,
                 tags: List[str] = None,
                 with_name=False,
                 neural_input_cache_rocks_file=None):
        """
        :param data: A list of JTransSeq.
        :param samples_id: The ids of each jTransSeq samples.
        :param tags: The tags of each jTransSeq samples.
        :param with_name: Whether return name when __getitem__ is called.
        """
        super().__init__(data, sample_id=samples_id, with_name=with_name, tags=tags, neural_input_cache_rocks_file=neural_input_cache_rocks_file)

    @staticmethod
    def collate_fn(data: Iterable[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        batched_data, lengths = JTransSeq.collate_raw_neural_input(data)
        max_length = max(lengths)
        masks = []
        for length in lengths:
            masks.append([1] * length + [0] * (max_length - length))
        batched_data = torch.from_numpy(batched_data).to(dtype=torch.long)
        masks = torch.tensor(masks).to(dtype=torch.float)
        return batched_data, masks


class JTransSeqSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[JTransSeq]], sample_format: str = None):
        super().__init__(data, sample_format=sample_format)

    @property
    def SingleSampleClass(self):
        return JTransSeqSampleDataset

    def collate_fn(self, data: List[List[Tuple[DGLGraph, torch.Tensor, torch.Tensor]]]) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self._sample_format == SiameseSampleFormat.Pair:
            return super().collate_fn(data)
        raise NotImplementedError("Only support SiameseSampleFormat.PAIR for now!")
