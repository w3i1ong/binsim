import torch
from binsim.disassembly.ida.cfg import JTransSeq
from typing import List, Tuple, Dict, Any, Iterable
from dgl import DGLGraph
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase
from binsim.neural.nn.globals.siamese import SiameseSampleFormat


class JTransSeqSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[JTransSeq],
                 samples_id: torch.Tensor=None,
                 tags: List[str] = None,
                 with_name=False):
        """
        :param data: A list of JTransSeq.
        :param samples_id: The ids of each jTransSeq samples.
        :param tags: The tags of each jTransSeq samples.
        :param with_name: Whether return name when __getitem__ is called.
        """
        super().__init__(data, sample_id=samples_id, with_name=with_name, tags=tags)

    @staticmethod
    def collate_fn(data: Iterable[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        data = list(data)
        max_length = max([len(seq) for seq in data])
        batched_data, masks = [], []
        for seq in data:
            batched_data.append(seq + [0] * (max_length - len(seq)))
            masks.append([1] * len(seq) + [0] * (max_length - len(seq)))
        return torch.tensor(batched_data, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


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
