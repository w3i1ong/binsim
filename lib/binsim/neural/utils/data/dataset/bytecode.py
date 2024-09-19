from binsim.disassembly.binaryninja import ByteCode
from typing import List, Dict, Any, Iterable
import torch
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase


class ByteCodeSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[ByteCode], sample_id: List[Any], tags=None, max_len=10000, with_name=False):
        super().__init__(data, sample_id, max_byte_num=max_len, with_name=with_name, tags=tags)
        self._max_len = max_len

    @staticmethod
    def collate_fn(data: Iterable[torch.Tensor]) -> Any:
        data, degrees = list(zip(*data))
        degrees = torch.tensor(degrees, dtype=torch.float32)
        return torch.stack(data, dim=0).to(torch.float32), degrees


class ByteCodeSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[ByteCode]], max_len=10000, sample_format: str = None):
        super().__init__(data, max_byte_num=max_len, sample_format=sample_format)

    @property
    def SingleSampleClass(self):
        return ByteCodeSampleDataset
