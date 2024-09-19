import torch
import numpy as np
from typing import List, Dict, Any, Iterable
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase
from binsim.disassembly.backend.binaryninja import ByteCode

class ByteCodeSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[ByteCode], samples_id: List[Any]=None, tags = None,
                 max_len=10000, with_name=False, neural_input_cache_rocks_file = None):
        super().__init__(data, samples_id, max_byte_num=max_len, with_name=with_name, tags=tags,
                         neural_input_cache_rocks_file=neural_input_cache_rocks_file)
        self._max_len = max_len

    @staticmethod
    def collate_fn_py(data: Iterable[torch.Tensor], **kwargs) -> Any:
        data, degrees = list(zip(*data))
        degrees = torch.tensor(degrees, dtype=torch.float32)
        return torch.stack(data, dim=0).to(torch.float32), degrees

    @staticmethod
    def collate_fn_raw(data: List[bytes], **kwargs) -> Any:
        degrees, bytecodes = ByteCode.collate_raw_neural_input(data, **kwargs)
        return (torch.from_numpy(bytecodes.astype(np.float32)),
                torch.from_numpy(degrees.astype(np.float32)))


class ByteCodeSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[ByteCode]],
                 max_len=10000, sample_format: str = None, neural_input_cache_rocks_file = None):
        super().__init__(data, max_byte_num=max_len, sample_format=sample_format,
                         neural_input_cache_rocks_file=neural_input_cache_rocks_file)

    @property
    def SingleSampleClass(self):
        return ByteCodeSampleDataset
