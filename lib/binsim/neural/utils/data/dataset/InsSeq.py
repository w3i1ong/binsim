import torch
from typing import List, Tuple, Dict, Any
from binsim.disassembly.backend.binaryninja import InsSeq
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase


class InsSeqSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[InsSeq],
                 samples_id: List[int],
                 tags:List = None,
                 with_name=False,
                 **kwargs):
        super().__init__(data, tags=tags, sample_id=samples_id, with_name=with_name, **kwargs)

    @staticmethod
    def collate_fn_py(data: List[Tuple[str, List]], **kwargs) -> Tuple[List, Dict]:
        raise NotImplementedError

    @staticmethod
    def collate_fn_raw(data, chunks=10, pack=False, max_seq_length=None, **kwargs):
        from .InsCFG import BatchedBBIndex, BatchedInstruction
        if pack:
            data, (packed_seqs, lengths) = InsSeq.collate_raw_neural_input(data, chunks=chunks, pack=pack)
            packed_seqs = torch.from_numpy(packed_seqs.astype('int32'))
            lengths = torch.from_numpy(lengths.astype('int32'))
            return BatchedInstruction(data), packed_seqs, lengths

        if chunks <= 1:
            data, chunks = InsSeq.collate_raw_neural_input(data, chunks=chunks)
            chunks = [torch.from_numpy(chunk) for chunk in chunks]
            batched_bb_ins_index = BatchedBBIndex(chunks, None)
        else:
            data, (*chunks, indexes) = InsSeq.collate_raw_neural_input(data, chunks=chunks)
            chunks = [torch.from_numpy(chunk) for chunk in chunks]
            indexes = torch.from_numpy(indexes)
            batched_bb_ins_index = BatchedBBIndex(chunks, indexes)
        unique_instructions = BatchedInstruction(data)
        return unique_instructions, batched_bb_ins_index



class InsSeqSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[InsSeq]], sample_format: str = None, **kwargs):
        super().__init__(data, sample_format=sample_format, **kwargs)

    @property
    def SingleSampleClass(self):
        return InsSeqSampleDataset
