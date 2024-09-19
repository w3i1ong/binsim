from binsim.disassembly.binaryninja import InsCFG
from typing import List, Tuple, Dict, Any
import torch
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase
from binsim.neural.utils.data.dataset.InsCFG import BSInstruction, BatchedInstruction


class InsSeqSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[InsCFG],
                 samples_id: torch.Tensor,
                 with_name=False):
        super().__init__(data, sample_id=samples_id, with_name=with_name, as_seq=True)

    @staticmethod
    def collate_fn(data: List[Tuple[str, List[BSInstruction]]]) -> Tuple[List, Dict]:
        architectures, functions = zip(*data)
        lengths = [len(x) for x in functions]
        # build basic block
        max_length = max(lengths)
        batched_function_idx, cur_base = [], 0
        for function, length in zip(functions, lengths):
            batched_function_idx.append(
                torch.cat([torch.arange(cur_base, cur_base + length, dtype=torch.long),
                           torch.zeros(max_length - length, dtype=torch.long)]))
            cur_base += length
        batched_function_idx = torch.stack(batched_function_idx)

        # count how may instructions are there for each architecture
        arch_instructions_count = {}
        for arch, length in zip(architectures, lengths):
            arch_instructions_count[arch] = arch_instructions_count.get(arch, 0) + length

        # assign index range for each architecture
        arch_base, arch_range = {}, {}
        base = 0
        for arch, count in sorted(list(arch_instructions_count.items())):
            arch_base[arch] = base
            arch_range[arch] = (base, base + count)
            base += count
        # calculate the index for each instruction
        grouped_instructions = [None for _ in range(sum(lengths))]
        index_from_grouped_to_batched = []
        for arch, function, length in zip(architectures, functions, lengths):
            for ins in function:
                grouped_instructions[arch_base[arch]] = ins
                index_from_grouped_to_batched.append(arch_base[arch])
                arch_base[arch] += 1
        index_from_grouped_to_batched = torch.tensor(index_from_grouped_to_batched)

        arch_instructions = {}
        # generate representation for each instruction
        for arch, (start, end) in arch_range.items():
            arch_instructions[arch] = BatchedInstruction(arch, grouped_instructions[start:end])
        return index_from_grouped_to_batched[batched_function_idx], arch_instructions


class InsSeqSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[InsCFG]], sample_format: str = None):
        super().__init__(data, sample_format=sample_format, as_seq=True)

    @property
    def SingleSampleClass(self):
        return InsSeqSampleDataset
