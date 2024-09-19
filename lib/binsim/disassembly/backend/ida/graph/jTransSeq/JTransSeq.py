from binsim.disassembly.core import BinsimFunction
from typing import List

class JTransSeq(BinsimFunction):
    def __init__(self, name,
                 arch,
                 token_id: List[int],
                 node_num: int,
                 ins_num: int,
                 func_hash=None):
        super().__init__(name, func_hash=func_hash, func_arch=arch, node_num=node_num, ins_num=ins_num)
        self._features = token_id

    def __str__(self):
        return f'<JTransSeq::{self.name}' \
               f'(length={len(self._features)})>'

    def __repr__(self):
        return self.__str__()

    def as_neural_input(self) -> List[int]:
        return self._features

    def as_neural_input_raw(self, *args, **kwargs):
        from binsim.disassembly.backend.binaryninja.graph.datautils import jtrans_seq_pack_neural_input
        return jtrans_seq_pack_neural_input(self._features)

    @staticmethod
    def collate_raw_neural_input(inputs: List[bytes], **kwargs):
        from binsim.disassembly.backend.binaryninja.graph.datautils import jtrans_seq_cfg_collate_neural_input
        return jtrans_seq_cfg_collate_neural_input(inputs)

    @property
    def features(self):
        return self._features

    def minimize(self):
        self._features = None
        return self
