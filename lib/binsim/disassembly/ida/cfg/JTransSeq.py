from binsim.disassembly.binaryninja.base import CFGBase
from typing import List


class JTransSeq(CFGBase):
    def __init__(self, name,
                 arch,
                 token_id: List[int],
                 node_num: int,
                 func_hash=None):
        super().__init__(name, func_hash=func_hash, func_arch=arch, node_num=node_num)
        self._features = token_id

    def __str__(self):
        return f'<JTransSeq::{self.name}' \
               f'(length={len(self._features)})>'

    def __repr__(self):
        return self.__str__()

    def as_neural_input(self) -> List[int]:
        return self._features
