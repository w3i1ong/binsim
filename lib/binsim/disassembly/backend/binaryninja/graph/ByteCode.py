import logging
import numpy as np
from binsim.utils import init_logger
from typing import List, TYPE_CHECKING
from binsim.disassembly.backend.binaryninja.utils import compute_function_hash
from .datautils import bytecode_pack_neural_input, bytecode_collate_neural_input
from binsim.disassembly.core import BinsimFunction, CFGNormalizerBase
logger = init_logger(__name__, level=logging.INFO,
                     console=True, console_level=logging.INFO)

if TYPE_CHECKING:
    from binaryninja import Function

class ByteCode(BinsimFunction):
    def __init__(self, name,
                 arch, raw_bytes: bytes, node_num:int, ins_num, *,
                 in_degree=None, out_degree=None,
                 func_hash=None):
        super().__init__(name, func_hash=func_hash, func_arch=arch, node_num=node_num, ins_num=ins_num)
        self._raw_bytes = raw_bytes
        self._in_degree = in_degree
        self._out_degree = out_degree

    @property
    def out_degree(self):
        return self._out_degree

    @property
    def in_degree(self):
        return self._in_degree

    @property
    def features(self):
        return self._raw_bytes

    def __str__(self):
        if self.features is not None:
            return (f'<ByteCode::{self.name}>'
                    f'(byte_num={len(self.features)})')
        return f'<ByteCode::{self.name}>'

    def __repr__(self):
        return self.__str__()

    def as_neural_input(self, max_byte_num=100 * 100):
        import torch, torch.nn.functional as F
        result = torch.from_numpy(np.array(list(self._raw_bytes[:max_byte_num]), dtype=np.uint8))
        if len(result) < max_byte_num:
            result = F.pad(result, (0, max_byte_num - len(result)))
        return result, (self._in_degree, self._out_degree)

    def as_neural_input_raw(self, max_byte_num = 10000)->bytes:
        payload = self._raw_bytes[:max_byte_num]
        return bytecode_pack_neural_input(self.in_degree, self.out_degree, payload)

    @staticmethod
    def collate_raw_neural_input(inputs: List[bytes], max_byte_num=10000):
        return bytecode_collate_neural_input(inputs, max_byte_num)

    def minimize(self):
        self._raw_bytes = None
        self._in_degree = self._out_degree = None
        return self

class ByteCodeNormalizer(CFGNormalizerBase):
    def __init__(self, arch):
        super().__init__(arch=arch)

    def __call__(self, function: "Function") -> ByteCode:
        bv = function.view
        # as the byte of some functions may not be continuous, we need to read the bytes of each basic block and then
        # aggregate them together.
        basic_block_bytes, instruction_count = [], 0
        basic_blocks = list(function)
        basic_blocks.sort(key=lambda x: x.start)
        for basic_block in basic_blocks:
            basic_block_bytes.append(bv.read(basic_block.start, basic_block.end - basic_block.start))
            instruction_count += basic_block.instruction_count
        function_bytes = b''.join(basic_block_bytes)
        # in-degree, out-degree
        in_degree = len(function.callees)
        out_degree = len(function.callers)
        return ByteCode(function.name, bv.arch.name, function_bytes,
                        in_degree=in_degree, out_degree=out_degree,
                        func_hash=compute_function_hash(function),
                        node_num=len(basic_blocks), ins_num=instruction_count)
