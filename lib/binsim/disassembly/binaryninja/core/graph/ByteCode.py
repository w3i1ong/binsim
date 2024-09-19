import numpy as np
import torch
import logging
from ..utils import compute_function_hash
from torch.nn import functional as F
from ...base import CFGBase, CFGNormalizerBase
from binaryninja import Function
from typing import Tuple
from binsim.utils import init_logger
logger = init_logger(__name__, level=logging.INFO,
                     console=True, console_level=logging.INFO)


class ByteCode(CFGBase):
    def __init__(self, name,
                 arch, raw_bytes: bytes, node_num:int, *,
                 in_degree=None, out_degree=None,
                 func_hash=None):
        super().__init__(name, func_hash=func_hash, func_arch=arch, node_num=node_num)
        self._raw_bytes = raw_bytes
        self.in_degree = in_degree
        self.out_degree = out_degree

    def __str__(self):
        return f'<ByteCode::{self.name}' \
               f'(byte_num={len(self._raw_bytes)})>'

    @property
    def raw_bytes(self):
        return self._raw_bytes

    def as_neural_input(self, max_byte_num=100 * 100) -> Tuple[torch.Tensor,Tuple]:
        result = torch.from_numpy(np.array(list(self._raw_bytes[:max_byte_num]), dtype=np.uint8))
        if len(result) < max_byte_num:
            result = F.pad(result, (0, max_byte_num - len(result)))
        return result, (self.in_degree, self.out_degree)


class ByteCodeNormalizer(CFGNormalizerBase):
    def __init__(self, arch):
        super().__init__(arch=arch)

    def __call__(self, function: Function) -> ByteCode:
        bv = function.view
        # as the byte of some functions may not be continuous, we need to read the bytes of each basic block and then
        # aggregate them together.
        basic_block_bytes = []
        basic_blocks = list(function)
        basic_blocks.sort(key=lambda x: x.start)
        for basic_block in basic_blocks:
            basic_block_bytes.append(bv.read(basic_block.start, basic_block.end - basic_block.start))
        function_bytes = b''.join(basic_block_bytes)
        # in-degree, out-degree
        in_degree = len(function.callees)
        out_degree = len(function.callers)
        return ByteCode(function.name, bv.arch.name, function_bytes,
                        in_degree=in_degree, out_degree=out_degree,
                        func_hash=compute_function_hash(function),
                        node_num=len(basic_blocks))
