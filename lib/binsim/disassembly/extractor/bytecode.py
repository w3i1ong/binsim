from .base import ExtractorBase
from binsim.disassembly.backend.binaryninja import BinaryNinja
from binsim.disassembly.backend.binaryninja import ByteCodeNormalizer
class BytecodeExtractor(ExtractorBase):
    def __init__(self, normalizer_kwargs, **kwargs):
        assert len(normalizer_kwargs) == 0 or normalizer_kwargs is None, "The ByteCodeNormalizer does not need any arguments."
        super().__init__(BinaryNinja(ByteCodeNormalizer),
                         **kwargs)
