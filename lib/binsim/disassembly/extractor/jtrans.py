from .base import ExtractorBase
from binsim.disassembly.backend.ida.graph.jTransSeq import JTransNormalizer
from binsim.disassembly.backend.ida import IDADisassembler

class JTransExtractor(ExtractorBase):
    def __init__(self, ida_path, normalizer_kwargs, **kwargs):
        super().__init__(IDADisassembler(JTransNormalizer, ida_path=ida_path, **normalizer_kwargs),
                         **kwargs)
