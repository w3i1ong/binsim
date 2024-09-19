from .base import ExtractorBase
from binsim.disassembly.backend.ida.graph.CodeAST import CodeASTNormalizer
from binsim.disassembly.backend.ida import IDADisassembler

class CodeAstExtractor(ExtractorBase):
    def __init__(self, ida_path, normalizer_kwargs, disassemble_kwargs, **kwargs):
        super().__init__(IDADisassembler(CodeASTNormalizer, ida_path=ida_path,**normalizer_kwargs),
                         disassemble_kwargs={**disassemble_kwargs, "need_decompile": True},
                         **kwargs)
