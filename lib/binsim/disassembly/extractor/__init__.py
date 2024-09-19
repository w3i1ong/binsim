from binsim.disassembly.utils.globals import GraphType
from .base import ExtractorBase


def get_extractor_by_name(name: GraphType, extractor_kwargs, disassemble_kwargs, normalizer_kwargs, neural_input_kwargs):
    kwargs = {**extractor_kwargs, "disassemble_kwargs": disassemble_kwargs,
              "normalizer_kwargs": normalizer_kwargs, "neural_input_kwargs": neural_input_kwargs}
    match name:
        case GraphType.ByteCode:
            from .bytecode import BytecodeExtractor
            return BytecodeExtractor(**kwargs)
        case GraphType.ACFG:
            from .acfg import AttributedCFGExtractor
            return AttributedCFGExtractor(**kwargs)
        case GraphType.TokenCFG:
            from .tokencfg import TokenCFGExtractor
            return TokenCFGExtractor(**kwargs)
        case GraphType.TokenSeq:
            from .tokencfg import TokenSeqExtractor
            return TokenSeqExtractor(**kwargs)
        case GraphType.InsCFG:
            from .inscfg import InsCFGExtractor
            return InsCFGExtractor(**kwargs)
        case GraphType.InsSeq:
            from .insseq import InsSeqExtractor
            return InsSeqExtractor(**kwargs)
        case GraphType.JTransSeq:
            from .jtrans import JTransExtractor
            return JTransExtractor(**kwargs)
        case GraphType.CodeAST:
            from .codeast import CodeAstExtractor
            return CodeAstExtractor(**kwargs)
        case _:
            raise ValueError('Unknown extractor name: {}'.format(name))
