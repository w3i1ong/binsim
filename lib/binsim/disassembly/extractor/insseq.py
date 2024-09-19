from .inscfg import InsCFGExtractor
from binsim.disassembly.backend.binaryninja import BinaryNinja
from binsim.disassembly.backend.binaryninja import InsSeqNormalizer, InsSeq
class InsSeqExtractor(InsCFGExtractor):
    def __init__(self, normalizer_kwargs, token2id_file=None,
                 record_token2id=False, **kwargs):
        super().__init__(normalizer_kwargs, token2id_file, record_token2id,
                         disassembler=BinaryNinja(InsSeqNormalizer, normalizer_kwargs), **kwargs)
