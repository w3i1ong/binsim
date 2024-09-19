from .base import ExtractorBase
from binsim.disassembly.backend.binaryninja import BinaryNinja
from binsim.disassembly.backend.binaryninja import AttributedCFGNormalizer
class AttributedCFGExtractor(ExtractorBase):
    def __init__(self, normalizer_kwargs=None, **kwargs):
        assert normalizer_kwargs is None or len(normalizer_kwargs) == 0, "The AttributedCFGNormalizer does not need any arguments."
        super().__init__(BinaryNinja(AttributedCFGNormalizer),
                         **kwargs)
