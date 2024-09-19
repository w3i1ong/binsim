from binsim.disassembly.core import NormalizerBase

class IDANormalizerBase(NormalizerBase):
    def __init__(self, arch):
        super().__init__(arch)

    def __call__(self, func):
        pass
