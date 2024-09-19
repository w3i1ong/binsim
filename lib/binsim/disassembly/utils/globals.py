from enum import Enum


class GraphType(Enum):
    PDG = 'PDG'
    TokenCFG = 'TokenCFG'
    ACFG = 'ACFG'
    ByteCode = 'ByteCode'
    CodeAST = 'CodeAST'
    InsCFG = 'InsCFG'
    MnemonicCFG = 'MnemonicCFG'
    JTransSeq = 'JTransSeq'

    def __str__(self):
        return self.value
