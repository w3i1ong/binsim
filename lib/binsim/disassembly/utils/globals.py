from enum import Enum

class GraphType(Enum):
    ACFG = 'ACFG'
    CodeAST = 'CodeAST'
    ByteCode = 'ByteCode'

    TokenCFG = 'TokenCFG'
    TokenSeq = 'TokenSeq'

    InsCFG = 'InsCFG'
    InsSeq = 'InsSeq'

    PDG = 'PDG'
    MnemonicCFG = 'MnemonicCFG'
    JTransSeq = 'JTransSeq'

    def __str__(self):
        return self.value
