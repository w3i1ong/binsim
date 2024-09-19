# distutils: language_level = 3
import re
from gensim.models.asm2vec import Instruction

INST_SPLITTER = re.compile(r"[#,{}+\-*\\\[\]:()\s]")

cpdef bb2Inst(bb):
    result = []
    for block in bb:
        operator, *operands = block
        operands_tokens = [token for token in operands if token not in ',+-[]:()#']
        result.append(Instruction(operator, operands_tokens))
    return result
