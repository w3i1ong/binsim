from .palmtree import PalmTree
from .ins2vec import Ins2vec
import sys
if sys.platform == "linux":
    from .asm2vec import Asm2Vec, ListFunctionFromFiles, ListFunctionFromFile
