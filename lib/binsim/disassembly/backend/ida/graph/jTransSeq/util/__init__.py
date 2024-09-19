import os
from .IDAIns import IDAIns


def load_ins_parser():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{cur_dir}/data/tokenizer/vocab.txt") as f:
        vocab = f.read().strip().split("\n") + ["[SEP]", "[PAD]", "[CLS]", "[MASK]"]
    vocab = dict((v, i) for i, v in enumerate(vocab))
    return IDAIns(vocab)
