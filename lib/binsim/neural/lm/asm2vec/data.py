import pickle
import random
from binsim.disassembly.binaryninja import TokenCFG
from gensim.models.asm2vec import Function
from typing import List, Generator
from .utils import bb2Inst
from binsim.disassembly.binaryninja.core import TokenCFGDataForm

MAX_WORDS_IN_BATCH = 10000


class ListFunctionFromFile(object):
    def __init__(self, source: str, max_sentence_length=MAX_WORDS_IN_BATCH):
        """Iterate over a file that contains sentences: one line = one sentence.
        Words must be already preprocessed and separated by whitespace.

        Parameters
        ----------
        source : string
            Path to the file on disk.
        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        with open(source, 'rb') as f:
            self._graphs: list[TokenCFG] = pickle.load(f)
        random.shuffle(self._graphs)

    def __iter__(self):
        """Iterate through the lines in the source."""
        for cfg in self._graphs:
            for seq in cfg.random_walk(walk_num=10, max_walk_length=self.max_sentence_length,
                                       basic_block_func=bb2Inst):
                function = Function(seq, cfg.name)
                yield function


class ListFunctionFromFiles(object):
    def __init__(self, files: List[str], max_sentence_length=10000, random_walk_num=10, edge_coverage=True):
        """Iterate over a file that contains sentences: one line = one sentence.
        Words must be already preprocessed and separated by whitespace.

        Parameters
        ----------
        files : List[string]
            Path to the file on disk.
        """
        self._source_files = files
        self.max_sentence_length = max_sentence_length
        self.random_walk_num = random_walk_num
        self.edge_coverage = edge_coverage

    def __iter__(self) -> Function:
        """Iterate through the lines in the source."""
        for file in self._source_files:
            with open(file, 'rb') as f:
                graphs: list[TokenCFG] = pickle.load(f)
            for cfg in graphs:
                for seq in cfg.random_walk(walk_num=self.random_walk_num,
                                           max_walk_length=self.max_sentence_length,
                                           edge_coverage=self.edge_coverage,
                                           basic_block_func=bb2Inst):
                    function = Function(seq, cfg.name)
                    yield function


class NamedTokenCFGs(object):
    def __init__(self, cfgs: List[TokenCFG], names:List[str], max_sequence_length=10000, random_walk_num=10, edge_coverage=True):
        self.cfgs = cfgs
        self.names = names
        assert len(cfgs) == len(names)
        self.max_sequence_length = max_sequence_length
        self.random_walk_num = random_walk_num
        self.edge_coverage = edge_coverage

    def __iter__(self) -> Generator[Function, None, None]:
        for cfg, name in zip(self.cfgs, self.names):
            cfg.data_form = TokenCFGDataForm.TokenGraph
            for seq in cfg.random_walk(walk_num=self.random_walk_num,
                                       max_walk_length=self.max_sequence_length,
                                       edge_coverage=self.edge_coverage,
                                       basic_block_func=bb2Inst):
                function = Function(seq, name)
                yield function
