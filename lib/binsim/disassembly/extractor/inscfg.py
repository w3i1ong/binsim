import os
import pickle
import filelock
from .base import ExtractorBase
from typing import List
from binsim.disassembly.backend.binaryninja import BinaryNinja
from binsim.disassembly.backend.binaryninja import InsCFGNormalizer, InsCFG

class InsCFGExtractor(ExtractorBase):
    def __init__(self, normalizer_kwargs, token2id_file=None,
                 record_token2id=False,
                 disassembler=None,**kwargs):
        if disassembler is None:
            disassembler = BinaryNinja(InsCFGNormalizer, normalizer_kwargs=normalizer_kwargs)
        super().__init__(disassembler, **kwargs)
        self.__record_token2id = record_token2id
        self.__token2id_file = token2id_file
        if record_token2id:
            assert token2id_file is not None, "The token2id file must be specified if record_token2id is True."

    def after_extract_process(self):
        if self.__record_token2id:
            lock_file = self.__token2id_file + '.lock'
            if os.path.exists(lock_file):
                os.unlink(lock_file)

    def after_disassemble(self, cfgs: List[InsCFG]):
        if cfgs is None:
            return None
        token2id = None
        if self.__record_token2id:
            tokens = set()
            for cfg in cfgs:
                tokens.update(cfg.unique_tokens())
            # If our extractor doesn't use multiprocessing, we can directly update the token2id dict
            with filelock.FileLock(self.__token2id_file + '.lock') as flock:
                token2id = {}
                if os.path.exists(self.__token2id_file):
                    with open(self.__token2id_file, 'rb') as f:
                        token2id = pickle.load(f)

                for token in tokens:
                    if token not in token2id:
                        token2id[token] = len(token2id)
                with open(self.__token2id_file, 'wb') as f:
                    pickle.dump(token2id, f)
        else:
            if self.__token2id_file is not None:
                with open(self.__token2id_file, 'rb') as f:
                    token2id = pickle.load(f)

        if token2id is not None:
            for cfg in cfgs:
                cfg.replace_tokens(token2id)

        return cfgs
