import os
import pickle
import filelock
from typing import List
from .base import ExtractorBase
from binsim.disassembly.backend.binaryninja import BinaryNinja
from binsim.disassembly.backend.binaryninja import TokenCFGNormalizer, TokenSeqNormalizer
from binsim.disassembly.backend.binaryninja.graph import TokenCFG

class TokenCFGExtractor(ExtractorBase):
    def __init__(self, normalizer_kwargs, token2id_file=None, corpus_file=None,
                 record_token2id=False, **kwargs):
        super().__init__(BinaryNinja(TokenCFGNormalizer, normalizer_kwargs=normalizer_kwargs),
                         **kwargs)
        self.__token2id_file = token2id_file
        self.__record_token2id = record_token2id
        self.__corpus_file = corpus_file
        if record_token2id:
            assert token2id_file is not None, "The token2id file must be specified if record_token2id is True."

    def after_extract_process(self):
        if self.__record_token2id:
            lock = self.__token2id_file + '.lock'
            if os.path.exists(lock):
                os.remove(lock)
        if self.__corpus_file is not None:
            corpus_lock = self.__corpus_file + '.lock'
            if os.path.exists(corpus_lock):
                os.remove(self.__corpus_file + '.lock')

    def after_disassemble(self, cfgs: List[TokenCFG]):
        if cfgs is None:
            return None
        token2id = None

        if self.__record_token2id:
            unique_tokens = set()
            for cfg in cfgs:
                unique_tokens.update(cfg.unique_tokens())
            with filelock.FileLock(self.__token2id_file + '.lock') as flock:
                token2id = {}
                if os.path.exists(self.__token2id_file):
                    with open(self.__token2id_file, 'rb') as f:
                        token2id = pickle.load(f)

                for token in unique_tokens:
                    if token not in token2id:
                        token2id[token] = len(token2id)
                with open(self.__token2id_file, 'wb') as f:
                    pickle.dump(token2id, f)
        elif self.__token2id_file is not None:
            with open(self.__token2id_file, 'rb') as f:
                token2id = pickle.load(f)

        if token2id is not None:
            for cfg in cfgs:
                cfg.replace_tokens(token2id)

        if self.__corpus_file is not None:
            with filelock.FileLock(self.__corpus_file + '.lock') as flock:
                with open(self.__corpus_file, 'a') as f:
                    for cfg in cfgs:
                        tokens = cfg.as_token_list()
                        tokens = [str(token) for token in tokens]
                        f.write(' '.join(tokens) + '\n')

        return cfgs


class TokenSeqExtractor(TokenCFGExtractor):
    def __init__(self, normalizer_kwargs, token2id_file=None, corpus_file=None,
                 record_token2id=False, **kwargs):
        super().__init__(normalizer_kwargs, token2id_file=token2id_file, corpus_file=corpus_file,
                         record_token2id=record_token2id, **kwargs)
        self._disassembler = BinaryNinja(TokenSeqNormalizer, normalizer_kwargs=normalizer_kwargs)
