import os
import shutil
import gensim
import pickle
import numpy
import torch
from ..base import LanguageModelBase
from torch import nn, Tensor
from typing import List, Union, Tuple, Iterable
from gensim.models.asm2vec import Asm2Vec as Asm2VecModel, Function, Instruction
from binsim.disassembly.binaryninja import TokenCFG
from .data import ListFunctionFromFile, ListFunctionFromFiles
from typing import Iterator
# get pack_padded_sequence and pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from gensim.models.callbacks import CallbackAny2Vec
import time


class Asm2VecCallback(CallbackAny2Vec):
    def __init__(self, logger=None):
        super().__init__()
        self._epoch_start_time = None
        self._logger = logger
        self._total_time = 0
        self._epoch_counter = 0

    def on_epoch_begin(self, model):
        self._epoch_start_time = time.time()
        self._logger.info(f"Epoch {self._epoch_counter} start.")

    def on_epoch_end(self, model):
        self._logger.info(f"Epoch {self._epoch_counter} finished, time cost: {time.time() - self._epoch_start_time} s.")
        self._total_time = self._total_time + time.time() - self._epoch_start_time
        self._epoch_counter += 1

    def on_train_begin(self, model):
        self._logger.info(f"Training for Asm2Vec model start. The model will be trained for {model.epochs} epochs.")

    def on_train_end(self, model):
        self._logger.info(f"Training for Asm2Vec model finished. Total time cost: {self._total_time} s.")


class Asm2Vec(LanguageModelBase):

    def __init__(self, embed_dim=200,
                 min_count=0,
                 window_size=1,
                 epoch=10,
                 workers=8,
                 logger=None):
        """
        Train Asm2Vec model on corpus that consists of TokenCFGs.
        :param embed_dim: The dimension of the word embedding.
        :param min_count: The minimum count of a token to be included in the vocabulary.
        :param window_size: The window size of the Word2Vec model.
        :param epoch: The number of epochs to train the model.
        :param workers: The number of workers to train the model.
        """
        super().__init__()
        self._embed_dim = embed_dim
        self._min_count = min_count
        self._window_size = window_size
        self._epoch = epoch
        self._workers = workers
        self._embeddings = None
        self._ins2idx = None
        self._model: Asm2VecModel = None
        if logger is None:
            self._callback = None
        else:
            self._callback = Asm2VecCallback(logger)

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def weights(self) -> torch.Tensor:
        return torch.from_numpy(self._embeddings)

    @property
    def ins2idx(self):
        return self._ins2idx

    @property
    def token_embeddings(self):
        return self._model.wv.vectors

    @property
    def tokens(self):
        return self._model.wv.index_to_key

    @property
    def function_embeddings(self):
        return self._model.dv.vectors

    @property
    def functions(self):
        return self._model.dv.index_to_key

    def train(self, functions: Iterable[Function]):
        """
        Train Word2Vec model with given corpus file.
        :param functions: The path of corpus file.
        :return:
        """
        # train Word2Vec model with the generated corpus file
        self._model = gensim.models.Asm2Vec(functions,
                                            embed_dim=self._embed_dim,
                                            min_count=self._min_count,
                                            window=self._window_size,
                                            epochs=self._epoch,
                                            workers=self._workers,
                                            callbacks=[self._callback])

    def train_dir(self, src_dir):
        """
        Train Word2Vec model with all files in the given corpus directory.
        :param src_dir: The path of corpus directory.
        :return:
        """
        raise NotImplementedError("train_dir has not been implemented for Asm2Vec.")

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.__class__.__name__, f)
            pickle.dump(self, f)

    @staticmethod
    def load(file) -> 'Asm2Vec':
        with open(file, 'rb') as f:
            name = pickle.load(f)
            assert name == Asm2Vec.__name__
            result = pickle.load(f)
        return result

    def generate_embedding(self, functions: Iterable[Function], epochs=1) -> numpy.array:
        return self._model.infer_vector(functions, 0.025, 0.001, epochs=epochs)

    def as_torch_model(self, freeze=False) -> nn.Module:
        raise NotImplementedError("as_torch_model has not been implemented for Asm2Vec.")

    def encode(self, token_lists: List[List[str]], with_length=False) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("encode has not been implemented for Asm2Vec.")

    def __repr__(self):
        return f"<Asm2Vec embed_dim={self._embed_dim} min_count={self._min_count} " \
               f"window_size={self._window_size}>"

    def __str__(self):
        return self.__repr__()
