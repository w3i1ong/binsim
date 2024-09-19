import os, time, torch, shutil, gensim, pickle
import numpy as np
from torch import nn, Tensor
from ..base import LanguageModelBase
from binsim.utils import init_logger
from typing import List, Union, Tuple
from gensim.models.callbacks import CallbackAny2Vec

logger = init_logger(__name__)


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training"""

    def __init__(self):
        self.epoch = 0
        self.start_time = None

    def on_epoch_begin(self, model):
        self.start_time = time.time()
        print(f"Epoch #{self.epoch} start")

    def on_epoch_end(self, model):
        end_time = time.time()
        duration = end_time - self.start_time
        logger.info(f"Epoch #{self.epoch} end")
        logger.info(f"Time taken for Epoch #{self.epoch}: {duration:.2f} seconds")
        self.epoch += 1


class Ins2vec(LanguageModelBase):

    def __init__(self, embed_dim=100,
                 min_count=10,
                 window_size=8,
                 epoch=10,
                 workers=8,
                 mode='skip-gram',
                 hs=True):
        """
        Train Word2Vec model on corpus that consists of TokenCFGs.
        :param embed_dim: The dimension of the word embedding.
        :param min_count: The minimum count of a token to be included in the vocabulary.
        :param window_size: The window size of the Word2Vec model.
        :param epoch: The number of epochs to train the model.
        :param workers: The number of workers to train the model.
        :param mode: The mode of the Word2Vec model, can be one of 'skip-gram' or 'cbow'.
        :param hs: Whether to use hierarchical softmax.
        """
        super().__init__()
        self._embed_dim = embed_dim
        self._min_count = min_count
        self._window_size = window_size
        self._epoch = epoch
        self._workers = workers
        self._embeddings = None
        self._ins2idx = None
        self._mode = 1 if mode == 'skip-gram' else 0
        self._hs = int(hs)

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def weights(self) -> torch.Tensor:
        return torch.from_numpy(self._embeddings)

    @property
    def ins2idx(self):
        return self._ins2idx

    def train(self, src_file):
        """
        Train Word2Vec model with given corpus file.
        :param src_file: The path of corpus file.
        :return:
        """
        assert os.path.exists(src_file), f'{src_file} does not exist.'
        if os.path.isfile(src_file):
            logger.info(f"Training Word2Vec model with file: {src_file}")
            corpus_file = gensim.models.word2vec.LineSentence(src_file)
        elif os.path.isdir(src_file):
            logger.info(f"Training Word2Vec model with directory: {src_file}")
            corpus_file = gensim.models.word2vec.PathLineSentences(src_file)
        else:
            raise ValueError('src_file must be a file or a directory')
        # train Word2Vec model with the generated corpus file
        model = gensim.models.Word2Vec(corpus_file,
                                       vector_size=self._embed_dim,
                                       min_count=self._min_count,
                                       window=self._window_size,
                                       epochs=self._epoch,
                                       workers=self._workers,
                                       sg=self._mode,
                                       hs=self._hs,
                                       callbacks=[EpochLogger()])
        self._ins2idx = model.wv.key_to_index
        self._embeddings = model.wv.vectors

    def train_dir(self, src_dir):
        """
        Train Word2Vec model with all files in the given corpus directory.
        :param src_dir: The path of corpus directory.
        :return:
        """
        self.train(src_dir)

    @staticmethod
    def _merge_dir(src_dir, dst_file):
        """
        Merge all corpus files in the given directory into one file.
        :param src_dir: The directory that contains all corpus files.
        :param dst_file: The file to store the merged corpus.
        :return:
        """
        files = os.listdir(src_dir)
        with open(dst_file, 'wb') as f:
            for file in files:
                with open(os.path.join(src_dir, file), 'rb') as f1:
                    shutil.copyfileobj(f1, f)
                    f.write(b'\n')

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.__class__.__name__, f)
            pickle.dump(self, f)

    @staticmethod
    def load(file) -> 'Ins2vec':
        with open(file, 'rb') as f:
            name = pickle.load(f)
            assert name == Ins2vec.__name__
            result = pickle.load(f)
        return result

    def as_torch_model(self, freeze=False, device=None, dtype=None):
        from binsim.neural.nn.layer.embedding import SparseEmbedding
        new_ins2idx = {}
        for key, value in self._ins2idx.items():
            new_ins2idx[int(key)] = value
        return SparseEmbedding(self._embeddings, new_ins2idx, freeze=freeze, device=device, dtype=dtype)

    def encode(self, token_lists: List[List[str]], with_length=False) -> Tuple[Tensor, Tensor]:
        """
        Encode the given data so that it can be used as input of the torch model.
        :param token_lists: token lists.
        :param with_length:  Whether to return the length of each instruction list.
        :return: The encoded token lists and the length of each token lists.
        """
        encoded = []
        for ins_list in token_lists:
            encoded.append([self._ins2idx[ins] for ins in ins_list])
        # padding to the same length
        max_len = max([len(ins_list) for ins_list in encoded])
        length_list = [len(ins_list) for ins_list in encoded]
        for ins_list in encoded:
            ins_list.extend([0] * (max_len - len(ins_list)))
        if with_length:
            return Tensor(encoded), Tensor(length_list)
        else:
            return Tensor(encoded)

    def __repr__(self):
        return 'ins2vec'

    def __str__(self):
        return 'ins2vec'


if __name__ == '__main__':
    ins2vec = Ins2vec()
    ins2vec.train("./test.py")
