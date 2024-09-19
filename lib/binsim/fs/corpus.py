import os
from enum import Enum
import shutil


class CorpusGraphType(Enum):
    TokenCFG = 'TokenCFG'
    PDG = 'PDG'

    def __str__(self):
        return self.value


class CorpusDir:
    DB = 'db'
    GRAPH = 'graph'
    DATA = 'data'
    CORPUS = 'corpus'
    MODEL = 'model'
    MIDDLE = 'cache'

    def __init__(self, root, name, graph_type: CorpusGraphType, data_form, init=False):
        """
        Directory class used to simplify the corpus operations.
        :param root: The root directory of the corpus.
        :param name: The corpus name to be processed.
        :param graph_type: Which type of graph the corpus use.
        :param data_form: Which kind of data form the corpus will use.
        :param init: Whether to initialize necessary directories in the corpus.
        """
        self._root = root
        self._name = name
        self._graph_type = graph_type
        self._data_form = data_form
        if init:
            self.initialize()

    def initialize(self):
        os.makedirs(self.database_dir,exist_ok=True)
        os.makedirs(self.graph_dir,exist_ok=True)
        os.makedirs(self.data_dir,exist_ok=True)
        os.makedirs(self.corpus_dir,exist_ok=True)
        os.makedirs(self.model_dir,exist_ok=True)

    def get_graph_dir(self, arch, filename=''):
        return f'{self._root}/{self.GRAPH}/{self._graph_type}/{self._name}/{arch}/{filename}'

    @property
    def data_dir(self):
        return f'{self._root}/{self.DATA}/{self._graph_type}/{self._name}'

    @property
    def graph_dir(self):
        return f'{self._root}/{self.GRAPH}/{self._graph_type}/{self._name}'

    @property
    def cache_dir(self):
        return f'{self.data_dir}/{self.MIDDLE}/{self._data_form}'

    @property
    def corpus_dir(self):
        return f'{self.data_dir}/{self.CORPUS}/{self._data_form}'

    @property
    def model_dir(self):
        return f'{self.data_dir}/{self.MODEL}/{self._data_form}'

    def get_corpus_file(self, arch=None):
        if arch is None:
            return f'{self.corpus_dir}/all-in-one.txt'
        else:
            return f'{self.corpus_dir}/{arch}.txt'

    def get_cache_file_for(self, arch, filename=''):
        return f'{self.cache_dir}/{arch}/{filename}'

    def get_cache_files_for(self, arch):
        return [self.get_cache_file_for(arch, filename=file) for file in os.listdir(f'{self.cache_dir}/{arch}')]

    def get_graph_file_for(self, arch, filename):
        return f'{self.graph_dir}/{arch}/{filename}'

    def get_db_file_for(self, arch, filename):
        return f'{self.database_dir}/{arch}/{filename}'



    def get_model(self, model_name):
        return f'{self.model_dir}/{model_name}'

    @property
    def database_dir(self):
        return f'{self._root}/{self.DB}/'

    def remove(self):
        graph_dir = self.graph_dir
        data_dir = self.data_dir
        if os.path.exists(graph_dir):
            shutil.rmtree(graph_dir)
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
