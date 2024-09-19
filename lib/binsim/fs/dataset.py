import os


class DatasetDir:
    CACHE_DIR = 'cache'
    GRAPH_DIR = 'graph'
    DATA_DIR = 'data'
    LOG_DIR = 'log'
    SUBSET_TRAIN = 'train'
    SUBSET_VAL = 'validation'
    SUBSET_TEST = 'test'

    def __init__(self, root: str, graph_type: str, dataset_name: str, merge_name):
        self._root = root
        self._graph_type = graph_type
        self._dataset_name = dataset_name
        self._merge_name = merge_name
        self._init_dir()

    def _init_dir(self):
        if not os.path.exists(self._root):
            os.makedirs(self._root, exist_ok=True)
        assert not os.path.isfile(self._root), f"The root directory is a file:{self._root}."
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    @property
    def cache_dir(self)->str:
        return os.path.join(self._root, self.CACHE_DIR)

    @property
    def graph_dir(self)->str:
        return os.path.join(self._root, self.GRAPH_DIR, str(self._graph_type), self._dataset_name)

    @property
    def data_dir(self)->str:
        return os.path.join(self._root, self.DATA_DIR, str(self._graph_type), self._dataset_name, self._merge_name)

    @property
    def log_dir(self)->str:
        return os.path.join(self._root, self.LOG_DIR)

    def rel_to_data_dir(self, rel:str)->str:
        return os.path.join(self.data_dir, rel)

    def rel_to_cache_dir(self, rel):
        return os.path.join(self.cache_dir, rel)

    def rel_to_graph_dir(self, rel):
        return os.path.join(self.graph_dir, rel)

    def rel_to_log_dir(self, rel):
        return os.path.join(self.log_dir, rel)

    def _check_subset_name(self, subset_name):
        assert subset_name in [self.SUBSET_TRAIN, self.SUBSET_VAL, self.SUBSET_TEST], \
            f"Invalid subset name:{subset_name}. The subset name must be one of " \
            f"{{{self.SUBSET_TRAIN}, {self.SUBSET_VAL} , {self.SUBSET_TEST}}}."

    def get_data_file(self, subset_name) -> str:
        self._check_subset_name(subset_name)
        return os.path.join(self.data_dir, f'{subset_name}.pkl')
