import os


class RecordDir:
    LOG_DIR = "log"
    TRAIN_DIR = "train"
    TEST_DIR = "test"
    MODEL_FILE = "model.pkl"

    def __init__(self, root, model_name):
        self._root = f'{root}/{model_name}'
        self._init_dir()

    def _init_dir(self):
        os.makedirs(self._root, exist_ok=True)
        os.makedirs(self.log_record, exist_ok=True)
        os.makedirs(self.train_record, exist_ok=True)
        os.makedirs(self.test_record, exist_ok=True)

    @property
    def model_file(self) -> str:
        return f'{self._root}/{self.MODEL_FILE}'

    @property
    def train_record(self) -> str:
        return f'{self._root}/{self.TRAIN_DIR}'

    @property
    def test_record(self) -> str:
        return f'{self._root}/{self.TEST_DIR}'

    @property
    def log_record(self) -> str:
        return f'{self._root}/{self.LOG_DIR}'
