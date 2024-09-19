import os, pickle
import random
from ..base import LanguageModelBase
from binsim.neural.lm.palmtree.bert import BERT, BERTTrainer
from torch.utils.data import DataLoader, Dataset
from ....disassembly.binaryninja import ProgramDependencyGraph


class InstructionPairDataset(Dataset):
    def __init__(self, data_file: str):
        super(InstructionPairDataset, self).__init__()
        with open(data_file, 'rb') as f:
            pickle.load(f)

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class PalmTree(LanguageModelBase):
    def __init__(self, vocab_size: int,
                 epochs: int = 20,
                 hidden_size: int = 128,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 lr: float = 1e-5,
                 betas: tuple = (0.9, 0.999),
                 weight_decay: float = 0.0,
                 with_cuda: bool = True,
                 cuda_devices=None,
                 log_freq: int = 100):
        super(PalmTree, self).__init__()
        self._model = BERT(vocab_size, hidden_size, num_layers, num_heads, dropout)

        if cuda_devices is None:
            cuda_devices = [0]
        self._epochs = epochs
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout
        self._lr = lr
        self._betas = betas
        self._weight_decay = weight_decay
        self._with_cuda = with_cuda
        self._cuda_devices = cuda_devices
        self._log_freq = log_freq

    def train(self, train_data, valid_data):
        trainer = BERTTrainer(bert=self._model,
                              vocab_size=self._vocab_size,
                              train_dataloader=train_data,
                              eval_dataloader=valid_data,
                              lr=self._lr,
                              betas=self._betas,
                              weight_decay=self._weight_decay,
                              with_cuda=self._with_cuda,
                              cuda_devices=self._cuda_devices,
                              log_freq=self._log_freq)
        for e in range(self._epochs):
            trainer.train(epoch=e)

    def _load_data(self, filename) -> DataLoader:
        # prepare data for training, validation and testing
        with open(filename, 'rb') as f:
            functions = pickle.load(f)
        cfg_data = []
        dfg_data = []
        for function in functions:
            cfg_data.append(self._load_control_flow_data(function))
            dfg_data.append(self._load_data_flow_data(function))
        return cfg_data, dfg_data

    def _load_control_flow_data(self, function: ProgramDependencyGraph):
        return self._flow_random_walk(function.features, function.adj_list, 40)

    def _flow_random_walk(self, instructions, adj_list, max_path_length: int):
        sequence = []
        for addr in instructions:
            flow_path = []
            for _ in range(max_path_length):
                instruction = instructions[addr]
                flow_path.append(instruction)
                if len(adj_list[addr]) == 0:
                    break
                addr = random.choice(adj_list[addr])
            sequence.append(flow_path)
        return sequence

    def _load_data_flow_data(self, function: ProgramDependencyGraph):
        return self._flow_random_walk(function.features, function.data_flow, 40)

    def train_dir(self, src_dir):
        files = os.listdir(src_dir)

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump([
                self._epochs,
                self._vocab_size,
                self._hidden_size,
                self._num_layers,
                self._num_heads,
                self._dropout,
                self._lr,
                self._betas,
                self._weight_decay,
                self._with_cuda,
                self._cuda_devices,
                self._log_freq
            ], f)
            pickle.dump(self._model.state_dict(), f)

    def load(self, file):
        with open(file, 'rb') as f:
            self._epochs, self._vocab_size, self._hidden_size, self._num_layers, \
            self._num_heads, self._dropout, self._lr, self._betas, self._weight_decay, \
            self._with_cuda, self._cuda_devices, self._log_freq = pickle.load(f)
            self._model.load_state_dict(pickle.load(f))

    def as_torch_model(self):
        return self._model

    def encode(self, data):
        pass
