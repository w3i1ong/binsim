from abc import abstractmethod, ABC
from torch import Tensor, nn


class LanguageModelBase(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Train the model with the given file.
        """
        pass

    @abstractmethod
    def train_dir(self, *args, **kwargs):
        """
        Train the model with all files in the given directory.
        """
        pass

    @abstractmethod
    def save(self, file):
        """
        Save the model to the given file.
        :param file: The file to save the model to.
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def load(file:str):
        """
        Load the model from the given file.
        :param file: The file to load the model from.
        :return:
        """
        pass

    @abstractmethod
    def as_torch_model(self) -> nn.Module:
        """
        Build a model that can be used in torch.
        :return: The torch model.
        """
        pass

    @abstractmethod
    def encode(self, data):
        """
        Encode the given data so that it can be used as input of the torch model.
        :param data: The data to be encoded, can be in any format you defined.
        :return:
        """
        pass
