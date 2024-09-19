from abc import ABC, abstractmethod
from torch import Tensor
from torch import nn
from typing import Tuple
class SamplerBase(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs)->Tensor:
        pass
