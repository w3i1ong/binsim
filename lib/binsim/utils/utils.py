import torch
from torch import nn
from typing import Union
from torch.nn.functional import pad as pad_tensor
from binsim.neural.nn.distance import PairwiseCosineDistance
from typing import Tuple, List


def diagonalize_matrices(matrices: List[torch.Tensor], row=None, col=None):
    """
    Diagonalize matrices, i.e. make the matrices diagonal matrices.
    :param matrices: The matrices to be diagonalized.
    :param row:
    :param col:
    :return: The diagonalized matrices.
    """
    if row is None:
        row = sum([matrix.size(0) for matrix in matrices])
    if col is None:
        col = sum([matrix.size(1) for matrix in matrices])
    assert sum([matrix.size(0) for matrix in matrices]) <= row, \
        "The row dimension of the matrices is larger than the specified row dimension."
    assert sum([matrix.size(1) for matrix in matrices]) <= col, \
        "The column dimension of the matrices is larger than the specified column dimension."

    normalized_matrices = []

    base = 0
    for matrix in matrices:
        assert matrix.shape[0] == matrix.shape[1], "The matrices must be square matrices."
        normalized_matrices.append(pad_tensor(matrix, [base,
                                                       col - base - matrix.shape[0]], value=0))
        base += matrix.shape[0]
    result = torch.concat(normalized_matrices, dim=0)
    if result.shape[0] < row:
        result = pad_tensor(result, [0, 0, 0, row - result.shape[0]], value=0)
    return result


def get_loss_by_name(name: str, **kwargs) -> nn.Module:
    if name == 'cel':
        assert 'margin' in kwargs, f"You must provide margin for cel loss!"
        return nn.CosineEmbeddingLoss(margin=kwargs['margin'])
    elif name == 'mse':
        return nn.CosineEmbeddingLoss()
    raise NotImplementedError(f"The loss function {name} is not supported!")


def get_distance_by_name(name) -> Tuple[nn.Module, nn.Module]:
    if name == 'cosine':
        return nn.CosineSimilarity(dim=1), PairwiseCosineDistance()
    elif name == 'euclid':
        return nn.PairwiseDistance(), PairwiseCosineDistance()
    raise NotImplementedError(f"The distance function {name} is not supported!")


def get_optimizer_by_name(name: str):
    name = name.lower()
    assert name in ['sgd', 'adam', 'adagrad', 'rmsprop', 'adamw'], f"The optimizer {name} is not supported!"
    if name == 'sgd':
        return torch.optim.SGD
    elif name == 'adam':
        return torch.optim.Adam
    elif name == 'adagrad':
        return torch.optim.Adagrad
    elif name == 'rmsprop':
        return torch.optim.RMSprop
    elif name == 'adamw':
        return torch.optim.AdamW
    raise ValueError(f"The optimizer {name} is not supported!")


def dict2str(d: Union[dict, list], level=0):
    if isinstance(d, list):
        if len(d) == 0:
            return '[]'
        result = "[\n"
        for item in d:
            result += "\t" * (level + 1) + f"{dict2str(item, level + 1)}\n"
        result += "\t" * level + "]"
        return result
    elif isinstance(d, dict):
        if len(d) == 0:
            return '{}'
        result = "{\n"
        for key, value in d.items():
            result += "\t" * (level + 1) + f'{key}:' + dict2str(value, level=level + 1) + '\n'
        result += "\t" * level + "}"
        return result
    return str(d)

def get_model_type(name: str):
    from binsim.neural.nn.model import Gemini, I2vAtt, I2vRNN, SAFE, AlphaDiff, JTrans, RCFG2Vec, GraphMatchingNet, Asteria
    match name.lower():
        case 'alphadiff':
            return AlphaDiff
        case 'gemini':
            return Gemini
        case 'i2vrnn' | "i2v_rnn":
            return I2vRNN
        case 'rcfg2vec':
            return RCFG2Vec
        case 'safe':
            return SAFE
        case 'jtrans':
            return JTrans
        case 'GMN' | 'GraphMatchingNet':
            return GraphMatchingNet
        case 'asteria':
            return Asteria
        case _:
            raise ValueError(f"Unsupported model type: {name}.")

def load_pretrained_model(model_name, weights_file, device=None):
    model_type = get_model_type(model_name)
    model = model_type.from_pretrained(weights_file, device)
    return model
