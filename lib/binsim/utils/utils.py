import torch
from typing import Union, List
from torch.nn.functional import pad as pad_tensor


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


def get_loss_by_name(name: str, **kwargs):
    match name:
        case 'csl':
            assert 'margin' in kwargs, f"You must provide margin for cel loss!"
            from binsim.neural.nn.siamese.loss import CosineEmbeddingLoss
            return CosineEmbeddingLoss(margin=kwargs['margin'])
        case 'mse':
            from binsim.neural.nn.siamese.loss import MSELoss
            return MSELoss()
        case "info-nce-loss":
            from binsim.neural.nn.siamese.loss import InfoNCELoss
            return InfoNCELoss(**kwargs)
        case "triplet":
            from binsim.neural.nn.siamese.loss import TripletLoss
            return TripletLoss(**kwargs)
        case "safe-loss":
            from binsim.neural.nn.siamese.loss import SAFELoss
            return SAFELoss(**kwargs)
        case "asteria-loss":
            from binsim.neural.nn.siamese.loss import AsteriaLoss
            return AsteriaLoss(**kwargs)
    raise NotImplementedError(f"The loss function {name} is not supported!")

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

def get_sampler_by_name(name: str, kwargs):
    from binsim.neural.nn.siamese.sampler.margin_sampler import MarginSampler
    match name:
        case "semi-hard-pair":
            return MarginSampler(margin=kwargs['margin'], triple=False)
        case "semi-hard-triplet":
            return MarginSampler(margin=kwargs['margin'], triple=True)
        case _:
            raise ValueError(f"Unsupported sampler: {name}.")

def get_distance_by_name(name:str, kwargs):
    from binsim.neural.nn.siamese.distance import CosineDistance, EuclidianDistance, AsteriaDistance
    match name:
        case "cosine":
            return CosineDistance(**kwargs)
        case "euclid":
            return EuclidianDistance(**kwargs)
        case "asteria-distance":
            return AsteriaDistance(**kwargs)
        case _:
            raise ValueError(f"Unsupported distance: {name}.")

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
    from binsim.neural.nn.model import Gemini, BinMamba, CFGFormer, I2vRNN, SAFE, AlphaDiff, JTrans, RCFG2Vec, GraphMatchingNet, Asteria, ASTSAFE
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
        case 'astsafe':
            return ASTSAFE
        case 'GMN' | 'GraphMatchingNet':
            return GraphMatchingNet
        case 'asteria':
            return Asteria
        case "binmamba":
            return BinMamba
        case "cfgformer":
            return CFGFormer
        case _:
            raise ValueError(f"Unsupported model type: {name}.")

def load_pretrained_model(model_name, weights_file, device=None):
    model_type = get_model_type(model_name)
    model = model_type.from_pretrained(weights_file, device)
    return model
