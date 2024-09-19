from torch import nn


def get_activation_by_name(name):
    name2activation = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'softmax': nn.Softmax()
    }
    return name2activation[name]
