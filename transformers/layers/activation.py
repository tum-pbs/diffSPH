
import torch
import torch.nn as nn
from collections import OrderedDict

def getActivationFunctions():
    return ['elu', 'relu', 'hardtanh', 'hardswish', 'selu', 'celu', 'leaky_relu', 'prelu', 'rrelu', 'glu', 'gelu', 'logsigmoid', 'hardshrink', 'tanhshrink', 'softsign', 'softplus', 'softmin', 'softmax', 'softshrink', 'log_softmax', 'tanh', 'sigmoid', 'hardsigmoid', 'silu', 'mish']
def getActivationLayer(function: str):
    if function == 'elu':
        return nn.ELU()
    elif function == 'relu':
        return nn.ReLU()
    elif function == 'hardtanh':
        return nn.Hardtanh()
    elif function == 'hardswish':
        return nn.Hardswish()
    elif function == 'selu':
        return nn.SELU()
    elif function == 'celu':
        return nn.CELU()
    elif function == 'leaky_relu':
        return nn.LeakyReLU()
    elif function == 'prelu':
        return nn.PReLU()
    elif function == 'rrelu':
        return nn.RReLU()
    elif function == 'glu':
        return nn.GLU()
    elif function == 'gelu':
        return nn.GELU()
    elif function == 'logsigmoid':
        return nn.LogSigmoid()
    elif function == 'hardshrink':
        return nn.Hardshrink()
    elif function == 'tanhshrink':
        return nn.Tanhshrink()
    elif function == 'softsign':
        return nn.Softsign()
    elif function == 'softplus':
        return nn.Softplus()
    elif function == 'softmin':
        return nn.Softmin()
    elif function == 'softmax':
        return nn.Softmax()
    elif function == 'softshrink':
        return nn.Softshrink()
    elif function == 'log_softmax':
        return nn.LogSoftmax()
    elif function == 'tanh':
        return nn.Tanh()
    elif function == 'sigmoid':
        return nn.Sigmoid()
    elif function == 'hardsigmoid':
        return nn.Hardsigmoid()
    elif function == 'silu':
        return nn.SiLU()
    elif function == 'mish':
        return nn.Mish()
    elif function == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'Unknown activation function: {function}')
    

def getActivationFunctions():
    return ['elu', 'relu', 'hardtanh', 'hardswish', 'selu', 'celu', 'leaky_relu', 'prelu', 'rrelu', 'glu', 'gelu', 'logsigmoid', 'hardshrink', 'tanhshrink', 'softsign', 'softplus', 'softmin', 'softmax', 'softshrink', 'gumbel_softmax', 'log_softmax', 'tanh', 'sigmoid', 'hardsigmoid', 'silu', 'mish']
def getActivationFunction(function : str):
    return getattr(nn.functional, function)
