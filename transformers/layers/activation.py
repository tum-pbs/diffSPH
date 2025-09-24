
import torch
import torch.nn as nn
from collections import OrderedDict

def getActivationFunctions():
    return ['elu', 'relu', 'hardtanh', 'hardswish', 'selu', 'celu', 'leaky_relu', 'prelu', 'rrelu', 'glu', 'gelu', 'logsigmoid', 'hardshrink', 'tanhshrink', 'softsign', 'softplus', 'softmin', 'softmax', 'softshrink', 'log_softmax', 'tanh', 'sigmoid', 'hardsigmoid', 'silu', 'mish']
def getActivationLayer(function: str, *args):
    if function == 'elu':
        return nn.ELU(*args)
    elif function == 'relu':
        return nn.ReLU(*args)
    elif function == 'hardtanh':
        return nn.Hardtanh(*args)
    elif function == 'hardswish':
        return nn.Hardswish(*args)
    elif function == 'selu':
        return nn.SELU(*args)
    elif function == 'celu':
        return nn.CELU(*args)
    elif function == 'leaky_relu':
        return nn.LeakyReLU(*args)
    elif function == 'prelu':
        return nn.PReLU(*args)
    elif function == 'rrelu':
        return nn.RReLU(*args)
    elif function == 'glu':
        return nn.GLU(*args)
    elif function == 'gelu':
        return nn.GELU(*args)
    elif function == 'logsigmoid':
        return nn.LogSigmoid(*args)
    elif function == 'hardshrink':
        return nn.Hardshrink(*args)
    elif function == 'tanhshrink':
        return nn.Tanhshrink(*args)
    elif function == 'softsign':
        return nn.Softsign(*args)
    elif function == 'softplus':
        return nn.Softplus(*args)
    elif function == 'softmin':
        return nn.Softmin(*args)
    elif function == 'softmax':
        return nn.Softmax(*args)
    elif function == 'softshrink':
        return nn.Softshrink(*args)
    elif function == 'log_softmax':
        return nn.LogSoftmax(*args)
    elif function == 'tanh':
        return nn.Tanh(*args)
    elif function == 'sigmoid':
        return nn.Sigmoid(*args)
    elif function == 'hardsigmoid':
        return nn.Hardsigmoid(*args)
    elif function == 'silu':
        return nn.SiLU(*args)
    elif function == 'mish':
        return nn.Mish(*args)
    elif function == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'Unknown activation function: {function}')
    

def getActivationFunctions():
    return ['elu', 'relu', 'hardtanh', 'hardswish', 'selu', 'celu', 'leaky_relu', 'prelu', 'rrelu', 'glu', 'gelu', 'logsigmoid', 'hardshrink', 'tanhshrink', 'softsign', 'softplus', 'softmin', 'softmax', 'softshrink', 'gumbel_softmax', 'log_softmax', 'tanh', 'sigmoid', 'hardsigmoid', 'silu', 'mish']
def getActivationFunction(function : str):
    return getattr(nn.functional, function)


from typing import Optional
def getActivationFromString(activation: Optional[str]):        
    if activation is None:
        activation_fn = nn.Identity()
        activationName = 'identity'
    else:
        activationName = activation.split('(')[0] if '(' in activation else activation
        activationArguments = () if '(' not in activation else activation[activation.index('(')+1:activation.index(')')].split(',')
        activationArguments = tuple([float(arg) for arg in activationArguments])
        activation_fn = getActivationLayer(activationName, *activationArguments)
        
    return activation_fn, activationName