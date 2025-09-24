    
from typing import Union, Tuple
def verbosePrint(message: str, verbose: bool, separator = False, width = 80, verbosePrefix = ''):
    if verbose:
        if separator:
            print('=' * width)
        print(f'{verbosePrefix}{message}')


        ################################################################################
        #                     Encode Edge Attributes for RPB                           #
        ################################################################################

def verboseBannerPrint(message: str, verbose: bool, width = 80):
    if verbose:
        print('=' * width)
        for line in message.split('\n'):
            print(f'#{line.center(width - 2)}#')
        print('=' * width)


from torch import Tensor
import torch

def verbosePrintSpatialTensorStats(tensor: Tensor, name: str = 'Tensor', verbose: bool = False, verbosePrefix: str = ''):
    if not verbose:
        return
    if tensor.numel() == 0:
        print(f'{verbosePrefix}{name} is empty')
        return
    print(f'{verbosePrefix}{name} shape: {tensor.shape}, min: {tensor.min().item()}, max: {tensor.max().item()}, mean: {tensor.mean().item()}, std: {tensor.std().item()}')
    for i in range(tensor.shape[-1]):
        print(f'{verbosePrefix}{name} dim {i}: min: {tensor[...,i].min().item()}, max: {tensor[...,i].max().item()}, mean: {tensor[...,i].mean().item()}, std: {tensor[...,i].std().item()}')
    lengths = torch.norm(tensor, dim=-1)
    print(f'{verbosePrefix}{name} lengths: min: {lengths.min().item()}, max: {lengths.max().item()}, mean: {lengths.mean().item()}, std: {lengths.std().item()}')
    

def shapeMatch(tensor: Tensor):        
    # the shape here is either [B,N,D] or [E,D] or [1,E,D]
    # we need to convert to an internal [E,D] shape
    if len(tensor.shape) == 3:
        mapped = True
        batches, entries, dim = tensor.shape
        matchedTensor = tensor.view(-1, dim)
        # verbosePrint(f'Input positions have batch dimension: {batches}', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
    elif len(tensor.shape) == 2:
        mapped = False
        entries, dim = tensor.shape
        batches = 1
        matchedTensor = tensor
        # verbosePrint('Input positions have no batch dimension', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
    else:
        raise ValueError(f'Input positions must be of shape [B,N,D] or [E,D], got {tensor.shape}')
    return matchedTensor, batches, entries, dim


from typing import List, Optional

def checkTensorShape(tensor: Tensor, expected_shape: List[str], shape_dict: dict, verbose: bool = False, logName: Optional[str] = None):
    if tensor is None:
        return
    # if verbose:
    #     name = f' for {logName}' if logName is not None else ''
    #     print(f'Checking tensor{name} shape: {tensor.shape} against expected: {expected_shape}')
    shape = tensor.shape
    if len(shape) != len(expected_shape):
        raise ValueError(f'Expected tensor to have {len(expected_shape)} dimensions, got {len(shape)} dimensions with shape {shape}')
    for i, dim in enumerate(expected_shape):
        if isinstance(dim, int):
            if shape[i] != dim:
                raise ValueError(f'Expected dimension {i} of tensor to have size {dim}, got {shape[i]}')
        elif '*' in dim or '//' in dim:
            LHS, RHS = dim.split('//') if '//' in dim else dim.split('*')
            if LHS.isdigit() and RHS.isdigit():
                lhs = int(LHS)
                rhs = int(RHS)
                if shape[i] % rhs != 0 or shape[i] // rhs != lhs:
                    raise ValueError(f'Expected dimension {i} of tensor to have size {lhs}*{rhs}, got {shape[i]}')  
            elif LHS.isdigit() and RHS in shape_dict:
                lhs = int(LHS)
                rhs = shape_dict[RHS]
                if rhs is not None and (shape[i] % rhs != 0 or shape[i] // rhs != lhs):
                    raise ValueError(f'Expected dimension {i} of tensor to have size {lhs}*{rhs} ({RHS}), got {shape[i]}')  
            elif LHS in shape_dict and RHS.isdigit():
                lhs = shape_dict[LHS]
                rhs = int(RHS)
                if lhs is not None and (shape[i] % rhs != 0 or shape[i] // rhs != lhs):
                    raise ValueError(f'Expected dimension {i} of tensor to have size {lhs} ({LHS})*{rhs}, got {shape[i]}')
            elif LHS in shape_dict and RHS in shape_dict:
                lhs = shape_dict[LHS]
                rhs = shape_dict[RHS]
                if lhs is not None and rhs is not None and (shape[i] % rhs != 0 or shape[i] // rhs != lhs):
                    raise ValueError(f'Expected dimension {i} of tensor to have size {lhs} ({LHS})*{rhs} ({RHS}), got {shape[i]}')
            else:
                raise ValueError(f'Unknown dimension specifier: {dim}')
        else:
            if dim.isdigit():
                expected_dim = int(dim)
                if shape[i] != expected_dim:
                    raise ValueError(f'Expected dimension {i} of tensor to have size {expected_dim}, got {shape[i]}')
            elif dim in shape_dict:
                expected_dim = shape_dict[dim]
                if expected_dim is not None and shape[i] != expected_dim:
                    raise ValueError(f'Expected dimension {i} of tensor to have size {expected_dim} ({dim}), got {shape[i]}')
            elif dim == '*':
                continue
            else:
                raise ValueError(f'Unknown dimension specifier: {dim}')
    if verbose:
        name = f' for {logName}' if logName is not None else ''
        print(f'Tensor{name} has expected shape: {shape}')

import copy
def mergeConfigWithKwargs(configClass, **kwargs):
    config = copy.deepcopy(configClass)
    for key, value in configClass.__dataclass_fields__.items():
        if str(key) not in kwargs:
            continue
        if isinstance(kwargs[str(key)], dict):
            for subkey, subvalue in kwargs[str(key)].items():
                if subkey in value.type.__dataclass_fields__:
                    setattr(config, subkey, subvalue)
        elif key in kwargs:
            setattr(config, key, kwargs[str(key)])
    return config
