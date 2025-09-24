# from distutils import config
import warnings
import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional
from .basisFunctions import basisEncoderLayer, evalBasisFunction
from .mlp import getDefaultMLPDict, buildMLPwDict
from .networkUtil import verbosePrint, verboseBannerPrint

from typing import Optional, Union, Tuple, List
from dataclasses import dataclass, field
from .activation import getActivationFromString

_ = """
The basis encoding layer acts as part of the absolute and relative position biases and  can be treated as part of the continuous convolution operation.

The input is a tensor that is either of shape
- [batch_size, num_nodes, dim] for absolute position encoding
- [num_edges, dim] for relative position encoding

The latter may also be padded to [1, num_edges, dim] for convenience and both inputs are supported and internally we begin by mapping to [*, entries, dim] ([B,N,D]) where * is any number of leading dimensions.

The basis encoder then works by first applying a basis function to each input dimension, i.e., for each input x_i, it computes a set of basis functions {f_1(x_i), f_2(x_i), ..., f_k(x_i)} where k is the number of basis terms. The shape of each basis function is [*, entries, basisTerms] and there are [dim] such basis functions.

The next step is combining the basis functions across the input dimensions. This can be done in several ways, specified by the mode parameter:
- 'cat': Concatenate the basis functions along the last dimension, resulting in a shape of [*, entries, dim * basisTerms].
- 'sum': Sum the basis functions across the input dimensions, resulting in a shape of [*, entries, basisTerms].
- 'prod': Compute the product of the basis functions across the input dimensions, resulting in a shape of [*, entries, basisTerms].
- 'outer': Compute the outer product of the basis functions, which first results in a shape of [*, entries, b,...,b] (dim times) and then is flattened to [*, entries, basisTerms^dim].
- 'i', 'j', 'k': Select the basis functions corresponding to the first, second, or third input dimension respectively. This is only valid for dim=1,2,3 and results in a shape of [*, entries, basisTerms].

On top of this basis encoding layer, we can then apply a linear transformation to map the output to the desired dimension for use in attention mechanisms or other parts of the model and alternatively use an MLP for this step for a non-linear mapping. No activation function is applied within the basis encoding layer itself, as the choice of activation may depend on the specific application and is typically handled in subsequent layers.

Forward pass inputs:
- positions: Tensor of shape [batch_size, num_nodes, dim] or [num_edges, dim] representing the positions to be encoded.
Forward pass outputs:
- encoded_positions: Tensor of shape determined by the mode parameter, representing the basis-encoded positions.

Configuration parameters:
- spatial_dim (int): The dimensionality of the input positions (e.g., 1 for 1D, 2 for 2D, 3 for 3D).
- basis_terms (int): The number of basis functions to use for encoding.
- basis_function (str): The type of basis function to use (e.g., 'ffourier', 'chebyshev', 'legendre').
- mode (str): The method for combining basis functions across dimensions ('cat', 'sum', 'prod', 'outer', 'i', 'j', 'k').
- project_out (bool): Whether to apply a projection to the output of the basis encoding layer.
- project_linear (bool): If project_out is True, whether to use a linear layer for projection. If False, an MLP is used.
- project_mlp_properties (dict): If project_out is True and project_linear is False, a dictionary specifying the properties of the MLP (e.g., number of layers, hidden units, activation functions).
- out_dim (int): The dimensionality of the output after projection, if project_out is True.

The layer also provides an outputShape attribute that indicates the shape of the output tensor as returned by the forward pass, excluding any leading batch dimensions. This is useful for understanding how the basis encoding transforms the input data.

As an additional option skip_basis can be set to True, which will bypass the basis encoding and only apply the projection if enabled. This can be useful for ablation studies or when testing the impact of the basis encoding on model performance.
"""

"""
Configuration class for Position Encoding settings.

This class encapsulates various parameters that control the behavior of position encoding in a model. this includes absolute _and_ relative position biases (APB and RPB). The only difference is the argument that they are fed with during a forward pass.
"""
@dataclass
class BasisEncoderConfig:
    spatial_dim:        int             = field(default=3,          metadata={"help": "Spatial dimensionality of the input positions"})
    
    base_encoding:      bool            = field(default=True,       metadata={"help": "Use basis function encoding"})
    base_function:      Union[str,List[str]] = field(default='fourier',  metadata={"help": "Basis function type, can be a single string or a list of strings for each dimension"})
    base_terms:         Union[int,List[int]] = field(default=16,         metadata={"help": "Number of basis functions"})
    base_mode:          str             = field(default='cat',      metadata={"help": "Basis function mode"})
    base_projection:    str             = field(default='cartesian',    metadata={"help": "Projection of the input positions before basis encoding. Options are 'cartesian', 'spherical', 'preserving'"})
    base_scaling:       bool            = field(default=False,      metadata={"help": "Apply a learnable scaling matrix (linear) to the positions after basis encoding"})
    
    projection:         bool            = field(default=False,      metadata={"help": "Project output of basis encoding"})
    projection_linear:  bool            = field(default=True,       metadata={"help": "Use linear encoding for APB"})
    projection_mlp:     Optional[dict]  = field(default=None,       metadata={"help": "MLP architecture for APB"})
    projection_dim:     Optional[int]   = field(default=None,       metadata={"help": "Output dimension of APB if projection is used"})

    normalize_positions:bool            = field(default=False,      metadata={"help": "If set to true, apply a manual normalization to the input to be within radius 1 and center 0."})
    clamp_positions:    bool            = field(default=False,      metadata={"help": "If set to true, clamp the input positions to be within radius 1 and center 0."})
    clamp_min:          float           = field(default=-1.0,       metadata={"help": "Minimum value for clamping positions"})
    clamp_max:          float           = field(default= 1.0,       metadata={"help": "Maximum value for clamping positions"})
    activation:         Optional[str]   = field(default=None,       metadata={"help": "Activation function to apply after position encoding"})
    layer_norm:         bool            = field(default=False,      metadata={"help": "Apply layer normalization to the input positions before encoding"})

def updateBasisEncoderConfig(config: BasisEncoderConfig, **kwargs) -> BasisEncoderConfig:
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f'Unknown configuration parameter: {key}')
    return config

import itertools
from math import prod
import copy
from .networkUtil import shapeMatch, verbosePrintSpatialTensorStats

def checkBasisDim(
    spatial_dim: int,
    basis_terms: Union[int, List[int]],
    basis_function: Union[str, List[str]],
    basis_mode: str
) -> Tuple[List[int], List[str]]:
    if isinstance(basis_terms, list):
        if len(basis_terms) != spatial_dim:
            raise ValueError('If basis_terms is a list, its length must match spatial_dim')
        base_terms: List[int] = basis_terms
    elif isinstance(basis_terms, int):
        base_terms: List[int] = [basis_terms] * spatial_dim
    else:
        raise ValueError('base_terms must be an int or a list of ints')

    if isinstance(basis_function, list):
        if len(basis_function) != spatial_dim:
            raise ValueError('If basis_function is a list, its length must match spatial_dim')
        base_function: List[str] = basis_function
    else:
        base_function: List[str] = [basis_function] * spatial_dim

    return base_terms, base_function

"""Compute the output shape of the BasisEncoder given a BasisEncoderConfig. This is useful for determining the expected output dimensions without instantiating the layer."""
def computeBasisEncoderOutputShape(
    config: BasisEncoderConfig = BasisEncoderConfig(),
    
    verbose: bool = False,
    verbosePrefix: str = '',
    **kwargs
):
    config = copy.deepcopy(config)
    config = updateBasisEncoderConfig(config, **kwargs)
    
    base_terms, base_function = checkBasisDim(config.spatial_dim, config.base_terms, config.base_function, config.base_mode)
    
    basisShape = [config.spatial_dim]
    if config.base_encoding:
        if config.base_mode == 'cat':
            basisShape = [sum(base_terms)]
        elif config.base_mode == 'sum' or config.base_mode == 'prod':
            if min(base_terms) != max(base_terms):
                raise ValueError('For sum or prod mode, all base_terms must be the same')
            basisShape = [base_terms[0]]
        elif config.base_mode == 'outer':
            basisShape = [prod(base_terms)]
        elif config.base_mode in ['i','j','k']:
            if config.base_mode == 'i' and config.spatial_dim < 1:
                raise ValueError('Mode i requires spatial_dim >= 1')
            if config.base_mode == 'j' and config.spatial_dim < 2:
                raise ValueError('Mode j requires spatial_dim >= 2')
            if config.base_mode == 'k' and config.spatial_dim < 3:
                raise ValueError('Mode k requires spatial_dim >= 3')
            if config.base_mode == 'i':
                basisShape = [base_terms[0]]
            elif config.base_mode == 'j':
                basisShape = [base_terms[1]]
            elif config.base_mode == 'k':
                basisShape = [base_terms[2]]
        else:
            raise ValueError(f'Unknown mode: {config.base_mode}')

    basisTerms = 1
    for s in basisShape:
        basisTerms *= s
        
    verbosePrint(f'{verbosePrefix}BasisEncoder: basisShape={basisShape}, total basis terms={basisTerms}', verbose)
    
    outputShape = None
    if not config.projection:
        outputShape = basisTerms
    else:
        outputShape = config.projection_dim if config.projection_dim is not None else basisTerms

    verbosePrint(f'{verbosePrefix}BasisEncoder: project_out={config.projection_dim}, outputShape={outputShape}', verbose)
            
    return basisShape, basisTerms, outputShape
    

""" Basis Encoding Layer
This layer implements a basis encoding mechanism for input positions, which can be used for absolute or relative position encoding in neural networks. The layer supports various basis functions and modes of combining them, as well as optional projection to a specified output dimension.

Args:
    config (BasisEncoderConfig): Configuration object containing parameters for the basis encoding.
    spatial_dim (int): The dimensionality of the input positions (e.g., 1 for 1D, 2 for 2D, 3 for 3D).
    basis_terms (int): The number of basis functions to use for encoding.
    basis_function (str): The type of basis function to use (e.g., 'fourier', 'chebyshev').
    mode (str): The method for combining basis functions across dimensions ('cat', 'sum', 'prod', 'outer', 'i', 'j', 'k').
    projection (bool): Whether to apply a projection to the output of the basis encoding layer.
    projection_dim (int): The dimensionality of the output after projection, if projection is True.
    verbose: bool: Whether to print verbose output during initialization and forward pass.
    verbosePrefix: str: Prefix for verbose output messages.
"""
from .mapping import map_positions

class BasisEncoder(nn.Module):
    def __init__(self,
                 config: BasisEncoderConfig = BasisEncoderConfig(),
                 
                 verbose: bool = False,
                 verbosePrefix: str = '',
                 **kwargs
                ):
        config = copy.deepcopy(config)
        config = updateBasisEncoderConfig(config, **kwargs)
        self.config = config
        self.verbose = verbose
        self.verbosePrefix = verbosePrefix

        super(BasisEncoder, self).__init__()
        verboseBannerPrint(f'{verbosePrefix}Initializing BasisEncoder Layer', verbose)

        self.base_terms, self.base_function = checkBasisDim(config.spatial_dim, config.base_terms, config.base_function, config.base_mode)
        verbosePrint(f'{verbosePrefix}BasisEncoder: basis_terms={self.base_terms}, basis_function={self.base_function}, mode={config.base_mode}, spatial_dim={config.spatial_dim}', verbose)

        self.basisShape, self.basisTerms, self.outputShape = computeBasisEncoderOutputShape(config, verbose=verbose, verbosePrefix=verbosePrefix+'\t')

        verbosePrint(f'{verbosePrefix}BasisEncoder: basisShape={self.basisShape}, total basis terms={self.basisTerms}, outputShape={self.outputShape}', verbose)
    
        if self.config.base_scaling:
            self.scaling = nn.Linear(self.basisTerms, self.basisTerms)
            
        verbosePrint(f'{verbosePrefix}BasisEncoder: project_out={self.config.projection}, outputShape={self.outputShape}', verbose)
    
        if self.config.projection:
            if self.config.projection_linear:
                self.projector = nn.Linear(self.basisTerms, self.outputShape)
            else:
                if self.config.projection_mlp is None:
                    self.config.projection_mlp = getDefaultMLPDict()

                self.projector = buildMLPwDict(self.config.projection_mlp,inputDim=self.basisTerms,outputDim=self.outputShape, verbose=verbose, verbosePrefix=verbosePrefix+'\t')
        else:
            self.projector = nn.Identity()
            
        if self.config.layer_norm:
            self.layerNorm = nn.LayerNorm(self.config.spatial_dim)
        else:
            self.layerNorm = nn.Identity()
            
        self.activation_fn, self.activationName = getActivationFromString(self.config.activation)

        verbosePrint(f'{verbosePrefix}BasisEncoder: projector={self.projector}', verbose)
        verbosePrint(f'{verbosePrefix}BasisEncoder Layer Initialized', verbose)
                
        
    def forward(self, inputPositions: Tensor) -> Tensor:        
        verboseBannerPrint('BasisEncoder Forward Pass', verbose=self.verbose)
        verbosePrintSpatialTensorStats(inputPositions, name='Input Positions', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        # the shape here is either [B,N,D] or [E,D] or [1,E,D]
        # we need to convert to an internal [E,D] shape
        normalizedPositions, batches, entries, dim = shapeMatch(inputPositions)

        if normalizedPositions.shape[1] != self.config.spatial_dim:
            raise ValueError(f'Input positions dimension {normalizedPositions.shape[1]} does not match configured spatial_dim {self.config.spatial_dim}')
        
        if self.config.clamp_positions:
            normalizedPositions = torch.clamp(normalizedPositions, self.config.clamp_min, self.config.clamp_max)
            verbosePrintSpatialTensorStats(normalizedPositions, name='Clamped Positions', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        if self.config.normalize_positions:
            max_length = torch.max(torch.norm(normalizedPositions, dim=-1, keepdim=True))
            if max_length > 0:
                normalizedPositions = normalizedPositions / max_length
            verbosePrintSpatialTensorStats(normalizedPositions, name='Normalized Positions', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        if self.config.layer_norm:
            normalizedPositions = self.layerNorm(normalizedPositions)
            verbosePrintSpatialTensorStats(normalizedPositions, name='LayerNorm Positions', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        if self.config.base_projection != 'cartesian':
            normalizedPositions = map_positions(normalizedPositions, self.config.base_projection)
            verbosePrintSpatialTensorStats(normalizedPositions, name=f'Mapped Positions ({self.config.base_projection})', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        bTerms = []
        if self.config.base_encoding:
            verbosePrint(f'Applying basis encoding with functions {self.base_function} and terms {self.base_terms}', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            for e, b, f in zip(normalizedPositions.T, self.base_terms, self.base_function):
                bTerm = evalBasisFunction(b, e, f).mT
                bTerms.append(bTerm)
        else:
            # skip basis encoding, just use the input positions directly
            for e in normalizedPositions.T:
                bTerms.append(e[:,None])
        combinedTerms = None
        
        if self.config.base_mode == 'cat':
            combinedTerms = torch.cat(bTerms, dim = 1)
        elif self.config.base_mode == 'sum':
            combinedTerms = torch.stack(bTerms, dim = 0).sum(dim = 0)
        elif self.config.base_mode == 'prod':
            combinedTerms = torch.stack(bTerms, dim = 0).prod(dim = 0)
        elif self.config.base_mode == 'outer':
            combinedTerms = bTerms[0]
            for bt in bTerms[1:]:
                combinedTerms = torch.einsum('ij,ik->ijk', combinedTerms, bt).reshape(-1, combinedTerms.shape[-1] * bt.shape[-1])
            # make sure the product is correctly flattened
            combinedTerms = combinedTerms.view(-1, self.basisTerms)
        elif self.config.base_mode == 'i':
            combinedTerms = bTerms[0]
        elif self.config.base_mode == 'j':
            combinedTerms = bTerms[1]
        elif self.config.base_mode == 'k':
            combinedTerms = bTerms[2]
        else:
            raise ValueError(f'Unknown mode: {self.config.base_mode}')
        
        verbosePrint(f'Combined basis terms shape: {combinedTerms.shape}', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        
        if self.config.base_scaling:
            combinedTerms = self.scaling(combinedTerms)
            verbosePrint(f'Scaled basis terms shape: {combinedTerms.shape}', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        if self.activation_fn is not None:
            combinedTerms = self.activation_fn(combinedTerms)
            verbosePrint(f'Activated basis terms shape: {combinedTerms.shape}', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        # now combinedTerms is of shape [E, combinedBasisTerms]
        # map back to [B,N,combinedBasisTerms] or [E,combinedBasisTerms]
        if inputPositions.shape != normalizedPositions.shape:
            verbosePrint(f'Reshaping output to include batch dimension: {batches}', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            combinedTerms = combinedTerms.view(batches, -1, combinedTerms.shape[-1])
        # now apply the projection if needed
        if self.config.projection:
            verbosePrint(f'Applying projection: {self.config.projection}', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            combinedTerms = self.projector(combinedTerms)
            
        verbosePrint(f'BasisEncoder output shape: {combinedTerms.shape}', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        return combinedTerms
