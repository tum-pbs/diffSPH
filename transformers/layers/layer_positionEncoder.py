import warnings
import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional
from .basisFunctions import basisEncoderLayer, evalBasisFunction
from .mlp import getDefaultMLPDict, buildMLPwDict
from .networkUtil import verbosePrint, verboseBannerPrint

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
class BasisEncoder(nn.Module):
    def __init__(self,
                 spatial_dim: int = 3,
                 basis_terms: int = 16,
                 basis_function: str = 'ffourier',
                 skip_basis: bool = False,
                 mode: str = 'cat',
                 
                 project_out: bool = False,
                 project_linear: bool = True,
                 project_mlp_properties: Optional[dict] = None,
                 
                 out_dim: Optional[int] = None,
                 verbose: bool = False,
                 verbosePrefix: str = ''):
        super(BasisEncoder, self).__init__()
        verboseBannerPrint(f'{verbosePrefix}Initializing BasisEncoder Layer', verbose)

        self.basisTerms = basis_terms
        self.basisFunction = basis_function
        self.skip_basis = skip_basis
        self.mode = mode
        self.spatial_dim = spatial_dim
        self.project_out = project_out
        self.outputShape = None
        self.mlp_properties = None
        self.project_linear = project_linear
        
        verbosePrint(f'{verbosePrefix}BasisEncoder: basis_terms={basis_terms}, basis_function={basis_function}, mode={mode}, spatial_dim={spatial_dim}', verbose)

        self.basisShape = None
        if self.skip_basis:
            self.basisShape = [self.spatial_dim]
        else:
            if mode == 'cat':
                self.basisShape = [self.basisTerms * self.spatial_dim]
            elif mode == 'sum' or mode == 'prod':
                self.basisShape = [self.basisTerms]
            elif mode == 'outer':
                self.basisShape = [self.basisTerms] * self.spatial_dim
            elif mode in ['i','j','k']:
                if mode == 'i' and spatial_dim < 1:
                    raise ValueError('Mode i requires spatial_dim >= 1')
                if mode == 'j' and spatial_dim < 2:
                    raise ValueError('Mode j requires spatial_dim >= 2')
                if mode == 'k' and spatial_dim < 3:
                    raise ValueError('Mode k requires spatial_dim >= 3')
                self.basisShape = [self.basisTerms]
            else:
                raise ValueError(f'Unknown mode: {mode}')
        
        basisTerms = 1
        for s in self.basisShape:
            basisTerms *= s
            
        verbosePrint(f'{verbosePrefix}BasisEncoder: basisShape={self.basisShape}, total basis terms={basisTerms}', verbose)
        
        if not self.project_out:
            self.outputShape = basisTerms
        else:
            self.outputShape = out_dim if out_dim is not None else basisTerms
            
        verbosePrint(f'{verbosePrefix}BasisEncoder: project_out={self.project_out}, outputShape={self.outputShape}', verbose)
            
        if self.project_out:
            if project_linear:
                self.projector = nn.Linear(basisTerms, self.outputShape)
            else:
                if project_mlp_properties is None:
                    project_mlp_properties = getDefaultMLPDict()
                project_mlp_properties['inputFeatures'] = basisTerms
                project_mlp_properties['output'] = self.outputShape
                self.mlp_properties = project_mlp_properties
                
                self.projector = buildMLPwDict(self.mlp_properties, verbose=verbose, verbosePrefix=verbosePrefix+'\t')
        else:
            self.projector = None
            
        verbosePrint(f'{verbosePrefix}BasisEncoder: projector={self.projector}', verbose)
        verbosePrint(f'{verbosePrefix}BasisEncoder Layer Initialized', verbose)
                
        
    def forward(self, inputPositions: Tensor) -> Tensor:
        
        # the shape here is either [B,N,D] or [E,D] or [1,E,D]
        # we need to convert to an internal [E,D] shape
        if len(inputPositions.shape) == 3:
            mapped = True
            batches, entries, dim = inputPositions.shape
            normalizedPositions = inputPositions.view(-1, dim)
        elif len(inputPositions.shape) == 2:
            mapped = False
            entries, dim = inputPositions.shape
            batches = 1
            normalizedPositions = inputPositions
        else:
            raise ValueError(f'Input positions must be of shape [B,N,D] or [E,D], got {inputPositions.shape}')
                   
            
            
        bTerms = []
        for e in normalizedPositions.T:
            bTerm = evalBasisFunction(self.basisTerms, e, self.basisFunction).mT
            bTerms.append(bTerm)
            
        combinedTerms = None
        
        if self.mode == 'cat':
            combinedTerms = torch.cat(bTerms, dim = 1)
        elif self.mode == 'sum':
            combinedTerms = torch.stack(bTerms, dim = 0).sum(dim = 0)
        elif self.mode == 'prod':
            combinedTerms = torch.stack(bTerms, dim = 0).prod(dim = 0)
        elif self.mode == 'outer':
            combinedTerms = bTerms[0]
            for bt in bTerms[1:]:
                combinedTerms = torch.einsum('ij,ik->ijk', combinedTerms, bt).reshape(-1, combinedTerms.shape[-1] * bt.shape[-1])
            # make sure the product is correctly flattened
            combinedTerms = combinedTerms.view(-1, self.basisTerms ** self.spatial_dim)
        elif self.mode == 'i':
            combinedTerms = bTerms[0]
        elif self.mode == 'j':
            combinedTerms = bTerms[1]
        elif self.mode == 'k':
            combinedTerms = bTerms[2]
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        
        # now combinedTerms is of shape [E, combinedBasisTerms]
        # map back to [B,N,combinedBasisTerms] or [E,combinedBasisTerms]
        if mapped:
            combinedTerms = combinedTerms.view(batches, -1, combinedTerms.shape[-1])
        # now apply the projection if needed
        if self.project_out:
            combinedTerms = self.projector(combinedTerms)
        return combinedTerms
