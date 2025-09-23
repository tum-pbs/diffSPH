import warnings
import torch
from torch import Tensor
import torch.nn as nn
try:
    import torch_geometric
    from torch_geometric.utils import scatter, segment
    from torch_geometric.utils.num_nodes import maybe_num_nodes
except ImportError:
    torch_geometric = None
from typing import Optional, Union, Tuple
 

from .activation import getActivationLayer
from .basisFunctions import basisEncoderLayer
from .layer_positionEncoder import BasisEncoder
from .networkUtil import verbosePrint
from .sparse import buildSparseTensor
from .softmax import softmax
from .mlp import buildMLPwDict, getDefaultMLPDict


# Input Encoding Layer
# This is the first step of a transformer architecture
# Input:  A sequence of input tokens (nodes) described by a feature and position vector
# Output: A sequence of output tokens (nodes) described by a latent space feature vector
# Configuration options:
# Basic Parameters:
# - input_dim: Dimensionality of the input feature vector per token
# - output_dim: Dimensionality of the output feature vector per token (latent space size)
# - spatial_dim: Dimensionality of the position vector per token (e.g. 3 for 3D positions)
#
# Encode Parameters:
# - linearEncode: If True, use a linear layer to encode input features to latent space, if False use an MLP
# - encoderMLPDict: Dictionary defining the MLP architecture for input feature encoding (if linearEncode is False)
#
# Absolute Position Bias (APB) Parameters:
# - absolutePositionBias: If True, the absolute position of each token is encoded and added to the input features
# - absolutePositionBiasScaledPositions: If True, the input positions are scaled by a given cutoff radius before encoding
# - absolutePositionBiasMultiplicative: If True, the absolute position encoding is multiplied to the input features instead of added
# - absolutePositionBiasBaseEncoding: If True, the absolute position is encoded using a basis function encoding (e.g. Fourier or Gaussian basis)
# - absolutePositionBiasBaseFunction: Type of basis function encoding to use for absolute position (e.g. 'fourier', 'gaussian')
# - absolutePositionBiasBaseTerms: Number of basis functions to use for absolute position encoding
# - absolutePositionBiasLinear: If true the APB is a result of the (potentially encoded) positions passed through a linear layer to match the input feature dimension, if false an MLP is used
# - absolutePositionBiasMLPDict: Dictionary defining the MLP architecture for absolute position bias encoding (if absolutePositionBiasLinear is False)
# 
# Misc Parameters:
# - verbose: If True, print detailed information during initialization and forward pass
#
#
# Forward Pass Parameters:
# - inputTokens: A tensor of shape [num_tokens, input_dim] representing the input feature vectors for each token
# - inputPositions: A tensor of shape [num_tokens, spatial_dim] representing the position vectors for each token
# - cutoffRadius: A scalar value representing the cutoff radius for neighborhood search (if applicable) [Optional]

class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 
                 pre_norm: bool = False,
                 post_norm: bool = False,

                 skip_connection: bool = True,
                 
                 linear: bool = False,
                 MLPDict: Optional[dict] = None,

                 verbose: bool = False
    ):
        super(FeedForwardNetwork, self).__init__()
        verbosePrint('Initializing Feed Forward Network...', verbose)

        self.input_dim = input_dim
        self.output_dim = output_dim
        verbosePrint(f'\tDimensions: input_dim={self.input_dim}, output_dim={self.output_dim}', verbose)

        self.linear = linear
        self.MLPDict = MLPDict if MLPDict is not None else getDefaultMLPDict()
        verbosePrint(f'\tFFN: linear={self.linear}', verbose)
        verbosePrint(f'\tFFN: MLPDict={self.MLPDict}', verbose)

        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.skip_connection = skip_connection
        if self.pre_norm:
            self.preNormLayer = nn.LayerNorm(self.input_dim)
            verbosePrint(f'\tUsing pre-norm layer', verbose)
        if self.post_norm:
            self.postNormLayer = nn.LayerNorm(self.output_dim)
            verbosePrint(f'\tUsing post-norm layer', verbose)
        if self.skip_connection:
            if self.input_dim != self.output_dim:
                warnings.warn(f'Input dim ({self.input_dim}) and output dim ({self.output_dim}) are different, skipping connection will not be possible!')
            verbosePrint(f'\tUsing skip connection', verbose)
        self.verbose = verbose

        # Input feature encoding
        verbosePrint(f'\tFFN:', self.verbose, separator=True)
        if self.linear:
            verbosePrint('\t\tUsing linear layer for FFN', self.verbose)
            self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        else:
            verbosePrint('\t\tUsing MLP for FFN', self.verbose)
            self.proj = buildMLPwDict(self.MLPDict, inputDim=input_dim, outputDim=output_dim, verbose=verbose, verbosePrefix='\t\t')
            numberOfParameters = sum(p.numel() for p in self.proj.parameters())
            verbosePrint(f'\t\tNumber of parameters in FFN MLP: {numberOfParameters}', self.verbose)

        verbosePrint(f'Done initializing Input Encode Layer.', self.verbose, separator=True)
        
        

    def forward(self, 
                inputTokens: torch.Tensor, # Shape: [num_tokens, input_dim] or [batch_size, num_tokens, input_dim]
                ):
        verbosePrint(f'Running Input Encode Layer...', self.verbose, separator=True)
        verbosePrint(f'\tInput tokens shape: {inputTokens.shape}', self.verbose)
        if self.pre_norm:
            outputTokens = self.preNormLayer(inputTokens)
        else:
            outputTokens = inputTokens
        verbosePrint(f'\tInput tokens shape (after pre-norm): {outputTokens.shape}', self.verbose)
        outputTokens = self.proj(outputTokens)
        verbosePrint(f'\tOutput tokens shape (after FFN): {outputTokens.shape}', self.verbose)
        if self.skip_connection:
            if self.input_dim != self.output_dim:
                raise ValueError(f'Input dim ({self.input_dim}) and output dim ({self.output_dim}) are different, skipping connection is not possible!')
            outputTokens = inputTokens + outputTokens
            verbosePrint(f'\tOutput tokens shape (after skip connection): {outputTokens.shape}', self.verbose)
        if self.post_norm:
            outputTokens = self.postNormLayer(outputTokens)
        verbosePrint(f'\tOutput tokens shape (after post-norm): {outputTokens.shape}', self.verbose)
        return outputTokens