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

class InputEncodeLayer(torch.nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 spatial_dim: int = 3,
                 
                 linearEncode: bool = True,
                 encoderMLPDict: Optional[dict] = None,

                 absolutePositionBias: bool = False,
                 absolutePositionBiasScaledPositions: bool = False,
                 absolutePositionBiasMultiplicative: bool = False,
                 absolutePositionBiasBaseEncoding: bool = True,
                 absolutePositionBiasBaseFunction: str = 'fourier',
                 absolutePositionBiasBaseTerms: int = 16,
                 absolutePositionBiasBaseMode: str = 'cat', 
                 absolutePositionBiasLinear: bool = True, 
                 absolutePositionBiasMLPDict: Optional[dict] = None,

                 verbose: bool = False
    ):
        super(InputEncodeLayer, self).__init__()
        verbosePrint('Initializing Input Encode Layer...', verbose)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spatial_dim = spatial_dim
        verbosePrint(f'\tDimensions: input_dim={self.input_dim}, output_dim={self.output_dim}, spatial_dim={self.spatial_dim}', verbose)

        self.linearEncode = linearEncode
        self.encoderMLPDict = encoderMLPDict if encoderMLPDict is not None else getDefaultMLPDict()
        verbosePrint(f'\tInput feature encoding: linearEncode={self.linearEncode}', verbose)
        verbosePrint(f'\tInput feature encoding: encoderMLPDict={self.encoderMLPDict}', verbose)

        self.absolutePositionBias = absolutePositionBias
        self.absolutePositionBiasScaledPositions = absolutePositionBiasScaledPositions
        self.absolutePositionBiasMultiplicative = absolutePositionBiasMultiplicative
        self.absolutePositionBiasBaseEncoding = absolutePositionBiasBaseEncoding
        self.absolutePositionBiasBaseFunction = absolutePositionBiasBaseFunction
        self.absolutePositionBiasBaseTerms = absolutePositionBiasBaseTerms
        self.absolutePositionBiasBaseMode = absolutePositionBiasBaseMode
        self.absolutePositionBiasLinear = absolutePositionBiasLinear
        self.absolutePositionBiasMLPDict = absolutePositionBiasMLPDict if absolutePositionBiasMLPDict is not None else getDefaultMLPDict()
        verbosePrint(f'\tAbsolute Position Bias (APB): absolutePositionBias={self.absolutePositionBias}', verbose)
        if self.absolutePositionBias:
            verbosePrint(f'\tAbsolute Position Bias (APB): absolutePositionBiasScaledPositions={self.absolutePositionBiasScaledPositions}', verbose)
            verbosePrint(f'\tAbsolute Position Bias (APB): absolutePositionBiasMultiplicative={self.absolutePositionBiasMultiplicative}', verbose)
            verbosePrint(f'\tAbsolute Position Bias (APB): absolutePositionBiasBaseEncoding={self.absolutePositionBiasBaseEncoding}', verbose)
            if self.absolutePositionBiasBaseEncoding:
                verbosePrint(f'\tAbsolute Position Bias (APB): absolutePositionBiasBaseFunction={self.absolutePositionBiasBaseFunction}', verbose)
                verbosePrint(f'\tAbsolute Position Bias (APB): absolutePositionBiasBaseTerms={self.absolutePositionBiasBaseTerms}', verbose)
                verbosePrint(f'\tAbsolute Position Bias (APB): absolutePositionBiasBaseMode={self.absolutePositionBiasBaseMode}', verbose)
            verbosePrint(f'\tAbsolute Position Bias (APB): absolutePositionBiasLinear={self.absolutePositionBiasLinear}', verbose)
            verbosePrint(f'\tAbsolute Position Bias (APB): absolutePositionBiasMLPDict={self.absolutePositionBiasMLPDict}', verbose)

        self.verbose = verbose

        # Input feature encoding
        verbosePrint(f'\tInput feature encoding:', self.verbose, separator=True)
        if self.linearEncode:
            verbosePrint('\t\tUsing linear layer for input feature encoding', self.verbose)
            self.inputEncoder = nn.Linear(self.input_dim, self.output_dim, bias=False)
        else:
            verbosePrint('\t\tUsing MLP for input feature encoding', self.verbose)
            if self.encoderMLPDict is not None:
                self.inputEncoder = buildMLPwDict({
                    'inputFeatures': self.input_dim,
                    'output': self.output_dim,
                    **self.encoderMLPDict
                }, verbose = verbose, verbosePrefix='\t\t')
            else:
                self.inputEncoder = buildMLPwDict({
                    'inputFeatures': self.input_dim,
                    'output': self.output_dim,
                }, verbose = verbose, verbosePrefix='\t\t')
            numberOfParameters = sum(p.numel() for p in self.inputEncoder.parameters())
            verbosePrint(f'\t\tNumber of parameters in input encoder MLP: {numberOfParameters}', self.verbose)

        self.apbEncoder = None
        self.apbBasisEncoderOutputDim = None
        # Absolute position bias encoding
        if self.absolutePositionBias:   
            verbosePrint(f'\tAbsolute position bias (APB) encoding:', self.verbose, separator=True)
            if self.absolutePositionBiasScaledPositions:
                verbosePrint(f'\t\tUsing scaled positions for APB encoding', self.verbose)
            if self.absolutePositionBiasMultiplicative:
                verbosePrint(f'\t\tUsing multiplicative APB', self.verbose)
            else:
                verbosePrint(f'\t\tUsing additive APB', self.verbose)
            
            self.apbEncoder = BasisEncoder(
                spatial_dim = self.spatial_dim,
                basis_terms = self.absolutePositionBiasBaseTerms,
                basis_function = self.absolutePositionBiasBaseFunction,
                skip_basis = not self.absolutePositionBiasBaseEncoding,
                mode = self.absolutePositionBiasBaseMode,
                project_out = True,
                out_dim = self.output_dim,
                project_linear = self.absolutePositionBiasLinear,
                project_mlp_properties = self.absolutePositionBiasMLPDict,
                
                verbose = self.verbose, 
                verbosePrefix = '\t\t'
            )
            self.apbBasisEncoderOutputDim = self.apbEncoder.outputShape

            numberOfParameters = sum(p.numel() for p in self.apbEncoder.parameters())
            verbosePrint(f'\t\tNumber of parameters in APB encoder MLP: {numberOfParameters}', self.verbose)

        verbosePrint(f'Done initializing Input Encode Layer.', self.verbose, separator=True)
        
        

    def forward(self, 
                inputTokens: torch.Tensor, # Shape: [num_tokens, input_dim]
                inputPositions: torch.Tensor, # Shape: [num_tokens, spatial_dim],
                cutoffRadius: Optional[float] = None # Shape: [num_tokens]
                ):
        verbosePrint(f'Running Input Encode Layer...', self.verbose, separator=True)
        verbosePrint(f'\tInput tokens shape: {inputTokens.shape}', self.verbose)
        verbosePrint(f'\tInput positions shape: {inputPositions.shape}', self.verbose)
        if cutoffRadius is not None:
            verbosePrint(f'\tCutoff radius shape: {cutoffRadius.shape}', self.verbose)

        verbosePrint(f'\tEncoding input features...', self.verbose, separator=True)
        encodedFeatures = self.inputEncoder(inputTokens)
        verbosePrint(f'\tEncoded features shape: {encodedFeatures.shape}', self.verbose)

        if self.absolutePositionBias:
            verbosePrint(f'\tEncoding absolute position bias (APB)...', self.verbose, separator=True)
            encodedPositions = inputPositions
            if self.absolutePositionBiasScaledPositions:
               verbosePrint(f'\tScaling positions by cutoff radius for APB encoding', self.verbose)
               encodedPositions = inputPositions / cutoffRadius.view(-1,1)

            if self.absolutePositionBiasBaseEncoding:
                verbosePrint(f'\tUsing basis function encoding for APB', self.verbose)
                encodedPositions = self.apbBasisEncoder(encodedPositions)
            verbosePrint(f'\tEncoded positions shape (before flattening): {encodedPositions.shape}', self.verbose)

            if len(encodedPositions.shape) > 2:
                verbosePrint(f'\tFlattening encoded positions for APB', self.verbose)
                encodedPositions = encodedPositions.view(encodedPositions.shape[0], -1)
            verbosePrint(f'\tEncoded positions shape (after flattening): {encodedPositions.shape}', self.verbose)

            verbosePrint(f'\tPassing encoded positions through APB encoder', self.verbose)
            apb = self.apbEncoder(encodedPositions)
            verbosePrint(f'\tAPB shape: {apb.shape}', self.verbose)

            if self.absolutePositionBiasMultiplicative:
                verbosePrint(f'\tUsing multiplicative APB', self.verbose)
                encodedFeatures = encodedFeatures * apb
            else:
                verbosePrint(f'\tUsing additive APB', self.verbose)
                encodedFeatures = encodedFeatures + apb
            verbosePrint(f'\tFeatures shape after adding APB: {encodedFeatures.shape}', self.verbose)
        verbosePrint(f'Done running Input Encode Layer.', self.verbose, separator=True)
        return encodedFeatures # Shape: [num_tokens, output_dim]