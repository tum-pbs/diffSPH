import warnings
import copy
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
 

from .activation import getActivationFromString, getActivationLayer
from .basisFunctions import basisEncoderLayer
from .layer_positionEncoder import BasisEncoder
from .networkUtil import verbosePrint, verboseBannerPrint
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

from typing import Optional, Union, Tuple
from dataclasses import dataclass, field

"""Configuration for Absolute Position Bias (APB) in InputEncodeLayer.

This configuration controls the behavior of the absolute position bias (APB) mechanism,
which encodes the absolute position of each token in the input sequence.

Note: APB violates translation equivariance, so it should be used with caution in tasks where
translation equivariance is important, e.g., physics-based tasks.
"""

from .layer_positionEncoder import BasisEncoder, BasisEncoderConfig, computeBasisEncoderOutputShape


@dataclass(slots=True)
class TokenEncoderConfig:
    token_input_dim:        int = field(default =0, metadata={"help": "Dimensionality of the input feature vector per token"})
    token_output_dim:       Optional[int] = field(default = None, metadata={"help": "Dimensionality of the output feature vector per token"})
    token_latent_dim:       Optional[int] = field(default = None, metadata={"help": "Dimensionality of the latent space feature vector per token. If None, set to token_output_dim"})
    skip_connection:        bool = field(default=True, metadata={"help": "If True, use skip connection from input to output"})

    projection:             bool = field(default=True, metadata={"help": "If True, project input features to output dimension using a linear layer or MLP"})
    projection_linear:      bool = field(default=True, metadata={"help": "If True, use a linear layer for input feature projection, if False use an MLP"})
    projection_mlp_dict:    Optional[dict] = field(default=None, metadata={"help": "Dictionary defining the MLP architecture for input feature projection (if projection_linear is False)"})

    position_bias:          Optional[BasisEncoderConfig] = field(default=None, metadata={"help": "Configuration for absolute position bias (APB) encoding. If None, APB is disabled."})
    position_bias_mixing:   Optional[str] = field(default=None, metadata={"help": "Mode for combining position bias with input features. Options: 'cat' (concatenate), 'add' (additive), 'mul' (multiplicative), 'mix' (use linear or MLP to combine)"})
    position_bias_linear:   bool = field(default=True, metadata={"help": "If True, use a linear layer for position bias projection, if False use an MLP"})
    position_bias_mlp_dict: Optional[dict] = field(default=None, metadata={"help": "Dictionary defining the MLP architecture for position bias projection (if position_bias_linear is False)"})
    position_bias_dim:     Optional[int] = field(default=None, metadata={"help": "Override position_bias_dim, used when there is no position bias encoder within this layer but the information is provided externally."})

    use_ffn:                bool = field(default=False, metadata={"help": "If True, use a feed-forward network (FFN) after input encoding"})
    ffn_linear:             bool = field(default=False, metadata={"help": "If True, use a linear layer for the FFN, if False use an MLP"})
    ffn_mlp_dict:           Optional[dict] = field(default=None, metadata={"help": "Dictionary defining the MLP architecture for the FFN (if ffn_linear is False)"})
    ffn_skip_connection:    bool = field(default=False, metadata={"help": "If True, use skip connection in the FFN"})
    pre_norm:               bool = field(default=False, metadata={"help": "If True, apply layer normalization before the feed-forward network"})
    post_norm:              bool = field(default=False, metadata={"help": "If True, apply layer normalization after the feed-forward network"})

    final_activation:    Optional[str] = field(default=None, metadata={"help": "Activation function to apply after the FFN. Options: None, 'relu', 'gelu', etc."})


from .networkUtil import shapeMatch, verbosePrintSpatialTensorStats, mergeConfigWithKwargs, checkTensorShape

class TokenEncoder(torch.nn.Module):
    def __init__(self, 
                 token_input_dim:         int = field(metadata={"help": "Dimensionality of the input feature vector per token"}),
                 config: Optional[TokenEncoderConfig] = None,
                 verbose: bool = False,
                 verbosePrefix: str = '',
                 **kwargs
    ):
        super(TokenEncoder, self).__init__()
        verboseBannerPrint('Initializing Input Encode Layer...', verbose)

        if config is None:
            config = TokenEncoderConfig(token_input_dim=token_input_dim)
        else:
            config = copy.deepcopy(config)
            config.token_input_dim = token_input_dim
        self.config = mergeConfigWithKwargs(config, **kwargs)
        self.verbose = verbose
        self.verbosePrefix = verbosePrefix

        self.input_token_dim = self.config.token_input_dim
        self.output_token_dim = self.config.token_output_dim if self.config.token_output_dim is not None else self.config.token_input_dim
        self.latent_token_dim = self.config.token_latent_dim if self.config.token_latent_dim is not None else self.output_token_dim
        verbosePrint(f'\tToken dimensions: input_token_dim={self.input_token_dim}, output_token_dim={self.output_token_dim}, latent_token_dim={self.latent_token_dim}', self.verbose, self.verbosePrefix)

        if self.output_token_dim != self.latent_token_dim and not self.config.use_ffn:
            raise ValueError(f'Output token dim ({self.output_token_dim}) and latent token dim ({self.latent_token_dim}) are different, but FFN is disabled! Cannot project latent to output space.')

        ### Position Bias (APB) Setup ###
        self.latent_out_dim = self.latent_token_dim
        if self.config.position_bias is not None:
            positionBiasDim = computeBasisEncoderOutputShape(self.config.position_bias, self.verbose, self.verbosePrefix)[-1]
            verbosePrint(f'\tPosition bias encoding enabled with output dimension {positionBiasDim}', self.verbose, self.verbosePrefix)
            if self.config.position_bias_mixing in ['add', 'mul']:
                if self.latent_token_dim != positionBiasDim:
                    raise ValueError(f'Latent token dimension {self.latent_token_dim} does not match position bias output dimension {positionBiasDim}')
            elif self.config.position_bias_mixing == 'cat':
                self.latent_out_dim += positionBiasDim
                verbosePrint(f'\tPosition bias concatenation increases latent output dimension to {self.latent_out_dim}', self.verbose, self.verbosePrefix)
            elif self.config.position_bias_mixing == 'mix':
                self.latent_out_dim = self.latent_token_dim
                verbosePrint(f'\tPosition bias mixing keeps latent output dimension at {self.latent_out_dim}', self.verbose, self.verbosePrefix)
            else:
                raise ValueError(f'Invalid position_bias_mixing mode: {self.config.position_bias_mixing}')
            
            self.apbEncoder = BasisEncoder(self.config.position_bias, verbose=self.verbose, verbosePrefix=self.verbosePrefix + '\t')
            self.apbDim = positionBiasDim
        else:
            self.apbEncoder = None
            self.apbDim = self.config.position_bias_dim
            verbosePrint(f'\tPosition bias encoding disabled', self.verbose, self.verbosePrefix)

        ### PB Mixing Setup ###
        if self.apbDim is not None and self.config.position_bias_mixing == 'mix':
            if self.config.position_bias_linear:
                verbosePrint(f'\tUsing linear layer for position bias mixing', self.verbose, self.verbosePrefix)
                self.pbMixer = nn.Linear(self.latent_token_dim + self.apbDim, self.latent_token_dim, bias=False)
            else:
                verbosePrint(f'\tUsing MLP for position bias mixing', self.verbose, self.verbosePrefix)
                mlpDict = self.config.position_bias_mlp_dict if self.config.position_bias_mlp_dict is not None else getDefaultMLPDict()
                self.pbMixer = buildMLPwDict(mlpDict, inputDim=self.latent_token_dim + self.apbDim, outputDim=self.latent_token_dim, verbose=self.verbose, verbosePrefix=self.verbosePrefix + '\t')
                numberOfParameters = sum(p.numel() for p in self.pbMixer.parameters())
                verbosePrint(f'\tNumber of parameters in position bias mixer MLP: {numberOfParameters}', self.verbose, self.verbosePrefix)
        else:
            self.pbMixer = None

        ### Input Feature Encoding Setup ###
        if self.config.projection:
            verbosePrint(f'\tInput feature projection enabled [{self.input_token_dim} -> {self.latent_token_dim}]', self.verbose, self.verbosePrefix)
            if self.config.projection_linear:
                verbosePrint(f'\tUsing linear layer for input feature projection', self.verbose, self.verbosePrefix)
                self.input_encoder = nn.Linear(self.input_token_dim, self.latent_token_dim, bias=False)
            else:
                verbosePrint(f'\tUsing MLP for input feature projection', self.verbose, self.verbosePrefix)
                mlpDict = self.config.projection_mlp_dict if self.config.projection_mlp_dict is not None else getDefaultMLPDict()
                self.input_encoder = buildMLPwDict(mlpDict, inputDim=self.input_token_dim, outputDim=self.latent_token_dim, verbose=self.verbose, verbosePrefix=self.verbosePrefix + '\t')
                numberOfParameters = sum(p.numel() for p in self.input_encoder.parameters())
                verbosePrint(f'\tNumber of parameters in input encoder MLP: {numberOfParameters}', self.verbose, self.verbosePrefix)

        ## Skip Connection Setup ##
        if self.config.skip_connection:
            if self.input_token_dim != self.output_token_dim:
                warnings.warn(f'Input token dim ({self.input_token_dim}) and output token dim ({self.output_token_dim}) are different, skipping connection will not be possible!')
                self.config.skip_connection = False
            verbosePrint(f'\tUsing skip connection from input to output', self.verbose, self.verbosePrefix)

        ### FFN Setup ###
        if self.config.use_ffn:
            verbosePrint(f'\tFeed Forward Network (FFN) enabled [{self.latent_out_dim} -> {self.output_token_dim}]', self.verbose, self.verbosePrefix)

            if self.config.pre_norm:
                self.preNormLayer = nn.LayerNorm(self.latent_out_dim)
                verbosePrint(f'\tUsing pre-norm layer before FFN', self.verbose, self.verbosePrefix)
            if self.config.ffn_linear:
                verbosePrint(f'\tUsing linear layer for FFN', self.verbose, self.verbosePrefix)
                self.ffn = nn.Linear(self.latent_out_dim, self.output_token_dim, bias=False)
            else:
                verbosePrint(f'\tUsing MLP for FFN', self.verbose, self.verbosePrefix)
                mlpDict = self.config.ffn_mlp_dict if self.config.ffn_mlp_dict is not None else getDefaultMLPDict()
                self.ffn = buildMLPwDict(mlpDict, inputDim=self.latent_out_dim, outputDim=self.output_token_dim, verbose=self.verbose, verbosePrefix=self.verbosePrefix + '\t')
                numberOfParameters = sum(p.numel() for p in self.ffn.parameters())
                verbosePrint(f'\tNumber of parameters in FFN MLP: {numberOfParameters}', self.verbose, self.verbosePrefix)

            if self.config.ffn_skip_connection:
                if self.latent_out_dim != self.output_token_dim:
                    warnings.warn(f'Cannot use skip connection in FFN, latent output dim ({self.latent_out_dim}) and output token dim ({self.output_token_dim}) are different!')
                    self.config.ffn_skip_connection = False
                verbosePrint(f'\tUsing skip connection in FFN', self.verbose, self.verbosePrefix)

            if self.config.post_norm:
                self.postNormLayer = nn.LayerNorm(self.output_token_dim)
                verbosePrint(f'\tUsing post-norm layer after FFN', self.verbose, self.verbosePrefix)

        ### Activation Setup ###
        self.final_activation = nn.Identity()
        if self.config.final_activation is not None:
            self.final_activation, act_name = getActivationFromString(self.config.final_activation)

        verbosePrint(f'Done initializing Input Encode Layer.', self.verbose, separator=True)
        
    def forward(self, 
                inputTokens: torch.Tensor, # Shape: [num_tokens, input_dim]
                inputPositions: Optional[torch.Tensor], # Shape: [num_tokens, spatial_dim],
                encodedInputPositions: Optional[torch.Tensor] = None, # Shape: [num_tokens, position_encoding_dim],
                ):
        verboseBannerPrint(f'Running Input Encode Layer...', self.verbose)
        verbosePrint(f'\tInput tokens shape: {inputTokens.shape}', self.verbose)
        if inputPositions is not None:
            verbosePrint(f'\tInput positions shape: {inputPositions.shape}', self.verbose)


        normalizedTokens, batchSize, numTokens, featureDim = shapeMatch(inputTokens)
        if inputPositions is not None:
            normalizedPositions, _, _, spatial_dim = shapeMatch(inputPositions)
        else:
            spatial_dim = 0

        shapeDict = {
            'N': numTokens,
            'F': featureDim,
            'B': batchSize,
            'D': spatial_dim,
            'L': self.latent_token_dim,
            'O': self.output_token_dim,
            'APB': self.apbDim
        }

        ######################################################################################
        #### Step 1: Project input features to latent space
        #### Step 2: Encode absolute position bias (APB)
        #### Step 3: Combine encoded features and APB
        #### Step 4: Apply FFN (if enabled)
        #### Step 5: Apply final activation (if specified)
        ######################################################################################

        ################################################################################
        #                     Step 1: Project input features to latent space            #
        ################################################################################
        verbosePrint(f'\tProjecting input features to latent space...', self.verbose, separator=True)
        if self.config.projection:
            encodedFeatures = self.input_encoder(normalizedTokens)
        else:
            encodedFeatures = normalizedTokens
        verbosePrint(f'\tEncoded features shape: {encodedFeatures.shape}', self.verbose)
        checkTensorShape(encodedFeatures, ['N*B', 'L'], shapeDict, False, 'encodedFeatures')

        ################################################################################
        #                     Step 2: Encode absolute position bias (APB)                #
        ################################################################################
        if self.apbEncoder is not None and encodedInputPositions is not None:
            raise ValueError(f'Both inputPositions and encodedInputPositions are provided, only one should be given when position bias encoding is enabled!')
        if self.config.position_bias is not None:
            positionBias = self.apbEncoder(normalizedPositions)
            verbosePrint(f'\tPosition bias shape: {positionBias.shape}', self.verbose)
            checkTensorShape(positionBias, ['N*B', 'APB'], shapeDict, False, 'positionBias')
        else:
            positionBias = encodedInputPositions

        ################################################################################
        #                     Step 3: Combine encoded features and APB                  #
        ################################################################################

        if positionBias is not None and self.config.position_bias_mixing is not None:
            verbosePrint(f'\tCombining encoded features and position bias using mode: {self.config.position_bias_mixing}', self.verbose, separator=True)
            if self.config.position_bias_mixing == 'add':
                encodedFeatures = encodedFeatures + positionBias
            elif self.config.position_bias_mixing == 'mul':
                encodedFeatures = encodedFeatures * positionBias
            elif self.config.position_bias_mixing == 'cat':
                encodedFeatures = torch.cat((encodedFeatures, positionBias), dim=-1)
            elif self.config.position_bias_mixing == 'mix':
                combined = torch.cat((encodedFeatures, positionBias), dim=-1)
                encodedFeatures = self.pbMixer(combined)
            else:
                raise ValueError(f'Invalid position_bias_mixing mode: {self.config.position_bias_mixing}')
            verbosePrint(f'\tCombined features shape: {encodedFeatures.shape}', self.verbose)
            checkTensorShape(encodedFeatures, ['N*B', 'L'] if self.config.position_bias_mixing != 'cat' else ['N*B', 'L+APB'], shapeDict, False, 'combinedFeatures')


        ################################################################################
        #                     Step 4: Apply FFN (if enabled)                             #
        ################################################################################
        if self.config.use_ffn:
            ffn_input_features = encodedFeatures
            verbosePrint(f'\tApplying Feed Forward Network (FFN)...', self.verbose, separator=True)
            if self.config.pre_norm:
                verbosePrint(f'\tApplying pre-norm layer before FFN', self.verbose)
                ffn_input_features = self.preNormLayer(encodedFeatures)
            ffn_output_features = self.ffn(ffn_input_features)
            verbosePrint(f'\tFFN output shape: {ffn_output_features.shape}', self.verbose)
            checkTensorShape(ffn_output_features, ['N*B', 'O'], shapeDict, False, 'ffnOutput')

            if self.config.ffn_skip_connection:
                verbosePrint(f'\tApplying skip connection in FFN', self.verbose)
                ffn_output_features = ffn_output_features + encodedFeatures
                verbosePrint(f'\tFFN output shape after skip connection: {ffn_output_features.shape}', self.verbose)
                checkTensorShape(ffn_output_features, ['N*B', 'O'], shapeDict, False, 'ffnSkipConnectionOutput')
            
            if self.config.post_norm:
                verbosePrint(f'\tApplying post-norm layer after FFN', self.verbose)
                ffn_output_features = self.postNormLayer(ffn_output_features)
                verbosePrint(f'\tPost-norm output shape: {encodedFeatures.shape}', self.verbose)
                checkTensorShape(encodedFeatures, ['N*B', 'O'], shapeDict, False, 'postNormOutput') 

            encodedFeatures = ffn_output_features

        ################################################################################
        #                     Step 5: Apply final activation (if specified)              #
        ################################################################################
        if self.config.final_activation is not None:
            verbosePrint(f'\tApplying final activation: {self.config.final_activation}', self.verbose, separator=True)
            encodedFeatures = self.final_activation(encodedFeatures)
            verbosePrint(f'\tFinal output shape after activation: {encodedFeatures.shape}', self.verbose)
            # checkTensorShape(encodedFeatures, ['N*B', 'O'], shapeDict, False, 'finalOutput')
        # Reshape back to [B, N, O] if needed
        if normalizedTokens.shape != inputTokens.shape:
            verbosePrint(f'\tReshaping output to include batch dimension: {batchSize}', verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            encodedFeatures = encodedFeatures.view(batchSize, -1, encodedFeatures.shape[-1])

        if self.config.skip_connection:
            verbosePrint(f'\tApplying skip connection from input to output', self.verbose, separator=True)
            if encodedFeatures.shape[-1] != inputTokens.shape[-1]:
                raise ValueError(f'Cannot apply skip connection, output feature dimension {encodedFeatures.shape[-1]} does not match input feature dimension {inputTokens.shape[-1]}')
            encodedFeatures = encodedFeatures + inputTokens
            verbosePrint(f'\tOutput shape after skip connection: {encodedFeatures.shape}', self.verbose)
            # checkTensorShape(encodedFeatures, ['N*B', 'O'], shapeDict, False, 'skipConnectionOutput')

        verbosePrint(f'\tOutput encoded features shape: {encodedFeatures.shape}', self.verbose)
        verbosePrint(f'Done running Input Encode Layer.', self.verbose, separator=True)

        return encodedFeatures # Shape: [num_tokens, output_dim]

