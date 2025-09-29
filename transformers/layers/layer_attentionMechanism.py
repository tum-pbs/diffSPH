from copy import error
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
from .networkUtil import verbosePrint, verboseBannerPrint, shapeMatch, checkTensorShape, mergeConfigWithKwargs
import copy
from .sparse import buildSparseTensor
from .softmax import softmax
from .mlp import buildMLPwDict, getDefaultMLPDict
from .layer_positionEncoder import BasisEncoder, computeBasisEncoderOutputShape
from .windows import getWindowFunction
from .softmax import softmax_


# This is the basic attention mechanism that computes the attention scores (after softmax and scaling)
# The inputs are:
# queryTokens: (num_query_nodes, latent_dim) - the query node features
# keyTokens:   (num_key_nodes, latent_dim)   - the key node features
# edge_index:  (2, num_edges)         - the edges defining which key nodes are connected to which query nodes
# edge_attr:   (num_edges, edge_dim)   - the edge features for each edge
# s_k:         (num_edges) - scaling factors for the attention across all attention heads
# 
# The output is:
# attentionScoresSparse: (num_edges, num_heads) - the sparse attention scores
# 
# Configuration Parameters are:
# - latent_dim: int - the dimensionality of the input features
# - edge_dim: int - the dimensionality of the edge features
# - transformer_dim: Optional[int] - the dimensionality of the transformer (if None, set to latent_dim)
# - num_heads: int - the number of attention heads
# - attentionMechanism: str - the type of attention mechanism to use ('dot', 'scaled_dot', 'mlp', 'biLinearForm')
#
# Query/Key Parameters:
# - linearEncode: bool - whether to linearly encode the query and key features
# - linearEncodeDict: dict - dictionary with parameters for the linear encoding MLP
# - linearEncodeShared: bool - whether to share the linear encoding MLP between query and key
#
# Attention Score Parameters:
# - attentionScoreMLPDict: dict - dictionary with parameters for the attention score MLP
# - attentionDropout: float - dropout rate for the attention scores
# - attentionScaling: bool - whether to scale the attention scores by sqrt(latent_dim / num_heads)
# - attentionClipping: bool - whether to clip the attention scores
# - attentionClippingValue: float - the value to clip the attention scores to (if attentionClipping is True)
#
# Relative Position Bias Parameters:                 
# - relativePositionBias: If True, the relative distance of each edge is encoded and added to the input features
# - relativePositionBiasScaledPositions: If True, the input positions are scaled by a given cutoff radius before encoding
# - relativePositionBiasMultiplicative: If True, the relative position encoding is multiplied to the input features instead of added
# - relativePositionBiasBaseEncoding: If True, the relative position is encoded using a basis function encoding (e.g. Fourier or Gaussian basis)
# - relativePositionBiasBaseFunction: Type of basis function encoding to use for relative position (e.g. 'fourier', 'gaussian')
# - relativePositionBiasBaseTerms: Number of basis functions to use for relative position encoding
# - relativePositionBiasLinear: If true the rpb is a result of the (potentially encoded) positions passed through a linear layer to match the input feature dimension, if false an MLP is used
# - relativePositionBiasMLPDict: Dictionary defining the MLP architecture for relative position bias encoding (if relativePositionBiasLinear is False)
#
# Window Function Parameters:
# - windowFunction: bool - If True, a window function is applied to the attention based on the edge parameters
# - windowFunctionType: str - Type of window function to use ('cubic', 'quartic', etc.)



from typing import Optional, Union, Tuple
from dataclasses import dataclass, field
from .layer_mixing import TokenMixer, TokenMixerConfig
from .layer_positionEncoder import BasisEncoderConfig



@dataclass(slots=True)
class AttentionMechanismConfig:
    mechanism: str = field(default='dot', metadata={"help": "Type of attention mechanism to use ('dot', 'scaled_dot', 'mix', 'cosine')"})

    dropout: float = field(default=0.0, metadata={"help": "Dropout rate for the attention scores"})
    # scaling: bool = field(default=True, metadata={"help": "Whether to scale the attention scores by sqrt(latent_dim / num_heads)"})

    clipping: bool = field(default=False, metadata={"help": "Whether to clip the attention scores"})
    clipping_value: float = field(default=1.0, metadata={"help": "The value to clip the attention scores to (if clipping is True)"})
    activation: str = field(default='leaky_relu(0.2)', metadata={"help": "Activation function to use for attention score MLP"})


@dataclass(slots=True)
class AttentionLayerConfig:
    token_input_dim: int = field(default=0, metadata={"help": "Dimensionality of the input feature vector per token"})
    spatial_dim: int = field(default=0, metadata={"help": "Dimensionality of the position vector per token (e.g. 3 for 3D positions)"})

    edge_feature_dim: int = field(default=0, metadata={"help": "Dimensionality of the edge feature vector per edge"})
    
    attention_heads: int = field(default=4, metadata={"help": "Number of attention heads"})
    transformer_features: Optional[int] = field(default=None, metadata={"help": "Dimensionality of the attention features per head (if None, set to token_input_dim / attention_heads)"})
    attention_mechanism: AttentionMechanismConfig = field(default_factory=AttentionMechanismConfig, metadata={"help": "Configuration for the attention mechanism"})
    attention_softmax: bool = field(default=True, metadata={"help": "Whether to apply softmax to the attention scores"})

    encode_tokens: bool = field(default=True, metadata={"help": "Whether to encode the query and key tokens"})
    encode_tokens_linear: bool = field(default=True, metadata={"help": "Whether to use a linear layer for token encoding (if False, use MLP)"})
    encode_tokens_mlp_dict: Optional[dict] = field(default=None, metadata={"help": "Dictionary defining the MLP architecture for token encoding (if encode_tokens_linear is False)"})
    encode_tokens_shared: bool = field(default=False, metadata={"help": "Whether to share the token encoding MLP between query and key"})
    encode_using_cconv: bool = field(default=False, metadata={"help": "Whether to use continuous convolution mode (use edge features to compute W_Q and W_K)"})
    cconv_source: str = field(default='rpb', metadata={"help": "Whether to use edge features or relative position bias to compute the cconv weights ('edge', 'rpb', 'spatial', 'window', 'spatial_length')"})
    cconv_linear: bool = field(default=True, metadata={"help": "Whether to use a linear layer for cconv weight computation (if False, use MLP)"})
    cconv_mlp_dict: Optional[dict] = field(default=None, metadata={"help": "Dictionary defining the MLP architecture for cconv weight computation (if cconv_linear is False)"})

    # position_bias: bool = field(default=True, metadata={"help": "Whether to use relative position bias"})
    # position_bias_mixing: str = field(default='add', metadata={"help": "Whether to add or multiply the position bias to the attention scores ('add', 'mul')"})
    # position_bias_after_attention: bool = field(default=True, metadata={"help": "Whether to add the position bias after the attention scores (if False, add before softmax)"})
    # position_bias_per_head: bool = field(default=True, metadata={"help": "Whether to have a separate position bias per attention head (if False, use a single position bias for all heads)"})

    window_function: bool = field(default=False, metadata={"help": "Whether to apply a window function to the attention scores"})
    window_function_type: str = field(default='cubicSpline', metadata={"help": "Type of window function to use ('cubicSpline', 'wendland4', etc.)"})
    window_function_normalized: bool = field(default=True, metadata={"help": "Whether the window function is normalized to 1 over the edges"})
    window_function_before_softmax: bool = field(default=True, metadata={"help": "Whether to apply the window function before the softmax (if False, apply after softmax)"})
    window_function_mixing : str = field(default='mul', metadata={"help": "Whether to add or multiply the window function to the attention scores ('add', 'mul')"})


    preAttentionMixer: Optional[TokenMixerConfig] = field(default=None,)
    postAttentionMixer: Optional[TokenMixerConfig] = field(default=None,)
    position_bias_config: Optional[BasisEncoderConfig] = field(default=None, metadata={"help": "Configuration for the relative position bias basis encoder"})

def build_projection(linear, inputDim, outputDim, dict = None, verbose = False, verbosePrefix = ''):
    if linear:
        verbosePrint(f'Building linear projection from {inputDim} to {outputDim}', verbose, verbosePrefix=verbosePrefix+'\t')
        return nn.Linear(inputDim, outputDim)
    else:
        if dict is None:
            dict = getDefaultMLPDict()
        verbosePrint(f'Building MLP projection from {inputDim} to {outputDim} with config: {dict}', verbose, verbosePrefix=verbosePrefix+'\t')
        return buildMLPwDict(dict, verbose=verbose, verbosePrefix=verbosePrefix+'\t', inputDim=inputDim, outputDim=outputDim)

class AttentionMechanismLayer(torch.nn.Module):
    def __init__(self, 
                config : AttentionLayerConfig,
                verbose: bool = False,
                verbosePrefix: str = '',
                **kwargs
                 ):
        verboseBannerPrint(f'Initializing Attention Mechanism Layer...', verbose)
        super(AttentionMechanismLayer, self).__init__()
        verbosePrint(f'Initializing Attention Mechanism Layer with parameters:', verbose, separator=True)

        config = copy.deepcopy(config)
        self.config = mergeConfigWithKwargs(config, **kwargs)
        self.verbose = verbose
        self.verbosePrefix = verbosePrefix

        ################################################################################
        #                           Set Class Parameters                               #
        ################################################################################
        self.token_input_dim = self.config.token_input_dim
        self.edge_feature_dim = self.config.edge_feature_dim
        self.num_heads = self.config.attention_heads
        self.transformer_features = self.config.transformer_features if self.config.transformer_features is not None else self.config.token_input_dim // self.num_heads

        # self.transformer_dim = self.transformer_features * self.num_heads
        verbosePrint(f'\tLatent dimension: {self.token_input_dim}', self.verbose)
        verbosePrint(f'\tEdge dimension: {self.edge_feature_dim}', self.verbose)
        verbosePrint(f'\tTransformer features: {self.transformer_features}', self.verbose)
        verbosePrint(f'\tNumber of heads: {self.num_heads}', self.verbose)

        ###################################################################################
        # While complex, we can break the attention mechanism into a few key steps:
        # 1. (Optional) Encode the query and key tokens using a shared or separate MLP
        # 2. Scatter the query and key tokens to the edges using the edge_index
        # 3. (Optional) Compute a relative position bias for each edge using a basis function encoding
        # 4. (Optional) Use the spatial edge features to compute the cconv weights for W_Q and W_K and apply them
        # 5. (Optional) Compute the window function values for each edge
        #
        # At this point we have as inputs to the attention mechanism:
        # - Q_i: (num_edges, num_heads, transformer_features) - the query tokens scattered to the edges
        # - K_j: (num_edges, num_heads, transformer_features) - the key tokens scattered to the edges
        # - e_ij: (num_edges, edge_feature_dim) - the edge features
        # - rpb_features: (num_edges, rpb_feature_dim) - the relative position bias features (if using relative position bias)
        # - spatial_features: (num_edges, spatial_dim) - the spatial edge features (if using cconv)
        # - window_values: (num_edges) - the window function values (if using window function)
        #
        # 6. Compute the attention scores using the specified attention mechanism with the TokenMixer
        # 7. (Optional) if there is a post attention mixer, apply it to the attention scores
        # 8. (Optional) Apply the window function to the attention scores
        # 9. (Optional) Clip the attention scores
        # 10. (Optional) Apply dropout to the attention scores
        # 11. (Optional) Apply softmax to the attention scores
        #
        # The output is:
        # - attentionScoresSparse: (num_edges, num_heads) - the sparse attention scores

        ################################################################################
        #                        Encode Query and Key Tokens                           #
        ################################################################################
        verboseBannerPrint(f'Encoding Query and Key Tokens...', self.verbose)
        self.encode_query = nn.Identity()
        self.encode_key = nn.Identity()
        if self.config.encode_tokens:
            if self.config.encode_using_cconv:
                # If using continuous convolution mode, we do not encode the tokens here
                verbosePrint(f'Using continuous convolution mode, skipping token encoding.', self.verbose)
            else:
                self.encode_query = build_projection(
                    linear = self.config.encode_tokens_linear,
                    inputDim = self.token_input_dim,
                    outputDim = self.num_heads * self.transformer_features,
                    dict = self.config.encode_tokens_mlp_dict,
                    verbose = self.verbose,
                    verbosePrefix = self.verbosePrefix+'\t'
                )
                if self.config.encode_tokens_shared:
                    self.encode_key = self.encode_query
                    verbosePrint(f'Sharing token encoding MLP between query and key.', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
                else:
                    self.encode_key = build_projection(
                        linear = self.config.encode_tokens_linear,
                        inputDim = self.token_input_dim,
                        outputDim = self.num_heads * self.transformer_features,
                        dict = self.config.encode_tokens_mlp_dict,
                        verbose = self.verbose,
                        verbosePrefix = self.verbosePrefix+'\t'
                    )
        verbosePrint(f'Query token encoder: {self.encode_query}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        verbosePrint(f'Key token encoder: {self.encode_key}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        ################################################################################
        #                    (Optional) Relative Position Bias                         #
        ################################################################################
        verboseBannerPrint(f'Setting up Relative Position Bias...', self.verbose)

        if self.config.position_bias_config is not None:
            self.position_bias_encoder = BasisEncoder(self.config.position_bias_config, verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            rpb_feature_dim = computeBasisEncoderOutputShape(self.config.position_bias_config)[-1]
            verbosePrint(f'Using relative position bias with feature dimension {rpb_feature_dim}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        else:
            self.position_bias_encoder = nn.Identity()
            rpb_feature_dim = 0
            verbosePrint(f'Not using relative position bias.', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        ################################################################################
        #                             (Optional) Cconv Mode                            #
        ################################################################################
        verboseBannerPrint(f'Setting up Continuous Convolution Mode...', self.verbose)
        if self.config.encode_using_cconv:
            cconv_input_dim = 0
            if self.config.cconv_source == 'edge':
                if self.edge_feature_dim == 0:
                    raise ValueError("AttentionMechanismLayer: cconv_source is 'edge' but edge_feature_dim is 0")
                cconv_input_dim = self.edge_feature_dim
                verbosePrint(f'Using edge features for cconv source with dimension {cconv_input_dim}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            elif self.config.cconv_source == 'rpb':
                if rpb_feature_dim == 0:
                    raise ValueError("AttentionMechanismLayer: cconv_source is 'rpb' but relative position bias is not used")
                cconv_input_dim = rpb_feature_dim
                verbosePrint(f'Using relative position bias for cconv source with dimension {cconv_input_dim}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            elif self.config.cconv_source == 'spatial':
                if self.config.spatial_dim == 0:
                    raise ValueError("AttentionMechanismLayer: cconv_source is 'spatial' but spatial_dim is 0")
                cconv_input_dim = self.config.spatial_dim
                verbosePrint(f'Using spatial features for cconv source with dimension {cconv_input_dim}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            elif self.config.cconv_source == 'spatial_length':
                if self.config.spatial_dim == 0:
                    raise ValueError("AttentionMechanismLayer: cconv_source is 'spatial_length' but spatial_dim is 0")
                cconv_input_dim = 1
                verbosePrint(f'Using spatial length for cconv source with dimension {cconv_input_dim}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            elif self.config.cconv_source == 'window':
                cconv_input_dim = 1
                verbosePrint(f'Using window function values for cconv source with dimension {cconv_input_dim}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            else:
                raise ValueError(f"AttentionMechanismLayer: cconv_source must be one of 'edge', 'rpb', 'spatial', 'spatial_length', 'window', got {self.config.cconv_source}")

            self.cconv_W_Q = build_projection(
                linear = self.config.cconv_linear,
                inputDim = cconv_input_dim,
                outputDim = self.num_heads * self.transformer_features * self.token_input_dim,
                dict = self.config.cconv_mlp_dict,
                verbose = self.verbose,
                verbosePrefix = self.verbosePrefix+'\t'
            )
            if self.config.encode_tokens_shared:
                self.cconv_W_K = self.cconv_W_Q
                verbosePrint(f'Sharing cconv weight MLP between W_Q and W_K.', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            else:
                self.cconv_W_K = build_projection(
                    linear = self.config.cconv_linear,
                    inputDim = cconv_input_dim,
                    outputDim = self.num_heads * self.transformer_features * self.token_input_dim,
                    dict = self.config.cconv_mlp_dict,
                    verbose = self.verbose,
                    verbosePrefix = self.verbosePrefix+'\t'
                )
            verbosePrint(f'cconv W_Q projection: {self.cconv_W_Q}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            verbosePrint(f'cconv W_K projection: {self.cconv_W_K}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        ################################################################################
        # Attention Mixer (to compute attention scores) #
        ################################################################################

        verboseBannerPrint(f'Setting up Attention Mechanism...', self.verbose)

        self.preAttentionMixer = copy.deepcopy(self.config.preAttentionMixer) if self.config.preAttentionMixer is not None else TokenMixerConfig()

        self.preAttentionMixer.num_heads = self.num_heads
        self.preAttentionMixer.transformer_features = self.transformer_features
        self.preAttentionMixer.mixing_out_features = 1
        self.preAttentionMixer.input_channels = 2

        self.preAttentionMixer.spatial_dim = self.config.spatial_dim
        self.preAttentionMixer.edge_feature_dim = self.edge_feature_dim
        self.preAttentionMixer.rpb_feature_dim = rpb_feature_dim
        self.preAttentionMixer.channel_mixing = True
        verbosePrint(f'Pre Attention Mixer config: {self.preAttentionMixer}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        mode = self.config.attention_mechanism.mechanism
        if mode == 'dot':
            self.preAttentionMixer.channel_mixing_operation = 'dot'
            self.preAttentionMixer.channel_normalization = None
            verbosePrint(f'Using dot product attention mechanism.', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        elif mode == 'scaled_dot':  
            self.preAttentionMixer.channel_mixing_operation = 'scaled_dot'
            self.preAttentionMixer.channel_normalization = 'scaled'
            verbosePrint(f'Using scaled dot product attention mechanism.', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        elif mode == 'cosine':
            self.preAttentionMixer.channel_mixing_operation = 'cosine'
            self.preAttentionMixer.channel_normalization = 'cosine'
            verbosePrint(f'Using cosine similarity attention mechanism.', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        elif mode == 'mlp':
            self.preAttentionMixer.channel_mixing_operation = 'project'
            self.preAttentionMixer.channel_projection_linear = False
            self.preAttentionMixer.channel_projection_mlp_dict = self.config.encode_tokens_mlp_dict
            self.preAttentionMixer.channel_normalization = None
            verbosePrint(f'Using MLP attention mechanism.', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        elif mode == 'linear':
            self.preAttentionMixer.channel_mixing_operation = 'project'
            self.preAttentionMixer.channel_projection_linear = True
            self.preAttentionMixer.channel_normalization = None
            verbosePrint(f'Using linear attention mechanism.', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        else:
            raise ValueError(f"AttentionMechanismLayer: attention_mechanism must be one of 'dot', 'scaled_dot', 'mlp', 'linear', 'cosine', got {mode}")

        self.preMixer = TokenMixer(self.preAttentionMixer, verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        verboseBannerPrint(f'Post Attention Mixer...', self.verbose)
        self.postAttentionMixer = nn.Identity()
        if self.config.postAttentionMixer is not None:
            self.config.postAttentionMixer.num_heads = self.num_heads
            self.config.postAttentionMixer.transformer_features = 1
            self.config.postAttentionMixer.mixing_out_features = 1
            self.config.postAttentionMixer.input_channels = 1

            self.config.postAttentionMixer.spatial_dim = self.config.spatial_dim
            self.config.postAttentionMixer.edge_feature_dim = self.edge_feature_dim
            self.config.postAttentionMixer.rpb_feature_dim = rpb_feature_dim

            self.postAttentionMixer = TokenMixer(self.config.postAttentionMixer, verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            verbosePrint(f'Post attention mixer: {self.postAttentionMixer}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        else:
            verbosePrint(f'Not using post attention mixer.', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        ################################################################################
        #                           (Optional) Dropout                     #
        ################################################################################
        self.attention_dropout = nn.Dropout(self.config.attention_mechanism.dropout) if self.config.attention_mechanism.dropout > 0.0 else nn.Identity()
        verbosePrint(f'Attention dropout: {self.attention_dropout}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        ################################################################################
        #                                  Finalize                                    #
        ################################################################################

        verbosePrint(f'Attention Mechanism Layer initialized.', verbose, separator=True)
        


    def forward(self, 
                queryTokens: Tensor, # (num_query_nodes, latent_dim) (current tokens)
                keyTokens: Tensor,  # (num_key_nodes, latent_dim) (neighbor tokens)
                edge_index: Tensor, # (2, num_edges)
                edge_attr: Optional[Tensor] = None, # (num_edges, edge_dim)
                edge_features: Optional[Tensor] = None, # (num_edges, F)
                ):
        verboseBannerPrint(f'Running Attention Mechanism Layer...', self.verbose)
        verbosePrint(f'\tQuery tokens shape: {queryTokens.shape} [B, Q, L]', self.verbose)
        verbosePrint(f'\tKey tokens shape: {keyTokens.shape} [B, K, L]', self.verbose)
        verbosePrint(f'\tEdge index shape: {edge_index.shape} [2, E]', self.verbose)
        verbosePrint(f'\tEdge attr shape: {edge_attr.shape if edge_attr is not None else None} [E, D]', self.verbose)
        verbosePrint(f'\tEdge features shape: {edge_features.shape if edge_features is not None else None} [E, F]', self.verbose)
        rows = edge_index[0]
        cols = edge_index[1]
        num_edges = edge_index.shape[1]
        batch_size = queryTokens.shape[0]
        num_query_nodes = queryTokens.shape[1]
        num_key_nodes = keyTokens.shape[1]

        # 1. (Optional) Encode the query and key tokens using a shared or separate MLP
        # 2. Scatter the query and key tokens to the edges using the edge_index
        # 3. (Optional) Compute a relative position bias for each edge using a basis function encoding
        # 4. (Optional) Use the spatial edge features to compute the cconv weights for W_Q and W_K and apply them
        # 5. (Optional) Compute the window function values for each edge
        # 6. Compute the attention scores using the specified attention mechanism with the TokenMixer
        # 7. (Optional) if there is a post attention mixer, apply it to the attention scores
        # 8. (Optional) Apply the window function to the attention scores
        # 9. (Optional) Clip the attention scores
        # 10. (Optional) Apply dropout to the attention scores
        # 11. (Optional) Apply softmax to the attention scores
        #
        # The output is:
        # - attentionScoresSparse: (num_edges, num_heads) - the sparse attention scores

        ################################################################################
        #                          Encode Query and Key Tokens                        #
        ################################################################################
        verboseBannerPrint(f'Encoding query and key tokens...', self.verbose)
        verbosePrint(f'Query tokens shape before encoding: {queryTokens.shape} [B, Q, L]', self.verbose)
        verbosePrint(f'Key tokens shape before encoding: {keyTokens.shape} [B, K, L]', self.verbose)

        queryTokensEncoded = self.encode_query(queryTokens)
        keyTokensEncoded = self.encode_key(keyTokens)

        if self.config.encode_using_cconv:
            verbosePrint(f'Using continuous convolution mode so nothing happens here!', self.verbose)

        verbosePrint(f'Query tokens encoded shape: {queryTokensEncoded.shape} [B, Q, H*T]', self.verbose)
        verbosePrint(f'Key tokens encoded shape: {keyTokensEncoded.shape} [B, K, H*T]', self.verbose)

        ################################################################################
        #                        Scatter Tokens to Edges                               #
        ################################################################################
        verboseBannerPrint(f'Scattering query and key tokens to edges...', self.verbose)

        flattenedQueryTokens = queryTokensEncoded.flatten(0, 1)
        flattenedKeyTokens = keyTokensEncoded.flatten(0, 1)
        verbosePrint(f'Flattened query tokens shape: {flattenedQueryTokens.shape} [B*Q, H*T]', self.verbose)
        verbosePrint(f'Flattened key tokens shape: {flattenedKeyTokens.shape} [B*K, H*T]', self.verbose)

        Q_i = flattenedQueryTokens[edge_index[0]]  # (num_edges, H*T)
        K_j = flattenedKeyTokens[edge_index[1]]    # (num_edges, H*T)
        verbosePrint(f'Scattered query tokens Q_i shape: {Q_i.shape} [E, H*T]', self.verbose)
        verbosePrint(f'Scattered key tokens K_j shape: {K_j.shape} [E, H*T]', self.verbose)

        ################################################################################
        #                  (Optional) Relative Position Bias                         #
        ################################################################################
        verboseBannerPrint(f'Computing Relative Position Bias...', self.verbose)

        rpb_features = None
        if self.config.position_bias_config is not None:
            if edge_attr is None:
                raise ValueError("AttentionMechanismLayer: position_bias_config is set but edge_attr is None")
            rpb_features = self.position_bias_encoder(edge_attr)
            verbosePrint(f'Relative position bias features shape: {rpb_features.shape} [E, R]', self.verbose)

        ################################################################################
        #                             (Optional) Cconv Mode                            #
        ################################################################################
        verboseBannerPrint(f'Applying Continuous Convolution Mode...', self.verbose)

        if self.config.encode_using_cconv:
            cconv_input = None
            if self.config.cconv_source == 'edge':
                if edge_features is None:
                    raise ValueError("AttentionMechanismLayer: cconv_source is 'edge' but edge_features is None")
                cconv_input = edge_features
                verbosePrint(f'Using edge features for cconv input with shape {cconv_input.shape} [E, D]', self.verbose)
            elif self.config.cconv_source == 'rpb':
                if rpb_features is None:
                    raise ValueError("AttentionMechanismLayer: cconv_source is 'rpb' but relative position bias features are None")
                cconv_input = rpb_features
                verbosePrint(f'Using relative position bias features for cconv input with shape {cconv_input.shape} [E, R]', self.verbose)
            elif self.config.cconv_source == 'spatial':
                if edge_attr is None:
                    raise ValueError("AttentionMechanismLayer: cconv_source is 'spatial' but edge_attr is None")
                cconv_input = edge_attr
                verbosePrint(f'Using spatial edge features for cconv input with shape {cconv_input.shape} [E, F]', self.verbose)
            elif self.config.cconv_source == 'spatial_length':
                if edge_attr is None:
                    raise ValueError("AttentionMechanismLayer: cconv_source is 'spatial_length' but edge_attr is None")
                cconv_input = torch.norm(edge_attr, dim=-1, keepdim=True)
                verbosePrint(f'Using spatial length for cconv input with shape {cconv_input.shape} [E, 1]', self.verbose)
            elif self.config.cconv_source == 'window':
                # We will compute the window function later, so just use a placeholder here
                cconv_input = torch.ones((edge_index.shape[1], 1), device=edge_index.device)
                verbosePrint(f'Using placeholder ones for window function for cconv input with shape {cconv_input.shape} [E, 1]', self.verbose)
            else:
                raise ValueError(f"AttentionMechanismLayer: cconv_source must be one of 'edge', 'rpb', 'spatial', 'spatial_length', 'window', got {self.config.cconv_source}")

            W_Q = self.cconv_W_Q(cconv_input)  # (num_edges, H*T*L)
            W_K = self.cconv_W_K(cconv_input)  # (num_edges, H*T*L)
            verbosePrint(f'cconv W_Q shape: {W_Q.shape} [E, H*T*L]', self.verbose)
            verbosePrint(f'cconv W_K shape: {W_K.shape} [E, H*T*L]', self.verbose)

            W_Q = W_Q.view(-1, self.num_heads * self.transformer_features, self.token_input_dim)  # (num_edges, H * T, L)
            W_K = W_K.view(-1, self.num_heads * self.transformer_features, self.token_input_dim)  # (num_edges, H * T, L)

            Q_i = torch.einsum('el, etl -> et', Q_i, W_Q)  # (num_edges, H*T)
            K_j = torch.einsum('el, etl -> et', K_j, W_K)  # (num_edges, H*T)
            verbosePrint(f'cconv applied Q_i shape: {Q_i.shape} [E, H*T]', self.verbose)
            verbosePrint(f'cconv applied K_j shape: {K_j.shape} [E, H*T]', self.verbose)

        ################################################################################
        #                       Reshape Q_i and K_j for Attention                      #
        ################################################################################
        verboseBannerPrint(f'Reshaping Q_i and K_j for attention...', self.verbose)

        Q_i = Q_i.view(-1, self.num_heads, self.transformer_features)  # (num_edges, H, T)
        K_j = K_j.view(-1, self.num_heads, self.transformer_features)  # (num_edges, H, T)
        verbosePrint(f'reshaped Q_i shape: {Q_i.shape} [E, H, T]', self.verbose)
        verbosePrint(f'reshaped K_j shape: {K_j.shape} [E, H, T]', self.verbose)

        ################################################################################
        # Compute Window Function Values #
        ################################################################################
        windowScaling = None
        if self.config.window_function_type is not None:
            verboseBannerPrint(f'Computing Window Function Values...', self.verbose)
            if edge_attr is None:
                raise ValueError("AttentionMechanismLayer: window_function is True but edge_attr is None")
            windowScaling = getWindowFunction(self.config.window_function_type, norm= None)(torch.linalg.norm(edge_attr, dim=-1))
            verbosePrint(f'\tWindow function shape: {windowScaling.shape} [E]', self.verbose)
            # The scaling here is not normalized to 1 for the window function, we need to make sure the sum is still 1 after applying the window function
            if self.config.window_function_normalized:
                verbosePrint(f'Window function is normalized to 1 over the edges.', self.verbose)
                numNeighbors = scatter(torch.ones_like(rows), rows, dim=0, dim_size=batch_size * num_query_nodes, reduce='sum')  # Shape: [num_query_nodes]
                windowScaling_sum = scatter(windowScaling, rows, dim=0, dim_size=batch_size * num_query_nodes, reduce='sum')  # Shape: [num_query_nodes]
                windowScaling_sum = windowScaling_sum[rows]  # Shape: [num_edges]
                # print(windowScaling_sum)
                windowScaling = numNeighbors[rows] * windowScaling / (windowScaling_sum + 1e-16)  # Normalize to sum to 1 for each query node

            verbosePrint(f'Window function shape: {windowScaling.shape} [E]', self.verbose)

        ################################################################################
        # Compute Attention Scores #
        ################################################################################

        verboseBannerPrint(f'Computing Attention Scores...', self.verbose)
        attention_scores = self.preMixer(
            tokens = [Q_i, K_j],
            edgeTokens = edge_features,
            spatialTokens = edge_attr,
            positionBiasTokens = rpb_features,
            windowValues= windowScaling
        )  # (num_edges, H, 1)
        verbosePrint(f'Raw attention scores shape: {attention_scores.shape} [E, H, 1]', self.verbose)

        if self.config.postAttentionMixer is not None:
            attention_scores = self.postAttentionMixer(
                tokens = [attention_scores],
                edgeTokens = edge_features,
                spatialTokens = edge_attr,
                positionBiasTokens = rpb_features,
                windowValues= windowScaling
            )[0]  # (num_edges, H, 1)
            verbosePrint(f'Post-mixed attention scores shape: {attention_scores.shape} [E, H, 1]', self.verbose)

        ################################################################################
        # Softmax, Dropout, Clipping, Window Function #
        ################################################################################

        if self.config.window_function and self.config.window_function_before_softmax:
            attention_scores = attention_scores * windowScaling.view(-1, 1, 1)  # (num_edges, 1, 1)
            verbosePrint(f'Applied window function before softmax, attention scores shape: {attention_scores.shape} [E, H, 1]', self.verbose)

        if self.config.attention_mechanism.clipping:
            clip_value = self.config.attention_mechanism.clipping_value
            attention_scores = torch.clamp(attention_scores, -clip_value, clip_value)
            verbosePrint(f'Clipped attention scores to range [-{clip_value}, {clip_value}]', self.verbose)

        if self.config.attention_mechanism.dropout > 0:
            attention_scores = self.attention_dropout(attention_scores)
            verbosePrint(f'Applied dropout to attention scores with probability {self.config.attention_mechanism.dropout}', self.verbose)

        if self.config.attention_softmax:
            sparse_values = attention_scores.flatten()  # Shape: [num_edges * num_heads]
            size = (1, self.num_heads, num_query_nodes * batch_size, num_key_nodes * batch_size)
            attentionScoresSparse, sparse_indices = buildSparseTensor(rows, cols, sparse_values, size)

            attention_scores = softmax(attentionScoresSparse, sparse_values, rows, cols, sparse_indices)  # (num_edges, H)
            verbosePrint(f'Applied softmax to attention scores, shape: {attention_scores.shape} [E, H]', self.verbose)
        else:
            attention_scores = attention_scores.view(-1, self.num_heads)  # (num_edges, H)
            verbosePrint(f'Skipped softmax on attention scores, shape: {attention_scores.shape} [E, H]', self.verbose)

        if self.config.window_function and not self.config.window_function_before_softmax:
            attention_scores = attention_scores * windowScaling.view(-1, 1)  # (num_edges, 1)
            verbosePrint(f'Applied window function after softmax, attention scores shape: {attention_scores.shape} [E, H]', self.verbose)
            
        attention_scores = attention_scores.unsqueeze(-1)  # (num_edges, H, 1)

        # attention_scores = attention_scores.permute(1,0,2).contiguous()  # (H, num_edges, 1)

        verbosePrint(f'Final attention scores shape: {attention_scores.shape} [E, H]', self.verbose)
        verbosePrint(f'Attention Mechanism Layer complete.', self.verbose, separator=True)

        return attention_scores  # (num_edges, H, 1)
