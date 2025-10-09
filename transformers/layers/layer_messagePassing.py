import warnings
from numpy import integer
import torch
from torch import Tensor
import torch.nn as nn

from .windows import getWindowFunction

try:
    import torch_geometric
    from torch_geometric.utils import scatter, segment
    from torch_geometric.utils.num_nodes import maybe_num_nodes
except ImportError:
    torch_geometric = None
from typing import Optional, Union, Tuple
 

from .activation import getActivationLayer
from .basisFunctions import basisEncoderLayer
from .layer_positionEncoder import BasisEncoder, computeBasisEncoderOutputShape
from .networkUtil import verboseBannerPrint
from .networkUtil import verbosePrint
from .sparse import buildSparseTensor
from .softmax import softmax
from .mlp import buildMLPwDict, getDefaultMLPDict
from .networkUtil import checkTensorShape

from typing import List, Optional
from dataclasses import dataclass, field




from .layer_positionEncoder import BasisEncoder, computeBasisEncoderOutputShape, BasisEncoderConfig
from .layer_mixing import TokenMixer, TokenMixerConfig
@dataclass(slots=True)
class MessagePassingConfig:
    token_input_dim: int = field(default=0, metadata={"help": "Dimensionality of the input feature vector per token"})
    spatial_dim: int = field(default=0, metadata={"help": "Dimensionality of the position vector per token (e.g. 3 for 3D positions)"})

    edge_feature_dim: int = field(default=0, metadata={"help": "Dimensionality of the edge feature vector per edge"})
    
    attention_heads: int = field(default=4, metadata={"help": "Number of attention heads"})
    transformer_features: Optional[int] = field(default=None, metadata={"help": "Dimensionality of the attention features per head (if None, set to token_input_dim / attention_heads)"})

    encode_tokens: bool = field(default=True, metadata={"help": "Whether to encode the query and key tokens"})
    encode_tokens_linear: bool = field(default=True, metadata={"help": "Whether to use a linear layer for token encoding (if False, use MLP)"})
    encode_tokens_mlp_dict: Optional[dict] = field(default=None, metadata={"help": "Dictionary defining the MLP architecture for token encoding (if encode_tokens_linear is False)"})

    messageMixer: Optional[TokenMixerConfig] = field(default=None, metadata={"help": "Configuration for the message token mixer (if None, no mixing is applied before message generation)"})
    messageActivation: Optional[str] = field(default=None, metadata={"help": "Activation function to use after message generation (e.g. 'relu', 'gelu', etc.)"})

    multiHeadAggregation: str = field(default='concat', metadata={"help": "Method to aggregate attention head outputs, valid options: 'concat', 'mean', 'sum', 'max', 'min'"})
    postMessageMixer: Optional[TokenMixerConfig] = field(default=None, metadata={"help": "Configuration for the post-message token mixer (if None, no mixing is applied after message generation)"})

    use_attention: bool = field(default=True, metadata={"help": "Whether to use attention values in token mixing"})
    use_node_i: bool = field(default=True, metadata={"help": "Whether to use the features of the target node (node i) in token mixing"})
    use_node_j: bool = field(default=True, metadata={"help": "Whether to use the features of the source node (node j) in token mixing"})
    use_node_sum: bool = field(default=False, metadata={"help": "Whether to use the sum of the features of node i and j in token mixing"})
    use_node_diff: bool = field(default=False, metadata={"help": "Whether to use the difference of the features of node i and j in token mixing"})
    use_edge_features: bool = field(default=False, metadata={"help": "Whether to use edge features in token mixing"})
    use_window_function: bool = field(default=False, metadata={"help": "Whether to use a window function based on distance in token mixing"})
    use_spatial: bool = field(default=False, metadata={"help": "Whether to use spatial information (e.g. relative position encoding) in token mixing"})
    use_rpb: bool = field(default=False, metadata={"help": "Whether to use relative position bias in token mixing"})

    rpb_feature_dim: int = field(default=0, metadata={"help": "Dimensionality of the relative position bias feature vector per edge (if using relative position bias)"})
    position_bias_config: Optional[BasisEncoderConfig] = field(default=None, metadata={"help": "If provided, a BasisEncoderConfig to encode the spatial information before mixing. The input dimension of the encoder must match the spatial_dim."})

    window_function_type: str = field(default='cubicSpline', metadata={"help": "Type of window function to use ('cubicSpline', 'wendland4', etc.)"})

import copy
from .networkUtil import mergeConfigWithKwargs

def build_projection(linear, inputDim, outputDim, dict = None, verbose = False, verbosePrefix = ''):
    if linear:
        verbosePrint(f'Building linear projection from {inputDim} to {outputDim}', verbose, verbosePrefix=verbosePrefix+'\t')
        return nn.Linear(inputDim, outputDim, bias = False)
    else:
        if dict is None:
            dict = getDefaultMLPDict()
        verbosePrint(f'Building MLP projection from {inputDim} to {outputDim} with config: {dict}', verbose, verbosePrefix=verbosePrefix+'\t')
        return buildMLPwDict(dict, verbose=verbose, verbosePrefix=verbosePrefix+'\t', inputDim=inputDim, outputDim=outputDim)

class MessagePassingLayer(torch.nn.Module):
    def __init__(self, 
                config : MessagePassingConfig,
                verbose: bool = False,
                verbosePrefix: str = '',
                **kwargs
                 ):
        super(MessagePassingLayer, self).__init__()
        verboseBannerPrint('Initializing MessagePassingLayer', verbose)

        config = copy.deepcopy(config)
        self.config = mergeConfigWithKwargs(config, **kwargs)
        self.verbose = verbose
        self.verbosePrefix = verbosePrefix

        self.token_input_dim = self.config.token_input_dim
        self.edge_feature_dim = self.config.edge_feature_dim
        self.num_heads = self.config.attention_heads
        self.transformer_features = self.config.transformer_features if self.config.transformer_features is not None else self.config.token_input_dim // self.num_heads

        # self.transformer_dim = self.transformer_features * self.num_heads
        verbosePrint(f'\tLatent dimension: {self.token_input_dim}', self.verbose)
        verbosePrint(f'\tEdge dimension: {self.edge_feature_dim}', self.verbose)
        verbosePrint(f'\tTransformer features: {self.transformer_features}', self.verbose)
        verbosePrint(f'\tNumber of heads: {self.num_heads}', self.verbose)


        ################################################################################
        #                        Encode Value Tokens                           #
        ################################################################################
        verboseBannerPrint(f'Encoding Value Tokens...', self.verbose)
        self.encode_value = nn.Identity()
        if self.config.encode_tokens:
            self.encode_value = build_projection(
                linear = self.config.encode_tokens_linear,
                inputDim = self.token_input_dim,
                outputDim = self.num_heads * self.transformer_features,
                dict = self.config.encode_tokens_mlp_dict,
                verbose = self.verbose,
                verbosePrefix = self.verbosePrefix+'\t'
            )
        verbosePrint(f'Value token encoder: {self.encode_value}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

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
            rpb_feature_dim = self.config.rpb_feature_dim
            verbosePrint(f'Not using relative position bias.', self.verbose, verbosePrefix=self.verbosePrefix+'\t')


        self.messageMixer = copy.deepcopy(self.config.messageMixer) if self.config.messageMixer is not None else TokenMixerConfig()

        self.messageMixer.num_heads = self.num_heads
        self.messageMixer.transformer_features = self.transformer_features
        self.messageMixer.mixing_out_features = self.transformer_features
        self.messageMixer.input_channels = 1

        self.messageMixer.spatial_dim = self.config.spatial_dim
        self.messageMixer.edge_feature_dim = self.edge_feature_dim
        self.messageMixer.rpb_feature_dim = rpb_feature_dim

        edge_features = 0
        if self.config.use_edge_features:
            edge_features += self.edge_feature_dim
        if self.config.use_window_function:
            edge_features += 1
        if self.config.use_spatial:
            edge_features += self.config.spatial_dim
        if self.config.use_rpb:
            edge_features += rpb_feature_dim
        if self.config.use_attention:
            edge_features += 1
        if self.config.use_node_i:
            edge_features += self.token_input_dim
        if self.config.use_node_j:
            edge_features += self.token_input_dim
        if self.config.use_node_sum:
            edge_features += self.token_input_dim
        if self.config.use_node_diff:
            edge_features += self.token_input_dim
        verbosePrint(f'Edge features to be used in message mixer: {edge_features}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        self.messageMixer.edge_feature_dim = edge_features

        verbosePrint(f'Message Mixer config: {self.messageMixer}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        self.mixer = TokenMixer(self.messageMixer, verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        self.messageActivation = getActivationLayer(self.config.messageActivation) if self.config.messageActivation is not None else None

        ################################################################################
        #                        Post-Message Mixer                           ##
        ################################################################################
        verboseBannerPrint(f'Setting up Post-Message Mixer...', self.verbose)
        if self.config.postMessageMixer is not None:
            self.postMessageMixer = TokenMixer(self.config.postMessageMixer, verbose=self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            verbosePrint(f'Post-Message Mixer config: {self.config.postMessageMixer}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
        else:
            self.postMessageMixer = None
            verbosePrint(f'No Post-Message Mixer.', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        ################################################################################
        #                        Multi-Head Aggregation                           ##
        ################################################################################
        verboseBannerPrint(f'Setting up Multi-Head Aggregation...', self.verbose)
        valid_aggregations = ['concat', 'mean', 'sum', 'max', 'min']
        if self.config.multiHeadAggregation not in valid_aggregations:
            raise ValueError(f'Invalid multiHeadAggregation: {self.config.multiHeadAggregation}, must be one of {valid_aggregations}')
        self.multiHeadAggregation = self.config.multiHeadAggregation
        verbosePrint(f'Using {self.multiHeadAggregation} for multi-head aggregation.', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        verboseBannerPrint('MessagePassingLayer Built', verbose)




    def forward(self, 
                queryTokens: Tensor,  # (num_query_nodes, latent_dim) (central tokens)
                valueTokens: Tensor,  # (num_key_nodes, latent_dim) (neighbor tokens)
                edge_index: Tensor, # (2, num_edges)
                edgeAttention: Optional[Tensor] = None, # (num_edges, num_heads) or (num_edges, 1)
                edgeTokens: Optional[Tensor] = None, # shape [*, H?, F_e]
                spatialTokens: Optional[Tensor] = None, # shape [*, H?, D]
                positionBiasTokens: Optional[Tensor] = None, # shape [*, H?, F_rpb]
                windowValues: Optional[Tensor] = None, # shape [*,H?]
    ):
        verboseBannerPrint('Running MessagePassingLayer...', self.verbose)
        rows = edge_index[0]
        cols = edge_index[1]
        num_edges = edge_index.shape[1]
        batch_size = valueTokens.shape[0]
        num_key_nodes = valueTokens.shape[1]
        num_query_nodes = queryTokens.shape[1]

        verbosePrint(f'Query tokens shape: {queryTokens.shape} [B, Q, L]', self.verbose)
        verbosePrint(f'Value tokens shape: {valueTokens.shape} [B, V, L]', self.verbose)
        verbosePrint(f'Edge index shape: {edge_index.shape} [2, E]', self.verbose)
        verbosePrint(f'Number of edges: {num_edges}', self.verbose)
        verbosePrint(f'Number of query nodes: {num_query_nodes}', self.verbose)
        verbosePrint(f'Number of key/value nodes: {num_key_nodes}', self.verbose)


        ################################################################################
        #                          Encode Value Tokens                        #
        ################################################################################
        verboseBannerPrint(f'Encoding value tokens...', self.verbose)
        verbosePrint(f'Value tokens shape before encoding: {valueTokens.shape} [B, V, L]', self.verbose)

        valueTokensEncoded = self.encode_value(valueTokens)

        verbosePrint(f'Value tokens encoded shape: {valueTokensEncoded.shape} [B, V, H*T]', self.verbose)

        ################################################################################
        #                        Scatter Tokens to Edges                               #
        ################################################################################
        verboseBannerPrint(f'Scattering value tokens to edges...', self.verbose)

        flattenedValueTokens = valueTokensEncoded.flatten(0, 1)
        verbosePrint(f'Flattened value tokens shape: {flattenedValueTokens.shape} [B*V, H*T]', self.verbose)

        V_j = flattenedValueTokens[edge_index[1]]  # (num_edges, H*T)
        verbosePrint(f'Scattered value tokens V_j shape: {V_j.shape} [E, H*T]', self.verbose)

        ################################################################################
        #                  (Optional) Relative Position Bias                         #
        ################################################################################
        verboseBannerPrint(f'Computing Relative Position Bias...', self.verbose)

        rpb_features = None
        if self.config.position_bias_config is not None:
            if spatialTokens is None:
                raise ValueError("AttentionMechanismLayer: position_bias_config is set but edge_attr is None")
            rpb_features = self.position_bias_encoder(spatialTokens)
            verbosePrint(f'Relative position bias features shape: {rpb_features.shape} [E, R]', self.verbose)

        ################################################################################
        #                       Reshape V_j for Attention                      #
        ################################################################################
        verboseBannerPrint(f'Reshaping V_j for attention...', self.verbose)

        V_j = V_j.view(-1, self.num_heads, self.transformer_features)  # (num_edges, H, T)
        verbosePrint(f'reshaped V_j shape: {V_j.shape} [E, H, T]', self.verbose)

        self.edge_features = []
        if self.config.use_edge_features:
            if edgeTokens is None:
                raise ValueError("MessagePassingLayer: use_edge_features is True but edgeTokens is None")
            self.edge_features.append(edgeTokens)
            verbosePrint(f'Using edge features with shape: {edgeTokens.shape} [E, F_e]', self.verbose)
        if self.config.use_window_function:
            if windowValues is None:
                raise ValueError("MessagePassingLayer: use_window_function is True but windowValues is None")
            self.edge_features.append(windowValues.unsqueeze(-1))
            verbosePrint(f'Using window function values with shape: {windowValues.shape} [E, 1]', self.verbose)
        if self.config.use_spatial:
            if spatialTokens is None:
                raise ValueError("MessagePassingLayer: use_spatial is True but spatialTokens is None")
            self.edge_features.append(spatialTokens)
            verbosePrint(f'Using spatial tokens with shape: {spatialTokens.shape} [E, D]', self.verbose)
        if self.config.use_rpb:
            if rpb_features is None:
                raise ValueError("MessagePassingLayer: use_rpb is True but rpb_features is None")
            self.edge_features.append(rpb_features)
            verbosePrint(f'Using relative position bias features with shape: {rpb_features.shape} [E, R]', self.verbose)
        if self.config.use_attention:
            if edgeAttention is None:
                raise ValueError("MessagePassingLayer: use_attention is True but edgeAttention is None")
            self.edge_features.append(edgeAttention)
            verbosePrint(f'Using attention values with shape: {edgeAttention.shape} [E, H]', self.verbose)
        if self.config.use_node_i:
            node_i = queryTokens.flatten(0,1)[edge_index[0]]  # (num_edges, L)
            self.edge_features.append(node_i)
            verbosePrint(f'Using node i features with shape: {node_i.shape} [E, L]', self.verbose)
        if self.config.use_node_j:
            node_j = valueTokens.flatten(0,1)[edge_index[1]]  # (num_edges, L)
            self.edge_features.append(node_j)
            verbosePrint(f'Using node j features with shape: {node_j.shape} [E, L]', self.verbose)
        if self.config.use_node_sum:
            node_sum = queryTokens.flatten(0,1)[edge_index[0]] + valueTokens.flatten(0,1)[edge_index[1]]  # (num_edges, L)
            self.edge_features.append(node_sum)
            verbosePrint(f'Using node sum features with shape: {node_sum.shape} [E, L]', self.verbose)
        if self.config.use_node_diff:
            node_diff = queryTokens.flatten(0,1)[edge_index[0]] - valueTokens.flatten(0,1)[edge_index[1]]  # (num_edges, L)
            self.edge_features.append(node_diff)
            verbosePrint(f'Using node diff features with shape: {node_diff.shape} [E, L]', self.verbose)
        if len(self.edge_features) > 0:
            edge_features = torch.cat(self.edge_features, dim=-1)  # (num_edges, F_total)
            verbosePrint(f'Combined edge features shape: {edge_features.shape} [E, F_total]', self.verbose)
        else:
            edge_features = None
            verbosePrint(f'No edge features to use.', self.verbose)

        message = self.mixer(
            V_j,
            edgeTokens = edge_features,
            spatialTokens = spatialTokens,
            positionBiasTokens = rpb_features,
            windowValues = windowValues,
        )
        verbosePrint(f'Message shape after mixer: {message.shape} [E, H, T]', self.verbose)
        if self.messageActivation is not None:
            message = self.messageActivation(message)
        verbosePrint(f'Message shape after mixer: {message.shape} [E, H, T]', self.verbose)

        if self.postMessageMixer is not None:
            message = self.postMessageMixer(
                message,
                edgeTokens = edge_features,
                spatialTokens = spatialTokens,
                positionBiasTokens = rpb_features,
                windowValues = windowValues,
            )
            verbosePrint(f'Message shape after post-message mixer: {message.shape} [E, H, T]', self.verbose)

        gathered = torch_geometric.utils.scatter(
            message, rows, dim=0, dim_size=num_query_nodes * batch_size, reduce='sum')
        verbosePrint(f'Gathered messages shape before multi-head aggregation: {gathered.shape} [B*Q, H, T]', self.verbose)

        if self.multiHeadAggregation == 'concat':
            gathered = gathered.view(batch_size, num_query_nodes, self.num_heads * self.transformer_features)
            verbosePrint(f'Gathered messages shape after concat: {gathered.shape} [B, Q, H*T]', self.verbose)
            # gathered shape: (batch_size, num_query_nodes, num_heads * transformer_features)
        elif self.multiHeadAggregation == 'mean':
            gathered = gathered.view(batch_size, num_query_nodes, self.num_heads, self.transformer_features)
            gathered = gathered.mean(dim=2)  # average over heads
            verbosePrint(f'Gathered messages shape after mean: {gathered.shape} [B, Q, T]', self.verbose)
            # gathered shape: (batch_size, num_query_nodes, transformer_features)
        elif self.multiHeadAggregation == 'sum':
            gathered = gathered.view(batch_size, num_query_nodes, self.num_heads, self.transformer_features)
            gathered = gathered.sum(dim=2)  # sum over heads
            verbosePrint(f'Gathered messages shape after sum: {gathered.shape} [B, Q, T]', self.verbose)
            # gathered shape: (batch_size, num_query_nodes, transformer_features)
        elif self.multiHeadAggregation == 'max':
            gathered = gathered.view(batch_size, num_query_nodes, self.num_heads, self.transformer_features)
            gathered, _ = gathered.max(dim=2)  # max over heads
            verbosePrint(f'Gathered messages shape after max: {gathered.shape} [B, Q, T]', self.verbose)
            # gathered shape: (batch_size, num_query_nodes, transformer_features)
        elif self.multiHeadAggregation == 'min':
            gathered = gathered.view(batch_size, num_query_nodes, self.num_heads, self.transformer_features)
            gathered, _ = gathered.min(dim=2)  # min over heads
            verbosePrint(f'Gathered messages shape after min: {gathered.shape} [B, Q, T]', self.verbose)
            # gathered shape: (batch_size, num_query_nodes, transformer_features)

        return gathered

        raise NotImplementedError