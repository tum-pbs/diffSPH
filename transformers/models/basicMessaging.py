from layers.layer_attentionMechanism import AttentionMechanismLayer, AttentionMechanismConfig, AttentionLayerConfig
from layers.layer_positionEncoder import BasisEncoder, BasisEncoderConfig
from layers.layer_tokenEncoder import TokenEncoder, TokenEncoderConfig
from layers.layer_mixing import TokenMixer, TokenMixerConfig
from layers.layer_messagePassing import MessagePassingLayer, MessagePassingConfig
from layers.layer_mlp import MLP, MLPConfig
import torch
import copy
from layers.networkUtil import verbosePrint, verboseBannerPrint
from typing import Optional
from torch import Tensor


class BasicMessagePassing(torch.nn.Module):
    def __init__(self, 
                 token_input_dim: int,
                 spatial_dim: int,
                 edge_feature_dim: int = 0,

                 transformer_features: int = 32,       
                 attention_heads: int = 4,

                 messageConfig: Optional[MessagePassingConfig] = None,
                 mlpConfig: Optional[MLPConfig] = None,

                 verbose: bool = False,
                 verbosePrefix: str = ''
    ):
        if mlpConfig is None:
            raise ValueError('[DEBUG] mlpConfig must be provided.')
        super(BasicMessagePassing, self).__init__()
        verbosePrint('Initializing Basic Message Passing...', verbose)
        self.config = copy.deepcopy(messageConfig) if messageConfig is not None else MessagePassingConfig()
        self.verbose = verbose
        self.verbosePrefix = verbosePrefix

        self.config.token_input_dim = token_input_dim
        self.config.spatial_dim = spatial_dim
        self.config.transformer_features = transformer_features
        self.config.edge_feature_dim = edge_feature_dim
        self.config.attention_heads = attention_heads

        if messageConfig is None:
            verbosePrint(f'\tUsing default Message Passing config.', verbose)
            self.config.encode_tokens = True
            self.config.messageMixer = TokenMixerConfig(
                mode = 'multiply',
                include_spatial = False,
                include_edges = True
            )

            self.config.multiHeadAggregation = 'concat'
            self.config.use_attention = True
            self.config.use_node_i = False
            self.config.use_node_j = False
            self.config.use_node_sum = False
            self.config.use_node_diff = False
            self.config.use_edge_features = False
            self.config.use_window_function = False
            self.config.use_spatial = False
            self.config.use_rpb = False

            self.config.position_bias_config = None

        self.messenger = MessagePassingLayer(self.config, verbose = verbose, verbosePrefix = verbosePrefix, mlpConfig=self.mlpConfig)

        verbosePrint(f'\tToken Encoder config: {self.config}', verbose)
        numberOfParameters = sum(p.numel() for p in self.messenger.parameters())
        verbosePrint(f'\tNumber of parameters in Message Passing Layer: {numberOfParameters}', verbose)
        verboseBannerPrint(f'Done initializing Basic Encoder.', verbose)
        
        
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
        return self.messenger(
            queryTokens = queryTokens,
            valueTokens = valueTokens,
            edge_index = edge_index,
            edgeAttention = edgeAttention,
            edgeTokens = edgeTokens,
            spatialTokens = spatialTokens,
            positionBiasTokens = positionBiasTokens,
            windowValues = windowValues
        )