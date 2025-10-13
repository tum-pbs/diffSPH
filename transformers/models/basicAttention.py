from layers.layer_attentionMechanism import AttentionMechanismLayer, AttentionMechanismConfig, AttentionLayerConfig
from layers.layer_positionEncoder import BasisEncoder, BasisEncoderConfig
from layers.layer_tokenEncoder import TokenEncoder, TokenEncoderConfig
from layers.layer_mixing import TokenMixer, TokenMixerConfig
import torch
import copy
from layers.networkUtil import verbosePrint, verboseBannerPrint
from layers.layer_mlp import MLP, MLPConfig
from typing import Optional
from torch import Tensor


class BasicAttention(torch.nn.Module):
    def __init__(self, 
                 token_input_dim: int,
                 spatial_dim: int,
                 edge_feature_dim: int = 0,

                 transformer_features: int = 32,       
                 attention_heads: int = 4,

                 attentionConfig: Optional[AttentionLayerConfig] = None,
                 mlpConfig: Optional[MLPConfig] = None,


                 verbose: bool = False,
                 verbosePrefix: str = ''
    ):
        super(BasicAttention, self).__init__()
        if mlpConfig is None:
            raise ValueError('[DEBUG] mlpConfig must be provided.')
        verbosePrint('Initializing Basic Attention...', verbose)
        self.config = copy.deepcopy(attentionConfig) if attentionConfig is not None else AttentionLayerConfig()
        self.mlpConfig = copy.deepcopy(mlpConfig) if mlpConfig is not None else MLPConfig()
        self.verbose = verbose
        self.verbosePrefix = verbosePrefix

        self.config.token_input_dim = token_input_dim
        self.config.spatial_dim = spatial_dim
        self.config.transformer_features = transformer_features
        self.config.edge_feature_dim = edge_feature_dim
        self.config.attention_heads = attention_heads

        if attentionConfig is None:
            verbosePrint(f'\tUsing default Attention config.', verbose)
            self.config.position_bias_config = BasisEncoderConfig(
                spatial_dim = spatial_dim,
                projection=True,
                projection_dim = 1
            )
            self.config.attention_mechanism = AttentionMechanismConfig()
            self.config.attention_mechanism.mechanism = 'dot'
            self.config.preAttentionMixer = TokenMixerConfig(
                input_channels = 2,

                include_edges=False,
                include_spatial=False,
                include_rpb=True,
                include_window=False,

                channel_mixing = True,
                channel_mixing_operation = 'dot',
                channel_normalization = None,
                rpb_feature_dim = 1,
                mode = 'add'
            )
            self.config.encode_tokens = True
            self.config.encode_tokens_shared = False

        self.attention = AttentionMechanismLayer(self.config, verbose = verbose, verbosePrefix = verbosePrefix, mlpConfig=self.mlpConfig)

        verbosePrint(f'\tToken Encoder config: {self.config}', verbose)
        numberOfParameters = sum(p.numel() for p in self.attention.parameters())
        verbosePrint(f'\tNumber of parameters in Attention Mechanism: {numberOfParameters}', verbose)
        verboseBannerPrint(f'Done initializing Basic Encoder.', verbose)
        
        
    def forward(self, 
                queryTokens: Tensor, # (num_query_nodes, latent_dim) (current tokens)
                keyTokens: Tensor,  # (num_key_nodes, latent_dim) (neighbor tokens)
                edge_index: Tensor, # (2, num_edges)
                edge_attr: Optional[Tensor] = None, # (num_edges, edge_dim)
                edge_features: Optional[Tensor] = None, # (num_edges, F)
                ):
        return self.attention(
            queryTokens = queryTokens,
            keyTokens = keyTokens,
            edge_index = edge_index,
            edge_attr = edge_attr,
            edge_features = edge_features
        )