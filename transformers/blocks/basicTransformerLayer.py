from layers.attentionMechanism import AttentionMechanismLayer, AttentionMechanismConfig, AttentionLayerConfig
from layers.positionEncoder import BasisEncoder, BasisEncoderConfig
from layers.tokenEncoder import TokenEncoder, TokenEncoderConfig
from layers.tokenMixer import TokenMixer, TokenMixerConfig
from layers.messagePassing import MessagePassingLayer, MessagePassingConfig
from layers.mlp import MLP, MLPConfig
import torch
import copy
from mlUtil.networkUtil import verbosePrint, verboseBannerPrint
from typing import Optional, Tuple, Union
from torch import Tensor

from .attention import BasicAttention
from .encoder import BasicEncoder
from .attention import BasicAttention
from .messagePassing import BasicMessagePassing
from mlUtil.networkUtil import mergeConfigWithKwargs
class BasicTransformerLayer(torch.nn.Module):
    def __init__(self, 
                 token_input_dim: int,
                 token_output_dim: int,
                 spatial_dim: int,
                 edge_feature_dim: int = 0,

                 transformer_features: int = 32,       
                 attention_heads: int = 4,
                 transformer_layers: int = 1,

                 attentionConfig: Optional[AttentionLayerConfig] = None,
                 messageConfig: Optional[MessagePassingConfig] = None,
                 mlpConfig: Optional[MLPConfig] = None,  

                 verbose: bool = False,
                 verbosePrefix: str = '',
                 **kwargs
    ):
        if mlpConfig is None:
            raise ValueError('[DEBUG] mlpConfig must be provided.')
        attentionConfig = mergeConfigWithKwargs(attentionConfig if attentionConfig is not None else AttentionLayerConfig(), **kwargs)
        # messageConfig = mergeConfigWithKwargs(messageConfig if messageConfig is not None else MessagePassingConfig(), **kwargs)



        super(BasicTransformerLayer, self).__init__()
        verbosePrint('Initializing Basic Transformer Layer...', verbose)
        self.verbose = verbose
        self.verbosePrefix = verbosePrefix
        self.mlpConfig = copy.deepcopy(mlpConfig) if mlpConfig is not None else MLPConfig()


        self.token_input_dim = token_input_dim
        self.token_output_dim = token_output_dim
        self.spatial_dim = spatial_dim
        self.edge_feature_dim = edge_feature_dim
        self.transformer_features = transformer_features
        self.attention_heads = attention_heads
        self.transformer_layers_list = torch.nn.ModuleList()
        self.message_layers_list = torch.nn.ModuleList()

        layer = 0
        verbosePrint(f'\tAdding transformer layer {layer+1}/{transformer_layers}.', verbose)
        self.transformer_layers_list.append(
            BasicAttention(
            token_input_dim = token_input_dim,
            spatial_dim = spatial_dim,
            edge_feature_dim= edge_feature_dim,
            transformer_features = transformer_features,
            attention_heads = attention_heads,
            verbose = verbose,
            verbosePrefix = f'{verbosePrefix}  L{layer+1}|',
            attentionConfig = attentionConfig,
            mlpConfig = self.mlpConfig
            )
        )
        self.message_layers_list.append(
            BasicMessagePassing(
                token_input_dim = token_input_dim,
                spatial_dim = spatial_dim,
                edge_feature_dim = edge_feature_dim,
                transformer_features = transformer_features,
                attention_heads = attention_heads,
                messageConfig = messageConfig,
                verbose = verbose,
                verbosePrefix = f'{verbosePrefix}  L{layer+1}|',
                mlpConfig = self.mlpConfig
            )
        )

    def forward(self, 
        node_features: Union[Tensor, Tuple[Tensor, Tensor]], 
        edge_indices: Tensor, 
        edge_features: Optional[Tensor] = None, 
        edge_spatial_features: Optional[Tensor] = None
    ):
        nodes_query = node_features if isinstance(node_features, Tensor) else node_features[0]
        nodes_key_value = node_features if isinstance(node_features, Tensor) else node_features[1]

        for layer, (attention_layer, message_layer) in enumerate(zip(self.transformer_layers_list, self.message_layers_list)):
            verbosePrint(f'{self.verbosePrefix}Applying Attention layer {layer+1}/{len(self.transformer_layers_list)}.', self.verbose)

            attention_values = attention_layer(
                queryTokens = nodes_query,
                keyTokens = nodes_key_value,
                edge_index = edge_indices,
                edge_features = edge_features,
                edge_attr = edge_spatial_features
            )
            verbosePrint(f'{self.verbosePrefix}Attention output shape: {attention_values.shape}', self.verbose)

            verbosePrint(f'{self.verbosePrefix}Applying Message Passing layer {layer+1}/{len(self.message_layers_list)}.', self.verbose)
            message_values = message_layer(
                queryTokens = nodes_query,
                valueTokens = nodes_key_value,
                edge_index = edge_indices,
                edgeAttention = attention_values,
                edgeTokens = edge_features,
                spatialTokens = edge_spatial_features,
                positionBiasTokens = None,
                windowValues = None
            )
            verbosePrint(f'{self.verbosePrefix}Message Passing output shape: {message_values.shape}', self.verbose)

        return message_values