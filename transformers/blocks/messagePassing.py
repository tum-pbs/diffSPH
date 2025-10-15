from layers.attentionMechanism import AttentionMechanismLayer, AttentionMechanismConfig, AttentionLayerConfig
from layers.positionEncoder import BasisEncoder, BasisEncoderConfig
from layers.tokenEncoder import TokenEncoder, TokenEncoderConfig
from layers.tokenMixer import TokenMixer, TokenMixerConfig
from layers.messagePassing import MessagePassingLayer, MessagePassingConfig
from layers.mlp import MLP, MLPConfig
import torch
import copy
from mlUtil.networkUtil import verbosePrint, verboseBannerPrint
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
    


from layers.messagePassing import MessagePassingLayer, MessagePassingConfig, getDefaultMessagePassingConfig
from typing import Optional, Union, List
from mlUtil.networkUtil import mergeConfigWithKwargs
from mlUtil.networkUtil import verbosePrint, verboseBannerPrint
from blocks.feedForwardNetwork import applyAdaptiveScaling
import torch.nn as nn
from layers.norm import NormLayer
from blocks.common import CommonConfiguration
import warnings

class MessagePassingBlock(torch.nn.Module):
    def __init__(self, 
    token_input_dim: int,
    token_output_dim: int,
    spatial_dim: int,
    edge_feature_dim: int,
    
    config: Optional[CommonConfiguration] = None, 
    mlpConfig: Optional[MLPConfig] = None, 
    messageConfig: Optional[MessagePassingConfig] = None,
    embedding_dim: Optional[int] = None,
    verbose: bool = False, verbosePrefix: str = '',
    **kwargs):
        super(MessagePassingBlock, self).__init__()
        self.verbose = verbose
        self.verbosePrefix = verbosePrefix
        verboseBannerPrint(f'{self.verbosePrefix}Initializing MessagePassingBlock', self.verbose)

        self.config = mergeConfigWithKwargs(config if config is not None else CommonConfiguration(), **kwargs)
        self.messageConfig = mergeConfigWithKwargs(messageConfig if messageConfig is not None else getDefaultMessagePassingConfig('gnn'), **kwargs)
        self.mlpConfig = mlpConfig if mlpConfig is not None else (self.config.mlpConfig if self.config.mlpConfig is not None else MLPConfig())

        self.messageConfig.token_input_dim = token_input_dim if self.messageConfig.token_input_dim < 0 else self.messageConfig.token_input_dim
        self.messageConfig.spatial_dim = spatial_dim if self.messageConfig.spatial_dim < 0 else self.messageConfig.spatial_dim
        self.messageConfig.edge_feature_dim = edge_feature_dim if self.messageConfig.edge_feature_dim < 0 else self.messageConfig.edge_feature_dim

        self.messageConfig.transformer_features = token_output_dim if self.messageConfig.transformer_features is None else self.messageConfig.transformer_features
        self.messageConfig.attention_heads = 1 if self.messageConfig.attention_heads < 0 else self.messageConfig.attention_heads

        self.token_input_dim = self.messageConfig.token_input_dim
        self.token_output_dim = self.messageConfig.transformer_features

        verbosePrint(f'{self.verbosePrefix}MessagePassingBlock config: {self.messageConfig}', self.verbose)

        self.messenger = MessagePassingLayer(
            self.messageConfig,
            mlpConfig = self.mlpConfig,
            verbose = self.verbose, verbosePrefix = f'{self.verbosePrefix} [Messenger] ')
        
        self.embedding = None
        self.embeddingConfig = self.config.embeddingConfig if self.config.embeddingConfig is not None else self.mlpConfig
        if self.config.embedding_dim > 0:
            self.embeddingConfig.input_dim = self.config.embedding_dim
        else:
            self.embeddingConfig.input_dim = embedding_dim if (self.embeddingConfig.input_dim is None or self.embeddingConfig.input_dim < 0) else self.embeddingConfig.input_dim

        self.norm_type = self.config.norm_type
        self.pre_norm = self.config.pre_norm
        self.post_norm = self.config.post_norm
        self.use_conditioning = self.config.use_conditioning

        if self.use_conditioning and self.embeddingConfig.input_dim > 0:
            self.embedding = MLP(in_features=self.embeddingConfig.input_dim, out_features=self.token_input_dim * 2 + self.token_output_dim, config=self.embeddingConfig, verbose=verbose, verbosePrefix=verbosePrefix+'[Embedding] ')
            verbosePrint(f'{verbosePrefix}Using embedding MLP with config: {self.embeddingConfig}', verbose)

            if self.config.adaLn_zero_init:
                verbosePrint(f'{verbosePrefix}Initializing embedding MLP last layer to zero for AdaLN', verbose)
                # the final layer produces gamma, beta, alpha scaling factors stacked
                # gamma and beta are of size input_dim, alpha is of size output_dim
                # So the final layer has output size input_dim * 2 + output_dim
                # we want to initialize this layer such that only alpha is zero (see here https://ar5iv.labs.arxiv.org/html/2212.09748 for the reference)
                # that means we can only partially zero out the weights
                if self.embedding.finalLinear is None:
                    raise ValueError('Embedding MLP final layer is None, cannot initialize to zero')
                nn.init.zeros_(self.embedding.finalLinear.weight[self.token_input_dim * 2:, :])
                if self.embedding.finalLinear.bias is not None:
                    nn.init.zeros_(self.embedding.finalLinear.bias[self.token_input_dim * 2:])
                verbosePrint(f'{verbosePrefix}Initialized embedding MLP last layer to zero for AdaLN', verbose)
                # Note: This initialization is crucial for stable training when using AdaLN conditioning.

            if not self.pre_norm:
                warnings.warn('Using embedding MLP without pre-norm in the main MLP. This may lead to instability.', UserWarning)
        elif self.use_conditioning:
            raise ValueError('use_conditioning is True but embeddingConfig.input_dim is not set. Cannot use conditioning without embedding input.')
            warnings.warn('use_conditioning is True but embeddingConfig.input_dim is not set. Ignoring conditioning.', UserWarning)
            self.use_conditioning = False
        else:
            verbosePrint(f'{verbosePrefix}Not using embedding MLP', verbose)

        if self.pre_norm:
            self.pre_norm_layer = NormLayer(self.norm_type, self.mlpConfig.batch_size, self.mlpConfig.seq_length, self.messageConfig.token_input_dim, verbose=verbose, verbosePrefix=verbosePrefix+'[PreNorm] ')
            verbosePrint(f'{verbosePrefix}Using pre-norm layer with type: {self.norm_type}', verbose)
        else:
            self.pre_norm_layer = nn.Identity()
            verbosePrint(f'{verbosePrefix}No pre-norm layer', verbose)
        if self.post_norm:
            self.post_norm_layer = NormLayer(self.norm_type, self.mlpConfig.batch_size, self.mlpConfig.seq_length, self.messageConfig.transformer_features, verbose=verbose, verbosePrefix=verbosePrefix+'[PostNorm] ')
            verbosePrint(f'{verbosePrefix}Using post-norm layer with type: {self.norm_type}', verbose)
        else:
            self.post_norm_layer = nn.Identity()
            verbosePrint(f'{verbosePrefix}No post-norm layer', verbose)
        
        if self.config.message_skip_connections:
            verbosePrint(f'{verbosePrefix}Using skip connection in FeedForwardNetwork', verbose)
            if self.token_input_dim != self.messageConfig.transformer_features:
                if self.config.ffn_skip_projection:
                    self.skip_connection = nn.Linear(self.token_input_dim, self.messageConfig.transformer_features, bias = False)
                    verbosePrint(f'{verbosePrefix}Using skip connection with projection from {self.token_input_dim} to {self.messageConfig.transformer_features}', verbose)
                else:
                    raise ValueError('ffn_skip_connection is True but input and output dimensions do not match and ffn_skip_projection is False')
            else:
                self.skip_connection = nn.Identity()
                verbosePrint(f'{verbosePrefix}Using skip connection without projection', verbose)



    def forward(self,
                queryTokens: torch.Tensor,  # (num_query_nodes, latent_dim) (central tokens)
                valueTokens: torch.Tensor,  # (num_key_nodes, latent_dim) (neighbor tokens)
                edge_index: torch.Tensor, # (2, num_edges)
                edgeAttention: Optional[torch.Tensor] = None, # (num_edges, num_heads) or (num_edges, 1)
                edgeTokens: Optional[torch.Tensor] = None, # shape [*, H?, F_e]
                spatialTokens: Optional[torch.Tensor] = None, # shape [*, H?, D]
                positionBiasTokens: Optional[torch.Tensor] = None, # shape [*, H?, F_rpb]
                windowValues: Optional[torch.Tensor] = None, # shape [*,H?]
                embedding_input: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
                ) -> torch.Tensor:
        verboseBannerPrint(f'{self.verbosePrefix}MessagePassing Forward Pass', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Input tensor shape: {queryTokens.shape}', self.verbose)
        # x is of shape [B, N, F] or [N,F]
        # if x is of shape [N,F], add a batch dimension
        if queryTokens.dim() == 2:
            verbosePrint(f'{self.verbosePrefix}Input tensor has no batch dimension, adding one', self.verbose)
            unsqueezedQuery = True
            queryTokens = queryTokens.unsqueeze(0)
        else:
            unsqueezedQuery = False
        if valueTokens.dim() == 2:
            verbosePrint(f'{self.verbosePrefix}Value tensor has no batch dimension, adding one', self.verbose)
            unsqueezedValue = True
            valueTokens = valueTokens.unsqueeze(0)
        else:
            unsqueezedValue = False

        B, N, F = queryTokens.shape
        O = self.token_output_dim
        verbosePrint(f'{self.verbosePrefix}Query tensor shape after unsqueeze: {queryTokens.shape}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Value tensor shape after unsqueeze: {valueTokens.shape}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Batch size: {B}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Sequence length: {N}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Feature dimension: {F}', self.verbose)

        queryTokens, alpha_scale_query = applyAdaptiveScaling(
            queryTokens, 
            shapes = [B, N, F, O],
            embedding = self.use_conditioning,
            normLayer = self.pre_norm_layer,
            embedding_mlp = self.embedding,
            embedding_input = embedding_input,
            verbose = self.verbose,
            verbosePrefix = self.verbosePrefix + '[AdaptiveScaling] '
        )
        valueTokens, alpha_scale_value = applyAdaptiveScaling(
            valueTokens, 
            shapes = [B, valueTokens.shape[1], F, O],
            embedding = self.use_conditioning,
            normLayer = None,
            embedding_mlp = self.embedding,
            embedding_input = embedding_input,
            verbose = self.verbose,
            verbosePrefix = self.verbosePrefix + '[AdaptiveScaling] '
        )

        result = self.messenger(
            queryTokens = queryTokens,
            valueTokens = valueTokens,
            edge_index = edge_index,
            edgeAttention = edgeAttention,
            edgeTokens = edgeTokens,
            spatialTokens = spatialTokens,
            positionBiasTokens = positionBiasTokens,
            windowValues = windowValues
        )
        if self.messenger.edgeMLP is not None:
            verbosePrint(f'{self.verbosePrefix}Edge MLP is used in the messenger', self.verbose)
            messaged = result[0]
            edges = result[1]
        else:
            messaged = result
            edges = edgeTokens

        verbosePrint(f'{self.verbosePrefix}Tensor shape after message passing: {messaged.shape}', self.verbose)

        messaged = self.post_norm_layer(messaged)
        verbosePrint(f'{self.verbosePrefix}Tensor shape after post-norm: {messaged.shape}', self.verbose)

        if alpha_scale_query is not None:
            messaged = messaged * alpha_scale_query
            verbosePrint(f'{self.verbosePrefix}Applied alpha scaling from query tokens', self.verbose)
        

        if self.config.message_skip_connections:
            verbosePrint(f'{self.verbosePrefix}Applying skip connection', self.verbose)
            messaged = messaged + self.skip_connection(queryTokens)
            verbosePrint(f'{self.verbosePrefix}Tensor shape after skip connection: {messaged.shape}', self.verbose)

        if unsqueezedQuery:
            messaged = messaged.squeeze(0)
            verbosePrint(f'{self.verbosePrefix}Squeezed output tensor to remove batch dimension: {messaged.shape}', self.verbose)

        return messaged,edges 


