from layers.attentionMechanism import AttentionMechanismLayer, AttentionMechanismConfig, AttentionLayerConfig
from layers.positionEncoder import BasisEncoder, BasisEncoderConfig
from layers.tokenEncoder import TokenEncoder, TokenEncoderConfig
from layers.tokenMixer import TokenMixer, TokenMixerConfig
from layers.messagePassing import MessagePassingLayer, MessagePassingConfig
from layers.mlp import MLP, MLPConfig
import torch
import copy
from mlUtil.networkUtil import verbosePrint, verboseBannerPrint
from typing import Optional, Tuple, Union, List
from torch import Tensor

from .attention import BasicAttention
from .encoder import BasicEncoder
from .attention import BasicAttention
from .messagePassing import BasicMessagePassing

from dataclasses import dataclass, field
from mlUtil.networkUtil import mergeConfigWithKwargs
# from layers.mlp import getDefaultMLPDict, buildMLPwDict
from mlUtil.activation import getActivationFromString
from .common import CommonConfiguration


class BasicCConvLayer(torch.nn.Module):
    def __init__(self, 
                 token_input_dim: int,
                 spatial_dim: int,

                cconv_basis: Union[str, List[str]] = 'ffourier',
                cconv_terms: Union[int, List[int]] = 6,
                cconv_projection: str = 'cartesian',
                cconv_mode: str = 'outer',
                mlpConfig: Optional[MLPConfig] = None,

                 token_output_dim: int = 0,
                 verbose: bool = False,
                 verbosePrefix: str = ''
    ):
        if mlpConfig is None:
            raise ValueError('[DEBUG] mlpConfig must be provided.')
        super(BasicCConvLayer, self).__init__()
        verbosePrint('Initializing Basic CConv Layer...', verbose)
        self.verbose = verbose
        self.verbosePrefix = verbosePrefix
        self.mlpConfig = copy.deepcopy(mlpConfig) if mlpConfig is not None else MLPConfig()

        self.token_input_dim = token_input_dim
        self.spatial_dim = spatial_dim

        self.cconv_messaging = MessagePassingLayer(
            config = MessagePassingConfig(
                token_input_dim=token_input_dim,
                spatial_dim = spatial_dim,

                edge_feature_dim=0,
                attention_heads = 1,
                transformer_features = token_input_dim,
                use_attention = False,

                encode_tokens = False,

                messageMixer = TokenMixerConfig(
                    num_heads = 1,
                    transformer_features = token_input_dim,
                    mixing_out_features= token_output_dim if token_output_dim > 0 else token_input_dim,
                    
                    input_channels = 1,
                    spatial_dim = spatial_dim,
                    edge_feature_dim = 0,

                    mode = 'cconv',
                    cconv_source = 'rpb',
                    basis_encoder = BasisEncoderConfig(
                        spatial_dim = spatial_dim,
                        base_encoding = True,
                        base_function = cconv_basis,
                        base_terms = cconv_terms,
                        base_mode = cconv_mode,
                        base_projection = cconv_projection,
                        projection = False,

                        normalize_positions=False,
                    )
                ),
                mlpConfig = self.mlpConfig,
            ),
            verbose = verbose,
            verbosePrefix = f'CConv|{verbosePrefix}'
        )

        verbosePrint(f'\tCConv Messaging config: {self.cconv_messaging.config}', verbose)

    def forward(self,
        node_features: Union[Tensor, Tuple[Tensor, Tensor]], 
        edge_indices: Tensor, 
        edge_features: Optional[Tensor] = None, 
        edge_spatial_features: Optional[Tensor] = None
    ):
        nodes_query = node_features if isinstance(node_features, Tensor) else node_features[0]
        nodes_key_value = node_features if isinstance(node_features, Tensor) else node_features[1]

        return self.cconv_messaging(
            queryTokens = nodes_query,
            valueTokens = nodes_key_value,
            edge_index = edge_indices,
            edgeAttention = None,
            edgeTokens = edge_features,
            spatialTokens = edge_spatial_features,
            positionBiasTokens = None,
            windowValues = None
        )
    
    

class CConvModel(torch.nn.Module):
    def __init__(self, 
        config: Optional[CommonConfiguration] = None,
                 verbose: bool = False,
                 verbosePrefix: str = '',
                 **kwargs
    ):
        super(CConvModel, self).__init__()
        verbosePrint('Initializing CConv Model...', verbose)

        self.config = copy.deepcopy(config) if config is not None else CommonConfiguration()
        self.config = mergeConfigWithKwargs(self.config, **kwargs)

        mlp_dict = self.config.mlp_dict if self.config.mlp_dict is not None else getDefaultMLPDict()
        mlp_dict['layout'] = [self.config.latent_features] * self.config.mlp_hidden_layers
        mlp_dict['hidden_dim'] = self.config.mlp_latent_dim
        mlp_dict['activation'] = self.config.mlp_activation
        self.config.mlp_dict = mlp_dict


        self.verbose = verbose
        self.verbosePrefix = verbosePrefix

        self.use_encoder = self.config.use_encoder
        self.use_decoder = self.config.use_decoder

        self.token_input_dim = self.config.token_input_dim
        self.token_output_dim = self.config.token_output_dim
        self.spatial_dim = self.config.spatial_dim
        self.edge_feature_dim = self.config.edge_feature_dim
        self.latent_features = self.config.latent_features
        self.convolution_layers = self.config.hidden_layers

        if self.use_encoder:
            verbosePrint(f'\tUsing input encoder.', verbose)
            self.inputEncoder = BasicEncoder(
                input_dim = self.token_input_dim,
                output_dim = self.latent_features,
                latent_dim = mlp_dict['layout'][0],

                tokenEncoderConfig = TokenEncoderConfig(
                    use_ffn = False,
                    projection = True,
                    projection_linear = True
                ),

                hidden_layers = len(mlp_dict['layout']),
                verbose = verbose
            )
            token_input_dim = self.latent_features
        else:
            verbosePrint(f'\tSkipping input encoder.', verbose)
            self.inputEncoder = torch.nn.Identity()
            if self.config.inputEncoderTokenConfig is not None:
                verbosePrint(f'\tWarning: inputEncoderTokenConfig provided but use_encoder is False. Ignoring inputEncoderTokenConfig.', verbose)
            token_input_dim = self.token_input_dim

        self.convolution_layers_list = torch.nn.ModuleList()
        self.ffns = torch.nn.ModuleList() if self.config.node_ffn else None
        self.message_norms = torch.nn.ModuleList() if self.config.post_message_norm is not None else None
        self.ffn_norms = torch.nn.ModuleList() if self.config.post_ffn_norm is not None else None
        self.ffn_projs = torch.nn.ModuleList() if self.config.ffn_skip_connection and self.config.ffn_skip_projection else None
        self.message_projs = torch.nn.ModuleList() if self.config.message_skip_connections and self.config.message_skip_projection else None


        current_token_dim = token_input_dim

        for layer in range(self.convolution_layers):
            verbosePrint(f'\tAdding convolution layer {layer+1}/{self.convolution_layers}.', verbose)
            self.convolution_layers_list.append(
                BasicCConvLayer(
                    token_input_dim = current_token_dim,
                    token_output_dim = self.latent_features,
                    spatial_dim = self.spatial_dim,

                    cconv_basis = self.config.basis_function,
                    cconv_terms = self.config.basis_terms,
                    cconv_projection = self.config.basis_projection,
                    cconv_mode = self.config.basis_mode,
                    
                    verbose = False,
                    verbosePrefix = f'{verbosePrefix}  L{layer+1}|'
                )
            )
            if self.config.post_message_norm is not None:
                self.message_norms.append(torch.nn.LayerNorm(self.latent_features))
            if self.config.message_skip_connections:
                if self.config.message_skip_projection:
                    self.message_projs.append(
                        torch.nn.Linear(current_token_dim, self.latent_features)
                    )
            # if ffn_skip_last is set to true then create no ffn for the last layer, i.e., when layer == self.convolution_layers - 1
            if self.config.node_ffn:
                if layer == self.convolution_layers - 1 and self.config.ffn_skip_last:
                    verbosePrint(f'\t\tSkipping FFN at last layer {layer+1} due to ffn_skip_last=True.', verbose)
                    continue
                self.ffns.append(
                    buildMLPwDict(mlp_dict, inputDim=self.latent_features, outputDim=self.latent_features)
                )
                if self.config.post_ffn_norm is not None:
                    self.ffn_norms.append(torch.nn.LayerNorm(self.latent_features))

                if self.config.ffn_skip_connection:
                    verbosePrint(f'\t\tUsing skip connection for FFN at layer {layer+1}.', verbose)
                    if self.config.ffn_skip_projection:
                        self.ffn_projs.append(
                            torch.nn.Linear(self.latent_features, self.latent_features)
                        )


            current_token_dim = self.latent_features


        if self.use_decoder:
            verbosePrint(f'\tUsing output decoder.', verbose)
            self.outputDecoder = BasicEncoder(
                input_dim = current_token_dim,
                output_dim = self.token_output_dim,
                latent_dim = mlp_dict['layout'][-1],

                tokenEncoderConfig = self.config.outputDecoderTokenConfig,

                hidden_layers = len(mlp_dict['layout']),
                verbose = verbose
            )
        else:
            verbosePrint(f'\tSkipping output decoder.', verbose)
            self.outputDecoder = torch.nn.Identity()
            if self.config.outputDecoderTokenConfig is not None:
                verbosePrint(f'\tWarning: outputDecoderTokenConfig provided but use_decoder is False. Ignoring outputDecoderTokenConfig.', verbose)
    
    def forward(self, 
        node_features: Union[Tensor, Tuple[Tensor, Tensor]], 
        node_positions: Union[Tensor, Tuple[Tensor, Tensor]], 
        edge_indices: Tensor, 
        edge_features: Optional[Tensor] = None, 
        edge_spatial_features: Optional[Tensor] = None
    ):
        nodes_query = node_features if isinstance(node_features, Tensor) else node_features[0]
        nodes_key_value = node_features if isinstance(node_features, Tensor) else node_features[1]

        positions_query = node_positions if isinstance(node_positions, Tensor) else node_positions[0]
        positions_key_value = node_positions if isinstance(node_positions, Tensor) else node_positions[1]

        if self.use_encoder:
            nodes_query = self.inputEncoder(nodes_query, inputPositions=positions_query)
            if isinstance(node_features, Tensor):
                nodes_key_value = nodes_query
            else:
                nodes_key_value = self.inputEncoder(nodes_key_value, inputPositions=positions_key_value)

        for i, layer in enumerate(self.convolution_layers_list):
            verbosePrint(f'\tPassing through convolution layer {i+1}/{self.convolution_layers}.', self.verbose)
            ans = layer(
                node_features = (nodes_query, nodes_key_value),
                edge_indices = edge_indices,
                edge_features = edge_features,
                edge_spatial_features = edge_spatial_features
            )
            if self.config.message_activation is not None:
                verbosePrint(f'\tApplying message activation {self.config.message_activation} at layer {i+1}.', self.verbose)
                ans = self.message_activation(ans)
            if self.config.post_message_norm is not None and self.message_norms is not None and i < len(self.message_norms):
                verbosePrint(f'\tApplying post-message normalization {self.config.post_message_norm} at layer {i+1}.', self.verbose)
                ans = self.message_norms[i](ans)
            if self.config.message_skip_connections:
                verbosePrint(f'\tUsing skip connection for message passing at layer {i+1}.', self.verbose)
                if self.config.message_skip_projection and self.message_projs is not None and i < len(self.message_projs):
                    verbosePrint(f'\t\tUsing projection for message passing skip connection at layer {i+1}.', self.verbose)
                    ans = ans + self.message_projs[i](nodes_query)
                else:
                    ans = ans + nodes_query
            if self.config.node_ffn and self.ffns is not None and i < len(self.ffns):
                verbosePrint(f'\tApplying FFN at layer {i+1}.', self.verbose)
                ffn_out = self.ffns[i](ans)
                if self.config.post_ffn_norm is not None and self.ffn_norms is not None and i < len(self.ffn_norms):
                    verbosePrint(f'\tApplying post-FFN normalization {self.config.post_ffn_norm} at layer {i+1}.', self.verbose)
                    ffn_out = self.ffn_norms[i](ffn_out)
                if self.config.ffn_skip_connection:
                    verbosePrint(f'\tUsing skip connection for FFN at layer {i+1}.', self.verbose)
                    if self.config.ffn_skip_projection and self.ffn_projs is not None and i < len(self.ffn_projs):
                        verbosePrint(f'\t\tUsing projection for FFN skip connection at layer {i+1}.', self.verbose)
                        ans = ffn_out + self.ffn_projs[i](ans)
                    else:
                        ans = ffn_out + ans
                else:
                    ans = ffn_out

            nodes_query = ans
            verbosePrint(f'\tDone convolution layer {i+1}/{self.convolution_layers}. Shape: {nodes_query.shape}', self.verbose)

            if isinstance(node_features, Tensor):
                nodes_key_value = nodes_query
            elif self.convolution_layers > 1:
                raise NotImplementedError('Currently only supports single pass when using separate query and key/value nodes.')
            else:
                pass

        if self.use_decoder:
            nodes_query = self.outputDecoder(nodes_query, inputPositions=positions_query)

        return nodes_query

        