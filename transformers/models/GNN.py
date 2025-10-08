from layers.layer_attentionMechanism import AttentionMechanismLayer, AttentionMechanismConfig, AttentionLayerConfig
from layers.layer_positionEncoder import BasisEncoder, BasisEncoderConfig
from layers.layer_tokenEncoder import TokenEncoder, TokenEncoderConfig
from layers.layer_mixing import TokenMixer, TokenMixerConfig
from layers.layer_messagePassing import MessagePassingLayer, MessagePassingConfig
import torch
import copy
from layers.networkUtil import verbosePrint, verboseBannerPrint
from typing import Optional, Tuple, Union, List
from torch import Tensor

from layers.mlp import getDefaultMLPDict, buildMLPwDict

from .basicAttention import BasicAttention
from .basicEncoder import BasicEncoder
from .basicAttention import BasicAttention
from .basicMessaging import BasicMessagePassing

from dataclasses import dataclass, field
from layers.networkUtil import mergeConfigWithKwargs
from layers.activation import getActivationFromString
from .common import CommonConfiguration


class GNNModel(torch.nn.Module):
    def __init__(self,
        config: Optional[CommonConfiguration] = None,
        latent_edge_features = 0,

        use_edge_mlp: bool = False,
        use_edge_encoder = True,

        inputEdgeEncoderTokenConfig: Optional[TokenEncoderConfig] = None,
        use_basis_encoder: bool = True,

        verbose: bool = False,
        verbosePrefix: str = '',
        **kwargs):
        super(GNNModel, self).__init__()
        verbosePrint('Initializing GNN Model...', verbose)
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
        self.use_edge_encoder = use_edge_encoder

        self.token_input_dim = self.config.token_input_dim
        self.token_output_dim = self.config.token_output_dim
        self.spatial_dim = self.config.spatial_dim
        self.edge_feature_dim = self.config.edge_feature_dim

        self.message_activation = getActivationFromString(self.config.message_activation)[0] if self.config.message_activation is not None else torch.nn.Identity()

        self.latent_features = self.config.latent_features
        self.latent_edge_features = latent_edge_features if latent_edge_features > 0 else self.latent_features
        self.hidden_layers = self.config.hidden_layers

        if self.use_encoder:
            verbosePrint(f'\tAdding input node encoder.', verbose)
            self.input_node_encoder = BasicEncoder(
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
            current_token_dim = self.latent_features
        else:
            current_token_dim = self.token_input_dim

        if self.use_edge_encoder and self.config.spatial_dim > 0:
            verbosePrint(f'\tAdding input edge encoder.', verbose)
            self.input_edge_encoder = BasisEncoder(
                config = BasisEncoderConfig(
                    spatial_dim = self.config.spatial_dim,
                    base_encoding = use_basis_encoder,
                    base_function = self.config.basis_function,
                    base_terms = self.config.basis_terms,
                    base_mode = self.config.basis_mode,
                    base_projection = self.config.basis_projection,

                    projection = True,
                    projection_linear = False,
                    projection_dim = self.latent_edge_features,
                    projection_mlp = mlp_dict,
                ),
                verbose = verbose,
                verbosePrefix = 'EdgeEnc|'
            )

            current_edge_dim = self.latent_edge_features
        else:
            current_edge_dim = self.edge_feature_dim

        verbosePrint(f'Initial token dimensions: {self.token_input_dim}', verbose)
        verbosePrint(f'Initial edge dimensions: {self.edge_feature_dim}', verbose)

        verbosePrint(f'Set up input dimensions: token_input_dim={current_token_dim}, edge_feature_dim={current_edge_dim}', verbose)

        self.message_passing_layers = torch.nn.ModuleList()
        self.message_edge_mlps = torch.nn.ModuleList() if use_edge_mlp else None

        self.ffns = torch.nn.ModuleList() if self.config.node_ffn else None
        self.message_norms = torch.nn.ModuleList() if self.config.post_message_norm is not None else None
        self.ffn_norms = torch.nn.ModuleList() if self.config.post_ffn_norm is not None else None
        self.ffn_projs = torch.nn.ModuleList() if self.config.ffn_skip_connection and self.config.ffn_skip_projection else None
        self.message_projs = torch.nn.ModuleList() if self.config.message_skip_connections and self.config.message_skip_projection else None

        for layer in range(self.hidden_layers):
            verbosePrint(f'\tAdding message passing layer {layer+1}/{self.hidden_layers}.', verbose)
            self.message_passing_layers.append(
                MessagePassingLayer(
                config = MessagePassingConfig(
                    encode_tokens = False,

                    token_input_dim= current_token_dim,
                    spatial_dim = self.config.spatial_dim,
                    edge_feature_dim = current_edge_dim,
                    attention_heads = 1,
                    transformer_features = self.latent_features,

                    messageMixer = TokenMixerConfig(
                        input_channels = 1,
                        mode = 'mlp',
                        mlp_dict = mlp_dict,

                        include_spatial = False,
                        include_edges = True,
                    ),

                    use_attention = False,
                    use_node_i = False,
                    use_node_j = False,
                    use_node_sum = False,
                    use_node_diff = False,
                    use_edge_features = True,
                    use_window_function = False,
                    use_spatial = False,
                    use_rpb = False,


                ),
                    verbose = verbose,
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
            if self.config.node_ffn:
                if layer == self.hidden_layers - 1 and self.config.ffn_skip_last:
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


            current_token_dim = self.latent_features  # Assuming message passing does not change token dim

            if use_edge_mlp and layer < self.hidden_layers - 1:  # No edge MLP after last layer
                verbosePrint(f'\t\tAdding edge MLP to update edge features from {current_edge_dim} to {self.latent_edge_features}.', verbose)
                out_edgeMessages = TokenMixer(
                    transformer_features = current_edge_dim,
                    mixing_out_features = self.latent_edge_features,
                    input_channels = 1,

                    channel_mixing = False,
                    mode = 'mlp',
                    mlp_dict = mlp_dict,

                    include_edges = True,
                    include_spatial = False,

                    edge_feature_dim = 2*current_token_dim,
                    verbose = verbose
                )
                self.message_edge_mlps.append(out_edgeMessages)
                current_edge_dim = self.latent_edge_features
                # verbosePrint(f'\t\tAdding edge MLP to update edge features to {current_edge_dim}.', verbose)
            else:
                verbosePrint(f'\t\tSkipping edge MLP, keeping edge features at {current_edge_dim}.', verbose)


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

        verbosePrint(f'\tFinal token dimension: {current_token_dim}', verbose)
        verbosePrint(f'\tFinal edge dimension: {current_edge_dim}', verbose)

        verboseBannerPrint('GNN Model initialization complete.', verbose)

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
            nodes_query = self.input_node_encoder(nodes_query, inputPositions=positions_query)
            if isinstance(node_features, Tensor):
                nodes_key_value = nodes_query
            else:
                nodes_key_value = self.input_node_encoder(nodes_key_value, inputPositions=positions_key_value)

        if self.use_edge_encoder:
            if edge_features is not None and edge_spatial_features is None:
                edge_features = self.input_edge_encoder(edge_features)
            elif edge_features is None and edge_spatial_features is not None:
                edge_features = self.input_edge_encoder(edge_spatial_features)
            else:
                raise ValueError('Either edge_features or edge_spatial_features must be provided when use_edge_encoder is True.')
        else:
            if edge_features is None and edge_spatial_features is not None:
                edge_features = edge_spatial_features
        
        for i, layer in enumerate(self.message_passing_layers):
            verbosePrint(f'\tPassing through message passing layer {i+1}/{self.hidden_layers}: query: {nodes_query.shape}, key/value: {nodes_key_value.shape}, edge: {edge_features.shape}.', self.verbose)

            ans = layer(
                queryTokens = nodes_query,
                valueTokens = nodes_key_value,
                edge_index = edge_indices,
                edgeAttention = None,
                edgeTokens = edge_features,
                spatialTokens = None,
                positionBiasTokens = None,
                windowValues = None,
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

            verbosePrint(f'\tDone message passing layer {i+1}/{self.hidden_layers}. Shape: {ans.shape}', self.verbose)

            if self.use_edge_encoder and self.message_edge_mlps is not None and edge_features is not None and i < len(self.message_edge_mlps):

                q = nodes_query.view(nodes_query.shape[0] * nodes_query.shape[1],1, -1)[edge_indices[0]]
                k = nodes_key_value.view(nodes_key_value.shape[0] * nodes_key_value.shape[1],1, -1)[edge_indices[1]]
                qk = torch.cat([q,k], dim=-1)
                e = edge_features.view(edge_features.shape[0], 1, -1)


                newEdges = self.message_edge_mlps[i](
                    tokens = e,
                    edgeTokens = qk,
                    spatialTokens = None,
                    windowValues = None,
                    positionBiasTokens = None,
                ).squeeze(1)

                edge_features = newEdges

                verbosePrint(f'\tUpdated edge features through edge MLP at layer {i+1}/{self.hidden_layers}. Shape: {edge_features.shape}', self.verbose)   

            nodes_query = ans
            if isinstance(node_features, Tensor):
                nodes_key_value = nodes_query
            elif self.convolution_layers > 1:
                raise NotImplementedError('Currently only supports single pass when using separate query and key/value nodes.')
            else:
                pass


        if self.use_decoder:
            nodes_query = self.outputDecoder(nodes_query, inputPositions=positions_query)

        return nodes_query

        