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


@dataclass
class CommonConfiguration:
    # General model options
    token_input_dim: int = field(default=16, metadata={"help": "Dimensionality of input node features."})
    token_output_dim: int = field(default=16, metadata={"help": "Dimensionality of output node features."})
    spatial_dim: int = field(default=2, metadata={"help": "Dimensionality of spatial coordinates."})
    edge_feature_dim: int = field(default=0, metadata={"help": "Dimensionality of edge features."})

    # Latent space options
    latent_features: int = field(default=32, metadata={"help": "Dimensionality of latent node features."})
    hidden_layers: int = field(default=2, metadata={"help": "Number of message passing layers."})

    # Encoder/Decoder options
    use_encoder: bool = field(default=True, metadata={"help": "Whether to use an input encoder."})
    use_decoder: bool = field(default=True, metadata={"help": "Whether to use an output decoder."})

    inputEncoderTokenConfig: Optional[TokenEncoderConfig] = field(default=None, metadata={"help": "Configuration for input node encoder."})
    outputDecoderTokenConfig: Optional[TokenEncoderConfig] = field(default=None, metadata={"help": "Configuration for output node decoder."})

    # Node based feed forward options
    node_ffn: bool = field(default=True, metadata={"help": "Whether to use a feed-forward network after each message passing layer."})
    ffn_skip_connection: bool = field(default=False, metadata={"help": "Whether to use skip connections for the feed-forward network."})
    ffn_skip_projection: bool = field(default=False, metadata={"help": "Whether to use skip connections for the feed-forward network projection."})
    post_ffn_norm: Optional[str] = field(default=None, metadata={"help": "Normalization to apply after feed-forward network."})
    ffn_skip_last: bool = field(default=False, metadata={"help": "Whether to run an FFN after the last message passing layer, duplicates with the output decoder!."})

    # Message passing options
    message_skip_connections: bool = field(default=True, metadata={"help": "Whether to use skip connections for message passing."})
    message_skip_projection: bool = field(default=False, metadata={"help": "Whether to use skip connections for the message passing projection."})
    post_message_norm: Optional[str] = field(default=None, metadata={"help": "Normalization to apply after message passing."})
    message_activation: Optional[str] = field(default=None, metadata={"help": "Activation function to use for message passing."})

    # Relative Position Bias Options
    basis_function: str = field(default='ffourier', metadata={"help": "Type of basis function for relative position encoding."})
    basis_terms: int = field(default=6, metadata={"help": "Number of basis terms for relative position encoding."})
    basis_projection: str = field(default='cartesian', metadata={"help": "Projection type for basis encoding."})
    basis_mode: str = field(default='cat', metadata={"help": "Mode for combining basis encodings."})

    # MLP configuration
    mlp_dict: Optional[dict] = field(default=None, metadata={"help": "Configuration dictionary for MLPs used in the model."})
    mlp_hidden_layers : int = field(default=2, metadata={"help": "Number of hidden layers in MLPs."})
    mlp_latent_dim : int = field(default=32, metadata={"help": "Latent dimensionality in MLPs."})
    mlp_activation : str = field(default='silu', metadata={"help": "Activation function in MLPs."})
    