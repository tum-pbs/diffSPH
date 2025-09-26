from copy import error
import warnings
import torch
from torch import Tensor
import torch.nn as nn

from layers.layer_tokenEncoder import TokenEncoder, TokenEncoderConfig
try:
    import torch_geometric
    from torch_geometric.utils import scatter, segment
    from torch_geometric.utils.num_nodes import maybe_num_nodes
except ImportError:
    torch_geometric = None
from typing import Optional, Union, Tuple
 

from .activation import getActivationLayer
from .basisFunctions import basisEncoderLayer
from .networkUtil import verbosePrint, verboseBannerPrint
from .sparse import buildSparseTensor
from .softmax import softmax
from .mlp import buildMLPwDict, getDefaultMLPDict
from .layer_positionEncoder import BasisEncoder, computeBasisEncoderOutputShape
from .windows import getWindowFunction
from typing import Optional, Union, Tuple
from dataclasses import dataclass, field
from .networkUtil import shapeMatch, verbosePrintSpatialTensorStats, mergeConfigWithKwargs, checkTensorShape
import copy
from .layer_positionEncoder import BasisEncoder, computeBasisEncoderOutputShape, BasisEncoderConfig




@dataclass(slots=True)
class TokenMixerConfig:
    num_heads: int = field(default=1, metadata={"help": "Number of attention heads"})
    transformer_features: int = field(default=16, metadata={"help": "Dimensionality of the attention features per head"})
    mixing_out_features: Optional[int] = field(default=None, metadata={"help": "Dimensionality of the output features per head after mixing"})

    input_channels: int = field(default=1, metadata={"help": "Number of token channels provided, e.g., 2 means we provide incoming and outgoing tokens for the mixer"})
    per_head: bool = field(default=False, metadata={"help": "Whether to have a separate mixing MLP per attention head (if False, use a single mixing MLP for all heads)"})

    spatial_dim: int = field(default=3, metadata={"help": "Dimensionality of the position vector per token (e.g. 3 for 3D positions)"})
    edge_feature_dim: int = field(default=0, metadata={"help": "Dimensionality of the edge feature vector per edge"})
    rpb_feature_dim: int = field(default=0, metadata={"help": "Dimensionality of the relative position bias feature vector per edge (if using relative position bias)"})

    skip_token_mixing: bool = field(default=False, metadata={"help": "If True, skip the token mixing and just return the input tokens (still performs channel mixing)"})
    mode: str = field(default='linear', metadata={"help": "Type of token mixing to use ('linear', 'mlp' or 'cconv', 'add', 'multiply')"})
    mlp_dict : Optional[dict] = field(default=None, metadata={"help": "Dictionary defining the MLP architecture for attention score computation "})

    include_edges: bool = field(default=False, metadata={"help": "Whether to use edge features in the attention score computation"})
    include_spatial: bool = field(default=True, metadata={"help": "Whether to use spatial information in the attention score computation"})
    include_rpb: bool = field(default=False, metadata={"help": "Whether to use relative position bias in the attention score computation"})
    include_window: bool = field(default=False, metadata={"help": "Whether to include a window function based on the distance between tokens in the attention score computation"})

    cconv_source: str = field(default='rpb', metadata={"help": "Source of the continuous convolution (e.g. 'rpb' for relative position bias)"})

    channel_mixing: bool = field(default=False, metadata={"help": "Indicates if channels should be mixed before the token mixing, if False the biasing is done per channel"})
    channel_broadcast: bool = field(default=True, metadata={"help": "If channel_mixing is True, whether to broadcast the channel mixed tokens, e.g., if the first channel is of shape [N,H,3] and the second channel is of shape [N,H,1], the second channel will be broadcasted to [N,H,3] before mixing. If False, an error will be raised if the channels have different shapes."})
    channel_mixing_operation: str = field(default='add', metadata={"help": "Operation to use for channel mixing ('add', 'multiply', 'concat', 'subtract', 'project', 'mean')"})

    channel_projection_linear: bool = field(default=False, metadata={"help": "If channel_mixing is True and channel_mixing_operation is 'project', whether to use a linear layer for projection (if False, use an MLP)"})
    channel_projection_mlp_dict: Optional[dict] = field(default=None, metadata={"help": "Dictionary defining the MLP architecture for channel projection (if channel_mixing_operation is 'project' and channel_projection_linear is False)"})
    channel_projection_out_features: Optional[int] = field(default=None, metadata={"help": "Output feature dimension after channel projection (if None, use transformer_features)"})
    channel_normalization: Optional[Union[float,str]] = field(default=None, metadata={"help": "Normalization to apply after channel mixing (if any), could be 'cosine', 'scaled'(uses d_k) or a float value to scale the output by"})

    basis_encoder: Optional[BasisEncoderConfig] = field(default=None, metadata={"help": "If provided, a BasisEncoderConfig to encode the spatial information before mixing. The input dimension of the encoder must match the spatial_dim."})

    channel_encoder: Optional[TokenEncoderConfig] = field(default=None, metadata={"help": "If provided, a TokenEncoderConfig to encode the channels before mixing. The input dimension of the encoder must match the number of channels."})
    channel_encoder_shared: bool = field(default=True, metadata={"help": "If True, use the same TokenEncoder for all channels, if False, use a separate TokenEncoder for each channel."})

    output_decoder: Optional[TokenEncoderConfig] = field(default=None, metadata={"help": "If provided, a TokenEncoderConfig to decode the output tokens after mixing. The input dimension of the decoder must match the mixing_out_features."})




""""""

from typing import List

class TokenMixer(torch.nn.Module):
    def __init__(self, 
                 config: Optional[TokenMixerConfig] = None,
                 verbose: bool = False,
                 verbosePrefix: str = '',
                 **kwargs
    ):
        super(TokenMixer, self).__init__()
        verboseBannerPrint('Initializing Input Mix Layer...', verbose)

        if config is None:
            config = TokenMixerConfig()
        else:
            config = copy.deepcopy(config)
        self.config = mergeConfigWithKwargs(config, **kwargs)

        self.verbose = verbose
        self.verbosePrefix = verbosePrefix

        input_dim = self.config.transformer_features

        if self.config.channel_encoder is not None:
            self.encoder_output_dim = self.config.channel_encoder.token_output_dim if self.config.channel_encoder.token_output_dim is not None else self.config.transformer_features
            self.config.channel_encoder.token_input_dim = self.config.transformer_features
            self.config.channel_encoder.token_output_dim = self.encoder_output_dim
            verbosePrint(f'Using channel encoder with input dimension {self.config.channel_encoder.token_input_dim} output dimension {self.config.channel_encoder.token_output_dim}', verbose, verbosePrefix=self.verbosePrefix+'\t')

            cfg = copy.deepcopy(self.config.channel_encoder)

            encoders = []
            if self.config.channel_encoder_shared:
                verbosePrint(f'Using shared channel encoder for all {self.config.input_channels} channels', verbose, verbosePrefix=self.verbosePrefix+'\t')
                encoder = TokenEncoder(token_input_dim=self.config.transformer_features, token_output_dim=cfg.token_output_dim, verbose=verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                encoders = [encoder for _ in range(self.config.input_channels)]
            else:
                verbosePrint(f'Using separate channel encoder for each of the {self.config.input_channels} channels', verbose, verbosePrefix=self.verbosePrefix+'\t')
                for i in range(self.config.input_channels):
                    verbosePrint(f'\tBuilding encoder for channel {i}', verbose, verbosePrefix=self.verbosePrefix+'\t')
                    encoder = TokenEncoder(token_input_dim=self.config.transformer_features, token_output_dim=cfg.token_output_dim, verbose=verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                    encoders.append(encoder)

            input_dim = self.encoder_output_dim
            self.channelEncoders = nn.ModuleList(encoders)
        if self.config.include_rpb or self.config.mode == 'cconv' and self.config.cconv_source == 'rpb':
            if self.config.rpb_feature_dim <= 0:
                if self.config.basis_encoder is not None:
                    self.basis_encoder = BasisEncoder(config=self.config.basis_encoder, verbose=verbose, verbosePrefix=self.verbosePrefix+'\t')
                    self.config.rpb_feature_dim = computeBasisEncoderOutputShape(self.config.basis_encoder)[-1]
                else:
                    raise ValueError('TokenMixer: relative position bias is included, but rpb_feature_dim is 0')
            elif self.config.basis_encoder is not None:
                raise ValueError('TokenMixer: rpb_feature_dim is provided, but basis_encoder is also defined in the config. Provide only one of them.')
            # input_dim += self.config.rpb_feature_dim
            # verbosePrint(f'Including relative position bias features of dimension {self.config.rpb_feature_dim}', verbose, verbosePrefix=self.verbosePrefix+'\t')

        if self.config.channel_mixing:
            verbosePrint(f'Channel mixing is enabled with operation {self.config.channel_mixing_operation}', verbose, verbosePrefix=self.verbosePrefix+'\t')
            if self.config.channel_mixing_operation not in ['add', 'multiply', 'concat', 'subtract', 'project', 'mean']:
                raise ValueError(f"TokenMixer: channel_mixing_operation must be one of 'add', 'multiply', 'concat', 'subtract', 'project' or 'mean', got {self.config.channel_mixing_operation}")
            if self.config.channel_mixing_operation == 'project':
                verbosePrint(f'Using channel projection for mixing {self.config.input_channels} channels', verbose, verbosePrefix=self.verbosePrefix+'\t')
                if self.config.channel_projection_out_features is None:
                    self.config.channel_projection_out_features = input_dim
                if self.config.channel_projection_linear:
                    verbosePrint(f'Using linear layer for channel projection [{self.config.input_channels} channels * {input_dim} features -> {self.config.channel_projection_out_features} features]', verbose, verbosePrefix=self.verbosePrefix+'\t')
                    self.channel_mixing_layer = nn.Linear(self.config.input_channels * input_dim, self.config.channel_projection_out_features)
                else:
                    verbosePrint(f'Using MLP for channel projection [{self.config.input_channels} channels * {input_dim} features -> {self.config.channel_projection_out_features} features]', verbose, verbosePrefix=self.verbosePrefix+'\t')
                    mlp_dict = self.config.channel_projection_mlp_dict
                    if mlp_dict is None:
                        mlp_dict = getDefaultMLPDict()
                    self.channel_mixing_layer = buildMLPwDict(mlp_dict, inputDim=self.config.input_channels * input_dim, outputDim=self.config.channel_projection_out_features)
                    
                input_dim = self.config.channel_projection_out_features
                verbosePrint(f'Using channel projection with output features {self.config.channel_projection_out_features}', verbose, verbosePrefix=self.verbosePrefix+'\t')
            elif self.config.channel_mixing_operation == 'concat':
                input_dim = self.config.input_channels * input_dim
                verbosePrint(f'Using channel concatenation, increasing input dimension to {input_dim}', verbose, verbosePrefix=self.verbosePrefix+'\t')
            elif self.config.channel_mixing_operation == 'mean':
                input_dim = input_dim
                verbosePrint(f'Using channel mean, keeping input dimension {input_dim}', verbose, verbosePrefix=self.verbosePrefix+'\t')
            elif self.config.channel_mixing_operation in ['dot', 'scaled_dot', 'inner', 'cosine']:
                input_dim = 1
                verbosePrint(f'Using channel {self.config.channel_mixing_operation}, reducing input dimension to {input_dim}', verbose, verbosePrefix=self.verbosePrefix+'\t')
            else:
                input_dim = input_dim
                verbosePrint(f'Using channel {self.config.channel_mixing_operation}, keeping input dimension {input_dim}', verbose, verbosePrefix=self.verbosePrefix+'\t')


        if self.config.include_edges:
            if self.config.edge_feature_dim <= 0:
                raise ValueError('TokenMixer: edge features are included, but edge_feature_dim is 0')
            input_dim += self.config.edge_feature_dim
            verbosePrint(f'Including edge features of dimension {self.config.edge_feature_dim}', verbose, verbosePrefix=self.verbosePrefix+'\t')
        if self.config.include_spatial:
            if self.config.spatial_dim <= 0:
                raise ValueError('TokenMixer: spatial information is included, but spatial_dim is 0')
            input_dim += self.config.spatial_dim
            verbosePrint(f'Including spatial information of dimension {self.config.spatial_dim}', verbose, verbosePrefix=self.verbosePrefix+'\t')
        if self.config.include_rpb:
            input_dim += self.config.rpb_feature_dim
            verbosePrint(f'Including relative position bias features of dimension {self.config.rpb_feature_dim}', verbose, verbosePrefix=self.verbosePrefix+'\t')

        if self.config.include_window:
            if not self.config.include_spatial:
                raise ValueError('TokenMixer: include_window is True, but include_spatial is False. Spatial information is needed to compute the window function.')
            input_dim += 1
            verbosePrint(f'Including window function in the mixing computation', verbose, verbosePrefix=self.verbosePrefix+'\t')

        verbosePrint(f'Input dimension to the token mixing: {input_dim}', verbose, verbosePrefix=self.verbosePrefix+'\t')
        self.input_dim = input_dim

        if self.config.mixing_out_features is None:
            self.config.mixing_out_features = self.config.transformer_features
        self.output_dim = self.config.mixing_out_features
        verbosePrint(f'Output dimension of the token mixing: {self.output_dim}', verbose, verbosePrefix=self.verbosePrefix+'\t')

        mixingLayers = []
        head_range = range(max(1,self.config.num_heads)) if self.config.per_head else range(1)

        for _ in head_range:
            mixingLayers.append(self._build_mixing_layer())

        self.mixingLayers = nn.ModuleList(mixingLayers)


        if self.config.output_decoder is not None:
            self.config.output_decoder.token_input_dim = self.config.mixing_out_features
            self.config.output_decoder.token_output_dim = self.config.mixing_out_features
            verbosePrint(f'Using output decoder with input and output dimension {self.config.mixing_out_features}', verbose, verbosePrefix=self.verbosePrefix+'\t')
            self.outputDecoder = TokenEncoder(token_input_dim=self.config.output_decoder.token_input_dim, verbose=verbose, verbosePrefix=self.verbosePrefix+'\t')

        verboseBannerPrint('Done initializing Input Mix Layer.', verbose)

    def _build_mixing_layer(self):
        mode = self.config.mode.lower()
        if mode == 'linear':
            mixingLayer = nn.Linear(self.input_dim, self.output_dim)
        elif mode == 'mlp':
            mlp_dict = self.config.mlp_dict
            if mlp_dict is None:
                mlp_dict = getDefaultMLPDict()
            mixingLayer = buildMLPwDict(mlp_dict, inputDim=self.input_dim, outputDim=self.output_dim)
        elif mode == 'cconv':
            if self.config.cconv_source not in ['rpb', 'spatial', 'edge']:
                raise ValueError(f"TokenMixer: cconv_source must be one of 'rpb', 'spatial' or 'edge', got {self.config.cconv_source}")
            if self.config.cconv_source == 'rpb' and self.config.rpb_feature_dim <= 0:
                raise ValueError('TokenMixer: cconv_source is rpb, but rpb_feature_dim is 0')
            if self.config.cconv_source == 'spatial' and self.config.spatial_dim <= 0:
                raise ValueError('TokenMixer: cconv_source is spatial, but spatial_dim is 0')
            if self.config.cconv_source == 'edge' and self.config.edge_feature_dim <= 0:
                raise ValueError('TokenMixer: cconv_source is edge, but edge_feature_dim is 0')

            kernel_input_dim = 0
            if self.config.cconv_source == 'rpb':
                kernel_input_dim = self.config.rpb_feature_dim
            elif self.config.cconv_source == 'spatial':
                kernel_input_dim = self.config.spatial_dim
            elif self.config.cconv_source == 'edge':
                kernel_input_dim = self.config.edge_feature_dim

            kernel_output_dim = self.input_dim * self.output_dim

            mixingLayer = nn.Linear(kernel_input_dim, kernel_output_dim)
        elif mode == 'add' or mode == 'multiply':
            if self.config.transformer_features != self.output_dim:
                raise ValueError(f'TokenMixer: For mode "{mode}", input_dim ({self.input_dim}) must be equal to output_dim ({self.output_dim})')
            mixingLayer = nn.Linear(self.input_dim - self.config.transformer_features, self.output_dim)

        else:
            raise ValueError(f"TokenMixer: mode must be one of 'linear', 'mlp' or 'cconv', got {self.config.mode}")
        return mixingLayer

    def mix(self,
        headInput: List[Tensor],
        headIndex: int,
        edgeTokens: Optional[Tensor] = None,
        spatialTokens: Optional[Tensor] = None,
        positionBiasTokens: Optional[Tensor] = None,          
    ):
        verbosePrint(f'Mixing head {headIndex} with mode {self.config.mode}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t', separator=True)

        if self.config.mode == 'linear' or self.config.mode == 'mlp':
            verbosePrint(f'Using {"linear" if self.config.mode == "linear" else "MLP"} mixing layer', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            headInputTensor = torch.cat(headInput, dim=-1) # shape [*, input_dim]
            verbosePrint(f'Head {headIndex} input tensor shape after concatenation: {headInputTensor.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
            
            mixedHead = self.mixingLayers[headIndex](headInputTensor) # shape [*, output_dim]
        elif self.config.mode == 'cconv':
            verbosePrint(f'Using continuous convolution mixing layer with source {self.config.cconv_source}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
            if self.config.cconv_source == 'rpb':
                verbosePrint(f'Using relative position bias features as kernel input', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
                if positionBiasTokens is None:
                    raise ValueError('TokenMixer: cconv_source is rpb, but positionBiasTokens is None')
                kernelInput = positionBiasTokens
            elif self.config.cconv_source == 'spatial':
                verbosePrint(f'Using spatial features as kernel input', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
                if spatialTokens is None:
                    raise ValueError('TokenMixer: cconv_source is spatial, but spatialTokens is None')
                kernelInput = spatialTokens
            elif self.config.cconv_source == 'edge':
                verbosePrint(f'Using edge features as kernel input', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
                if edgeTokens is None:
                    raise ValueError('TokenMixer: cconv_source is edge, but edgeTokens is None')
                kernelInput = edgeTokens
            else:
                raise ValueError(f"TokenMixer: cconv_source must be one of 'rpb', 'spatial' or 'edge', got {self.config.cconv_source}")
            
            verbosePrint(f'Kernel input shape: {kernelInput.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')

            kernel = self.mixingLayers[headIndex](kernelInput) # shape [*, input_dim * output_dim]
            kernel = kernel.view(*kernel.shape[:-1], self.input_dim, self.output_dim) # shape [*, input_dim, output_dim]
            verbosePrint(f'Kernel shape after linear layer and reshaping: {kernel.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')

            headInputTensor = torch.cat(headInput, dim=-1) # shape [*, input_dim]
            verbosePrint(f'Head {headIndex} input tensor shape after concatenation: {headInputTensor.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
            mixedHead = torch.einsum('n...i,nio->n...o', headInputTensor, kernel) # shape [*, output_dim]
        elif self.config.mode == 'add':
            verbosePrint(f'Using add mixing layer', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
            inputTokens = headInput[0] # shape [*, transformer_features]
            otherTokens = torch.cat(headInput[1:], dim=-1) # shape [*, input_dim - transformer_features]
            verbosePrint(f'Head {headIndex} input tensor shape after concatenation: {inputTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
            verbosePrint(f'Other tokens shape: {otherTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
            bias = self.mixingLayers[headIndex](otherTokens) # shape [*, output_dim]
            verbosePrint(f'Head {headIndex} bias shape: {bias.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
            mixedHead = inputTokens + bias # shape [*, output_dim]
        elif self.config.mode == 'multiply':
            verbosePrint(f'Using multiply mixing layer', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
            inputTokens = headInput[0] # shape [*, transformer_features]
            otherTokens = torch.cat(headInput[1:], dim=-1) # shape [*, input_dim - transformer_features]
            verbosePrint(f'Head {headIndex} input tensor shape after concatenation: {inputTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
            verbosePrint(f'Other tokens shape: {otherTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
            scale = self.mixingLayers[headIndex](otherTokens) # shape [*, output_dim]
            verbosePrint(f'Head {headIndex} scale shape: {scale.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
            mixedHead = inputTokens * scale # shape [*, output_dim]
        else:
            raise ValueError(f"TokenMixer: mode must be one of 'linear', 'mlp' or 'cconv', got {self.config.mode}")
        verbosePrint(f'Head {headIndex} mixed output shape: {mixedHead.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
        return mixedHead

        

    def forward(self,
        tokens: Union[Tensor, List[Tensor]], # shape [*, H, T]
        edgeTokens: Optional[Tensor] = None, # shape [*, H?, F_e]
        spatialTokens: Optional[Tensor] = None, # shape [*, H?, D]
        positionBiasTokens: Optional[Tensor] = None, # shape [*, H?, F_rpb]
        windowValues: Optional[Tensor] = None, # shape [*,H?]
    ):
        """ Mixes the input tokens using the configured mixing operation.
        
The tokens being input could either be an individual tensor or a list of tensors (for multiple channels). If a list of tensors is provided, channel mixing is applied first (if enabled in the config), followed by token mixing. All the channels are required to have the same shape!

The tokens could be of the following shapes in practice:

- Edge tokens: [num_edges, H, T] or [num_edges, T]
- Node tokens: [num_nodes, H, T] or [num_nodes, T]
- Batched node tokens: [batch_size, num_nodes, H, T] or [batch_size, num_nodes, T]

if self.config.num_heads is 0, we assume the inputs to be in the shape without heads, i.e. [*, T], and we add a head dimension of size 1 and remove it at the end.

        """
        verboseBannerPrint('TokenMixer: Forward Pass', self.verbose)
        if positionBiasTokens is not None and self.config.basis_encoder is not None:
            raise ValueError('TokenMixer: positionBiasTokens is provided, but basis_encoder is also defined in the config. Provide only one of them.')
        if positionBiasTokens is None and self.config.basis_encoder is not None:
            verbosePrint(f'Computing relative position bias features using basis encoder', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')
            positionBiasTokens = self.basis_encoder(spatialTokens)
            verbosePrint(f'Position bias tokens shape: {positionBiasTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')

        if self.config.num_heads == 0:
            if isinstance(tokens, list):
                tokens = [t.unsqueeze(-2) for t in tokens]
            else:
                tokens = tokens.unsqueeze(-2)
        inputTokens = [tokens] if not isinstance(tokens, list) else tokens

        if self.config.channel_encoder is not None:
            verbosePrint(f'Encoding input tokens using channel encoder', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            for i in range(len(inputTokens)):
                verbosePrint(f'Input tokens {i} shape before encoding: {inputTokens[i].shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                inputTokens[i] = self.channelEncoders[i](
                    inputTokens[i],
                    inputPositions=spatialTokens,
                    encodedInputPositions=positionBiasTokens)
                verbosePrint(f'Input tokens {i} shape after encoding: {inputTokens[i].shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')

        for i, input in enumerate(inputTokens):
            verbosePrint(f'Input tokens {i} shape: {input.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            if input.shape[-2] != self.config.num_heads and self.config.num_heads > 0:
                raise ValueError(f'TokenMixer: Number of heads in input tokens ({input.shape[-2]}) does not match config.num_heads ({self.config.num_heads})')
            if self.config.channel_broadcast:
                if input.shape[-1] != self.config.transformer_features and input.shape[-1] != 1:
                    raise ValueError(f'TokenMixer: Feature dimension of input tokens ({input.shape[-1]}) does not match config.transformer_features ({self.config.transformer_features}) or 1 (for broadcasting)')
                inputTokens[i] = input.expand(-1, -1, self.config.transformer_features)
                verbosePrint(f'Input tokens {i} shape after broadcasting (if needed): {inputTokens[i].shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

            else:
                if input.shape[-1] != self.config.transformer_features:
                    raise ValueError(f'TokenMixer: Feature dimension of input tokens ({input.shape[-1]}) does not match config.transformer_features ({self.config.transformer_features})')
        # inputTokens = [tokens] if not isinstance(tokens, list) else tokens
        if self.config.channel_mixing:
            if len(inputTokens) != self.config.input_channels:
                raise ValueError(f'TokenMixer: Number of input token channels ({len(inputTokens)}) does not match config.input_channels ({self.config.input_channels})')
            
            
        if edgeTokens is not None and self.config.include_edges:
            if edgeTokens.shape[-1] != self.config.edge_feature_dim:
                raise ValueError(f'TokenMixer: Feature dimension of edge tokens ({edgeTokens.shape[-1]}) does not match config.edge_feature_dim ({self.config.edge_feature_dim})')
        if spatialTokens is not None and self.config.include_spatial:
            if spatialTokens.shape[-1] != self.config.spatial_dim:
                raise ValueError(f'TokenMixer: Feature dimension of spatial tokens ({spatialTokens.shape[-1]}) does not match config.spatial_dim ({self.config.spatial_dim})')
        if positionBiasTokens is not None and self.config.include_rpb:
            if positionBiasTokens.shape[-1] != self.config.rpb_feature_dim:
                raise ValueError(f'TokenMixer: Feature dimension of position bias tokens ({positionBiasTokens.shape[-1]}) does not match config.rpb_feature_dim ({self.config.rpb_feature_dim})')
        if windowValues is not None and self.config.include_window:
            
            if windowValues.shape[-1] != 1:
                verbosePrint(f'TokenMixer: Warning: window values should have shape [*] or [*, 1], got {windowValues.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
                windowValues = windowValues.unsqueeze(-1)
                verbosePrint(f'TokenMixer: Adjusted window values shape to {windowValues.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')


        channelMixedTokens = inputTokens

        if self.config.channel_mixing:
            verbosePrint(f'Applying channel mixing with operation {self.config.channel_mixing_operation}', self.verbose, verbosePrefix=self.verbosePrefix+'\t', separator=True)

            for i, input in enumerate(inputTokens):
                verbosePrint(f'Input tokens {i} shape before channel mixing: {input.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')


            if self.config.channel_mixing_operation == 'add':
                channelMixedTokens = torch.stack(inputTokens, dim=0).sum(dim=0) # shape [*, H, T]
                verbosePrint(f'Channel mixed tokens shape after addition: {channelMixedTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            elif self.config.channel_mixing_operation == 'multiply':
                channelMixedTokens = torch.stack(inputTokens, dim=0).prod(dim=0) # shape [*, H, T]
                verbosePrint(f'Channel mixed tokens shape after multiplication: {channelMixedTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            elif self.config.channel_mixing_operation == 'mean':
                channelMixedTokens = torch.stack(inputTokens, dim=0).mean(dim=0) # shape [*, H, T]
                verbosePrint(f'Channel mixed tokens shape after mean: {channelMixedTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            elif self.config.channel_mixing_operation == 'subtract':
                if len(inputTokens) != 2:
                    raise ValueError(f'TokenMixer: channel_mixing_operation is subtract, but number of input token channels is {len(inputTokens)} (must be 2)')
                channelMixedTokens = inputTokens[0] - inputTokens[1] # shape [*, H, T]
                verbosePrint(f'Channel mixed tokens shape after subtraction: {channelMixedTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            elif self.config.channel_mixing_operation == 'concat':
                channelMixedTokens = torch.cat(inputTokens, dim=-1) # shape [*, H, input_channels * T]
                verbosePrint(f'Channel mixed tokens shape after concatenation: {channelMixedTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')    
            elif self.config.channel_mixing_operation == 'project':
                channelMixedTokens = torch.cat(inputTokens, dim=-1) # shape [*, H, input_channels * T]
                verbosePrint(f'Channel mixed tokens shape before projection: {channelMixedTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
                channelMixedTokens = self.channel_mixing_layer(channelMixedTokens) # shape [*, H, channel_projection_out_features]
                verbosePrint(f'Channel mixed tokens shape after projection: {channelMixedTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            elif self.config.channel_mixing_operation in ['dot', 'inner', 'scaled_dot', 'cosine']:
                channelMixedTokens = torch.stack(inputTokens, dim=0).prod(dim=0) # shape [*, H, T]
                verbosePrint(f'Channel mixed tokens shape after multiplication: {channelMixedTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
                channelMixedTokens = channelMixedTokens.sum(dim=-1, keepdim=True) # shape [*, H, 1]
                verbosePrint(f'Channel mixed tokens shape after summation: {channelMixedTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            else:
                raise ValueError(f"TokenMixer: channel_mixing_operation must be one of 'add', 'multiply', 'concat', 'subtract' or 'project', got {self.config.channel_mixing_operation}")
            channelMixedTokens = [channelMixedTokens] # make it a list for the next stage

            if self.config.channel_normalization is not None:
                if isinstance(self.config.channel_normalization, float):
                    verbosePrint(f'Applying scaling normalization with factor {self.config.channel_normalization} after channel mixing', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
                    verbosePrint(f'Channel mixed tokens shape before normalization: {channelMixedTokens[0].shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
                    channelMixedTokens = [c * self.config.channel_normalization for c in channelMixedTokens]

                elif self.config.channel_normalization == 'scaled':
                    scale = self.config.transformer_features ** 0.5
                    verbosePrint(f'Applying scaled normalization with factor sqrt(d_k)={scale} after channel mixing', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

                    channelMixedTokens = [c * scale for c in channelMixedTokens]
                elif self.config.channel_normalization == 'cosine':
                    norms = [torch.linalg.norm(t, dim=-1, keepdim=True) for t in inputTokens]

                    norm = torch.prod(torch.stack(norms, dim=0), dim=0)
                    verbosePrint(f'Computed cosine normalization factor with shape {norm.shape} after channel mixing', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
                    
                    channelMixedTokens = [c / (norm + 1e-8) for c in channelMixedTokens]
                    verbosePrint(f'Applying cosine normalization after channel mixing', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
                elif self.config.channel_normalization == 'length':
                    verbosePrint(f'Applying length normalization after channel mixing', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

                    channelMixedTokens = [c / (torch.linalg.norm(c, dim=-1, keepdim=True) + 1e-8) for c in channelMixedTokens]

                else:
                    raise ValueError(f"TokenMixer: channel_normalization must be a float value, 'scaled' or 'cosine', got {self.config.channel_normalization}")
                verbosePrint(f'Channel mixed tokens shape after normalization: {channelMixedTokens[0].shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

        else:
            verbosePrint(f'Channel mixing is disabled, processing each channel separately', self.verbose, verbosePrefix=self.verbosePrefix+'\t', separator=True)

        if self.config.skip_token_mixing:
            verbosePrint(f'Skipping token mixing as per configuration, returning channel mixed tokens', self.verbose, verbosePrefix=self.verbosePrefix+'\t', separator=True)

            if self.config.num_heads == 0:
                return channelMixedTokens[0].squeeze(-2) if len(channelMixedTokens) == 1 else [t.squeeze(-2) for t in channelMixedTokens]

            return channelMixedTokens[0] if len(channelMixedTokens) == 1 else channelMixedTokens

        outputTokenList = []
        for i in range(len(channelMixedTokens)):
            verbosePrint(f'Processing input token channel {i+1}/{len(channelMixedTokens)}', self.verbose, verbosePrefix=self.verbosePrefix+'\t', separator=True)
            inputTokens = channelMixedTokens[i] # shape [*, H, T]
            verbosePrint(f'Input tokens shape: {inputTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')

            if self.config.per_head:
                verbosePrint(f'Using separate mixing layers per head', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                if len(self.mixingLayers) != self.config.num_heads:
                    raise ValueError(f'TokenMixer: Number of mixing layers ({len(self.mixingLayers)}) does not match config.num_heads ({self.config.num_heads})')
                
                mixedTokens = []
                for h in range(self.config.num_heads):
                    verbosePrint(f'Processing head {h+1}/{self.config.num_heads}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                    headInput = [inputTokens[..., h, :]]
                    if self.config.include_edges:
                        verbosePrint(f'Including edge features in the mixing computation', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                        if edgeTokens is None:
                            raise ValueError('TokenMixer: include_edges is True, but edgeTokens is None')
                        headInput.append(edgeTokens)
                    if self.config.include_spatial:
                        verbosePrint(f'Including spatial features in the mixing computation', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                        if spatialTokens is None:
                            raise ValueError('TokenMixer: include_spatial is True, but spatialTokens is None')
                        headInput.append(spatialTokens)
                    if self.config.include_rpb:
                        verbosePrint(f'Including relative position bias features in the mixing computation', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                        if positionBiasTokens is None:
                            raise ValueError('TokenMixer: include_rpb is True, but positionBiasTokens is None')
                        headInput.append(positionBiasTokens)
                    if self.config.include_window:
                        verbosePrint(f'Including window function in the mixing computation', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                        if windowValues is None:
                            raise ValueError('TokenMixer: include_window is True, but windowValues is None')
                        headInput.append(windowValues)
                    verbosePrint(f'Head Input components: {len(headInput)}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                    for i, input in enumerate(headInput):
                        verbosePrint(f'Head {h} input {i} shape: {input.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')

                    # headInputTensor = torch.cat(headInput, dim=-1) # shape [*, input_dim]
                    # verbosePrint(f'Head {h} input tensor shape after concatenation: {headInputTensor.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

                    ### Apply mixing

                    mixedHead = self.mix(
                        headInput = headInput,
                        headIndex = h,
                        edgeTokens = edgeTokens,
                        spatialTokens = spatialTokens,
                        positionBiasTokens = positionBiasTokens,
                    )



                    verbosePrint(f'Head {h} mixed output shape: {mixedHead.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                    mixedTokens.append(mixedHead.unsqueeze(-2)) # shape [*, 1, output_dim]
                
                outputTokens = torch.cat(mixedTokens, dim=-2) # shape [*, H, output_dim]
                verbosePrint(f'Output tokens shape after concatenating heads: {outputTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            else:
                if len(self.mixingLayers) != 1:
                    raise ValueError(f'TokenMixer: Number of mixing layers ({len(self.mixingLayers)}) must be 1 when per_head is False')
                headInput = [inputTokens] # shape [*, H, transformer_features]
                if self.config.include_edges:
                    verbosePrint(f'Including edge features in the mixing computation', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                    if edgeTokens is None:
                        raise ValueError('TokenMixer: include_edges is True, but edgeTokens is None')
                    headInput.append(edgeTokens.unsqueeze(-2).expand(-1, self.config.num_heads, -1)) # shape [*, H, F_e]
                if self.config.include_spatial:
                    verbosePrint(f'Including spatial features in the mixing computation', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                    if spatialTokens is None:
                        raise ValueError('TokenMixer: include_spatial is True, but spatialTokens is None')
                    headInput.append(spatialTokens.unsqueeze(-2).expand(-1, self.config.num_heads, -1)) # shape [*, H, F_s]
                if self.config.include_rpb:
                    verbosePrint(f'Including relative position bias features in the mixing computation', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                    if positionBiasTokens is None:
                        raise ValueError('TokenMixer: include_rpb is True, but positionBiasTokens is None')
                    headInput.append(positionBiasTokens.unsqueeze(-2).expand(-1, self.config.num_heads, -1)) # shape [*, H, F_r]
                if self.config.include_window:
                    verbosePrint(f'Including window function in the mixing computation', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                    if windowValues is None:
                        raise ValueError('TokenMixer: include_window is True, but windowValues is None')
                    headInput.append(windowValues.unsqueeze(-2).expand(-1, self.config.num_heads, -1)) # shape [*, H, F_w]

                verbosePrint(f'Head Input components: {len(headInput)}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                for i, input in enumerate(headInput):
                    verbosePrint(f'Head {0} input {i} shape: {input.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t\t')

                # headInputTensor = torch.cat(headInput, dim=-1) # shape [*, input_dim]
                # verbosePrint(f'Head input tensor shape after concatenation: {headInputTensor.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')

                ### Apply mixing

                outputTokens = self.mix(
                    headInput = headInput,
                    headIndex = 0,
                    edgeTokens = edgeTokens,
                    spatialTokens = spatialTokens,
                    positionBiasTokens = positionBiasTokens,
                )

                verbosePrint(f'Output tokens shape after mixing: {outputTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                outputTokens = outputTokens.view(*inputTokens.shape[:-2], max(1,self.config.num_heads), self.output_dim) # shape [*, H, output_dim]
                verbosePrint(f'Output tokens shape after reshaping: {outputTokens.shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t')
            outputTokenList.append(outputTokens)

        if self.config.num_heads == 0:
            outputTokenList = [t.squeeze(-2) for t in outputTokenList]

        if self.config.output_decoder is not None:
            verbosePrint(f'Applying output decoder to the mixed tokens', self.verbose, verbosePrefix=self.verbosePrefix+'\t', separator=True)
            for i in range(len(outputTokenList)):
                verbosePrint(f'Output tokens {i} shape before decoding: {outputTokenList[i].shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')
                outputTokenList[i] = self.outputDecoder(outputTokenList[i], 
                    inputPositions=spatialTokens,
                    encodedInputPositions=positionBiasTokens)
                verbosePrint(f'Output tokens {i} shape after decoding: {outputTokenList[i].shape}', self.verbose, verbosePrefix=self.verbosePrefix+'\t\t')

        return outputTokenList if len(outputTokenList) > 1 else outputTokenList[0]