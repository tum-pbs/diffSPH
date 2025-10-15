from layers.tokenEncoder import TokenEncoder, TokenEncoderConfig
import torch
import copy
from mlUtil.networkUtil import verbosePrint, verboseBannerPrint
from layers.mlp import MLP, MLPConfig
from typing import Optional
from blocks.common import CommonConfiguration
from mlUtil.networkUtil import mergeConfigWithKwargs


class BasicEncoder(torch.nn.Module):
    def __init__(self, 
                 token_input_dim: int,
                 token_output_dim: int,
                 
                 mlpConfig: Optional[MLPConfig] = None,
                 tokenEncoderConfig: Optional[TokenEncoderConfig] = None,
                 config: Optional[CommonConfiguration] = None,

                 verbose: bool = False,
                 verbosePrefix: str = '',
                 **kwargs
    ):
        super(BasicEncoder, self).__init__()
        
        verbosePrint(f'{verbosePrefix}Initializing Basic Encoder...', verbose)
        
        self.config = copy.deepcopy(config) if config is not None else CommonConfiguration()
        self.config = mergeConfigWithKwargs(self.config, **kwargs)
        self.tokenConfig = copy.deepcopy(tokenEncoderConfig) if tokenEncoderConfig is not None else TokenEncoderConfig()
        self.tokenConfig = mergeConfigWithKwargs(self.tokenConfig, **kwargs)

        self.mlpConfig = copy.deepcopy(mlpConfig) if mlpConfig is not None else (self.config.mlpConfig if self.config.mlpConfig is not None else MLPConfig())

        self.verbose = verbose
        self.verbosePrefix = verbosePrefix

        self.tokenConfig.token_input_dim = token_input_dim
        self.tokenConfig.token_output_dim = token_output_dim
        self.tokenConfig.token_latent_dim = token_output_dim

        if tokenEncoderConfig is None:
            verbosePrint(f'{verbosePrefix}\tUsing default Token Encoder config.', verbose)
            self.tokenConfig.position_bias = None
            self.tokenConfig.use_ffn = False

            self.tokenConfig.projection = True
            self.tokenConfig.projection_linear = False

        self.encoder = TokenEncoder(self.tokenConfig.token_input_dim,
                                    self.tokenConfig, verbose=verbose, verbosePrefix=verbosePrefix + 'TokenEncoder|',
                                    mlpConfig=self.mlpConfig)

        verbosePrint(f'{verbosePrefix}\tToken Encoder config: {self.tokenConfig}', verbose)
        numberOfParameters = sum(p.numel() for p in self.encoder.parameters())
        verbosePrint(f'{verbosePrefix}\tNumber of parameters in Token Encoder: {numberOfParameters}', verbose)
        verboseBannerPrint(f'{verbosePrefix}Done initializing Basic Encoder.', verbose)


    def forward(self, 
                inputTokens: torch.Tensor, # Shape: [num_tokens, input_dim]
                inputPositions: Optional[torch.Tensor] = None, # Shape: [num_tokens, spatial_dim],
                encodedInputPositions: Optional[torch.Tensor] = None, # Shape: [num_tokens, position_encoding_dim],
                ):
        return self.encoder(
            inputTokens = inputTokens,
            inputPositions = inputPositions,
            encodedInputPositions = encodedInputPositions
        )