from layers.tokenEncoder import TokenEncoder, TokenEncoderConfig
import torch
import copy
from mlUtil.networkUtil import verbosePrint, verboseBannerPrint
from layers.mlp import MLP, MLPConfig
from typing import Optional


class BasicEncoder(torch.nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,

                 latent_dim: int = 32,       
                 hidden_layers: int = 2,  
                 
                 tokenEncoderConfig: Optional[TokenEncoderConfig] = None,
                 mlpConfig: Optional[MLPConfig] = None,
                 
                 verbose: bool = False,
                 verbosePrefix: str = ''
    ):
        super(BasicEncoder, self).__init__()
        if mlpConfig is None:
            raise ValueError('[DEBUG] mlpConfig must be provided.')
        verbosePrint(f'{verbosePrefix}Initializing Basic Encoder...', verbose)
        self.config = copy.deepcopy(tokenEncoderConfig) if tokenEncoderConfig is not None else TokenEncoderConfig()
        self.mlpConfig = copy.deepcopy(mlpConfig) if mlpConfig is not None else MLPConfig()
        self.verbose = verbose
        self.verbosePrefix = verbosePrefix

        self.config.token_input_dim = input_dim
        self.config.token_output_dim = output_dim
        self.config.token_latent_dim = output_dim

        if tokenEncoderConfig is None:
            verbosePrint(f'{verbosePrefix}\tUsing default Token Encoder config.', verbose)
            self.config.position_bias = None
            self.config.use_ffn = False

            self.config.projection = True
            self.config.projection_linear = False
            # self.config.projection_mlp_dict = {
            #     'num_layers': hidden_layers,
            #     'hidden_dim': latent_dim,
            #     'layout': [latent_dim] * hidden_layers,
            #     'activation': 'silu',
            #     'norm': False,
            #     'bias': False,
            # }
        self.encoder = TokenEncoder(self.config.token_input_dim, 
                                    self.config, verbose = verbose, verbosePrefix = verbosePrefix + 'TokenEncoder|',
                                    mlpConfig=self.mlpConfig)

        verbosePrint(f'{verbosePrefix}\tToken Encoder config: {self.config}', verbose)
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