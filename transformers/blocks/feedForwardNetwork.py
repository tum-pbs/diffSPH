from layers.mlp import *
from blocks.common import CommonConfiguration



class FeedForwardNetwork(nn.Module):
    def __init__(self,
                 in_features : Optional[int] = None,
                 out_features : Optional[int] = None,

                mlpConfig: Optional[MLPConfig] = None,
                embeddingConfig: Optional[MLPConfig] = None,
                config: Optional[CommonConfiguration] = None,

                verbose: bool = False,
                verbosePrefix: str = '',
                   **kwargs):
        super(FeedForwardNetwork, self).__init__()
        verboseBannerPrint(f'{verbosePrefix}Initializing FeedForwardNetwork', verbose)

        self.config = copy.deepcopy(config) if config is not None else CommonConfiguration()
        self.config = mergeConfigWithKwargs(self.config, **kwargs)

        # verbosePrint(self.config, verbose)

        self.mlpConfig = copy.deepcopy(mlpConfig) if mlpConfig is not None else (self.config.mlpConfig if self.config.mlpConfig is not None else MLPConfig())
        self.embeddingConfig = copy.deepcopy(embeddingConfig) if embeddingConfig is not None else (self.config.embeddingConfig if self.config.embeddingConfig is not None else MLPConfig())

        self.norm_type = self.config.norm_type
        self.pre_norm = self.config.pre_norm
        self.post_norm = self.config.post_norm
        self.use_conditioning = self.config.use_conditioning

        self.verbose = verbose
        self.verbosePrintTensor = False
        self.verbosePrefix = verbosePrefix


        if in_features is not None:
            self.mlpConfig.input_dim = in_features
        if out_features is not None:
            self.mlpConfig.output_dim = out_features
        if self.mlpConfig.output_dim == -1 and self.mlpConfig.input_dim == -1:
            raise ValueError('Either in_features or out_features must be specified')
        if self.mlpConfig.input_dim == -1:
            raise ValueError('in_features must be specified')
        if self.mlpConfig.output_dim == -1:
            self.mlpConfig.output_dim = self.mlpConfig.input_dim

        verbosePrint(f'{verbosePrefix}MLP Configuration: {self.mlpConfig}', verbose)
        verbosePrint(f'{verbosePrefix}MLP Input Dim: {self.mlpConfig.input_dim}', verbose)
        verbosePrint(f'{verbosePrefix}MLP Output Dim: {self.mlpConfig.output_dim}', verbose)

        self.mlp = MLP(in_features=self.mlpConfig.input_dim, out_features=self.mlpConfig.output_dim, config=self.mlpConfig, verbose=verbose, verbosePrefix=verbosePrefix)
        self.embedding = None


        if self.use_conditioning and self.embeddingConfig.input_dim > 0:
            self.embedding = MLP(in_features=self.embeddingConfig.input_dim, out_features=self.mlpConfig.input_dim * 2 + self.mlpConfig.output_dim, config=self.embeddingConfig, verbose=verbose, verbosePrefix=verbosePrefix+'[Embedding] ')
            verbosePrint(f'{verbosePrefix}Using embedding MLP with config: {self.embeddingConfig}', verbose)

            if not self.pre_norm:
                warnings.warn('Using embedding MLP without pre-norm in the main MLP. This may lead to instability.', UserWarning)
        elif self.use_conditioning:
            warnings.warn('use_conditioning is True but embeddingConfig.input_dim is not set. Ignoring conditioning.', UserWarning)
            self.use_conditioning = False
        else:
            verbosePrint(f'{verbosePrefix}Not using embedding MLP', verbose)

        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        verbosePrint(f'{verbosePrefix}FeedForwardNetwork Number of parameters: {params}', verbose)

        if self.pre_norm:
            self.pre_norm_layer = NormLayer(self.norm_type, self.mlpConfig.batch_size, self.mlpConfig.seq_length, self.mlpConfig.input_dim, verbose=verbose, verbosePrefix=verbosePrefix+'[PreNorm] ')
            verbosePrint(f'{verbosePrefix}Using pre-norm layer with type: {self.norm_type}', verbose)
        else:
            self.pre_norm_layer = nn.Identity()
            verbosePrint(f'{verbosePrefix}No pre-norm layer', verbose)
        if self.post_norm:
            self.post_norm_layer = NormLayer(self.norm_type, self.mlpConfig.batch_size, self.mlpConfig.seq_length, self.mlpConfig.output_dim, verbose=verbose, verbosePrefix=verbosePrefix+'[PostNorm] ')
            verbosePrint(f'{verbosePrefix}Using post-norm layer with type: {self.norm_type}', verbose)
        else:
            self.post_norm_layer = nn.Identity()
            verbosePrint(f'{verbosePrefix}No post-norm layer', verbose)


        verboseBannerPrint(f'{verbosePrefix}FeedForwardNetwork Initialization Complete', verbose)

    def forward(self, x: torch.Tensor, 
                embedding_input: Optional[Union[List[torch.Tensor], torch.Tensor]] = None
        ) -> torch.Tensor:
        verboseBannerPrint(f'{self.verbosePrefix}FeedForwardNetwork Forward Pass', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Input tensor shape: {x.shape}', self.verbose)
        # x is of shape [B, N, F] or [N,F]
        # if x is of shape [N,F], add a batch dimension
        if x.dim() == 2:
            verbosePrint(f'{self.verbosePrefix}Input tensor has no batch dimension, adding one', self.verbose)
            unsqueezed = True
            x = x.unsqueeze(0)
        else:
            unsqueezed = False

        B, N, F = x.shape
        O = self.mlpConfig.output_dim
        verbosePrint(f'{self.verbosePrefix}Input tensor shape after unsqueeze: {x.shape}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Batch size: {B}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Sequence length: {N}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Feature dimension: {F}', self.verbose)

        if self.mlpConfig.input_dim != -1 and F != self.mlpConfig.input_dim:
            raise ValueError(f'Input feature dimension mismatch: expected {self.mlpConfig.input_dim}, got {F}')
        
        gamma_scale = None
        beta_shift = None
        alpha_scale = None
        if self.embedding is not None:
            verboseBannerPrint(f'{self.verbosePrefix}Processing embedding input', self.verbose)
            if embedding_input is None:
                raise ValueError('embedding_input must be provided when using embedding MLP')
            verbosePrint(f'{self.verbosePrefix}Passing through embedding MLP', self.verbose)
            
            if isinstance(embedding_input, list):
                embedding_input = torch.cat(embedding_input, dim=-1)
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'embedding input', embedding_input)

            embedding_out = self.embedding(embedding_input)
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'embedding output', embedding_out)
            # embedding_out is of shape [B, F*2 + O]
            
            verbosePrint(f'{self.verbosePrefix}Embedding output shape after processing: {embedding_out.shape}', self.verbose)
            gamma_scale = embedding_out[:, :F]
            beta_shift = embedding_out[:, F:F*2]
            alpha_scale = embedding_out[:, F*2:]
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'gamma_scale', gamma_scale)
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'beta_shift', beta_shift)
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'alpha_scale', alpha_scale)


             # Process the optional scaling and shifting parameters
            if gamma_scale is not None:
                if gamma_scale.dim() == 1 and gamma_scale.shape[0] == F:
                    gamma_scale = gamma_scale.view(1, 1, F)
                elif gamma_scale.dim() == 2 and gamma_scale.shape[0] == B and gamma_scale.shape[1] == F:
                    gamma_scale = gamma_scale.view(B, 1, F)
                elif gamma_scale.dim() == 2 and gamma_scale.shape[0] == N and gamma_scale.shape[1] == F:
                    gamma_scale = gamma_scale.view(1, N, F)
                elif gamma_scale.dim() == 3 and gamma_scale.shape[0] == B and gamma_scale.shape[1] == N and gamma_scale.shape[2] == F:
                    pass
                else:
                    raise ValueError(f'Invalid shape for gamma_scale: {gamma_scale.shape}')
                # verbosePrintTensor(self.verbose, self.verbosePrefix, 'gamma_scale', gamma_scale)
                verbosePrint(f'{self.verbosePrefix}gamma_scale shape after processing: {gamma_scale.shape}', self.verbose)
            if beta_shift is not None:
                if beta_shift.dim() == 1 and beta_shift.shape[0] == F:
                    beta_shift = beta_shift.view(1, 1, F)
                elif beta_shift.dim() == 2 and beta_shift.shape[0] == B and beta_shift.shape[1] == F:
                    beta_shift = beta_shift.view(B, 1, F)
                elif beta_shift.dim() == 2 and beta_shift.shape[0] == N and beta_shift.shape[1] == F:
                    beta_shift = beta_shift.view(1, N, F)
                elif beta_shift.dim() == 3 and beta_shift.shape[0] == B and beta_shift.shape[1] == N and beta_shift.shape[2] == F:
                    pass
                else:
                    raise ValueError(f'Invalid shape for beta_shift: {beta_shift.shape}')
                # verbosePrintTensor(self.verbose, self.verbosePrefix, 'beta_shift', beta_shift)
                verbosePrint(f'{self.verbosePrefix}beta_shift shape after processing: {beta_shift.shape}', self.verbose)
            if alpha_scale is not None:
                if alpha_scale.dim() == 1 and alpha_scale.shape[0] == O:
                    alpha_scale = alpha_scale.view(1, 1, O)
                elif alpha_scale.dim() == 2 and alpha_scale.shape[0] == B and alpha_scale.shape[1] == O:
                    alpha_scale = alpha_scale.view(B, 1, O)
                elif alpha_scale.dim() == 2 and alpha_scale.shape[0] == N and alpha_scale.shape[1] == O:
                    alpha_scale = alpha_scale.view(1, N, O)
                elif alpha_scale.dim() == 3 and alpha_scale.shape[0] == B and alpha_scale.shape[1] == N and alpha_scale.shape[2] == O:
                    pass
                else:
                    raise ValueError(f'Invalid shape for alpha_scale: {alpha_scale.shape}')
                # verbosePrintTensor(self.verbose, self.verbosePrefix, 'alpha_scale', alpha_scale)
                verbosePrint(f'{self.verbosePrefix}alpha_scale shape after processing: {alpha_scale.shape}', self.verbose)

        else:
            if embedding_input is not None:
                verbosePrint(f'{self.verbosePrefix}Ignoring embedding_input since no embedding MLP is used', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Passing through pre-norm layer', self.verbose)
        x = self.pre_norm_layer(x)
        verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after pre-norm', x)

        if self.use_conditioning:
            verboseBannerPrint(f'{self.verbosePrefix}Applying conditioning', self.verbose)
        if gamma_scale is not None:
            verbosePrint(f'{self.verbosePrefix}Applying gamma scaling', self.verbose)
            x = x * gamma_scale
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after gamma scaling', x)
        if beta_shift is not None:
            verbosePrint(f'{self.verbosePrefix}Applying beta shifting', self.verbose)
            x = x + beta_shift
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after beta shifting', x)

        # Main MLP
        verbosePrint(f'{self.verbosePrefix}Passing through main MLP', self.verbose)
        out = self.mlp(x)

        verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after main MLP', out)
        verbosePrint(f'{self.verbosePrefix}Passing through post-norm layer', self.verbose)
        out = self.post_norm_layer(out)
        verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after post-norm', out)
        if alpha_scale is not None:
            verbosePrint(f'{self.verbosePrefix}Applying alpha scaling', self.verbose)
            out = out * alpha_scale
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after alpha scaling', out)

        if unsqueezed:
            verbosePrint(f'{self.verbosePrefix}Removing batch dimension', self.verbose)
            out = out.squeeze(0)
        verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'output', out)
        return out