from dataclasses import dataclass, field
from mlUtil.networkUtil import mergeConfigWithKwargs, verboseBannerPrint, verbosePrint
from mlUtil.activation import getActivationFromString
from typing import Optional, Tuple, Union, List
import math
import torch
import copy
import torch.nn as nn
import warnings


'''
The input to the MLP is of shape [B, N, F] where
- B is the batch size
- N is the number of points
- F is the feature dimension
The output is of shape [B, N, output_dim]. For an image problem [B,H,W,C] with C channels, this is effectively the same with F = C and N = H*W. There are different ways to normalize this problem:

batch norm: Normalize over B and N, for each feature independently
layer norm: Normalize over F, for each point independently
instance norm: Normalize over N and F, for each batch independently
group norm: Normalize over groups of features, for each point independently
position norm: Normalize over N, for each feature and batch independently

All of these normalization layers can be useful in different contexts. Batch norm is often used in image problems, while layer norm is often used in NLP problems. Instance norm is often used in style transfer problems. Group norm is often used in object detection problems. Position norm is often used in point cloud problems. To configure the normalization, use the following flags:
- pre_norm: Apply normalization before the linear layer
- post_norm: Apply normalization after the linear layer
- hidden_norm: Apply normalization after the activation function in the hidden layers
- norm_type: The type of normalization to use. Options are 'layer', 'batch', 'instance', 'group[num_groups]'. For group norm, specify the number of groups in square brackets, e.g. 'group[8]'.

To configure the Layer norm you also need to provide the dimensions of the input features! This means that you need to provide [batch_size, seq_length, input_dim]. However, not all are required for all configurations of norms.
'''

@dataclass
class MLPConfig:
    input_dim: int = -1
    output_dim: int = -1

    batch_size: int = -1
    seq_length: int = -1

    hidden_dim : int = -64
    hidden_layers : int = 2
    activation : str = 'relu'
    initializer: str = 'uniform'  # 'uniform', 'normal', 'xavier', 'xavier_normal'

    dropout : float = 0.0
    gain: float = 1.0

    pre_norm : bool = False
    post_norm : bool = False
    hidden_norm: bool = False
    norm_type : str = 'layer'  # 'layer', 'batch', 'instance', 'group[num_groups]'
    norm_affine: Optional[bool] = None
    

    skip_linear : bool = False
    bias: bool = False
    residual: bool = False

def verbosePrintTensor(verbose, verbosePrefix, name, tensor):
    if not verbose:
        return
    print(''.join(['-']*80))
    print(f'{verbosePrefix}Tensor {name}: shape {tensor.shape}, dtype {tensor.dtype}, device {tensor.device}')
    print(f'{verbosePrefix}\tmin: {tensor.min().item()}, max: {tensor.max().item()}, mean: {tensor.mean().item()}, std: {tensor.std().item()}')
    # print(tensor)
    for i in range(tensor.shape[-1]):
        if len(tensor.shape) == 2:
            print(f'{verbosePrefix}\tChannel {i}: min: {tensor[:,i].min().item()}, max: {tensor[:,i].max().item()}, mean: {tensor[:,i].mean().item()}, std: {tensor[:,i].std().item()}, data: ')
        else:
            print(f'{verbosePrefix}\tChannel {i}: min: {tensor[:,:,i].min().item()}, max: {tensor[:,:,i].max().item()}, mean: {tensor[:,:,i].mean().item()}, std: {tensor[:,:,i].std().item()}, data: ')

    # for i in range(tensor.shape[-2]):
    #     print(f'\tParticle {i}: min: {tensor[:,i].min().item()}, max: {tensor[:,i].max().item()}, mean: {tensor[:,i].mean().item()}, std: {tensor[:,i].std().item()}, data: ')
    print(''.join(['-']*80))

from layers.norm import NormLayer

class MLP(torch.nn.Module):
    def _buildNormLayer(self, channel_dim: int, prefix: str = 'NormLayer', verbose: Optional[bool] = None):
        return NormLayer(self.config.norm_type, self.config.batch_size, self.config.seq_length, channel_dim, verbose = verbose if verbose is not None else self.verbose, verbosePrefix = f'{self.verbosePrefix} [{prefix}] ')

    def _buildLinearLayer(self, in_dim: int, out_dim: int):
        linear = torch.nn.Linear(in_dim, out_dim, bias=self.config.bias)
        if self.config.initializer == 'uniform':
            torch.nn.init.uniform_(linear.weight, self.config.gain *-1/math.sqrt(in_dim), self.config.gain * 1/math.sqrt(in_dim))
        elif self.config.initializer == 'normal':
            torch.nn.init.normal_(linear.weight, 0.0, self.config.gain * 1/math.sqrt(in_dim))
        elif self.config.initializer == 'xavier':
            torch.nn.init.xavier_uniform_(linear.weight, gain=self.config.gain)
        elif self.config.initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(linear.weight, gain=self.config.gain)
        else:
            raise ValueError(f'Unknown initializer: {self.config.initializer}')
        if self.config.bias:
            torch.nn.init.constant_(linear.bias, 0.0)
        return linear

    def _build(self):
        verbosePrint(f'{self.verbosePrefix}Building MLP', self.verbose)

        self.activation, self.activationString = getActivationFromString(self.config.activation)
        verbosePrint(f'{self.verbosePrefix}Using activation: {self.activationString}', self.verbose)

        if self.config.pre_norm:
            self.preNormLayer = self._buildNormLayer(self.config.input_dim, prefix ='PreNorm')
            verbosePrint(f'{self.verbosePrefix}Using pre-norm: {self.config.norm_type}', self.verbose)
        else:
            self.preNormLayer = nn.Identity()
            verbosePrint(f'{self.verbosePrefix}No pre-norm', self.verbose)

        if self.config.post_norm:
            self.postNormLayer = self._buildNormLayer(self.config.output_dim, prefix ='PostNorm')
            verbosePrint(f'{self.verbosePrefix}Using post-norm: {self.config.norm_type}', self.verbose)
        else:
            self.postNormLayer = nn.Identity()
            verbosePrint(f'{self.verbosePrefix}No post-norm', self.verbose)

        if self.config.skip_linear:
            self.layers = nn.Identity()
            verbosePrint(f'{self.verbosePrefix}Using skip-linear, no hidden layers', self.verbose)
            self.finalLinear = nn.Identity() if self.config.input_dim == self.config.output_dim else self._buildLinearLayer(self.config.input_dim, self.config.output_dim)
            verbosePrint(f'{self.verbosePrefix}Using final linear layer with in_dim={self.config.input_dim}, out_dim={self.config.output_dim}', self.verbose)
            return
        layers = []
        verbosePrint(f'{self.verbosePrefix}Building hidden layers', self.verbose)
        in_dim = self.config.input_dim
        for i in range(self.config.hidden_layers):
            out_dim = self.config.hidden_dim
            verbosePrint(f'{self.verbosePrefix}\tBuilding hidden layer {i+1}/{self.config.hidden_layers} with in_dim={in_dim}, out_dim={out_dim}', self.verbose)
            linear = self._buildLinearLayer(in_dim, out_dim)
            layers.append(linear)
            if self.config.hidden_norm:
                verbosePrint(f'{self.verbosePrefix}\tUsing hidden-norm: {self.config.norm_type}', self.verbose)
                normLayer = self._buildNormLayer(out_dim, prefix =f'HiddenNorm_Layer{i+1}', verbose = False)
                layers.append(normLayer)
            verbosePrint(f'{self.verbosePrefix}\tUsing activation: {self.activationString}', self.verbose)
            layers.append(self.activation)
            if self.config.dropout > 0.0:
                verbosePrint(f'{self.verbosePrefix}\tUsing dropout: {self.config.dropout}', self.verbose)
                layers.append(torch.nn.Dropout(self.config.dropout))
            in_dim = out_dim

        self.layers = nn.Sequential(*layers)
        self.finalLinear = self._buildLinearLayer(in_dim, self.config.output_dim)
        verbosePrint(f'{self.verbosePrefix}Using final linear layer with in_dim={in_dim}, out_dim={self.config.output_dim}', self.verbose)

    def __init__(self, 
                 in_features : Optional[int] = None,
                 out_features : Optional[int] = None,

                 config: Optional[MLPConfig] = None,
                verbose: bool = False,
                verbosePrefix: str = '',
                   **kwargs):
        super(MLP, self).__init__()
        verboseBannerPrint(f'{verbosePrefix}Initializing MLP', verbose)

        verbosePrint(f'{verbosePrefix}MLP init kwargs: {kwargs}', verbose)
        verbosePrint(f'{verbosePrefix}MLP init config: {config}', verbose)
        verbosePrint(f'{verbosePrefix}MLP init in_features: {in_features}, out_features: {out_features}', verbose)
        self.verbose = verbose
        self.verbosePrintTensor = False
        self.verbosePrefix = verbosePrefix

        if config is None:
            config = MLPConfig()
        self.config = copy.deepcopy(config)
        self.config = mergeConfigWithKwargs(self.config, **kwargs)

        if in_features is not None:
            self.config.input_dim = in_features
        if out_features is not None:
            self.config.output_dim = out_features
        if self.config.output_dim == -1 and self.config.input_dim == -1:
            raise ValueError('Either in_features or out_features must be specified')
        if self.config.input_dim == -1:
            raise ValueError('in_features must be specified')
        if self.config.output_dim == -1:
            self.config.output_dim = self.config.input_dim

        verbosePrint(f'{verbosePrefix}MLP Configuration: {self.config}', verbose)
        verbosePrint(f'{verbosePrefix}MLP Input Dim: {self.config.input_dim}', verbose)
        verbosePrint(f'{verbosePrefix}MLP Output Dim: {self.config.output_dim}', verbose)

        self._build()
        verbosePrint(f'{verbosePrefix}MLP Built', verbose)

        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        verbosePrint(f'{verbosePrefix}MLP Number of parameters: {params}', verbose)

        verboseBannerPrint(f'{verbosePrefix}MLP Initialization Complete', verbose)

    def __repr__(self):
        string = f'MLP(in_features={self.config.input_dim}, out_features={self.config.output_dim}, hidden_dim={self.config.hidden_dim}, hidden_layers={self.config.hidden_layers}, activation={self.activationString}, initializer={self.config.initializer}, dropout={self.config.dropout}, gain={self.config.gain}, pre_norm={self.config.pre_norm}, post_norm={self.config.post_norm}, hidden_norm={self.config.hidden_norm}, norm_type={self.config.norm_type}, skip_linear={self.config.skip_linear}, bias={self.config.bias}, residual={self.config.residual})'
    
        string += f'\nMLP Number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}'
        string += f'\nInput Norm Layer: {self.preNormLayer}'
        string += f'\nHidden Layers: {self.layers}'
        string += f'\nFinal Linear Layer: {self.finalLinear}'
        string += f'\nOutput Norm Layer: {self.postNormLayer}'
        return string


    def forward(self, x: torch.Tensor,
            gamma_scale: Optional[torch.Tensor] = None,
            beta_shift: Optional[torch.Tensor] = None,
            alpha_scale: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        verboseBannerPrint(f'{self.verbosePrefix}MLP Forward Pass', self.verbose)
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
        O = self.config.output_dim
        verbosePrint(f'{self.verbosePrefix}Input tensor shape after unsqueeze: {x.shape}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Batch size: {B}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Sequence length: {N}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Feature dimension: {F}', self.verbose)

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

        if self.config.batch_size != -1 and B != self.config.batch_size:
            raise ValueError(f'Batch size mismatch: expected {self.config.batch_size}, got {B}')
        if self.config.seq_length != -1 and N != self.config.seq_length:
            raise ValueError(f'Sequence length mismatch: expected {self.config.seq_length}, got {N}')
        if self.config.input_dim != -1 and F != self.config.input_dim:
            raise ValueError(f'Input feature dimension mismatch: expected {self.config.input_dim}, got {F}')
        
        verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'input', x)
        if self.config.pre_norm:
            verbosePrint(f'{self.verbosePrefix}Passing through pre-norm layer', self.verbose)
        out = self.preNormLayer(x)
        if self.config.pre_norm:
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after pre-norm', out)

        if gamma_scale is not None:
            verbosePrint(f'{self.verbosePrefix}Applying gamma scaling', self.verbose)
            out = out * gamma_scale
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after gamma scaling', out)
        if beta_shift is not None:
            verbosePrint(f'{self.verbosePrefix}Applying beta shifting', self.verbose)
            out = out + beta_shift
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after beta shifting', out)
        if not self.config.skip_linear:
            verbosePrint(f'{self.verbosePrefix}Passing through hidden layers', self.verbose)
        out = self.layers(out)
        if not self.config.skip_linear:
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after hidden layers', out)
        verbosePrint(f'{self.verbosePrefix}Passing through final linear layer', self.verbose)
        out = self.finalLinear(out)
        verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after final linear', out)
        if self.config.post_norm:
            verbosePrint(f'{self.verbosePrefix}Passing through post-norm layer', self.verbose)
        out = self.postNormLayer(out)
        if self.config.post_norm:
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after post-norm', out)
            verbosePrint(f'{self.verbosePrefix}Output tensor shape: {out.shape}', self.verbose)

        if alpha_scale is not None:
            verbosePrint(f'{self.verbosePrefix}Applying alpha scaling', self.verbose)
            out = out * alpha_scale
            verbosePrintTensor(self.verbosePrintTensor, self.verbosePrefix, 'after alpha scaling', out)

        if self.config.residual:
            if self.config.input_dim != self.config.output_dim:
                raise ValueError(f'Cannot use residual connection with different input and output dimensions: {self.config.input_dim} != {self.config.output_dim}')
            
            verbosePrint(f'{self.verbosePrefix}Adding residual connection', self.verbose)
            out = out + x
        if unsqueezed:
            verbosePrint(f'{self.verbosePrefix}Removing batch dimension', self.verbose)
            out = out.squeeze(0)
        return out
    