from dataclasses import dataclass, field
from mlUtil.networkUtil import mergeConfigWithKwargs, verboseBannerPrint, verbosePrint
from mlUtil.activation import getActivationFromString
from typing import Optional, Tuple, Union, List
import math
import torch
import copy
import torch.nn as nn
import warnings





class NormLayer(torch.nn.Module):
    def __init__(self, norm_type, batch_size, seq_length, channel_dim, affine: Optional[bool] = None, verbose = False, verbosePrefix = ''):
        super(NormLayer, self).__init__()
        self.norm_type = norm_type
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.channel_dim = channel_dim
        self.verbose = verbose
        self.verbosePrefix = verbosePrefix

        print(f'{self.verbosePrefix}Building Norm Layer of type {norm_type} with channel_dim={channel_dim}, batch_size={batch_size}, seq_length={seq_length}, affine={affine}')
        self.affine = affine

        if norm_type == 'batch': # Normalize over B and N, for each feature independently
            self.norm = torch.nn.BatchNorm1d(channel_dim, affine=True if affine is None else affine)
            self.affine = True if affine is None else affine
        elif norm_type == 'layer': # Normalize over F, for each point independently
            self.norm = torch.nn.LayerNorm(channel_dim, elementwise_affine=True if affine is None else affine)
            self.affine = True if affine is None else affine
        elif norm_type == 'instance': # Normalize over N and F, for each batch independently
            self.norm = torch.nn.InstanceNorm1d(channel_dim, affine=False if affine is None else affine)
            self.affine = False if affine is None else affine
        elif norm_type.startswith('group'): # Normalize over groups of features, for each point independently
            num_groups = 8
            if '[' in norm_type and ']' in norm_type:
                num_groups = int(norm_type[norm_type.index('[')+1:norm_type.index(']')])
            self.norm = torch.nn.GroupNorm(num_groups, channel_dim, affine=True if affine is None else affine)
            self.affine = True if affine is None else affine
        elif norm_type == 'position': # Normalize over N, for each feature and batch independently
            self.norm = torch.nn.LayerNorm([seq_length, channel_dim], elementwise_affine=True if affine is None else affine)
            self.affine = True if affine is None else affine
        else:
            raise ValueError(f'Unknown norm_type: {norm_type}')


    def forward(self, x: torch.Tensor
                ) -> torch.Tensor:
        verboseBannerPrint(f'{self.verbosePrefix}Norm Layer Forward Pass', self.verbose)
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
        verbosePrint(f'{self.verbosePrefix}Input tensor shape after unsqueeze: {x.shape}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Batch size: {B}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Sequence length: {N}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Feature dimension: {F}', self.verbose)

        # The input is of shape [B, N, F]
        if self.norm_type == 'batch':
            # Batch Norm 1D expects input of shape [B, F, N]
            x = x.permute(0, 2, 1)
            verbosePrint(f'{self.verbosePrefix}Permuted input tensor shape for batch norm: {x.shape}', self.verbose)
            out = self.norm(x)
            out = out.permute(0, 2, 1)
            verbosePrint(f'{self.verbosePrefix}Output tensor shape after batch norm: {out.shape}', self.verbose)
        elif self.norm_type == 'instance':
            # Instance Norm 1D expects input of shape [B, F, N]
            x = x.permute(0, 2, 1)
            verbosePrint(f'{self.verbosePrefix}Permuted input tensor shape for instance norm: {x.shape}', self.verbose)
            out = self.norm(x)
            out = out.permute(0, 2, 1)
            verbosePrint(f'{self.verbosePrefix}Output tensor shape after instance norm: {out.shape}', self.verbose)
        elif 'group' in self.norm_type:
            # Group Norm expects input of shape [B, F, N]
            x = x.permute(0, 2, 1)
            verbosePrint(f'{self.verbosePrefix}Permuted input tensor shape for group norm: {x.shape}', self.verbose)
            out = self.norm(x)
            out = out.permute(0, 2, 1)
            # verbosePrint(f'{self.verbosePrefix}Output tensor shape after group norm: {out.shape}', self.verbose)
        else:
            # Layer Norm, Group Norm, Position Norm expect input of shape [B, N, F]
            out = self.norm(x)
            verbosePrint(f'{self.verbosePrefix}Output tensor shape after {self.norm_type} norm: {out.shape}', self.verbose) 
        verbosePrint(f'{self.verbosePrefix}Norm Layer output tensor shape: {out.shape}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Mean and std before norm: mean {x.mean().item()}, std {x.std().item()}', self.verbose)
        verbosePrint(f'{self.verbosePrefix}Mean and std after norm: mean {out.mean().item()}, std {out.std().item()}', self.verbose)

        return out

