from copy import error
import warnings
import torch
from torch import Tensor
import torch.nn as nn
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


# This is the basic attention mechanism that computes the attention scores (after softmax and scaling)
# The inputs are:
# queryTokens: (num_query_nodes, latent_dim) - the query node features
# keyTokens:   (num_key_nodes, latent_dim)   - the key node features
# edge_index:  (2, num_edges)         - the edges defining which key nodes are connected to which query nodes
# edge_attr:   (num_edges, edge_dim)   - the edge features for each edge
# s_k:         (num_edges) - scaling factors for the attention across all attention heads
# 
# The output is:
# attentionScoresSparse: (num_edges, num_heads) - the sparse attention scores
# 
# Configuration Parameters are:
# - latent_dim: int - the dimensionality of the input features
# - edge_dim: int - the dimensionality of the edge features
# - transformer_dim: Optional[int] - the dimensionality of the transformer (if None, set to latent_dim)
# - num_heads: int - the number of attention heads
# - attentionMechanism: str - the type of attention mechanism to use ('dot', 'scaled_dot', 'mlp', 'biLinearForm')
#
# Query/Key Parameters:
# - linearEncode: bool - whether to linearly encode the query and key features
# - linearEncodeDict: dict - dictionary with parameters for the linear encoding MLP
# - linearEncodeShared: bool - whether to share the linear encoding MLP between query and key
#
# Attention Score Parameters:
# - attentionScoreMLPDict: dict - dictionary with parameters for the attention score MLP
# - attentionDropout: float - dropout rate for the attention scores
# - attentionScaling: bool - whether to scale the attention scores by sqrt(latent_dim / num_heads)
# - attentionClipping: bool - whether to clip the attention scores
# - attentionClippingValue: float - the value to clip the attention scores to (if attentionClipping is True)
#
# Relative Position Bias Parameters:                 
# - relativePositionBias: If True, the relative distance of each edge is encoded and added to the input features
# - relativePositionBiasScaledPositions: If True, the input positions are scaled by a given cutoff radius before encoding
# - relativePositionBiasMultiplicative: If True, the relative position encoding is multiplied to the input features instead of added
# - relativePositionBiasBaseEncoding: If True, the relative position is encoded using a basis function encoding (e.g. Fourier or Gaussian basis)
# - relativePositionBiasBaseFunction: Type of basis function encoding to use for relative position (e.g. 'fourier', 'gaussian')
# - relativePositionBiasBaseTerms: Number of basis functions to use for relative position encoding
# - relativePositionBiasLinear: If true the rpb is a result of the (potentially encoded) positions passed through a linear layer to match the input feature dimension, if false an MLP is used
# - relativePositionBiasMLPDict: Dictionary defining the MLP architecture for relative position bias encoding (if relativePositionBiasLinear is False)
#
# Window Function Parameters:
# - windowFunction: bool - If True, a window function is applied to the attention based on the edge parameters
# - windowFunctionType: str - Type of window function to use ('cubic', 'quartic', etc.)
class AttentionMechanismLayer(torch.nn.Module):
    def __init__(self, 
                    latent_dim: int,
                    edge_dim: int,
                    transformer_features: Optional[int] = None,
                    num_heads: int = 4,

                    attentionMechanism: str = 'dot', # 'dot', 'scaled_dot',
                    attentionScoreMLPDict: Optional[dict] = None,
                    attentionDropout: float = 0.0,
                    attentionScaling: bool = True,
                    attentionClipping: bool = False,
                    attentionClippingValue: float = 1.0,
                    attentionActivation: str = 'leaky_relu(0.2)',

                    # Normal attention mechanisms apply W_Q and W_K to the input tokens to get the query and key tokens
                    # However, GAT v2 directly uses the input tokens as query and key, and only applies a linear layer to the concatenated [Q||K]
                    # To implement this we need to be able to turn off the linear encoding of Q and K
                    encodeTokens: bool = True,
                    encodeTokensLinear: bool = True,
                    encodeTokensMLPDict: Optional[dict] = None,
                    encodeTokensShared: bool = False, # True for GAT
                    cConvMode: bool = False, # True for cConv style attention (use edge features to compute W_Q and W_K)

                    relativePositionBias: bool = True,
                    relativePositionBiasAfterAttention: bool = True,
                    relativePositionBiasScaledPositions: bool = False,
                    relativePositionBiasMultiplicative: bool = False,
                    relativePositionBiasBaseEncoding: bool = True,
                    relativePositionBiasBaseFunction: str = 'fourier',
                    relativePositionBiasBaseTerms: int = 16,
                    relativePositionBiasBaseMode: str = 'cat', # 'cat', 'sum',
                    relativePositionBiasLinear: bool = True,
                    relativePositionBiasEncoder: Optional[bool] = None,
                    relativePositionBiasMLPDict: Optional[dict] = None, 
                    relativePositionBiasDim: Optional[int] = None,
                    relativePositionBiasSplit: bool = False,

                    windowFunction: bool = False,
                    windowFunctionType: str = 'cubic',
                    windowFunctionBeforeSoftmax: bool = True,

                    skipSoftmax: bool = False,

                    verbose: bool = False,
                 ):
        super(AttentionMechanismLayer, self).__init__()
        verbosePrint(f'Initializing Attention Mechanism Layer with parameters:', verbose, separator=True)

        ################################################################################
        #                           Set Class Parameters                               #
        ################################################################################
        self.latent_dim = latent_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.transformer_features = transformer_features if transformer_features is not None else latent_dim // num_heads
        if not encodeTokens and not cConvMode:
            self.transformer_features = latent_dim // num_heads

        self.transformer_dim = self.transformer_features * num_heads
        verbosePrint(f'\tLatent dimension: {self.latent_dim}', verbose)
        verbosePrint(f'\tEdge dimension: {self.edge_dim}', verbose)
        verbosePrint(f'\tTransformer dimension: {self.transformer_features}', verbose)
        verbosePrint(f'\tNumber of heads: {self.num_heads}', verbose)

        self.attentionMechanism = attentionMechanism
        self.attentionScoreMLPDict = attentionScoreMLPDict if attentionScoreMLPDict is not None else getDefaultMLPDict()
        self.attentionDropout = attentionDropout
        self.attentionScaling = attentionScaling
        self.attentionClipping = attentionClipping
        self.attentionClippingValue = attentionClippingValue
        self.attentionActivationName = attentionActivation.split('(')[0] if '(' in attentionActivation else attentionActivation
        activationArguments = () if '(' not in attentionActivation else attentionActivation[attentionActivation.index('(')+1:attentionActivation.index(')')].split(',')
        activationArguments = tuple([float(arg) for arg in activationArguments])
        self.attentionActivation = getActivationLayer(self.attentionActivationName, *activationArguments)
        self.skipSoftmax = skipSoftmax


        verbosePrint(f'\tAttention Mechanism: {self.attentionMechanism}', verbose, separator=True)
        verbosePrint(f'\tAttention Score MLP Dict: {self.attentionScoreMLPDict}', verbose)
        verbosePrint(f'\tAttention Dropout: {self.attentionDropout}', verbose)
        verbosePrint(f'\tAttention Scaling: {self.attentionScaling}', verbose)
        verbosePrint(f'\tAttention Clipping: {self.attentionClipping}', verbose)
        if self.attentionClipping:
            verbosePrint(f'\t\tAttention Clipping Value: {self.attentionClippingValue}', verbose)
        verbosePrint(f'\tAttention Activation: {self.attentionActivationName} with arguments {activationArguments}', verbose)
        verbosePrint(f'\tSkip Softmax: {self.skipSoftmax}', verbose)

        self.encodeTokens = encodeTokens
        self.encodeTokensLinear = encodeTokensLinear
        self.encodeTokensMLPDict = encodeTokensMLPDict if encodeTokensMLPDict is not None else getDefaultMLPDict()
        self.encodeTokensShared = encodeTokensShared
        self.cConvMode = cConvMode
        verbosePrint(f'\tEncode Tokens: {self.encodeTokens}', verbose, separator=True)
        verbosePrint(f'\tEncode Tokens Linear: {self.encodeTokensLinear}', verbose)
        verbosePrint(f'\tEncode Tokens MLP Dict: {self.encodeTokensMLPDict}', verbose)
        verbosePrint(f'\tEncode Tokens Shared: {self.encodeTokensShared}', verbose)
        verbosePrint(f'\tUsing continuous Convolution Mode: {self.cConvMode}', verbose)

        self.relativePositionBias = relativePositionBias
        self.relativePositionBiasScaledPositions = relativePositionBiasScaledPositions
        self.relativePositionBiasMultiplicative = relativePositionBiasMultiplicative
        self.relativePositionBiasBaseEncoding = relativePositionBiasBaseEncoding
        self.relativePositionBiasBaseFunction = relativePositionBiasBaseFunction
        self.relativePositionBiasBaseTerms = relativePositionBiasBaseTerms
        self.relativePositionBiasBaseMode = relativePositionBiasBaseMode
        self.relativePositionBiasLinear = relativePositionBiasLinear
        self.relativePositionBiasMLPDict = relativePositionBiasMLPDict if relativePositionBiasMLPDict is not None else getDefaultMLPDict()
        self.relativePositionBiasAfterAttention = relativePositionBiasAfterAttention
        self.relativePositionBiasEncoder = relativePositionBiasEncoder
        self.relativePositionBiasDim = relativePositionBiasDim if relativePositionBiasDim is not None else (self.num_heads if self.relativePositionBiasAfterAttention else self.
        transformer_dim)
        self.relativePositionBiasSplit = relativePositionBiasSplit
        verbosePrint(f'\tRelative Position Bias: {self.relativePositionBias}', verbose, separator=True)
        if self.relativePositionBias or self.cConvMode:
            verbosePrint(f'\t\tRelative Position Bias Scaled Positions: {self.relativePositionBiasScaledPositions}', verbose)
            verbosePrint(f'\t\tRelative Position Bias Multiplicative: {self.relativePositionBiasMultiplicative}', verbose)
            verbosePrint(f'\t\tRelative Position Bias Dimension: {self.relativePositionBiasDim}', verbose)
            verbosePrint(f'\t\tRelative Position Bias After Attention: {self.relativePositionBiasAfterAttention}', verbose)
            verbosePrint(f'\t\tRelative Position Bias Base Encoding: {self.relativePositionBiasBaseEncoding}', verbose)
            if self.relativePositionBiasBaseEncoding:
                verbosePrint(f'\t\tRelative Position Bias Base Function: {self.relativePositionBiasBaseFunction}', verbose)
                verbosePrint(f'\t\tRelative Position Bias Base Terms: {self.relativePositionBiasBaseTerms}', verbose)
                verbosePrint(f'\t\tRelative Position Bias Base Mode: {self.relativePositionBiasBaseMode}', verbose)
            verbosePrint(f'\t\tRelative Position Bias Linear: {self.relativePositionBiasLinear}', verbose)
            verbosePrint(f'\t\tRelative Position Bias MLP Dict: {self.relativePositionBiasMLPDict}', verbose)

        self.windowFunction = windowFunction
        self.windowFunctionType = windowFunctionType
        self.windowFunctionBeforeSoftmax = windowFunctionBeforeSoftmax
        verbosePrint(f'\tWindow Function: {self.windowFunction}', verbose, separator=True)
        if self.windowFunction:
            verbosePrint(f'\t\tWindow Function Type: {self.windowFunctionType}', verbose)
            verbosePrint(f'\t\tWindow Function Before Softmax: {self.windowFunctionBeforeSoftmax}', verbose)

        self.verbose = verbose

        ################################################################################
        #                           Start building the layer                           #
        ################################################################################

        ################################################################################
        #                        Build Query/Key Encoding Layer                       #
        ################################################################################

        verbosePrint(f'Building Query/Key Encoding...', verbose, separator=True)
        if not self.encodeTokens or self.cConvMode: # GAT does not encode the query and key tokens, it directly uses the input tokens as query and key
            verbosePrint(f'\tNot encoding query and key tokens, using input tokens directly as query and key', verbose)
            self.W_Q = nn.Identity()
            self.W_K = nn.Identity()
        else:
            if self.encodeTokensShared:
                verbosePrint(f'\tUsing shared encoding for query and key', verbose)
                if self.encodeTokensLinear:
                    verbosePrint(f'\t\tUsing linear encoding', verbose)
                    verbosePrint(f'\t\tShape: {self.latent_dim} -> {self.transformer_features * self.num_heads}', verbose)
                    self.W_qk = nn.Linear(self.latent_dim, self.transformer_features * self.num_heads, bias=False)
                else:
                    verbosePrint(f'\t\tUsing MLP encoding', verbose)
                    self.encodeTokensMLPDict['inputFeatures'] = self.latent_dim
                    self.encodeTokensMLPDict['output'] = self.transformer_features * self.num_heads
                    verbosePrint(f'\t\tShape: {self.encodeTokensMLPDict["inputFeatures"]} -> {self.encodeTokensMLPDict["output"]}', verbose)
                    self.W_qk = buildMLPwDict(self.encodeTokensMLPDict, verbose, verbosePrefix='\t\t')

                self.W_Q = self.W_K = self.W_qk
            else:
                verbosePrint(f'\tUsing separate encoding for query and key', verbose)
                if self.encodeTokensLinear:
                    verbosePrint(f'\t\tUsing linear encoding', verbose)
                    verbosePrint(f'\t\tShape: {self.latent_dim} -> {self.transformer_features * self.num_heads}', verbose)
                    self.W_Q = nn.Linear(self.latent_dim, self.transformer_features * self.num_heads, bias=False)
                    self.W_K = nn.Linear(self.latent_dim, self.transformer_features * self.num_heads, bias=False)
                else:
                    verbosePrint(f'\t\tUsing MLP encoding', verbose)
                    verbosePrint(f'\t\tShape: {self.latent_dim} -> {self.transformer_features * self.num_heads}', verbose)
                    self.encodeTokensMLPDict['inputFeatures'] = self.latent_dim
                    self.encodeTokensMLPDict['output'] = self.transformer_features * self.num_heads
                    self.W_Q = buildMLPwDict(self.encodeTokensMLPDict, verbose, verbosePrefix='\t\tQuery ')
                    self.W_K = buildMLPwDict(self.encodeTokensMLPDict, verbose, verbosePrefix='\t\tKey   ')

        ################################################################################
        #                     Build Relative Position Bias Layer                      #
        ################################################################################
        verbosePrint(f'Building Relative Position Bias...', verbose, separator=True)

        self.rpbEncoder = None

        _ = """
For the relative position bias there are a lot of options and its convenient to recap them here:

Generally we have two choices for the bias (this is not gating)
1. Compute the RPB _before_ the attention mechanism
2. Compute the RPB _after_ the attention mechanism

For the first case (before attention), this is only really sensible with a linear or MLP attention mechanism. In case of linear and MLP mechanisms the RPB can have any dimensions, however, there are still two sensible choices:
- The RPB has an arbitrary dimension and is applied equally to all attention heads
- The RPB has a dimension that is a multiple of the number of attention heads and is split across the attention heads

In case of applying the RPB after the attention mechanism, the RPB needs to have a dimension equal to the number of attention heads, as it is added to the attention scores for each head.

For the RPB computation we have the following options:
1 use the raw edge distances
2 encode the raw edge distances using a basis encoder
3 apply a projection (linear or MLP) to the (potentially encoded) edge distances to get the final RPB

For 1 the RPB dimension is [num_edges, spatial_dim]
For 2 the RPB dimension is [num_edges, encoded_dim] where encoded_dim dependss on the basis encoder
For 3 the RPB dimension is [num_edges, rpb_dim] where rpb_dim is the output dimension of the projection

We can also compute the output dimension of the RPB as requested by the user and get rpb_dim.

So we get the following logic:
if RPB before attention:
    if attention is dot or scaled dot:
        raise error
    else:
        if split across heads:
            if rpb_dim % num_heads != 0:
                raise error
            else:
                rpb_dim = rpb_dim
                rpb_shape = [num_heads, rpb_dim // num_heads]
        else: # applied equally to all heads
            rpb_dim = rpb_dim
            rpb_shape = [rpb_dim]
else: # RPB after attention
    if projection is None:
        if rpb_dim != num_heads:
            add projection to num_heads
    else:
        if projection is true:
            if rpb_dim != num_heads:
                raise error
        else: # projection is false
            if rpb_dim != num_heads:
                add projection to num_heads

This means that the RPB has the following parameters:
- spatial_dim: int - the spatial dimension of the edge distances (e.g. 3 for 3D positions)

- basis_terms: int - the number of basis functions to use for encoding the edge distances
- basis_function: str - the type of basis function to use for encoding the edge distances (e.g. 'fourier', 'gaussian')
- mode: str - the mode of combining the basis functions (e.g. 'cat', 'sum')

- skip_basis: bool - whether to skip the basis function encoding and use the raw edge distances
- split_across_heads: bool - whether to split the RPB across the attention heads (only relevant if RPB is computed before attention)
- after_attention: bool - whether the RPB is computed after the attention mechanism

- project_out: bool - whether to project the (potentially encoded) edge distances to the final RPB dimension
- project_linear: bool - whether to use a linear projection (if False, use MLP)
- project_mlp_properties: dict - dictionary defining the MLP architecture for the projection (if project_linear is False)

- out_dim: int - the final RPB dimension (only used if project_out is True)
"""

        if self.relativePositionBias or self.cConvMode:

            ################################################################################
            # Start by collecting all properties for the RPB
            ################################################################################

            spatial_dim = self.edge_dim

            basis_terms = self.relativePositionBiasBaseTerms
            basis_function = self.relativePositionBiasBaseFunction
            mode = self.relativePositionBiasBaseMode

            skip_basis = not self.relativePositionBiasBaseEncoding
            split_across_heads = self.relativePositionBiasSplit
            after_attention = self.relativePositionBiasAfterAttention

            project_out = self.relativePositionBiasEncoder
            project_linear = self.relativePositionBiasLinear
            project_mlp_dict = self.relativePositionBiasMLPDict

            out_dim = self.relativePositionBiasDim

            basisEncoderOutputShape = computeBasisEncoderOutputShape(
                spatial_dim=spatial_dim,
                basis_terms=basis_terms,
                basis_function=basis_function,
                skip_basis=skip_basis,
                mode=mode,
                project_out=self.relativePositionBiasEncoder if self.relativePositionBiasEncoder is not None else False,
                out_dim=out_dim,
                verbose=False
            )
            verbosePrint(f'\t\trpb basis function encoding output shape: {basisEncoderOutputShape}', self.verbose)
            out_dim = basisEncoderOutputShape

            ################################################################################
            # Dimension checking logic from above
            ################################################################################

            if self.cConvMode:
                verbosePrint(f'\tIn cConv mode the relative position bias is used to compute W_Q and W_K', verbose)
                # In CConv mode we always use the settings as defined
                # if project_out is None default to False
                self.relativePositionBiasEncoder = project_out if project_out is not None else True
                if self.relativePositionBiasSplit:
                    warnings.warn('Using relativePositionBiasSplit=True in cConvMode is not recommended, as it complicates the attention mechanism unnecessarily')
                    self.relativePositionBiasSplit = False
            else: # normal rpb
                verbosePrint(f'\tIn normal attention mode the relative position bias is used as bias to the attention scores', verbose)

                if not after_attention:
                    verbosePrint(f'\t\tRelative Position Bias is computed before the attention mechanism', verbose)
                    if self.attentionMechanism in ['dot', 'scaled_dot']:
                        verbosePrint(f'\t\tRelative Position Bias is not supported for dot or scaled dot attention', verbose)
                        raise ValueError("Relative position bias not supported for dot or scaled dot attention")
                    else:
                        if split_across_heads:
                            verbosePrint(f'\t\tRelative Position Bias is split across heads', verbose)
                            if out_dim % num_heads != 0:
                                verbosePrint(f'\t\tRelative Position Bias split across heads requires out_dim to be a multiple of num_heads', verbose)
                                raise ValueError(f'relativePositionBiasDim must be a multiple of num_heads ({self.num_heads}) if relativePositionBiasSplit is True and relativePositionBiasAfterAttention is False, got {out_dim}')
                            else:
                                verbosePrint(f'\t\tRelative Position Bias split across heads is valid', verbose)
                                rpb_dim = out_dim
                                rpb_shape = [num_heads, rpb_dim // num_heads]
                                out_dim = rpb_shape[0] * rpb_shape[1]
                                verbosePrint(f'\t\tSetting relative position bias dimension to {out_dim} [{rpb_shape[0]} {rpb_shape[1]}]', verbose)

                        else: # applied equally to all heads
                            rpb_dim = out_dim
                            rpb_shape = [rpb_dim]
                            verbosePrint(f'\t\tRelative Position Bias applied equally to all heads with dimension {out_dim} [{rpb_shape[0]}]', verbose)

                else: # RPB after attention
                    verbosePrint(f'\t\tRelative Position Bias is computed after the attention mechanism', verbose)
                    if project_out is None:
                        verbosePrint(f'\t\tRelative Position Bias projection to out_dim is not specified, setting to True', verbose)
                        if out_dim != num_heads:
                            verbosePrint(f'\t\trelativePositionBiasDim must be equal to num_heads ({self.num_heads}) if relativePositionBiasAfterAttention is True, got {out_dim}, setting relativePositionBiasDim to {num_heads}', verbose)
                            project_out = True
                            out_dim = num_heads
                    else:
                        if project_out is True:
                            verbosePrint(f'\t\tRelative Position Bias projection to out_dim is set to True', verbose)
                            if out_dim != num_heads:
                                verbosePrint(f'\t\trelativePositionBiasDim must be equal to num_heads ({self.num_heads}) if relativePositionBiasAfterAttention is True, got {out_dim}, setting relativePositionBiasDim to {num_heads}', verbose)
                                out_dim = num_heads
                                warnings.warn(f'relativePositionBiasDim must be equal to num_heads ({self.num_heads}) if relativePositionBiasAfterAttention is True and relativePositionBiasEncoder is True, got {out_dim}, setting relativePositionBiasDim to {out_dim}')
                        else: # projection is false
                            verbosePrint(f'\t\tRelative Position Bias projection to out_dim is set to False', verbose)
                            if out_dim != num_heads:
                                verbosePrint(f'\t\trelativePositionBiasDim must be equal to num_heads ({self.num_heads}) if relativePositionBiasAfterAttention is True and relativePositionBiasEncoder is False, got {out_dim}', verbose)
                                raise ValueError(f'relativePositionBiasDim must be equal to num_heads ({self.num_heads}) if relativePositionBiasAfterAttention is True and relativePositionBiasEncoder is False, got {out_dim}')
            self.relativePositionBiasEncoder = project_out
            split_across_heads = self.relativePositionBiasSplit
            
            self.rpbEncoder = BasisEncoder(
                spatial_dim=spatial_dim,
                basis_terms=basis_terms,
                basis_function=basis_function,
                skip_basis=skip_basis,
                mode=mode,
                
                out_dim=out_dim,
                
                project_mlp_properties=project_mlp_dict,
                project_linear=project_linear,
                project_out= self.relativePositionBiasEncoder,

                verbose=verbose, verbosePrefix='\t\t',
            )
            self.rpbDim = self.rpbEncoder.outputShape
            verbosePrint(f'\trpb encoder output shape: {self.rpbDim}', verbose)
            self.rpbDimPerHead = self.rpbDim // self.num_heads if split_across_heads else self.rpbDim
        else:
            self.rpbDim = 0
            self.rpbDimPerHead = 0

        ################################################################################
        # Build CConv Mode
        ################################################################################
        if self.cConvMode:
            verbosePrint(f'Building continuous Convolution Mode (cConv)...', verbose, separator=True)

            verbosePrint(f'Input Basis Shape: {self.rpbDim}', verbose)
            verbosePrint(f'Input Key/Query Shape: {self.latent_dim}', verbose)
            verbosePrint(f'Number of Heads: {self.num_heads} / Features per Head: {self.transformer_features // self.num_heads}', verbose)

            if self.encodeTokensShared:
                W_wqk = torch.nn.Linear(self.rpbDimPerHead, self.latent_dim * self.transformer_features * self.num_heads, bias=False)
                self.W_wq = self.W_wk = W_wqk
            else:
                self.W_wq = torch.nn.Linear(self.rpbDimPerHead, self.latent_dim * self.transformer_features * self.num_heads, bias=False)
                self.W_wk = torch.nn.Linear(self.rpbDimPerHead, self.latent_dim * self.transformer_features * self.num_heads, bias=False)

        ################################################################################
        #                     Build Attention Score Mechanism                          #
        ################################################################################

        verbosePrint(f'Building Attention Score Mechanism...', verbose, separator=True)
        attentionInputDim = self.transformer_features #if self.encodeTokens else self.latent_dim
        # if self.cConvMode:
            # attentionInputDim = self.rpbDimPerHead
            # verbosePrint(f'\tIn cConv mode the attention input dimension is set to the rpb dimension per head: {attentionInputDim}', verbose)
        if self.attentionMechanism == 'dot':
            verbosePrint(f'\tUsing dot product attention', verbose)
            if self.attentionScaling:
                verbosePrint(f'\t\twith scaling by sqrt(latent_dim / num_heads)', verbose)
            else:
                verbosePrint(f'\t\twithout scaling', verbose)
        elif self.attentionMechanism == 'scaled_dot':
            verbosePrint(f'\tUsing scaled dot product attention', verbose)
            verbosePrint(f'\t\twith scaling by sqrt(latent_dim / num_heads)', verbose)
            self.attentionScaling = nn.Linear(attentionInputDim, self.transformer_features, bias=False)
        elif self.attentionMechanism == 'mlp':
            verbosePrint(f'\tUsing MLP attention', verbose)

            self.attentionScoreMLPDict['inputFeatures'] = 2 * attentionInputDim + (self.rpbDimPerHead if not self.relativePositionBiasAfterAttention and not self.cConvMode else 0)
            self.attentionScoreMLPDict['output'] = 1
            verbosePrint(f'\t\tShape: {self.attentionScoreMLPDict["inputFeatures"]} -> {self.attentionScoreMLPDict["output"]}', verbose)
            self.attentionScoreMLP = buildMLPwDict(self.attentionScoreMLPDict, verbose, verbosePrefix='\t\t')
        elif self.attentionMechanism == 'linear':
            verbosePrint(f'\tUsing linear attention', verbose)
            attentionInputShape = 2 * attentionInputDim + (self.rpbDimPerHead if not self.relativePositionBiasAfterAttention and not self.cConvMode else 0)
            verbosePrint(f'\t\tShape: {attentionInputShape} -> {attentionInputDim}', verbose)
            self.attentionScoreLinear = nn.Linear(attentionInputShape, 1, bias=False)

        ################################################################################
        #                               Build Dropout                                 #
        ################################################################################
        if self.attentionDropout > 0.0:
            verbosePrint(f'Building Dropout Layer with p={self.attentionDropout}...', verbose, separator=True)
            self.attentionDropoutLayer = nn.Dropout(p=self.attentionDropout)
            self.dropout = nn.Dropout(p=self.attentionDropout)
        else:
            self.attentionDropoutLayer = None
            self.dropout = None

        ################################################################################
        #                                  Finalize                                    #
        ################################################################################

        verbosePrint(f'Attention Mechanism Layer initialized.', verbose, separator=True)
        


    def forward(self, 
                queryTokens: Tensor, # (num_query_nodes, latent_dim) (current tokens)
                keyTokens: Tensor,  # (num_key_nodes, latent_dim) (neighbor tokens)
                edge_index: Tensor, # (2, num_edges)
                edge_attr: Optional[Tensor] = None, # (num_edges, edge_dim)
                edge_scaling: Optional[Tensor] = None, # (num_edges)
                s_k: Optional[Tensor] = None, # (num_edges)
                ):
        verboseBannerPrint(f'Running Attention Mechanism Layer...', self.verbose)
        verbosePrint(f'\tQuery tokens shape: {queryTokens.shape} [B, Q, L]', self.verbose)
        verbosePrint(f'\tKey tokens shape: {keyTokens.shape} [B, K, L]', self.verbose)
        verbosePrint(f'\tEdge index shape: {edge_index.shape} [2, E]', self.verbose)
        verbosePrint(f'\tEdge attr shape: {edge_attr.shape if edge_attr is not None else None} [E, D]', self.verbose)
        verbosePrint(f'\tEdge scaling shape: {edge_scaling.shape if edge_scaling is not None else None} [E]', self.verbose)
        verbosePrint(f'\tS_k shape: {s_k.shape if s_k is not None else None} [B,K]', self.verbose)

        batch_size, num_nodes_current, latentSpaceSize = queryTokens.shape
        batch_size_edges = 1
        num_nodes_neighbor = keyTokens.shape[1]
        num_edges = edge_index.shape[1]
        rows = edge_index[0]
        cols = edge_index[1]
        verbosePrint(f'\tInput Shapes: [B, Q, K, L, D, E] = [{batch_size}, {num_nodes_current}, {num_nodes_neighbor}, {latentSpaceSize}, {self.edge_dim}, {num_edges}]', self.verbose)
        verbosePrint(f'\tTransformer Shapes: [H, T, T*H] = [{self.num_heads}, {self.transformer_features}, {self.transformer_dim}]', self.verbose)

        ################################################################################
        #                     Window Function on Edge Scaling                          #
        ################################################################################
        if self.windowFunction:
            verboseBannerPrint(f'Applying Window Function...', self.verbose)
            edgeLengths = torch.linalg.norm(edge_attr, dim=-1)
            # verbosePrint(f'Edge lengths min: {edgeLengths.min().item():.4f}, max: {edgeLengths.max().item():.4f}, mean: {edgeLengths.mean().item():.4f}, std: {edgeLengths.std().item():.4f}', self.verbose)
            windowScaling = getWindowFunction(self.windowFunctionType, norm= None)(torch.linalg.norm(edge_attr, dim=-1)) 
            verbosePrint(f'\tWindow function shape: {windowScaling.shape} [E]', self.verbose)
  # The scaling here is not normalized to 1 for the window function, we need to make sure the sum is still 1 after applying the window function
            # print(windowScaling )
            numNeighbors = scatter(torch.ones_like(rows), rows, dim=0, dim_size=num_nodes_current, reduce='sum')  # Shape: [num_nodes_current]
            windowScaling_sum = scatter(windowScaling, rows, dim=0, dim_size=num_nodes_current, reduce='sum')  # Shape: [num_nodes_current]
            windowScaling_sum = windowScaling_sum[rows]  # Shape: [num_edges]
            # print(windowScaling_sum)
            windowScaling = numNeighbors[rows] * windowScaling / (windowScaling_sum + 1e-16)  # Normalize to sum to 1 for each query node
            # print(f'Window Scaling min: {windowScaling.min().item():.4f}, max: {windowScaling.max().item():.4f}, mean: {windowScaling.mean().item():.4f}, std: {windowScaling.std().item():.4f}')

        ################################################################################
        #                     Encode Edge Attributes for RPB                           #
        ################################################################################
        verboseBannerPrint(f'Encoding Edge Attributes for Relative Position Bias', self.verbose)

        if self.relativePositionBias or self.cConvMode:
            verbosePrint(f'\tEncoding Relative Position Bias (RPB)...', self.verbose)
            if edge_attr is None:
                raise ValueError('edge_attr must be provided if relativePositionBias is True')
            if self.relativePositionBiasScaledPositions:
                raise NotImplementedError('relativePositionBiasScaledPositions is not implemented yet')
            rpbFeatures = self.rpbEncoder(edge_attr)
            verbosePrint(f'\t\tRPB basis function encoding output shape: {rpbFeatures.shape}', self.verbose)

            verbosePrint(f'\t\tRPB encoded features shape: {rpbFeatures.shape} [E, H * T]', self.verbose)
            if self.relativePositionBiasAfterAttention:
                rpbFeatures = rpbFeatures.view(1, -1, self.num_heads)
                verbosePrint(f'\t\tRPB reshaped features shape: {rpbFeatures.shape} [1, E, H]', self.verbose)
            else:
                if self.relativePositionBiasSplit:
                    rpbDimPerHead = self.rpbDimPerHead
                    rpbFeatures = rpbFeatures.view(-1, self.num_heads, rpbDimPerHead)
                    verbosePrint(f'\t\tRPB reshaped features shape: {rpbFeatures.shape} [E, H, {rpbDimPerHead}]', self.verbose)
                else:
                    rpbFeatures = rpbFeatures.view(-1, 1, self.rpbDim)
                    verbosePrint(f'\t\tRPB reshaped features shape: {rpbFeatures.shape} [E, 1, {self.rpbDim}]', self.verbose)
                    if not self.cConvMode:
                        rpbFeatures = rpbFeatures.repeat(1, self.num_heads, 1)
                        verbosePrint(f'\t\tRPB repeated features shape: {rpbFeatures.shape} [E, H, {self.rpbDim}]', self.verbose)

        else:
            rpbFeatures = None

        ################################################################################
        #                        Encode Query and Key Tokens                           #
        ################################################################################
        verboseBannerPrint(f'\tEncoding Query and Key Tokens...', self.verbose)

        if not self.cConvMode:
            Q = self.W_Q(queryTokens)
            K = self.W_K(keyTokens)
        
            verbosePrint(f'Input Token Shape [current ]: {queryTokens.shape} [B {batch_size} x N {num_nodes_current} x D {latentSpaceSize}]', self.verbose)
            verbosePrint(f'Input Token Shape [neighbor]: {keyTokens.shape} [B {batch_size} x N {num_nodes_neighbor} x D {latentSpaceSize}]', self.verbose)

            Q = Q.view(batch_size, Q.shape[1], self.num_heads, self.transformer_features).permute(0, 2, 1, 3)
            K = K.view(batch_size, K.shape[1], self.num_heads, self.transformer_features).permute(0, 2, 1, 3)

            verbosePrint(f'Query Shape: {Q.shape} [B {batch_size} x H {self.num_heads} x N {Q.shape[2]} x D {self.transformer_features}]', self.verbose)
            verbosePrint(f'Key Shape:   {K.shape} [B {batch_size} x H {self.num_heads} x N {K.shape[2]} x D {self.transformer_features}]', self.verbose)

            if self.attentionScaling:
                Q = Q / (self.transformer_features ** 0.5)  # Scale by sqrt(d_k)
                verbosePrint(f'\tScaled Query Shape Tokens by 1/{self.transformer_features} ** 0.5', self.verbose)

            Q_unified = Q.permute(1, 0, 2, 3).reshape(self.num_heads, batch_size * num_nodes_current, self.transformer_features)
            K_unified = K.permute(1, 0, 2, 3).reshape(self.num_heads, batch_size * num_nodes_neighbor, self.transformer_features)

            verbosePrint(f'Unified Query Shape: {Q_unified.shape} [H {self.num_heads} x B {batch_size} * N {num_nodes_current} x D {self.transformer_features}]', self.verbose, separator=False)
            verbosePrint(f'Unified Key Shape:   {K_unified.shape} [H {self.num_heads} x B {batch_size} * N {num_nodes_neighbor} x D {self.transformer_features}]', self.verbose)


            Q_i = Q_unified[:, rows, :] # Shape: [B, H, num_edges, F]
            K_j = K_unified[:, cols, :] # Shape: [B, H, num_edges, F]
        else:
            verbosePrint(f'Using continuous Convolution Mode (cConv) to compute Query and Key Tokens...', self.verbose)

            verbosePrint(f'Input Token Shape [current ]: {queryTokens.shape} [B {batch_size} x N {num_nodes_current} x D {latentSpaceSize}]', self.verbose)
            verbosePrint(f'Input Token Shape [neighbor]: {keyTokens.shape} [B {batch_size} x N {num_nodes_neighbor} x D {latentSpaceSize}]', self.verbose)
            verbosePrint(f'Input Token Shape [RBP]: {rpbFeatures.shape} [B {batch_size} x E {edge_attr.shape[0]} x D {self.relativePositionBiasDim}]', self.verbose)

            rpb = rpbFeatures.squeeze(1)
            w_q = self.W_wq(rpb)  # Shape: [num_edges, latent_dim * transformer_features * num_heads]
            w_k = self.W_wk(rpb)  # Shape: [num_edges, latent_dim * transformer_features * num_heads]
            verbosePrint(f'\tW_q shape: {w_q.shape} [E, L * T * H]', self.verbose)
            verbosePrint(f'\tW_k shape: {w_k.shape} [E, L * T * H]', self.verbose)

            w_q = w_q.view(-1, self.rpbDim, self.latent_dim)  # Shape: [num_edges, H * T, L]
            w_k = w_k.view(-1, self.rpbDim, self.latent_dim)  # Shape: [num_edges, H * T, L]

            node_i = queryTokens[0, rows, :]  # Shape: [num_edges, latent_dim]
            node_j = keyTokens[0, cols, :]

            verbosePrint(f'\tNode i shape: {node_i.shape} [E, L]', self.verbose)
            verbosePrint(f'\tNode j shape: {node_j.shape} [E, L]', self.verbose)

            verbosePrint(f'\tW_q reshaped shape: {w_q.shape} [E, H * T, L]', self.verbose)
            verbosePrint(f'\tW_k reshaped shape: {w_k.shape} [E, H * T, L]', self.verbose)


            Q_i = torch.einsum('el, efl -> ef', node_i, w_q).view(-1, self.num_heads, self.transformer_features)  # Shape: [B, E, H, F]
            K_j = torch.einsum('el, efl -> ef', node_j, w_k).view(-1, self.num_heads, self.transformer_features)  # Shape: [B, E, H, F]
            Q_i = Q_i.permute(1, 0, 2)  # Shape: [H, E, F]
            K_j = K_j.permute(1, 0, 2)  # Shape: [H, E, F]



        ################################################################################
        #                        Compute Attention Scores                              #
        ################################################################################
        verboseBannerPrint(f'Computing Attention Scores', self.verbose)
        verbosePrint(f'Collected Query Tokens: {Q_i.shape} [H {self.num_heads} x E {num_edges} x F {self.transformer_features}]', self.verbose)
        verbosePrint(f'Collected Key Tokens:   {K_j.shape} [H {self.num_heads} x E {num_edges} x F {self.transformer_features}]', self.verbose)

        verbosePrint(f'Computing Attention', self.verbose)

        if self.attentionMechanism == 'dot':
            if not self.cConvMode:
                if not self.relativePositionBiasAfterAttention and self.relativePositionBias:
                    raise NotImplementedError('Relative Position Bias before attention not possible for dot product attention')

            attentionScores = (Q_i * K_j).sum(dim=-1)  # Shape: [H, num_edges]
            verbosePrint(f'\tDot Product Attention Scores shape: {attentionScores.shape} [H {self.num_heads} x E {num_edges}]', self.verbose)
        elif self.attentionMechanism == 'scaled_dot':
            if not self.cConvMode:
                if not self.relativePositionBiasAfterAttention and self.relativePositionBias:
                    raise NotImplementedError('Relative Position Bias before attention not possible for dot product attention')

            attentionScoresProduct = (Q_i * K_j)  # Shape: [H, num_edges]
            attentionScores = self.attentionScaling(attentionScoresProduct).sum(-1)  # Shape: [H, num_edges]
            verbosePrint(f'\tScaled Dot Product Attention Scores shape: {attentionScores.shape} [H {self.num_heads} x E {num_edges}]', self.verbose)
        elif self.attentionMechanism == 'mlp':
            verbosePrint(f'\tMLP Attention Input shape: {Q_i.shape} [H {self.num_heads} x E {num_edges} x F {self.transformer_features}]', self.verbose)
            attentionInput = torch.cat([Q_i, K_j], dim=-1)  # Shape: [H, num_edges, 2*F]
            if not self.cConvMode:
                if self.relativePositionBias and not self.relativePositionBiasAfterAttention:
                    verbosePrint(f'\tAdding Relative Position Bias before Attention', self.verbose)

                    # if relativePositionBiasEncoder is set to false, the inputs are not encoded per head, so we need to repeat them for each head
                    # if not self.relativePositionBiasEncoder:
                        # rpbFeatures = rpbFeatures.repeat(1, 1, self.num_heads, 1)  # Shape: [1, E, H, D_rpb]
                    # rpbFeatures = rpbFeatures.squeeze(0)
                    verbosePrint(f'\t\tRPB Features shape: {rpbFeatures.shape} [E, H, T]', self.verbose)
                    attentionInput = torch.cat([attentionInput, rpbFeatures.permute(1,0,2)], dim=-1)  # Shape: [H, num_edges, 2*F + D_rpb]

            verbosePrint(f'\tAttention Input shape: {attentionInput.shape} [H {self.num_heads} x E {num_edges} x (2*F + D_rpb)]', self.verbose)

            attentionScores = self.attentionScoreMLP(attentionInput)  # Shape: [H, num_edges]
            verbosePrint(f'\tMLP Attention Scores shape: {attentionScores.shape} [H {self.num_heads} x E {num_edges}]', self.verbose)
        elif self.attentionMechanism == 'linear':
            verbosePrint(f'\tLinear Attention Input shape: {Q_i.shape} [H {self.num_heads} x E {num_edges} x F {self.transformer_features}]', self.verbose)
            attentionInput = torch.cat([Q_i, K_j], dim=-1)  # Shape: [H, num_edges, 2*F]
            if not self.cConvMode:
                if self.relativePositionBias and not self.relativePositionBiasAfterAttention:
                    verbosePrint(f'\tAdding Relative Position Bias before Attention', self.verbose)


                    # if relativePositionBiasEncoder is set to false, the inputs are not encoded per head, so we need to repeat them for each head
                    # if not self.relativePositionBiasEncoder:
                        # rpbFeatures = rpbFeatures.repeat(1, 1, self.num_heads, 1)  # Shape: [1, E, H, D_rpb]
                    # rpbFeatures = rpbFeatures.squeeze(0)
                    verbosePrint(f'\t\tRPB Features shape: {rpbFeatures.shape} [E, H,  T]', self.verbose)

                    
                    attentionInput = torch.cat([attentionInput, rpbFeatures.permute(1,0,2)], dim=-1)  # Shape: [H, num_edges, 2*F + D_rpb]

            verbosePrint(f'\tAttention Input shape: {attentionInput.shape} [H {self.num_heads} x E {num_edges} x (2*F + D_rpb)]', self.verbose)

            attentionScores = self.attentionScoreLinear(attentionInput).squeeze(-1)  # Shape: [H, num_edges]
            verbosePrint(f'\tLinear Attention Scores shape: {attentionScores.shape} [H {self.num_heads} x E {num_edges}]', self.verbose)
        else:
            raise ValueError(f'Unknown attention mechanism: {self.attentionMechanism}')




        if self.relativePositionBias and self.relativePositionBiasAfterAttention:
            verbosePrint(f'Adding Relative Position Bias after Attention', self.verbose)
            verbosePrint(f'\tRPB Features shape: {rpbFeatures.shape} [1, E, H]', self.verbose)
            verbosePrint(f'\tAttention Scores shape: {attentionScores.shape} [H, E]', self.verbose)

            if self.relativePositionBiasMultiplicative:
                attentionScores = attentionScores * rpbFeatures.squeeze(0).permute(1,0)  # Shape: [H, num_edges]
            else:
                attentionScores = attentionScores + rpbFeatures.squeeze(0).permute(1,0)  # Shape: [H, num_edges]
            verbosePrint(f'\tAttention Scores with RPB shape: {attentionScores.shape} [H {self.num_heads} x E {num_edges}]', self.verbose)

        attentionScores = attentionScores.view(1, self.num_heads, num_edges) # Shape: [1, H, num_edges]
        ################################################################################
        #                        Apply Attention Score Scaling                         #
        ################################################################################
        verboseBannerPrint(f'Applying Attention Score Scaling and Activation', self.verbose)

        if self.attentionActivation is not None:
            verbosePrint(f'\tApplying Attention Activation: {self.attentionActivationName}', self.verbose)
            attentionScores = self.attentionActivation(attentionScores)
            verbosePrint(f'\tAttention Scores after activation shape: {attentionScores.shape} [1 {1} x H {self.num_heads} x E {num_edges}]', self.verbose)

        if self.attentionClipping:
            verbosePrint(f'\tClipping Attention Scores to [{-self.attentionClippingValue}, {self.attentionClippingValue}]', self.verbose)
            attentionScores = torch.clamp(attentionScores, -self.attentionClippingValue, self.attentionClippingValue)
            verbosePrint(f'\tAttention Scores after clipping shape: {attentionScores.shape} [1 {1} x H {self.num_heads} x E {num_edges}]', self.verbose)

        if self.windowFunction and self.windowFunctionBeforeSoftmax:
            verbosePrint(f'\tApplying Window Function Scaling to Attention Scores', self.verbose)
            # Window function scaling shape: [E] -> [1, H, E]
            # First expand then repeat to match attention scores shape
            windowScaling_expanded = windowScaling.view(1, 1, -1).repeat(1, self.num_heads, 1)
            attentionScores = attentionScores * windowScaling_expanded
            verbosePrint(f'\tAttention Scores after window function shape: {attentionScores.shape} [1 {1} x H {self.num_heads} x E {num_edges}]', self.verbose)

        ################################################################################
        #                 Apply Softmax to get Attention Weights                       #
        ################################################################################
        verboseBannerPrint(f'Applying Sparse Softmax to get Attention Weights', self.verbose)
        sparse_values = attentionScores.flatten()  # Shape: [num_edges * num_heads]

        size = (batch_size_edges, self.num_heads, num_nodes_current * batch_size, num_nodes_neighbor * batch_size)
        verbosePrint(f'Sparse Attention Dense Shape: {size} [1 x H {self.num_heads} x N_c {num_nodes_current * batch_size} x N_n {num_nodes_neighbor * batch_size}]', self.verbose)

        verbosePrint(f'Creating torch sparse COO Tensor for attention scores', self.verbose, separator=False)
        attentionScoresSparse, sparse_indices = buildSparseTensor(rows, cols, sparse_values, size)

        verbosePrint(f'Attention scores sparse shape: {attentionScoresSparse.shape} [ {attentionScoresSparse._nnz()} non-zero entries   ]', self.verbose)
        verbosePrint(f'Applying softmax', self.verbose)
        if self.skipSoftmax:
            verbosePrint(f'\tSkipping softmax, using raw attention scores as weights', self.verbose)
            scores = sparse_values.reshape(attentionScoresSparse.shape[0] * attentionScoresSparse.shape[1], -1)
            scores  = scores.mT
            normalized_weights_ = scores.mT.reshape(attentionScoresSparse.shape[1], sparse_values.shape[0] // attentionScoresSparse.shape[1])
            # normalized_weights_ = sparse_values
        else:   
            normalized_weights_ = softmax(attentionScoresSparse, sparse_values, rows, cols, sparse_indices)
        normalized_weights = normalized_weights_.view(batch_size_edges, self.num_heads, num_edges)
        if self.attentionDropout > 0.0:
            verbosePrint(f'Applying dropout to normalized weights', self.verbose)
            normalized_weights = self.attention_dropout(normalized_weights)
        
        verbosePrint(f'Normalized weights shape: {normalized_weights.shape} [1 x H {self.num_heads} x E {num_edges}]', self.verbose)

        ################################################################################
        # Apply attention scaling if provided
        ################################################################################

        if s_k is not None:
            verboseBannerPrint(f'Applying Attention Scaling s_k', self.verbose)
            verbosePrint(f'Applying attention scaling s_k', self.verbose)
            s_k = s_k.view(batch_size_edges, 1, num_edges)  # Shape: [1, 1, num_edges]
            normalized_weights = normalized_weights * s_k  # Shape: [1, H, num_edges]
            verbosePrint(f'Normalized weights after scaling shape: {normalized_weights.shape} [1 x H {self.num_heads} x E {num_edges}]', self.verbose)  

        if self.windowFunction and not self.windowFunctionBeforeSoftmax:
            verboseBannerPrint(f'Applying Window Function after Softmax', self.verbose)
            verbosePrint(f'\tApplying Window Function Scaling to Attention Weights', self.verbose)
          
            # Window function scaling shape: [E] -> [1, H, E]
            # First expand then repeat to match attention weights shape
            windowScaling_expanded = windowScaling.view(1, 1, -1).repeat(1, self.num_heads, 1)
            normalized_weights = normalized_weights * windowScaling_expanded
            verbosePrint(f'\tNormalized Weights after window function shape: {normalized_weights.shape} [1 {1} x H {self.num_heads} x E {num_edges}]', self.verbose)

        return normalized_weights