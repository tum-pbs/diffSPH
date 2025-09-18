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
from .basisFunctions import BasisEncoder
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
                    relativePositionBiasEncoder: bool = True,
                    relativePositionBiasMLPDict: Optional[dict] = None, 
                    relativePositionBiasDim: Optional[int] = None,

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
        self.relativePositionBiasDim = relativePositionBiasDim if relativePositionBiasDim is not None else (self.num_heads if self.relativePositionBiasAfterAttention else self.transformer_dim)
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
        if self.relativePositionBias or self.cConvMode:
            verbosePrint(f'\trelative position bias (rpb) encoding:', self.verbose, separator=True)
            if self.relativePositionBiasScaledPositions:
                verbosePrint(f'\t\tUsing scaled positions for rpb encoding', self.verbose)
            if self.relativePositionBiasMultiplicative:
                verbosePrint(f'\t\tUsing multiplicative rpb', self.verbose)
            else:
                verbosePrint(f'\t\tUsing additive rpb', self.verbose)
            if self.relativePositionBiasBaseEncoding:
                verbosePrint(f'\t\tUsing basis function encoding for rpb', self.verbose)
                self.rpbBasisEncoder = BasisEncoder(
                    basisTerms=self.relativePositionBiasBaseTerms,
                    basisFunction=self.relativePositionBiasBaseFunction,
                    mode=self.relativePositionBiasBaseMode,
                    dim=self.edge_dim,
                )
                self.rpbBasisShape = self.rpbBasisEncoder.outputShape
                verbosePrint(f'\t\trpb basis function encoding output shape: {self.rpbBasisShape}', self.verbose)
                self.rpbInputDim = self.rpbBasisShape
            else:
                verbosePrint(f'\t\tUsing raw positions for rpb', self.verbose)
                self.rpbInputDim = (self.edge_dim,)
            # Input Dim might be multi dimensional, needs to be flattened before running through the input Encoder
            self.rpbFlatInputDim = self.rpbInputDim
            if len(self.rpbInputDim) > 0:
                self.rpbFlatInputDim = 1
                for s in self.rpbInputDim:
                    self.rpbFlatInputDim *= s
            if relativePositionBiasEncoder:
                if self.relativePositionBiasLinear:
                    verbosePrint(f'\t\tUsing linear layer for rpb encoding', self.verbose)
                    self.rpbEncoder = nn.Linear(self.rpbFlatInputDim, self.relativePositionBiasDim, bias=False)
                else:
                    verbosePrint(f'\t\tUsing MLP for rpb encoding', self.verbose)
                    if self.relativePositionBiasMLPDict is None:
                        raise ValueError('relativePositionBiasMLPDict must be provided if relativePositionBiasLinear is False')
                    if self.relativePositionBiasMLPDict is not None:
                        self.rpbEncoder = buildMLPwDict({
                            'inputFeatures': self.rpbFlatInputDim,
                            'output': self.relativePositionBiasDim,
                            **self.relativePositionBiasMLPDict
                        }, verbose = verbose, verbosePrefix='\t\t')
                    else:
                        self.rpbEncoder = buildMLPwDict({
                            'inputFeatures': self.rpbFlatInputDim,
                            'output': self.relativePositionBiasDim,
                        }, verbose = verbose, verbosePrefix='\t\t')
                    numberOfParameters = sum(p.numel() for p in self.rpbEncoder.parameters())
                    verbosePrint(f'\t\tNumber of parameters in rpb encoder MLP: {numberOfParameters}', self.verbose)
            else:
                verbosePrint(f'\t\tNot using any encoder for rpb, using raw (or basis encoded) positions directly', self.verbose)
                self.relativePositionBiasDim = self.rpbFlatInputDim

            rpbDim = self.relativePositionBiasDim #if (self.relativePositionBias and not self.relativePositionBiasAfterAttention) else 0
            if self.relativePositionBiasEncoder:
                rpbDim = rpbDim // self.num_heads
        else:
            rpbDim = 0

        ################################################################################
        # Build CConv Mode
        ################################################################################
        if self.cConvMode:
            verbosePrint(f'Building continuous Convolution Mode (cConv)...', verbose, separator=True)

            verbosePrint(f'Input Basis Shape: {rpbDim}', verbose)
            verbosePrint(f'Input Key/Query Shape: {self.latent_dim}', verbose)
            verbosePrint(f'Number of Heads: {self.num_heads} / Features per Head: {self.transformer_features // self.num_heads}', verbose)

            if self.encodeTokensShared:
                W_wqk = torch.nn.Linear(rpbDim, self.latent_dim * self.transformer_features * self.num_heads, bias=False)
                self.W_wq = self.W_wk = W_wqk
            else:
                self.W_wq = torch.nn.Linear(rpbDim, self.latent_dim * self.transformer_features * self.num_heads, bias=False)
                self.W_wk = torch.nn.Linear(rpbDim, self.latent_dim * self.transformer_features * self.num_heads, bias=False)

        ################################################################################
        #                     Build Attention Score Mechanism                          #
        ################################################################################

        verbosePrint(f'Building Attention Score Mechanism...', verbose, separator=True)
        attentionInputDim = self.transformer_features #if self.encodeTokens else self.latent_dim
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

            self.attentionScoreMLPDict['inputFeatures'] = 2 * attentionInputDim + rpbDim
            self.attentionScoreMLPDict['output'] = 1
            verbosePrint(f'\t\tShape: {self.attentionScoreMLPDict["inputFeatures"]} -> {self.attentionScoreMLPDict["output"]}', verbose)
            self.attentionScoreMLP = buildMLPwDict(self.attentionScoreMLPDict, verbose, verbosePrefix='\t\t')
        elif self.attentionMechanism == 'linear':
            verbosePrint(f'\tUsing linear attention', verbose)
            attentionInputShape = 2 * attentionInputDim + rpbDim
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
            if self.relativePositionBiasBaseEncoding:
                rpbEncoded = self.rpbBasisEncoder(edge_attr)
                verbosePrint(f'\t\tRPB basis function encoding output shape: {rpbEncoded.shape}', self.verbose)
            else:
                rpbEncoded = edge_attr
                verbosePrint(f'\t\tUsing raw positions for RPB, shape: {rpbEncoded.shape}', self.verbose)
            rpbEncoded = rpbEncoded.view(-1, self.rpbFlatInputDim)
            verbosePrint(f'\t\tRPB flattened input shape: {rpbEncoded.shape}', self.verbose)
            if self.relativePositionBiasEncoder:    
                rpbFeatures = self.rpbEncoder(rpbEncoded)
            else:
                rpbFeatures = rpbEncoded
            verbosePrint(f'\t\tRPB encoded features shape: {rpbFeatures.shape} [E, H * T]', self.verbose)
            if self.relativePositionBiasAfterAttention:
                rpbFeatures = rpbFeatures.view(1, -1, self.num_heads)
                verbosePrint(f'\t\tRPB reshaped features shape: {rpbFeatures.shape} [1, E, H]', self.verbose)
            else:
                if self.relativePositionBiasEncoder:
                    rpbFeatures = rpbFeatures.view(1, -1, self.num_heads, self.relativePositionBiasDim//self.num_heads)
                    verbosePrint(f'\t\tRPB reshaped features shape: {rpbFeatures.shape} [1, E, H, T]', self.verbose)
                else:
                    rpbFeatures = rpbFeatures.view(1, -1, 1, self.relativePositionBiasDim)
                    verbosePrint(f'\t\tRPB reshaped features shape: {rpbFeatures.shape} [1, E, 1, D]', self.verbose) # Apply the same input to all heads
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
            verbosePrint(f'Input Token Shape [RBP]: {rpbFeatures.shape} [B {batch_size} x N {num_nodes_current} x D {self.relativePositionBiasDim}]', self.verbose)

            rpb = rpbFeatures.flatten(-2).squeeze(0)
            w_q = self.W_wq(rpb)  # Shape: [num_edges, latent_dim * transformer_features * num_heads]
            w_k = self.W_wk(rpb)  # Shape: [num_edges, latent_dim * transformer_features * num_heads]
            verbosePrint(f'\tW_q shape: {w_q.shape} [E, L * T * H]', self.verbose)
            verbosePrint(f'\tW_k shape: {w_k.shape} [E, L * T * H]', self.verbose)

            w_q = w_q.view(-1, self.transformer_features * self.num_heads, self.latent_dim)  # Shape: [num_edges, H * T, L]
            w_k = w_k.view(-1, self.transformer_features * self.num_heads, self.latent_dim)  # Shape: [num_edges, H * T, L]

            node_i = queryTokens[0, rows, :]  # Shape: [num_edges, latent_dim]
            node_j = keyTokens[0, cols, :]

            verbosePrint(f'\tNode i shape: {node_i.shape} [E, L]', self.verbose)
            verbosePrint(f'\tNode j shape: {node_j.shape} [E, L]', self.verbose)

            verbosePrint(f'\tW_q reshaped shape: {w_q.shape} [E, H * T, L]', self.verbose)
            verbosePrint(f'\tW_k reshaped shape: {w_k.shape} [E, H * T, L]', self.verbose)


            Q_i = torch.einsum('el, efl -> ef', node_i, w_q).view(batch_size, -1, self.num_heads, self.transformer_features)  # Shape: [B, E, H, F]
            K_j = torch.einsum('el, efl -> ef', node_j, w_k).view(batch_size, -1, self.num_heads, self.transformer_features)  # Shape: [B, E, H, F]


        ################################################################################
        #                        Compute Attention Scores                              #
        ################################################################################
        verboseBannerPrint(f'Computing Attention Scores', self.verbose)
        verbosePrint(f'Collected Query Tokens: {Q_i.shape} [H {self.num_heads} x E {num_edges} x F {self.transformer_features}]', self.verbose)
        verbosePrint(f'Collected Key Tokens:   {K_j.shape} [H {self.num_heads} x E {num_edges} x F {self.transformer_features}]', self.verbose)

        verbosePrint(f'Computing Attention', self.verbose)

        if self.attentionMechanism == 'dot':
            if not self.relativePositionBiasAfterAttention and self.relativePositionBias:
                raise NotImplementedError('Relative Position Bias before attention not possible for dot product attention')

            attentionScores = (Q_i * K_j).sum(dim=-1)  # Shape: [H, num_edges]
            verbosePrint(f'\tDot Product Attention Scores shape: {attentionScores.shape} [H {self.num_heads} x E {num_edges}]', self.verbose)
        elif self.attentionMechanism == 'scaled_dot':
            if not self.relativePositionBiasAfterAttention and self.relativePositionBias:
                raise NotImplementedError('Relative Position Bias before attention not possible for dot product attention')

            attentionScoresProduct = (Q_i * K_j)  # Shape: [H, num_edges]
            attentionScores = self.attentionScaling(attentionScoresProduct).sum(-1)  # Shape: [H, num_edges]
            verbosePrint(f'\tScaled Dot Product Attention Scores shape: {attentionScores.shape} [H {self.num_heads} x E {num_edges}]', self.verbose)
        elif self.attentionMechanism == 'mlp':
            verbosePrint(f'\tMLP Attention Input shape: {Q_i.shape} [H {self.num_heads} x E {num_edges} x F {self.transformer_features}]', self.verbose)
            attentionInput = torch.cat([Q_i, K_j], dim=-1)  # Shape: [H, num_edges, 2*F]
            if self.relativePositionBias and not self.relativePositionBiasAfterAttention:
                verbosePrint(f'\tAdding Relative Position Bias before Attention', self.verbose)

                # if relativePositionBiasEncoder is set to false, the inputs are not encoded per head, so we need to repeat them for each head
                if not self.relativePositionBiasEncoder:
                    rpbFeatures = rpbFeatures.repeat(1, 1, self.num_heads, 1)  # Shape: [1, E, H, D_rpb]
                rpbFeatures = rpbFeatures.squeeze(0)
                verbosePrint(f'\t\tRPB Features shape: {rpbFeatures.shape} [E, H, T]', self.verbose)
                attentionInput = torch.cat([attentionInput, rpbFeatures.permute(1,0,2)], dim=-1)  # Shape: [H, num_edges, 2*F + D_rpb]

            verbosePrint(f'\tAttention Input shape: {attentionInput.shape} [H {self.num_heads} x E {num_edges} x (2*F + D_rpb)]', self.verbose)

            attentionScores = self.attentionScoreMLP(attentionInput)  # Shape: [H, num_edges]
            verbosePrint(f'\tMLP Attention Scores shape: {attentionScores.shape} [H {self.num_heads} x E {num_edges}]', self.verbose)
        elif self.attentionMechanism == 'linear':
            verbosePrint(f'\tLinear Attention Input shape: {Q_i.shape} [H {self.num_heads} x E {num_edges} x F {self.transformer_features}]', self.verbose)
            attentionInput = torch.cat([Q_i, K_j], dim=-1)  # Shape: [H, num_edges, 2*F]
            if self.relativePositionBias and not self.relativePositionBiasAfterAttention:
                verbosePrint(f'\tAdding Relative Position Bias before Attention', self.verbose)


                # if relativePositionBiasEncoder is set to false, the inputs are not encoded per head, so we need to repeat them for each head
                if not self.relativePositionBiasEncoder:
                    rpbFeatures = rpbFeatures.repeat(1, 1, self.num_heads, 1)  # Shape: [1, E, H, D_rpb]
                rpbFeatures = rpbFeatures.squeeze(0)
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