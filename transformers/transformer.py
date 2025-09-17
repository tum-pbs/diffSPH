import torch
from ml import getActivationLayer
import torch.nn as nn

try:
    import torch_geometric
except ImportError:
    torch_geometric = None

from typing import Optional

from torch import Tensor

from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes


def softmax_(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
) -> Tensor:
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        count = ptr[1:] - ptr[:-1]
        ptr = ptr.view(size)
        src_max = segment(src.detach(), ptr, reduce='max')
        src_max = src_max.repeat_interleave(count, dim=dim)
        out = (src - src_max).exp()
        out_sum = segment(out, ptr, reduce='sum') + 1e-16
        out_sum = out_sum.repeat_interleave(count, dim=dim)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
        # print(f'Scatter max shape: {src_max.shape} [ {src_max.numel()} elements ]')
        # print(src_max)
        out = src - src_max.index_select(dim, index)
        out = out.exp()
        # print(f'Out shape: {out.shape} [ {out.numel()} elements ]')
        # print(out)
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
        out_sum = out_sum.index_select(dim, index)
        # print(f'Out sum shape: {out_sum.shape} [ {out_sum.numel()} elements ]')
        # print(out_sum)
    else:
        raise NotImplementedError

    return out / out_sum


def softmax(attentionScoresSparse, sparse_values, rows, cols, sparse_indices): #batch_size, multiHeads, num_edges, sparse_values, cols, num_nodes_neighbor):
    if torch_geometric is not None:
        # torch_geometric expects the input to be a 1D tensor of scores and an index tensor for grouping
        # Our manual implementation returns a flat tensor of shape [batch_size * multiHeads, num_edges]
        # We need to flatten scores and provide the correct index for grouping by destination node (cols) for each head in each batch
        scores = sparse_values.reshape(attentionScoresSparse.shape[0] * attentionScoresSparse.shape[1], -1)
        # Torch Geometric requires the edges to be the first dimension
        scores = scores.mT
        softmaxxed = softmax_(scores, index=rows)
        softmaxxed = softmaxxed.mT.reshape(attentionScoresSparse.shape[1], sparse_values.shape[0] // attentionScoresSparse.shape[1])
        return softmaxxed

    batch_size, multiHeads, num_nodes_neighbor, num_nodes_current = attentionScoresSparse.shape
    # sparse_values = attentionScoresSparse.values()
    # cols = sparse_indices[1]  # Assuming the second index is the column indices
    num_edges = sparse_values.shape[0]

    # if self.verbose:
        # print('attentionScores shape:', attentionScores.shape)
    # attention_weights_sparse = torch.sparse.softmax(attentionScoresSparse, dim=2)

    # Let's work with a flattened view for easier scattering
    # Shape: [B * H, num_edges]
    scores = sparse_values.reshape(batch_size * multiHeads, -1)

    # The 'rows' index needs to be broadcast to match the new shape
    # It will guide the grouping for every head in every batch item
    index = rows.expand_as(scores)

    # 1. Subtract max for numerical stability (a standard softmax trick)
    # We need to find the max score for each destination node group
    # 'scatter_max' is not native, so we use a dense intermediate for this step.
    # It's an acceptable tradeoff as it's not on a huge tensor.
    alpha_max = torch.zeros(batch_size * multiHeads, num_nodes_neighbor, device=scores.device).scatter_reduce_(
        1, index, scores, reduce="amax", include_self=False
    )
    # Now gather the max value for each edge
    scores_sub = scores - alpha_max.gather(1, index)
    # 2. Exponentiate the scores
    exp_scores = torch.exp(scores_sub)
    # 3. Sum the exponentiated scores for each destination node group
    # This is the denominator of the softmax
    exp_sum = torch.zeros(batch_size * multiHeads, num_nodes_neighbor, device=scores.device).scatter_add_(
        1, index, exp_scores
    )
    # Add a small epsilon to prevent division by zero
    exp_sum = exp_sum + 1e-10

    # 4. Divide each score by its group's sum to get the final weights
    normalized_weights_flat = exp_scores / exp_sum.gather(1, index)
    return normalized_weights_flat

def buildSparseTensor(rows, cols, sparse_values, size):
    # size = (batch_size_edges, self.multiHeads, num_nodes_current * batch_size, num_nodes_neighbor * batch_size)
    num_edges = rows.shape[0]
    multiHeads = size[1]
    # 1. Create indices for the batch dimension (b)
    # Each of the H*num_edges scores in a batch item gets the same batch index
    b_idx = torch.arange(1, device=sparse_values.device).repeat_interleave(multiHeads * num_edges)

    # 2. Create indices for the head dimension (h)
    # Within each batch item, the indices 0..H-1 repeat for each edge
    h_idx = torch.arange(multiHeads, device=sparse_values.device).repeat_interleave(num_edges).repeat(1)

    # 3. Repeat the edge indices for each batch and head
    # Destination nodes (i)
    i_idx = rows.repeat(1 * multiHeads)
    # Source nodes (j)
    j_idx = cols.repeat(1 * multiHeads)

    # 4. Stack them all together to create the final sparse indices
    # Shape will be [4, B * H * num_edges]
    sparse_indices = torch.stack([b_idx, h_idx, i_idx, j_idx], dim=0)

    # Create the sparse tensor of raw scores
    return torch.sparse_coo_tensor(
        indices=sparse_indices,
        values=sparse_values,
        size=size
    ), sparse_indices
    
    
from typing import Union, Tuple
def verbosePrint(message: str, verbose: bool, separator = False):
    if verbose:
        if separator:
            print(f'===============================================================')
        print(message)

import warnings

from ml import evalBasisFunction
@torch.jit.script
def basisEncoderLayer(edgeLengths, basisTerms : int, basisFunction : str = 'ffourier', mode : str = 'cat'):
    bTerms = []
    for e in edgeLengths.T:
        bTerm = evalBasisFunction(basisTerms, e, basisFunction).mT
        bTerms.append(bTerm)
    if mode == 'cat':
        return torch.cat(bTerms, dim = 1)
    elif mode == 'sum':
        return torch.stack(bTerms, dim = 0).sum(dim = 0)
    elif mode == 'prod':
        return torch.stack(bTerms, dim = 0).prod(dim = 0)
    elif mode == 'outer':
        return torch.einsum('ij,ik->ijk', bTerms[0], bTerms[1]).reshape(-1, basisTerms * basisTerms)
    elif mode == 'i':
        return bTerms[0]
    elif mode == 'j':
        return bTerms[1]
    elif mode == 'k':
        return bTerms[2]
    else:
        raise ValueError(f'Unknown mode: {mode}')

class TransformerLayer(torch.nn.Module):
    def __init__(self, input_dim, transformer_features, edgeFeatureSize, multi_heads,
                 edge_bias=False, edge_gating=False,
                 additive_bias=True, verbose=False,
                 activation='celu', ffnHiddenLayers=1, ffnHiddenSize=0,
                 attentionOp='dot', attentionOpIncludeEdge=False,
                 sharedWeights=False, scaleQ=True,
                 dropoutActive=False, dropoutRate = 0.1, 
                 headAggregation='concat', clipAttention=True,
                 v2=False, attentionActivation = nn.LeakyReLU(0.2),
                 messagePassingGAT=False,
                 edgeBasisEncoder = False, edgeBasisTerms = 8, edgeBasisMode = 'cat', 
                 shepardAttention = False):
        super(TransformerLayer, self).__init__()
        
        self.multiHeads = multi_heads
        self.transformerFeatures = transformer_features
        self.edgeFeatureSize = edgeFeatureSize
        self.scaleQ = scaleQ

        self.edgeBias = edge_bias
        self.edgeGating = edge_gating
        self.additiveBias = additive_bias
        self.verbose = verbose

        self.attentionOp = attentionOp  # 'dot' or 'GAT' or 'MLP'
        self.attentionOpIncludeEdge = attentionOpIncludeEdge
        self.hiddenLayers = ffnHiddenLayers
        self.ffnHiddenSize = ffnHiddenSize if ffnHiddenSize > 0 else input_dim * 4
        self.useDropout = dropoutActive
        self.dropoutRate = dropoutRate
        self.clipAttention = clipAttention
        self.multiHeadAggregation = headAggregation
        self.gatV2 = v2
        self.messagePassingGAT = messagePassingGAT
        self.edgeBasisEncoder = edgeBasisEncoder
        self.edgeBasisTerms = edgeBasisTerms
        self.edgeBasisMode = edgeBasisMode

        self.shepardAttention = shepardAttention

        verbosePrint(f'Initializing TransformerLayer with input_dim={input_dim}, transformer_features={transformer_features}, edgeFeatureSize={edgeFeatureSize}, multi_heads={multi_heads}, edge_bias={edge_bias}, edge_gating={edge_gating}, additive_bias={additive_bias}', verbose)
        verbosePrint(f'Building linear projections for Q, K, V', verbose, separator=True)

        if edgeBasisEncoder:
            terms = edgeBasisTerms
            mode = edgeBasisMode
            edge_dimensioniality = 0
            if mode == 'cat':
                edge_dimensioniality = terms * edgeFeatureSize
            elif mode == 'sum' or mode == 'prod':
                edge_dimensioniality = terms
            elif mode == 'outer':
                edge_dimensioniality = int(terms ** edgeFeatureSize)
            elif mode == 'i' or mode == 'j' or mode == 'k':
                edge_dimensioniality = terms
            else:
                raise ValueError(f'Unknown mode: {mode}')
        else:
            edge_dimensioniality = edgeFeatureSize
        verbosePrint(f'Edge feature dimensionality: {edge_dimensioniality}', verbose)
            

        if not self.gatV2:
            if not sharedWeights:
                self.W_Q = torch.nn.Linear(input_dim, transformer_features * multi_heads, bias=False)
                self.W_K = torch.nn.Linear(input_dim, transformer_features * multi_heads, bias=False)
                self.W_V = torch.nn.Linear(input_dim, transformer_features * multi_heads, bias=False)
            else:
                self.W_Q = self.W_K = self.W_V = torch.nn.Linear(input_dim, transformer_features * multi_heads, bias=False)
            verbosePrint(f'W_Q shape: {self.W_Q.weight.shape}, W_K shape: {self.W_K.weight.shape}, W_V shape: {self.W_V.weight.shape}', verbose)
        else:
            self.W_QK = torch.nn.Linear(input_dim * 2, multi_heads, bias=False)
            self.W_V = torch.nn.Linear(input_dim, transformer_features * multi_heads, bias=False)
            if sharedWeights:
                warnings.warn("sharedWeights is True but does not take effect in GATv2 mode.", UserWarning)
            verbosePrint(f'W_QK shape: {self.W_QK.weight.shape}, W_V shape: {self.W_V.weight.shape}', verbose)
        
        
        verbosePrint(f'Building edge bias and gating projections', verbose, separator=True)
        if edge_bias:
            self.W_E = torch.nn.Linear(edge_dimensioniality, multi_heads)
            verbosePrint(f'\tUsing edge bias with W_E shape: {self.W_E.weight.shape}', verbose)
        else:
            self.W_E = None            
        # Edge gating is optional and can be used to gate the value matrix with edge features
        if edge_gating:
            self.W_E_gate = torch.nn.Linear(edge_dimensioniality, multi_heads * transformer_features)
            verbosePrint(f'\tUsing edge gating with W_E_gate shape: {self.W_E_gate.weight.shape}', verbose)
        else:
            self.W_E_gate = None

        if self.messagePassingGAT:
            in_dim = transformer_features + edge_dimensioniality + 1 # 1 for the attention
            layers = []
            hiddenSize = ffnHiddenSize if ffnHiddenSize > 0 else transformer_features * 4
            for i in range(ffnHiddenLayers):
                layers.append(torch.nn.Linear(in_dim, hiddenSize))
                layers.append(getActivationLayer(activation))
                in_dim = hiddenSize
            layers.append(torch.nn.Linear(in_dim, transformer_features))
            self.messagePassing = torch.nn.Sequential(*layers)

        verbosePrint(f'Building output projection steps', verbose, separator=True)
        self.activation = activation if isinstance(activation, torch.nn.Module) else getActivationLayer(activation)
        if self.multiHeadAggregation == 'mean':
            self.W_O = torch.nn.Linear(transformer_features, input_dim, bias=False)        
        else:
            self.W_O = torch.nn.Linear(transformer_features * multi_heads, input_dim, bias=False)
        self.layer_norm1 = torch.nn.LayerNorm(input_dim)
        self.layer_norm2 = torch.nn.LayerNorm(input_dim)
        verbosePrint(f'W_O shape: {self.W_O.weight.shape}, layer_norm1 shape: {self.layer_norm1.weight.shape}, layer_norm2 shape: {self.layer_norm2.weight.shape}', verbose)
        verbosePrint(f'Activation Function: {self.activation}', verbose)
        
        verbosePrint(f'Building Feedforward Network (FFN) with {self.hiddenLayers} hidden layers and hidden size {self.ffnHiddenSize}', verbose)        
        layers = []
        in_dim = input_dim
        for i in range(self.hiddenLayers):
            layers.append(torch.nn.Linear(in_dim, self.ffnHiddenSize))
            layers.append(self.activation)
            in_dim = self.ffnHiddenSize
        layers.append(torch.nn.Linear(in_dim, input_dim))
        self.ffn = torch.nn.Sequential(*layers)
        verbosePrint(f'FFN: {self.ffn}', verbose)

        verbosePrint(f'Building Attention Mechanism', verbose, separator=True)
        self.attentionActivation = attentionActivation

        inputSize = 2 * transformer_features
        if self.attentionOpIncludeEdge:
            inputSize += edge_dimensioniality  # Include edge features in GAT attention

        if self.attentionOp == 'GAT':
            verbosePrint(f'\tUsing GAT attention with input size: {inputSize}', verbose)
            self.W_a = torch.nn.Linear(inputSize, 1, bias=False)            
        elif self.attentionOp == 'MLP':
            verbosePrint(f'\tUsing MLP attention with input size: {inputSize}', verbose)
            hiddenSize = transformer_features * multi_heads
            hiddenLayers = ffnHiddenLayers
            layers = []
            in_dim = inputSize
            for i in range(hiddenLayers):
                layers.append(torch.nn.Linear(in_dim, hiddenSize))
                layers.append(self.activation)
                in_dim = hiddenSize
            layers.append(torch.nn.Linear(in_dim, 1))
            self.W_a = torch.nn.Sequential(*layers)
        else:
            verbosePrint(f'\tUsing dot attention', verbose)
            self.W_a = None  # For 'dot' attention, no additional weights are needed
            
        if self.useDropout:
            self.dropout = torch.nn.Dropout(self.dropoutRate)
            self.attention_dropout = torch.nn.Dropout(self.dropoutRate)

        verbosePrint(f'TransformerLayer initialized with input_dim={input_dim}, transformer_features={transformer_features}, edgeFeatureSize={edge_dimensioniality}, multi_heads={multi_heads}, edge_bias={edge_bias}, edge_gating={edge_gating}, additive_bias={additive_bias}', verbose)

    def forward(self, inputTokens_: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], inputEdges_, edgeIndices, shepardValues = None):
        if isinstance(inputTokens_, tuple):
            inputTokensCurrent, inputTokensNeighbor = inputTokens_
        else:
            inputTokensCurrent = inputTokensNeighbor = inputTokens_  # No row information if not provided
            verbosePrint(f'\tNo row information provided for input tokens', self.verbose)
            
        batch_size, num_nodes_current, latentSpaceSize = inputTokensCurrent.shape
        batch_size_edges = 1
        num_nodes_neighbor = inputTokensNeighbor.shape[1]
        num_edges = edgeIndices.shape[1]
        rows = edgeIndices[0]
        cols = edgeIndices[1]
        
        verbosePrint(f'Input tokens shape: {inputTokensCurrent.shape}', self.verbose, separator=True)
        if inputEdges_ is not None:
            verbosePrint(f'\tInput edges shape: {inputEdges_.shape}', self.verbose)
        else:
            verbosePrint(f'\tNo edge features provided', self.verbose)
        verbosePrint(f'Edge indices shape: {edgeIndices.shape}', self.verbose)

        if self.edgeBasisEncoder:
            inputEdges = basisEncoderLayer(inputEdges_, self.edgeBasisTerms, mode = self.edgeBasisMode)
            verbosePrint(f'Edge features encoded with basis terms: {self.edgeBasisTerms}, mode: {self.edgeBasisMode}', self.verbose )
            verbosePrint(f'Encoded edge features shape: {inputEdges.shape}', self.verbose)
        else:
            inputEdges = inputEdges_

        # Preamble done, Projection into Query, Key, Value next

        verbosePrint(f'Projection Step', self.verbose, separator=True)

        if not self.gatV2:
            Q = self.W_Q(inputTokensCurrent)
            K = self.W_K(inputTokensNeighbor)

            verbosePrint(f'Input Token Shape [current ]: {inputTokensCurrent.shape} [B {batch_size} x N {num_nodes_current} x D {latentSpaceSize}]', self.verbose)
            verbosePrint(f'Input Token Shape [neighbor]: {inputTokensNeighbor.shape} [B {batch_size} x N {num_nodes_neighbor} x D {latentSpaceSize}]', self.verbose)

            Q = Q.view(batch_size, Q.shape[1], self.multiHeads, self.transformerFeatures).permute(0, 2, 1, 3)
            K = K.view(batch_size, K.shape[1], self.multiHeads, self.transformerFeatures).permute(0, 2, 1, 3)

            verbosePrint(f'Query Shape: {Q.shape} [B {batch_size} x H {self.multiHeads} x N {Q.shape[2]} x D {self.transformerFeatures}]', self.verbose)
            verbosePrint(f'Key Shape:   {K.shape} [B {batch_size} x H {self.multiHeads} x N {K.shape[2]} x D {self.transformerFeatures}]', self.verbose)

            # Scale Q by sqrt(d_k) This is a common practice in Transformer architectures to stabilize training
            if self.scaleQ:
                Q = Q / (self.transformerFeatures ** 0.5)  # Scale by sqrt(d_k)
                verbosePrint(f'\tScaled Query Shape Tokens by 1/{self.transformerFeatures} ** 0.5', self.verbose)

            # Q has shape [batch_size, multiHeads, num_nodes_current, transformerFeatures]
            # K has shape [batch_size, multiHeads, num_nodes_neighbor, transformerFeatures]
            # because of the sparse neighborhood we need to unify the batch size and num_nodes entries
            # Q_i will then have shape [multiHeads, num_edges, transformerFeatures]
            # K_j will have shape [multiHeads, num_edges, transformerFeatures]
            # where num_edges is the number of edges in the sparse neighborhood

            # Q_unified will have shape [multiHeads, batch_size * num_nodes_current, transformerFeatures]
            # K_unified will have shape [multiHeads, batch_size * num_nodes_neighbor, transformerFeatures]
            Q_unified = Q.permute(1, 0, 2, 3).reshape(self.multiHeads, batch_size * num_nodes_current, self.transformerFeatures)
            K_unified = K.permute(1, 0, 2, 3).reshape(self.multiHeads, batch_size * num_nodes_neighbor, self.transformerFeatures)

            verbosePrint(f'Unified Query Shape: {Q_unified.shape} [H {self.multiHeads} x B {batch_size} * N {num_nodes_current} x D {self.transformerFeatures}]', self.verbose, separator=True)
            verbosePrint(f'Unified Key Shape:   {K_unified.shape} [H {self.multiHeads} x B {batch_size} * N {num_nodes_neighbor} x D {self.transformerFeatures}]', self.verbose)

            Q_i = Q_unified[:, rows, :] # Shape: [B, H, num_edges, F]
            K_j = K_unified[:, cols, :] # Shape: [B, H, num_edges, F]

            verbosePrint(f'Collected Query Tokens: {Q_i.shape} [H {self.multiHeads} x E {num_edges} x F {self.transformerFeatures}]', self.verbose)
            verbosePrint(f'Collected Key Tokens:   {K_j.shape} [H {self.multiHeads} x E {num_edges} x F {self.transformerFeatures}]', self.verbose)

            verbosePrint(f'Computing Attention', self.verbose)

            # Normal dot product attention scores
            if self.attentionOp == 'dot':
                verbosePrint(f'\tUsing dot product attention scores', self.verbose)
                sparseAttentionValues = (Q_i * K_j).sum(dim=-1)
            elif self.attentionOp == 'GAT':
                verbosePrint(f'\tUsing GAT attention scores', self.verbose)
                # GAT attention scores: W_a(Q_i || K_j)
                # Concatenate Q_i and K_j along the last dimension
                if not self.attentionOpIncludeEdge:
                    verbosePrint(f'\t\tConcatenating Q_i and K_j without edge features', self.verbose)
                    combined = torch.cat([Q_i, K_j], dim=-1)  # Shape: [B, H, num_edges, 2*F]
                else:
                    verbosePrint(f'\t\tConcatenating Q_i and K_j with edge features', self.verbose)
                    # input edges has shape [num_edges, edgeFeatureSize]
                    # We need to expand it to match the batch size and multi-heads
                    verbosePrint(f'\t\tExpanding input edges to match batch size and multi-heads', self.verbose)
                    verbosePrint(f'\t\tInput edges shape: {inputEdges.shape} [E {num_edges} x FE {self.edgeFeatureSize}]', self.verbose )
                    
                    expanded_inputEdges = inputEdges.view(batch_size_edges, 1, Q_i.shape[1], -1).expand(batch_size_edges, self.multiHeads, -1, -1)  # Shape: [B, H, edgeFeatureSize]
                    expanded_inputEdges  = expanded_inputEdges.view(self.multiHeads, num_edges, -1)

                    verbosePrint(f'\t\tExpanded input edges [accounting for H]: {expanded_inputEdges.shape} [1 x H {self.multiHeads} x E {num_edges} x FE {self.edgeFeatureSize}]', self.verbose)
                    combined = torch.cat([Q_i, K_j, expanded_inputEdges], dim=-1)  # Shape: [B, H, num_edges, 2*F + edgeFeatureSize]
                verbosePrint(f'\tCombined shape for attention scores: {combined.shape} [1 {batch_size_edges} x H {self.multiHeads} x E {num_edges} x (2*F + ?FE)]', self.verbose)

                verbosePrint(f'\tProjecting attention scores', self.verbose)
                sparseAttentionValues = self.W_a(combined)  # Shape: [B, H, num_edges, 1]
                sparseAttentionValues = self.attentionActivation(sparseAttentionValues).squeeze(-1)  # Shape: [B, H, num_edges]
                
            elif self.attentionOp == 'MLP':
                verbosePrint(f'\tUsing MLP attention scores', self.verbose)
                if not self.attentionOpIncludeEdge:
                    combined = torch.cat([Q_i, K_j], dim=-1)  # Shape: [B, H, num_edges, 2*F]
                else:
                    expanded_inputEdges = inputEdges.view(batch_size_edges, 1, Q_i.shape[1], -1).expand(batch_size_edges, self.multiHeads, -1, -1)  # Shape: [B, H, edgeFeatureSize]
                    expanded_inputEdges  = expanded_inputEdges.view(self.multiHeads, num_edges, -1)
                    verbosePrint(f'\t\tExpanded input edges [accounting for H]: {expanded_inputEdges.shape} [1 x H {self.multiHeads} x E {num_edges} x FE {self.edgeFeatureSize}]', self.verbose)
                    combined = torch.cat([Q_i, K_j, expanded_inputEdges], dim=-1)  # Shape: [B, H, num_edges, 2*F + edgeFeatureSize]

                verbosePrint(f'\tCombined shape for attention scores: {combined.shape} [1 {batch_size_edges} x H {self.multiHeads} x E {num_edges} x (2*F + ?FE)]', self.verbose)                
                sparseAttentionValues = self.W_a(combined).squeeze(-1)  # Shape: [B, H, num_edges]
                sparseAttentionValues = self.attentionActivation(sparseAttentionValues)  # Apply activation
            else:
                raise ValueError(f'Unknown attention operation: {self.attentionOp}')
        elif self.gatV2:
            verbosePrint(f'Using GATv2 attention', self.verbose)
            # GATv2 uses a different approach for attention scores
            # We concatenate Q and K, then apply a linear transformation


            inputTokensCurrentFlat = inputTokensCurrent.view(batch_size * num_nodes_current, -1)
            inputTokensNeighborFlat = inputTokensNeighbor.view(batch_size * num_nodes_neighbor, -1)
            verbosePrint(f'Flattened Input Tokens Current Shape: {inputTokensCurrentFlat.shape} [B * N_c {batch_size * num_nodes_current} x D {latentSpaceSize}]', self.verbose)
            verbosePrint(f'Flattened Input Tokens Neighbor Shape: {inputTokensNeighborFlat.shape} [B * N_n {batch_size * num_nodes_neighbor} x D {latentSpaceSize}]', self.verbose)

            inputTokens_i = inputTokensCurrentFlat[rows, :]  # Shape: [num_edges, D]
            inputTokens_j = inputTokensNeighborFlat[cols, :]  # Shape: [num_edges, D]
            verbosePrint(f'Collected Input Tokens Current: {inputTokens_i.shape} [E {num_edges} x D {latentSpaceSize}]', self.verbose)
            verbosePrint(f'Collected Input Tokens Neighbor: {inputTokens_j.shape} [E {num_edges} x D {latentSpaceSize}]', self.verbose)

            # Concatenate the input tokens for Q and K
            combined_tokens = torch.cat([inputTokens_i, inputTokens_j], dim=-1)
            verbosePrint(f'Combined Input Tokens Shape: {combined_tokens.shape} [E {num_edges} x D {latentSpaceSize * 2}]', self.verbose)
            # Apply the linear transformation
            sparseAttentionValues = self.W_QK(combined_tokens)  # Shape: [num_edges, H * F]
            sparseAttentionValues = sparseAttentionValues.view(num_edges, self.multiHeads)  # Reshape to [num_edges, H, F]
            verbosePrint(f'Sparse Attention Values Shape: {sparseAttentionValues.shape} [E {num_edges} x H {self.multiHeads}]', self.verbose)
            sparseAttentionValues = self.attentionActivation(sparseAttentionValues)  # Apply activation
            sparseAttentionValues = sparseAttentionValues.view(batch_size_edges, self.multiHeads, num_edges)  # Reshape to [B, H, E]
            verbosePrint(f'Sparse Attention Values after activation: {sparseAttentionValues.shape} [1 {batch_size_edges} x H {self.multiHeads} x E {num_edges}]', self.verbose)

        verbosePrint(f'Final sparse attention values shape: {sparseAttentionValues.shape} [1 {batch_size_edges} x H {self.multiHeads} x E {num_edges}]', self.verbose)

        sparse_values = sparseAttentionValues.flatten()
        if self.clipAttention:
            sparse_values = torch.clamp(sparse_values, min = -10., max = 10.)

        size = (batch_size_edges, self.multiHeads, num_nodes_current * batch_size, num_nodes_neighbor * batch_size)
        verbosePrint(f'Sparse Attention Dense Shape: {size} [1 x H {self.multiHeads} x N_c {num_nodes_current * batch_size} x N_n {num_nodes_neighbor * batch_size}]', self.verbose)

        if self.edgeBias:
            verbosePrint(f'\tUsing edge bias for attention scores', self.verbose, separator=True)
            verbosePrint(f'\tProjecting edge bias', self.verbose)
            edge_bias = self.W_E(inputEdges).reshape(1, num_edges, self.multiHeads) # shape: [batch, num_edges, multiHeads]
            verbosePrint(f'\tEdge bias shape: {edge_bias.shape} [1 x E {num_edges} x H {self.multiHeads}]', self.verbose)
            # print('edge_bias shape:', edge_bias.shape)
            # We need to align the dimensions for broadcasting with attention scores
            edge_bias = edge_bias.permute(0, 2, 1) # shape: [batch, multiHeads, num_edges]
            
            verbosePrint(f'\tEdge bias shape after permute: {edge_bias.shape}', self.verbose)

            # Compute Edge-aware Attention Scores
            if self.additiveBias:
                verbosePrint(f'\t\tAdding edge bias to sparse values', self.verbose)
                sparse_values = sparse_values + edge_bias.flatten()
            else:
                verbosePrint(f'\t\tMultiplying sparse values with edge bias', self.verbose)
                sparse_values = sparse_values * edge_bias.flatten()

        verbosePrint(f'Creating torch sparse COO Tensor for attention scores', self.verbose, separator=True)
        attentionScoresSparse, sparse_indices = buildSparseTensor(rows, cols, sparse_values, size)

        verbosePrint(f'Attention scores sparse shape: {attentionScoresSparse.shape} [ {attentionScoresSparse._nnz()} non-zero entries   ]', self.verbose)
        verbosePrint(f'Applying softmax (manual implementation)', self.verbose)
        normalized_weights_ = softmax(attentionScoresSparse, sparse_values, rows, cols, sparse_indices)
        normalized_weights = normalized_weights_.view(batch_size_edges, self.multiHeads, num_edges)
        if self.useDropout:
            verbosePrint(f'Applying dropout to normalized weights', self.verbose)
            normalized_weights = self.attention_dropout(normalized_weights)
        
        verbosePrint(f'Normalized weights shape: {normalized_weights.shape} [1 x H {self.multiHeads} x E {num_edges}]', self.verbose)
        verbosePrint(f'Collecting Value Tokens', self.verbose, separator=True)

        V = self.W_V(inputTokensNeighbor)
        V = V.view(batch_size, V.shape[1], self.multiHeads, self.transformerFeatures).permute(0, 2, 1, 3)
        verbosePrint(f'Value Shape: {V.shape} [B {batch_size} x H {self.multiHeads} x N {V.shape[2]} x D {self.transformerFeatures}]', self.verbose)

        V_unified = V.permute(1, 0, 2, 3).reshape(self.multiHeads, batch_size * num_nodes_neighbor, self.transformerFeatures)
        verbosePrint(f'V_unified shape: {V_unified.shape} [H {self.multiHeads} x B * N {batch_size * num_nodes_neighbor} x F {self.transformerFeatures}]', self.verbose)

        V_j = V_unified[:, cols, :] # Shape: [B, H, num_edges, F]
        verbosePrint(f'Collected Value Tokens: {V_j.shape} [H {self.multiHeads} x E {num_edges} x F {self.transformerFeatures}]', self.verbose)

        verbosePrint(f'Computing Messages', self.verbose, separator=True)
        messages = V_j  # This is the message vector for each edge
        if self.edgeGating:
            verbosePrint(f'\tUsing edge gating for messages', self.verbose)
            # Project edge features to create the gate
            # W_E_gate is a nn.Linear(edge_feature_dim, self.multiHeads * self.transformerFeatures)
            verbosePrint(f'\tProjecting edge features for gating', self.verbose)
            edge_gate_values = self.W_E_gate(inputEdges) 
            edge_gate_values = torch.sigmoid(edge_gate_values)
            verbosePrint(f'\tEdge gate values shape: {edge_gate_values.shape} [num_edges {num_edges} x H {self.multiHeads} x F {self.transformerFeatures}]', self.verbose)

            # Reshape gate to be compatible with V_j for broadcasting
            edge_gate_values = edge_gate_values.view(num_edges, self.multiHeads, self.transformerFeatures)
            edge_gate_values = edge_gate_values.permute(1, 0, 2).unsqueeze(0) 
            verbosePrint(f'\tReshaped edge gate values shape: {edge_gate_values.shape}', self.verbose)

            verbosePrint(f"\tShape of V_j to be gated: {V_j.shape}", self.verbose)
            
            # Apply the gate to the V_j vectors. Broadcasting handles the batch dim.
            gated_V_j = V_j * edge_gate_values
            messages = gated_V_j # Update messages to be the gated version

        verbosePrint(f'Final messages shape: {messages.shape} [H {self.multiHeads} x E {num_edges} x F {self.transformerFeatures}]', self.verbose)
        verbosePrint(f'Attention Shape: {normalized_weights.shape} [1 x H {self.multiHeads} x E {num_edges}]', self.verbose)
        final_messages = messages * normalized_weights.unsqueeze(-1)
        # print(f'Final messages after applying attention weights shape: {final_messages.shape} [H {self.multiHeads} x E {num_edges} x F {self.transformerFeatures}]', self.verbose)
        if not self.messagePassingGAT:
            final_messages = messages * normalized_weights.unsqueeze(-1)
        else:
            verbosePrint(f'\tUsing GAT message passing', self.verbose)
            # GAT message passing requires a different approach
            # We need to concatenate the messages with the edge features and apply the MLP
            messages_ = messages.permute(1, 0, 2)
            edge_features = inputEdges.view(num_edges, -1)
            # Goal is E H F shape: [E {num_edges} x H {self.multiHeads} x F {self.edgeFeatureSize}]

            attentionValues = normalized_weights.view(self.multiHeads, -1).mT.unsqueeze(-1)  # Shape: [H, B, E]

            # Need to map from E x FE to E x H x FE
            edge_features = edge_features.unsqueeze(1).expand(-1, self.multiHeads, -1)

            verbosePrint(f'\tMessages shape before GAT message passing: {messages_.shape} [E {num_edges} x H {self.multiHeads} x F {self.transformerFeatures}]', self.verbose)
            verbosePrint(f'\tAttention values shape: {attentionValues.shape} [E {num_edges} x H {self.multiHeads}]', self.verbose)
            verbosePrint(f'\tEdge features shape: {edge_features.shape} [E {num_edges} x H {self.multiHeads} x FE {self.edgeFeatureSize}]', self.verbose)


            # print(f'\tMessages: min: {messages_.min()}, max: {messages_.max()}, mean: {messages_.mean()}, std: {messages_.std()}')
            # print(f'\tAttention Values: min: {attentionValues.min()}, max: {attentionValues.max()}, mean: {attentionValues.mean()}, std: {attentionValues.std()}')
            # print(f'\tEdge Features: min: {edge_features.min()}, max: {edge_features.max()}, mean: {edge_features.mean()}, std: {edge_features.std()}')

            # Concatenate messages, attention values, and edge features
            combined_messages = torch.cat([messages_ * 0, attentionValues * 0, edge_features], dim=-1)



            # print(f'\tCombined messages shape: {combined_messages.shape} [H {self.multiHeads} x E {num_edges} x (F {self.transformerFeatures} + ?FE {self.edgeFeatureSize} + 1)]', self.verbose)
            # print(f'\tCombined messages: min: {combined_messages.min()}, max: {combined_messages.max()}, mean: {combined_messages.mean()}, std: {combined_messages.std()}')

            final_messages = self.messagePassing(combined_messages).permute(1, 0, 2)  # Apply the MLP



            verbosePrint(f'\tFinal messages shape after GAT message passing: {final_messages.shape} [H {self.multiHeads} x E {num_edges} x F {self.transformerFeatures}]', self.verbose)
            # print(f'Final messages: min: {final_messages.min()}, max: {final_messages.max()}, mean: {final_messages.mean()}, std: {final_messages.std()}')
        # print(f'Attention Weights: [{normalized_weights.shape}]', normalized_weights)
        # print(f'Messages [{messages.shape}]: ', messages)
        # print(f'Final Messages [{final_messages.shape}]: ', final_messages)
        # print(final_messages)

        verbosePrint(f'Final messages after applying attention weights shape: {final_messages.shape} [H {self.multiHeads} x E {num_edges} x F {self.transformerFeatures}]', self.verbose)
        message_values = final_messages.reshape(-1, self.transformerFeatures)
        verbosePrint(f'Message values shape: {message_values.shape} [B * H * E {batch_size_edges * self.multiHeads * num_edges} x F {self.transformerFeatures}]', self.verbose)

        verbosePrint(f'Summing Messages Step', self.verbose, separator=True)
        
        if torch_geometric is not None:
            verbosePrint(f'Using PyTorch Geometric for message aggregation', self.verbose)
            messages_transposed = final_messages.view(self.multiHeads, num_edges, self.transformerFeatures).permute(1, 0, 2)  # Shape: [E, H, F]
            aggregated_messages_sparse_geometric = torch_geometric.utils.scatter(
                messages_transposed, rows, dim=0, dim_size=batch_size * num_nodes_current, reduce='sum'
            )
            aggregated_messages_sparse = aggregated_messages_sparse_geometric.transpose(0, 1).reshape(batch_size_edges, self.multiHeads, num_nodes_current * batch_size, self.transformerFeatures)
        else:
            verbosePrint(f'Using manual sparse tensor aggregation', self.verbose)
            # The full size of the hybrid tensor: sparse part + dense part
            hybrid_size = (batch_size_edges, self.multiHeads, num_nodes_current * batch_size, num_nodes_neighbor * batch_size, self.transformerFeatures)
            # Create the hybrid sparse tensor representing messages
            sparse_message_tensor = torch.sparse_coo_tensor(indices=sparse_indices, values=message_values, size=hybrid_size)
            verbosePrint(f'Sparse message tensor shape: {sparse_message_tensor.shape} [1 x H {self.multiHeads} x N_curr {num_nodes_current * batch_size} x N_neigh {num_nodes_neighbor * batch_size} x F {self.transformerFeatures}]', self.verbose)

            # Sum over the source dimension 'j' (dim=3) to aggregate messages
            aggregated_messages_sparse = torch.sparse.sum(sparse_message_tensor, dim=3)
            verbosePrint(f'Aggregated messages shape: {aggregated_messages_sparse.shape} [1 x H {self.multiHeads} x N_curr {num_nodes_current * batch_size} x F {self.transformerFeatures}]', self.verbose)

        dense_output = aggregated_messages_sparse.to_dense()

        if self.shepardAttention and shepardValues is not None:
            verbosePrint(f'Applying Shepard attention modulation', self.verbose, separator=True)
            # shepardValues has shape [batch_size, num_nodes_current, num_nodes_neighbor]
            # Multiply shepardValues with the dense output after aggregation as a form of scaled softmax
            dense_output = dense_output * shepardValues.unsqueeze(-1)
            verbosePrint(f'Dense output shape after Shepard modulation: {dense_output.shape} [1 x H {self.multiHeads} x N {num_nodes_current * batch_size} x F {self.transformerFeatures}]', self.verbose)
            # print(f'Shepard Values: min: {shepardValues.min()}, max: {shepardValues.max()}, mean: {shepardValues.mean()}, std: {shepardValues.std()}')
            # print(f'Dense Output after Shepard: min: {dense_output.min()},        
            #       max: {dense_output.max()}, mean: {dense_output.mean()}, std: {dense_output.std()}')
        elif self.shepardAttention and shepardValues is None:
            warnings.warn("shepardAttention is True but no shepardValues provided. Skipping Shepard modulation.", UserWarning)
        else:
            verbosePrint(f'\tNo Shepard attention modulation applied', self.verbose)

        # print(dense_output)

        verbosePrint(f'Dense output shape: {dense_output.shape} [B {batch_size * num_nodes_current} x H {self.multiHeads} x F {self.transformerFeatures}]', self.verbose)
        attentionOutputSparse = dense_output.permute(0, 2, 1, 3).reshape(num_nodes_current, batch_size, -1).transpose(0, 1)
        verbosePrint(f'Attention output sparse shape: {attentionOutputSparse.shape} [B {batch_size} x N {num_nodes_current} x H*F {self.multiHeads * self.transformerFeatures}]', self.verbose)

        verbosePrint(f'Projecting attention output back to latent space', self.verbose, separator=True)
        # Project back to latent space
        
        if self.multiHeadAggregation == 'mean':
            attentionOutputSparse = attentionOutputSparse.view(batch_size, num_nodes_current, self.multiHeads, self.transformerFeatures)
            attentionOutput = self.W_O(attentionOutputSparse)
            verbosePrint(f'Attention output shape before mean aggregation: {attentionOutput.shape} [B {batch_size} x N {num_nodes_current} x L {latentSpaceSize}]', self.verbose)
            attentionOutput = attentionOutput.mean(dim=2)
            verbosePrint(f'Attention output shape after mean aggregation: {attentionOutput.shape} [B {batch_size} x N {num_nodes_current} x L {latentSpaceSize}]', self.verbose)
        else:
            attentionOutput = self.W_O(attentionOutputSparse)
        verbosePrint(f'Attention output shape after projection: {attentionOutput.shape} [B {batch_size} x N {num_nodes_current} x L {latentSpaceSize}]', self.verbose)
        if self.useDropout:
            attentionOutput = self.dropout(attentionOutput)        
        # Residual Connection and Layer Norm (Post-Attention)
        verbosePrint(f'Applying residual connection: {inputTokensCurrent.shape} [B {batch_size} x N {num_nodes_current} x L {latentSpaceSize}]', self.verbose)
        # attentionOutput = inputTokensCurrent + attentionOutput  # Residual connection
        verbosePrint(f'Running Layer Norm', self.verbose)

        # print(f'Pre Norm min: {attentionOutput.min()}, max: {attentionOutput.max()}, mean: {attentionOutput.mean()}, std: {attentionOutput.std()}')
        # attentionOutput = self.layer_norm1(attentionOutput)
        # print(f'Post Norm min: {attentionOutput.min()}, max: {attentionOutput.max()}, mean: {attentionOutput.mean()}, std: {attentionOutput.std()}')

        verbosePrint(f'Applying Feedforward Network (FFN)', self.verbose)
        ffnOutput = self.ffn(attentionOutput)
        verbosePrint(f'FFN output shape: {ffnOutput.shape} [B {batch_size} x N {num_nodes_current} x L {latentSpaceSize}]', self.verbose)

        # Residual Connection and Layer Norm (Post-FFN)
        transformerOutput = attentionOutput + ffnOutput  # Residual connection
        transformerOutput = attentionOutput
        # print(f'Pre Norm min: {transformerOutput.min()}, max: {transformerOutput.max()}, mean: {transformerOutput.mean()}, std: {transformerOutput.std()}')
        # transformerOutput = self.layer_norm2(transformerOutput)
        # print(f'Post Norm min: {transformerOutput.min()}, max: {transformerOutput.max()}, mean: {transformerOutput.mean()}, std: {transformerOutput.std()}' )
        verbosePrint(f'Final transformerOutput shape after residual connection and layer norm: {transformerOutput.shape} [B {batch_size} x N {num_nodes_current} x L {latentSpaceSize}]', self.verbose)

        return transformerOutput

