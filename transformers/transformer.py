import torch
from ml import getActivationLayer
import torch.nn as nn

def softmax(attentionScoresSparse, sparse_values, rows, cols, sparse_indices): #batch_size, multiHeads, num_edges, sparse_values, cols, num_nodes_neighbor):
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

    # The 'cols' index needs to be broadcast to match the new shape
    # It will guide the grouping for every head in every batch item
    index = cols.expand_as(scores)

    # print(f'===============================================================')
    # print(f'Computing normalized attention weights for {batch_size} batches, {multiHeads} heads, {num_edges} edges')
    # print(f'Index shape: {index.shape}, Scores shape: {scores.shape}')

    # 1. Subtract max for numerical stability (a standard softmax trick)
    # We need to find the max score for each destination node group
    # 'scatter_max' is not native, so we use a dense intermediate for this step.
    # It's an acceptable tradeoff as it's not on a huge tensor.
    alpha_max = torch.zeros(batch_size * multiHeads, num_nodes_neighbor, device=scores.device).scatter_reduce_(
        1, index, scores, reduce="amax", include_self=False
    )
    # print(f'Alpha max shape: {alpha_max.shape} [ {alpha_max.numel()} elements ]')

    # Now gather the max value for each edge
    scores_sub = scores - alpha_max.gather(1, index)
    # print(f'Scores shape after subtracting max: {scores_sub.shape} [ {scores_sub.numel()} elements ]')

    # 2. Exponentiate the scores
    exp_scores = torch.exp(scores_sub)
    # print(f'Exponentiated scores shape: {exp_scores.shape} [ {exp_scores.numel()} elements ]')

    # 3. Sum the exponentiated scores for each destination node group
    # This is the denominator of the softmax
    exp_sum = torch.zeros(batch_size * multiHeads, num_nodes_neighbor, device=scores.device).scatter_add_(
        1, index, exp_scores
    )
    # print(f'Exponentiated sum shape: {exp_sum.shape} [ {exp_sum.numel()} elements ]')
    # Add a small epsilon to prevent division by zero
    exp_sum = exp_sum + 1e-10
    # print(f'Exponentiated sum after adding epsilon: {exp_sum.shape} [ {exp_sum.numel()} elements ]')

    # 4. Divide each score by its group's sum to get the final weights
    normalized_weights_flat = exp_scores / exp_sum.gather(1, index)
    # print(f'Normalized weights shape: {normalized_weights_flat.shape} [ {normalized_weights_flat.numel()} elements ]')

    return normalized_weights_flat

    # attention_weights_sparse = torch.sparse_coo_tensor(
    #     indices=sparse_indices,
    #     values=normalized_weights_flat.flatten(),
    #     size=attentionScoresSparse.shape
    # )
    # return attention_weights_sparse, normalized_weights_flat


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
    i_idx = cols.repeat(1 * multiHeads)
    # Source nodes (j)
    j_idx = rows.repeat(1 * multiHeads)

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


class TransformerLayer(torch.nn.Module):
    def __init__(self, input_dim, transformer_features, edgeFeatureSize, multi_heads,
                 edge_bias=False, edge_gating=False,
                 additive_bias=True, verbose=False,
                 activation='celu', ffnHiddenLayers=1, ffnHiddenSize=0,
                 attentionOp='dot', attentionOpIncludeEdge=False,
                 sharedWeights=False, scaleQ=True):
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
        
        verbosePrint(f'Initializing TransformerLayer with input_dim={input_dim}, transformer_features={transformer_features}, edgeFeatureSize={edgeFeatureSize}, multi_heads={multi_heads}, edge_bias={edge_bias}, edge_gating={edge_gating}, additive_bias={additive_bias}', verbose)
        verbosePrint(f'Building linear projections for Q, K, V', verbose, separator=True)

        if not sharedWeights:
            self.W_Q = torch.nn.Linear(input_dim, transformer_features * multi_heads, bias=False)
            self.W_K = torch.nn.Linear(input_dim, transformer_features * multi_heads, bias=False)
            self.W_V = torch.nn.Linear(input_dim, transformer_features * multi_heads, bias=False)
        else:
            self.W_Q = self.W_K = self.W_V = torch.nn.Linear(input_dim, transformer_features * multi_heads, bias=False)
        verbosePrint(f'W_Q shape: {self.W_Q.weight.shape}, W_K shape: {self.W_K.weight.shape}, W_V shape: {self.W_V.weight.shape}', verbose)
        
        verbosePrint(f'Building edge bias and gating projections', verbose, separator=True)
        if edge_bias:
            self.W_E = torch.nn.Linear(edgeFeatureSize, multi_heads)
            verbosePrint(f'\tUsing edge bias with W_E shape: {self.W_E.weight.shape}', verbose)
        else:
            self.W_E = None            
        # Edge gating is optional and can be used to gate the value matrix with edge features
        if edge_gating:
            self.W_E_gate = torch.nn.Linear(edgeFeatureSize, multi_heads * transformer_features)
            verbosePrint(f'\tUsing edge gating with W_E_gate shape: {self.W_E_gate.weight.shape}', verbose)
        else:
            self.W_E_gate = None

        verbosePrint(f'Building output projection steps', verbose, separator=True)
        self.activation = activation if isinstance(activation, torch.nn.Module) else getActivationLayer(activation)
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
        
        inputSize = 2 * transformer_features
        if self.attentionOpIncludeEdge:
            inputSize += edgeFeatureSize  # Include edge features in GAT attention

        if self.attentionOp == 'GAT':
            verbosePrint(f'\tUsing GAT attention with input size: {inputSize}', verbose)
            self.W_a = torch.nn.Linear(inputSize, 1, bias=False)
            # The LeakyReLU activation, with the negative slope commonly used in GAT.
            self.leaky_relu = nn.LeakyReLU(0.2)
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
            layers.append(torch.nn.Linear(in_dim, inputSize))
            self.W_a = torch.nn.Sequential(*layers)

            # The LeakyReLU activation, with the negative slope commonly used in GAT.
            self.leaky_relu = nn.LeakyReLU(0.2)
        else:
            verbosePrint(f'\tUsing dot attention', verbose)
            self.W_a = None  # For 'dot' attention, no additional weights are needed

        verbosePrint(f'TransformerLayer initialized with input_dim={input_dim}, transformer_features={transformer_features}, edgeFeatureSize={edgeFeatureSize}, multi_heads={multi_heads}, edge_bias={edge_bias}, edge_gating={edge_gating}, additive_bias={additive_bias}', verbose)

    def forward(self, inputTokens_: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], inputEdges, edgeIndices):
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
        if inputEdges is not None:
            verbosePrint(f'\tInput edges shape: {inputEdges.shape}', self.verbose)
        else:
            verbosePrint(f'\tNo edge features provided', self.verbose)
        verbosePrint(f'Edge indices shape: {edgeIndices.shape}', self.verbose)

        # Preamble done, Projection into Query, Key, Value next

        verbosePrint(f'Projection Step', self.verbose, separator=True)

        Q = self.W_Q(inputTokensCurrent)
        K = self.W_K(inputTokensNeighbor)
        V = self.W_V(inputTokensNeighbor)

        verbosePrint(f'Input Token Shape [current ]: {inputTokensCurrent.shape} [B {batch_size} x N {num_nodes_current} x D {latentSpaceSize}]', self.verbose)
        verbosePrint(f'Input Token Shape [neighbor]: {inputTokensNeighbor.shape} [B {batch_size} x N {num_nodes_neighbor} x D {latentSpaceSize}]', self.verbose)

        Q = Q.view(batch_size, Q.shape[1], self.multiHeads, self.transformerFeatures).permute(0, 2, 1, 3)
        K = K.view(batch_size, K.shape[1], self.multiHeads, self.transformerFeatures).permute(0, 2, 1, 3)
        V = V.view(batch_size, V.shape[1], self.multiHeads, self.transformerFeatures).permute(0, 2, 1, 3)

        verbosePrint(f'Query Shape: {Q.shape} [B {batch_size} x H {self.multiHeads} x N {Q.shape[2]} x D {self.transformerFeatures}]', self.verbose)
        verbosePrint(f'Key Shape:   {K.shape} [B {batch_size} x H {self.multiHeads} x N {K.shape[2]} x D {self.transformerFeatures}]', self.verbose)
        verbosePrint(f'Value Shape: {V.shape} [B {batch_size} x H {self.multiHeads} x N {V.shape[2]} x D {self.transformerFeatures}]', self.verbose)

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
                
                expanded_inputEdges = inputEdges.view(batch_size_edges, 1, Q_i.shape[2], -1).expand(batch_size_edges, self.multiHeads, -1, -1)  # Shape: [B, H, edgeFeatureSize]
                verbosePrint(f'\t\tExpanded input edges [accounting for H]: {expanded_inputEdges.shape} [1 x H {self.multiHeads} x E {num_edges} x FE {self.edgeFeatureSize}]', self.verbose)
                combined = torch.cat([Q_i, K_j, expanded_inputEdges], dim=-1)  # Shape: [B, H, num_edges, 2*F + edgeFeatureSize]
            verbosePrint(f'\tCombined shape for attention scores: {combined.shape} [1 {batch_size_edges} x H {self.multiHeads} x E {num_edges} x (2*F + ?FE)]', self.verbose)

            verbosePrint(f'\tProjecting attention scores', self.verbose)
            sparseAttentionValues = self.W_a(combined)  # Shape: [B, H, num_edges, 1]
            sparseAttentionValues = self.leaky_relu(sparseAttentionValues).squeeze(-1)  # Shape: [B, H, num_edges]
            
        elif self.attentionOp == 'MLP':
            verbosePrint(f'\tUsing MLP attention scores', self.verbose)
            if not self.attentionOpIncludeEdge:
                combined = torch.cat([Q_i, K_j], dim=-1)  # Shape: [B, H, num_edges, 2*F]
            else:
                verbosePrint(f'\t\tExpanded input edges [accounting for H]: {expanded_inputEdges.shape} [1 x H {self.multiHeads} x E {num_edges} x FE {self.edgeFeatureSize}]', self.verbose)
                expanded_inputEdges = inputEdges.view(batch_size_edges, 1, Q_i.shape[2], -1).expand(batch_size_edges, self.multiHeads, -1, -1)  # Shape: [B, H, edgeFeatureSize]
                combined = torch.cat([Q_i, K_j, expanded_inputEdges], dim=-1)  # Shape: [B, H, num_edges, 2*F + edgeFeatureSize]

            verbosePrint(f'\tCombined shape for attention scores: {combined.shape} [1 {batch_size_edges} x H {self.multiHeads} x E {num_edges} x (2*F + ?FE)]', self.verbose)                
            sparseAttentionValues = self.W_a(combined).squeeze(-1)  # Shape: [B, H, num_edges]
            sparseAttentionValues = self.leaky_relu(sparseAttentionValues)  # Apply activation
        else:
            raise ValueError(f'Unknown attention operation: {self.attentionOp}')

        verbosePrint(f'Final sparse attention values shape: {sparseAttentionValues.shape} [1 {batch_size_edges} x H {self.multiHeads} x E {num_edges}]', self.verbose)

        sparse_values = sparseAttentionValues.flatten()
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
        
        verbosePrint(f'Normalized weights shape: {normalized_weights.shape} [1 x H {self.multiHeads} x E {num_edges}]', self.verbose)
        verbosePrint(f'Collecting Value Tokens', self.verbose, separator=True)

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
        message_values = final_messages.reshape(-1, self.transformerFeatures)
        verbosePrint(f'Message values shape: {message_values.shape} [B * H * E {batch_size_edges * self.multiHeads * num_edges} x F {self.transformerFeatures}]', self.verbose)

        verbosePrint(f'Summing Messages Step', self.verbose, separator=True)
        
        # The full size of the hybrid tensor: sparse part + dense part
        hybrid_size = (batch_size_edges, self.multiHeads, num_nodes_current * batch_size, num_nodes_neighbor * batch_size, self.transformerFeatures)
        # Create the hybrid sparse tensor representing messages
        sparse_message_tensor = torch.sparse_coo_tensor(indices=sparse_indices, values=message_values, size=hybrid_size)
        verbosePrint(f'Sparse message tensor shape: {sparse_message_tensor.shape} [1 x H {self.multiHeads} x N_curr {num_nodes_current * batch_size} x N_neigh {num_nodes_neighbor * batch_size} x F {self.transformerFeatures}]', self.verbose)

        # Sum over the source dimension 'j' (dim=3) to aggregate messages
        aggregated_messages_sparse = torch.sparse.sum(sparse_message_tensor, dim=3)
        verbosePrint(f'Aggregated messages shape: {aggregated_messages_sparse.shape} [1 x H {self.multiHeads} x N_curr {num_nodes_current * batch_size} x F {self.transformerFeatures}]', self.verbose)
        dense_output = aggregated_messages_sparse.to_dense()
        verbosePrint(f'Dense output shape: {dense_output.shape} [B {batch_size * num_nodes_current} x H {self.multiHeads} x F {self.transformerFeatures}]', self.verbose)
        attentionOutputSparse = dense_output.permute(0, 2, 1, 3).reshape(num_nodes_current, batch_size, -1).transpose(0, 1)
        verbosePrint(f'Attention output sparse shape: {attentionOutputSparse.shape} [B {batch_size} x N {num_nodes_current} x H*F {self.multiHeads * self.transformerFeatures}]', self.verbose)

        verbosePrint(f'Projecting attention output back to latent space', self.verbose, separator=True)
        # Project back to latent space
        
        attentionOutput = self.W_O(attentionOutputSparse)
        verbosePrint(f'Attention output shape after projection: {attentionOutput.shape} [B {batch_size} x N {num_nodes_current} x L {latentSpaceSize}]', self.verbose)
        # Residual Connection and Layer Norm (Post-Attention)
        verbosePrint(f'Applying residual connection: {inputTokensCurrent.shape} [B {batch_size} x N {num_nodes_current} x L {latentSpaceSize}]', self.verbose)
        attentionOutput = inputTokensCurrent + attentionOutput  # Residual connection
        verbosePrint(f'Running Layer Norm', self.verbose)
        attentionOutput = self.layer_norm1(attentionOutput)

        verbosePrint(f'Applying Feedforward Network (FFN)', self.verbose)
        ffnOutput = self.ffn(attentionOutput)
        verbosePrint(f'FFN output shape: {ffnOutput.shape} [B {batch_size} x N {num_nodes_current} x L {latentSpaceSize}]', self.verbose)

        # Residual Connection and Layer Norm (Post-FFN)
        transformerOutput = attentionOutput + ffnOutput  # Residual connection
        transformerOutput = self.layer_norm2(transformerOutput)
        verbosePrint(f'Final transformerOutput shape after residual connection and layer norm: {transformerOutput.shape} [B {batch_size} x N {num_nodes_current} x L {latentSpaceSize}]', self.verbose)

        return transformerOutput

