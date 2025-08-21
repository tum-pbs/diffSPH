from typing import Union, Tuple
class TransformerLayer(torch.nn.Module):
    def __init__(self, input_dim, transformer_features, edgeFeatureSize, multi_heads, 
                 edge_bias = False, edge_gating = False,
                 additive_bias = True, verbose = False,
                 activation = 'celu', ffnHiddenLayers = 1, ffnHiddenSize = 0):
        super(TransformerLayer, self).__init__()
        if verbose:
            print(f'Initializing TransformerLayer with input_dim={input_dim}, transformer_features={transformer_features}, edgeFeatureSize={edgeFeatureSize}, multi_heads={multi_heads}, edge_bias={edge_bias}, edge_gating={edge_gating}, additive_bias={additive_bias}')

        if verbose:
            print(f'===============================================================')
            print(f'Building linear projections for Q, K, V')

        self.W_Q = torch.nn.Linear(input_dim, transformer_features * multi_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, transformer_features * multi_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, transformer_features * multi_heads, bias=False)
        if verbose:
            print(f'\tW_Q shape: {self.W_Q.weight.shape}, W_K shape: {self.W_K.weight.shape}, W_V shape: {self.W_V.weight.shape}')

        if verbose:
            print(f'===============================================================')
            print(f'Building edge bias and gating projections')
        if edge_bias:
            self.W_E = torch.nn.Linear(edgeFeatureSize, multi_heads).to(device)
            if verbose:
                print(f'\tUsing edge bias with W_E shape: {self.W_E.weight.shape}')
        else:
            self.W_E = None
            
        # Edge gating is optional and can be used to gate the value matrix with edge features
        if edge_gating:
            self.W_E_gate = torch.nn.Linear(edgeFeatureSize, multi_heads * transformer_features).to(device)
            if verbose:
                print(f'\tUsing edge gating with W_E_gate shape: {self.W_E_gate.weight.shape}')
        else:
            self.W_E_gate = None

        if verbose:
            print(f'===============================================================')
            print(f'Building output projection steps')
        self.activation = activation if isinstance(activation, torch.nn.Module) else getActivationLayer(activation)
        self.W_O = torch.nn.Linear(transformer_features * multi_heads, input_dim, bias=False)
        self.layer_norm1 = torch.nn.LayerNorm(input_dim)
        self.layer_norm2 = torch.nn.LayerNorm(input_dim)

        if verbose:
            print(f'\tW_O shape: {self.W_O.weight.shape}')

        self.hiddenLayers = ffnHiddenLayers
        self.ffnHiddenSize = ffnHiddenSize if ffnHiddenSize > 0 else input_dim * 4

        if verbose:
            print(f'===============================================================')
            print(f'Building Feedforward Network (FFN) with {self.hiddenLayers} hidden layers and hidden size {self.ffnHiddenSize}')

        # Build FFN with configurable number of hidden layers and hidden size
        layers = []
        in_dim = input_dim
        for i in range(self.hiddenLayers):
            layers.append(torch.nn.Linear(in_dim, self.ffnHiddenSize))
            layers.append(self.activation)
            in_dim = self.ffnHiddenSize
        layers.append(torch.nn.Linear(in_dim, input_dim))
        self.ffn = torch.nn.Sequential(*layers)

        if verbose:
            print(f'\tFFN: {self.ffn}')

        self.multiHeads = multi_heads
        self.transformerFeatures = transformer_features
        self.edgeFeatureSize = edgeFeatureSize

        self.edgeBias = edge_bias
        self.edgeGating = edge_gating
        self.additiveBias = additive_bias
        self.verbose = verbose
        

        if verbose:
            print(f'===============================================================')
            print(f'TransformerLayer initialized with input_dim={input_dim}, transformer_features={transformer_features}, edgeFeatureSize={edgeFeatureSize}, multi_heads={multi_heads}, edge_bias={edge_bias}, edge_gating={edge_gating}, additive_bias={additive_bias}')

    def forward(self, inputTokens_: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], inputEdges, edgeIndices):
        if isinstance(inputTokens_, tuple):
            inputTokensCurrent, inputTokensNeighbor = inputTokens_
        else:
            inputTokensCurrent = inputTokensNeighbor = inputTokens_  # No row information if not provided
        batch_size, num_nodes, latentSpaceSize = inputTokensCurrent.shape
        # Multi-Head Attention with Edge Information
        if self.verbose:
            print(f'===============================================================')
            print(f'Input tokens shape: {inputTokensCurrent.shape}')
            if inputEdges is not None:
                print(f'Input edges shape: {inputEdges.shape}')
            else:
                print('No edge features provided')

            if edgeIndices is not None:
                print(f'Edge indices shape: {edgeIndices.shape}')
            else:
                print('No edge indices provided')
                raise ValueError('Edge indices must be provided for sparse neighborhoods')
            
        rows = edgeIndices[0]
        cols = edgeIndices[1]

        if self.verbose:
            print(f'===============================================================')

        Q = self.W_Q(inputTokensCurrent)
        K = self.W_K(inputTokensNeighbor)
        V = self.W_V(inputTokensNeighbor)

        Q = Q.view(batch_size, Q.shape[1], self.multiHeads, self.transformerFeatures).permute(0, 2, 1, 3)
        K = K.view(batch_size, K.shape[1], self.multiHeads, self.transformerFeatures).permute(0, 2, 1, 3)
        V = V.view(batch_size, V.shape[1], self.multiHeads, self.transformerFeatures).permute(0, 2, 1, 3)

        if self.verbose:
            print('Q shape:', Q.shape)
            print('K shape:', K.shape)
            print('V shape:', V.shape)

        # Scale Q by sqrt(d_k)
        # This is a common practice in Transformer architectures to stabilize training
        # and improve convergence.
        # The scaling factor is the square root of the dimension of the key vectors (d_k).
        # It helps to prevent the dot product from growing too large, which can lead to
        # very small gradients and slow down training.
        # This scaling is applied before the softmax operation in the attention mechanism.
        # It ensures that the attention scores are not too large, which can lead to numerical instability.
        # In this case, d_k is equal to transformerFeatures.
        # Therefore, we scale Q by the square root of transformerFeatures.
        Q = Q / (self.transformerFeatures ** 0.5)  # Scale by sqrt(d_k)

        # if self.edgeBias:
        # else:
        # For a dense neighborhood we can run a matmul directly
        attentionScores = torch.matmul(Q, K.transpose(-2, -1))
        # Attention scores shape: [batch_size, multiHeads, num_nodes, num_nodes]
        print(f'===============================================================')
        print('attentionScores shape:', attentionScores.shape)
        # For a sparse neighborhood we need to use the edge indices
        if self.verbose:
            print(f'===============================================================')
            print('Using sparse neighborhood for attention scores')

        # Q has shape [batch_size, multiHeads, num_nodes, transformerFeatures]
        # K has shape [batch_size, multiHeads, num_nodes, transformerFeatures]
        # We need to compute the attention scores using the edge indices
        # This is done by selecting the rows and columns from Q and K based on edgeIndices
        # rows and cols are the indices of the edges in the sparse neighborhood
        # However, the edgeIndices are for the flattened versions of Q and K
        Q_flat = Q.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, self.multiHeads, self.transformerFeatures)
        K_flat = K.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, self.multiHeads, self.transformerFeatures)
        # Q_flat = Q.reshape(batch_size * self.multiHeads, num_nodes, self.transformerFeatures)
        # K_flat = K.reshape(batch_size * self.multiHeads, num_nodes, self.transformerFeatures)
        if self.verbose:
            print(f'Q_flat shape: {Q_flat.shape}, K_flat shape: {K_flat.shape}')
            print(f'Rows: {rows}, Cols: {cols}')

        # Now we can index the query matrices using the rows and the key matrices using the cols
        # This will give us the correct rows and columns for the attention scores
        if self.verbose:
            print(f'Edge indices: {edgeIndices}')
            print(f'Rows: {rows}, Cols: {cols}')

        # Gather the vectors for each edge
        # Q_i are the queries for the destination nodes of each edge
        # K_j are the keys for the source nodes of each edge
        Q_i = Q[:, :, cols, :] # Shape: [B, H, num_edges, F]
        K_j = K[:, :, rows, :] # Shape: [B, H, num_edges, F]


        # Q_i_dense = Q_i.view(batch_size, self.multiHeads, num_nodes, num_nodes, self.transformerFeatures)
        # K_j_dense = K_j.view(batch_size, self.multiHeads, num_nodes, num_nodes, self.transformerFeatures)

        # print(f'Q_i_dense shape: {Q_i_dense.shape}, K_j_dense shape: {K_j_dense.shape}')
        # print(f'Q_i shape: {Q_i.shape}, K_j shape: {K_j.shape}')

        # if torch.any(torch.logical_not(torch.isclose(Q_i_dense, Q))) or torch.any(torch.logical_not(torch.isclose(K_j_dense, K))):
        #     print('Q_i or K_j mismatch with Q or K!')
        #     print('Q_i:', Q_i_dense)
        #     print('Q:', Q)
        #     print('K_j:', K_j_dense)
        #     print('K:', K)
        #     raise ValueError('Q_i or K_j mismatch with Q or K!')

        if self.verbose:
            print(f'Q_i shape: {Q_i.shape}, K_j shape: {K_j.shape}')
            print(f'Rows: {rows.shape}, Cols: {cols.shape}')

        if self.verbose:
            print(f'Edge indices shape: {edgeIndices.shape}')
            print(f'Rows: {rows}, Cols: {cols}')

        # sparseAttentionValues = torch.einsum('hij,hki->hi', Q_i, K_j.transpose(-2, -1))
        sparseAttentionValues = (Q_i * K_j).sum(dim=-1) # / (self.transformerFeatures**0.5)
        sparse_values = sparseAttentionValues.flatten()
        size = (batch_size, self.multiHeads, num_nodes, num_nodes)


        # print(f'sparseAttentionValues shape: {sparseAttentionValues.shape}')
        
        # denseAttentionValues = sparseAttentionValues.view(batch_size, self.multiHeads, num_nodes, num_nodes)
        # if torch.any(torch.logical_not(torch.isclose(denseAttentionValues, attentionScores))):
        #     print('Attention scores mismatch! ')
        #     print('denseAttentionValues:', denseAttentionValues)
        #     print('attentionScores:', attentionScores)
        #     raise ValueError('Attention scores mismatch! ')
        num_edges = edgeIndices.shape[1]

        # 1. Create indices for the batch dimension (b)
        # Each of the H*num_edges scores in a batch item gets the same batch index
        b_idx = torch.arange(batch_size, device=Q.device).repeat_interleave(self.multiHeads * num_edges)

        # 2. Create indices for the head dimension (h)
        # Within each batch item, the indices 0..H-1 repeat for each edge
        h_idx = torch.arange(self.multiHeads, device=Q.device).repeat_interleave(num_edges).repeat(batch_size)

        # 3. Repeat the edge indices for each batch and head
        # Destination nodes (i)
        i_idx = cols.repeat(batch_size * self.multiHeads)
        # Source nodes (j)
        j_idx = rows.repeat(batch_size * self.multiHeads)

        # 4. Stack them all together to create the final sparse indices
        # Shape will be [4, B * H * num_edges]
        sparse_indices = torch.stack([b_idx, h_idx, i_idx, j_idx], dim=0)

        if self.edgeBias:
            if self.verbose:
                print(f'===============================================================')
                print('Using edge bias for attention scores')
            edge_bias = self.W_E(inputEdges).reshape(1, nx**2, nx**2, self.multiHeads) # shape: [batch, num_nodes, num_nodes, multiHeads]
            # print('edge_bias shape:', edge_bias.shape)
            # We need to align the dimensions for broadcasting with attention scores
            edge_bias = edge_bias.permute(0, 3, 1, 2) # shape: [batch, multiHeads, num_nodes, num_nodes]

            if self.verbose:
                print(f'Edge bias shape after permute: {edge_bias.shape}')
                print(f'Attention scores shape before edge bias: {Q.shape} x {K.transpose(-2, -1).shape}')

            # Compute Edge-aware Attention Scores
            if self.additiveBias:
                sparse_values = sparse_values + edge_bias.flatten()
            else:
                sparse_values = sparse_values * edge_bias.flatten()

        # Create the sparse tensor of raw scores
        attentionScoresSparse = torch.sparse_coo_tensor(
            indices=sparse_indices,
            values=sparse_values,
            size=size
        )

        if self.edgeBias:
            if self.verbose:
                print(f'===============================================================')
                print('Using edge bias for attention scores')
            edge_bias = self.W_E(inputEdges).reshape(1, nx**2, nx**2, self.multiHeads) # shape: [batch, num_nodes, num_nodes, multiHeads]
            # print('edge_bias shape:', edge_bias.shape)
            # We need to align the dimensions for broadcasting with attention scores
            edge_bias = edge_bias.permute(0, 3, 2, 1) # shape: [batch, multiHeads, num_nodes, num_nodes]

            if self.verbose:
                print(f'Edge bias shape after permute: {edge_bias.shape}')
                print(f'Attention scores shape before edge bias: {Q.shape} x {K.transpose(-2, -1).shape}')

            # Compute Edge-aware Attention Scores
            if self.additiveBias:
                attentionScores = attentionScores + edge_bias          

            else:
                attentionScores = attentionScores * edge_bias


        # attentionScoresSparse = torch.sparse_coo_tensor(
        #     indices=torch.stack([rows, cols]),
        #     values=sparseAttentionValues.flatten(),
        #     size=(batch_size, self.multiHeads, num_nodes, num_nodes)
        # )
        print(f'Attention scores sparse shape: {attentionScoresSparse.shape}')
        # attentionScores = attentionScoresSparse.to_dense().view(batch_size, self.multiHeads, num_nodes, num_nodes)

        if self.verbose:
            print('attentionScores shape:', attentionScores.shape)
        attention_weights_sparse = torch.sparse.softmax(attentionScoresSparse, dim=2)
        attentionScores = torch.nn.functional.softmax(attentionScores, dim=-1)

        print(f'Attention scores shape after softmax: {attentionScores.shape}')
        print(f'Attention weights sparse shape: {attention_weights_sparse.shape}')
    
        if torch.any(torch.logical_not(torch.isclose(attentionScores, attention_weights_sparse.to_dense()))):
            print('Attention scores mismatch with sparse weights!')
            print('attentionScores:', attentionScores)
            print('attention_weights_sparse:', attention_weights_sparse.to_dense())
            raise ValueError('Attention scores mismatch with sparse weights!')
        

        if self.verbose:
            print(f'===============================================================')
            print('attentionScores shape after softmax:', attentionScores.shape)
        if self.edgeGating:
            if self.verbose:
                print(f'===============================================================')
                print('Using edge gating for value matrix')
            # ==============================================================================
            # --- NEW: Gating the Value matrix with Edge Features ---
            # 1. Create a new linear projection for the edge gates
            # The output must match the feature dimension of V

            # 2. Project edge features and apply sigmoid
            # dense_mod_xij shape: [B, N, N, edgeFeatureSize]
            edge_gate = self.W_E_gate(inputEdges) # Shape: [B, N, N, H*F]
            edge_gate = torch.sigmoid(edge_gate)
            if self.verbose:
                print(f'Edge gate shape after projection and sigmoid: {edge_gate.shape}')

            # 3. Reshape the gate to match the multi-head structure for broadcasting
            # New shape: [B, N (query), N (key), H, F] -> [B, H, N (query), N (key), F]
            edge_gate = edge_gate.view(batch_size, num_nodes, num_nodes, multiHeads, transformerFeatures)
            edge_gate = edge_gate.permute(0, 3, 1, 2, 4)
            if self.verbose:
                print(f'Edge gate shape after reshape and permute: {edge_gate.shape}')

            # 4. Prepare V for gating. We need to align it with the [query, key] structure of the gate
            # V shape is [B, H, N (key), F]. We unsqueeze to add a "query" dimension for broadcasting
            V_expanded = V.unsqueeze(2) # Shape: [B, H, 1, N (key), F]

            # 5. Apply the gate. Broadcasting rules will make this work.
            # V_expanded is broadcast across the "query" dimension (dim 2)
            # edge_gate is broadcast across the "value feature" dimension (dim 4)
            V_gated = V_expanded * edge_gate # Shape: [B, H, N (query), N (key), F]
            # print(f'V_gated after gating: {V_gated}')
            if self.verbose:
                print(f'V_gated shape after gating: {V_gated.shape}')

            # 6. Compute the final output using the gated values
            # The matmul is now between attentionScores [B,H,N,N] and a gated V
            # We need to sum over the "key" dimension. This is no longer a simple matmul.
            # We can do it with einsum for clarity or broadcasting + sum.
            # attentionScores needs an extra dimension: [B, H, N (query), N (key), 1]
            # Then multiply and sum over the key dimension (dim=3)
            attentionOutput = (attentionScores.unsqueeze(-1) * V_gated).sum(dim=3)
            # Final shape of attentionOutput: [B, H, N (query), F]
            # ==============================================================================

            # --- Reshape back (this part is now different) ---
            # Previous shape was [B, H, N, F]. Now it's [B, H, N (query), F], which is the same.
            attentionOutput = attentionOutput.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, -1)
                
        else:
        # ==============================================================================
        # --- Original Attention Output without Edge Gating ---
        # 1. Compute Edge-aware Attention Output
        # attentionOutput = torch.matmul(attentionScores, V)
        # attentionOutput = attentionOutput.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, -1)
        # ==============================================================================
            attentionOutput = torch.matmul(attentionScores, V)
            attentionOutput = attentionOutput.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, -1)


        # Get the normalized weights from the sparse tensor
        normalized_weights = attention_weights_sparse.values() # Shape: [B * H * num_edges]
        normalized_weights = normalized_weights.view(batch_size, self.multiHeads, num_edges)

        V_j = V[:, :, rows, :] # Shape: [B, H, num_edges, F]

        messages = V_j  # This is the message vector for each edge

        # Weight the V_j vectors with the normalized attention
        # Add a dimension for broadcasting: weights shape -> [B, H, num_edges, 1]
        if self.edgeGating:
            if self.verbose:
                print(f'===============================================================')
                print('Using edge gating for messages')
            # 3. Project edge features to create the gate
            # W_E_gate is a nn.Linear(edge_feature_dim, self.multiHeads * self.transformerFeatures)
            edge_gate_values = self.W_E_gate(inputEdges) # Shape: [num_edges, H*F]
            edge_gate_values = torch.sigmoid(edge_gate_values)

            # 4. Reshape gate to be compatible with V_j for broadcasting
            # Goal shape: [1, H, num_edges, F]
            edge_gate_values = edge_gate_values.view(
                num_edges, self.multiHeads, self.transformerFeatures
            ) # Shape: [num_edges, H, F]
            edge_gate_values = edge_gate_values.permute(1, 0, 2).unsqueeze(0) # Shape: [1, H, num_edges, F]

            if self.verbose:
                print(f"Shape of V_j to be gated: {V_j.shape}")
                print(f"Shape of sparse edge gate: {edge_gate_values.shape}")

            # 5. Apply the gate to the V_j vectors. Broadcasting handles the batch dim.
            gated_V_j = V_j * edge_gate_values

            gated_V_j_dense = gated_V_j.view(batch_size, self.multiHeads, num_nodes, num_nodes, self.transformerFeatures)
            print(f'Gated V_j dense shape: {gated_V_j_dense.shape} | V_gated shape: {V_gated.shape}')

            if torch.any(torch.logical_not(torch.isclose(gated_V_j_dense, V_gated))):
                print('Gated V_j mismatch with V!')
                print('gated_V_j:', gated_V_j_dense)
                print('V:', V_gated)
                raise ValueError('Gated V_j mismatch with V!')

            messages = gated_V_j # Update messages to be the gated version


        final_messages = messages * normalized_weights.unsqueeze(-1)

        # 4. Prepare the components for the hybrid sparse tensor
        # We can reuse the indices from our attention score calculation
        # sparse_indices has shape [4, B * H * num_edges] and maps to [b, h, i, j]
        #
        # The values are now the message vectors. We need to reshape them to match the indices.
        # Shape: [B, H, num_edges, F] -> [B * H * num_edges, F]
        message_values = final_messages.reshape(-1, self.transformerFeatures)

        # The full size of the hybrid tensor: sparse part + dense part
        hybrid_size = (
            batch_size,
            self.multiHeads,
            num_nodes,
            num_nodes,
            self.transformerFeatures # The dense dimension
        )

        # 5. Create the hybrid sparse tensor representing messages
        sparse_message_tensor = torch.sparse_coo_tensor(
            indices=attention_weights_sparse.indices(), # Reuse the indices
            values=message_values,
            size=hybrid_size
        )

        # 6. Sum over the source dimension 'j' (dim=3) to aggregate messages
        # This collapses the j-th dimension, summing all messages for each (b, h, i)
        aggregated_messages_sparse = torch.sparse.sum(sparse_message_tensor, dim=3)

        # 7. Convert the result back to a dense tensor
        # The result has aggregated messages for each node.
        # Shape: [B, H, N, F]
        dense_output = aggregated_messages_sparse.to_dense()

        # ==============================================================================

        # 8. Reshape for the final layers
        # Shape: [B, H, N, F] -> [B, N, H*F]
        attentionOutputSparse = dense_output.permute(0, 2, 1, 3).reshape(
            batch_size, num_nodes, -1
        )

        print('Attention attentionOutputSparse.shape:', attentionOutputSparse.shape)

        if torch.any(torch.logical_not(torch.isclose(attentionOutputSparse, attentionOutput))):
            print('Attention output mismatch! ')
            print('attentionOutputSparse:', attentionOutputSparse)
            print('attentionOutput:', attentionOutput)
            raise ValueError('Attention output mismatch! ')

        if self.verbose:
            print('attentionOutput shape:', attentionOutput.shape)

        # ==============================================================================
        # --- Original Transformer Output ---

        # Project back to latent space
        attentionOutput = self.W_O(attentionOutputSparse)
        if self.verbose:
            print(f'===============================================================')
            print('attentionOutput shape after projection:', attentionOutput.shape)

        # Residual Connection and Layer Norm (Post-Attention)
        attentionOutput = inputTokensCurrent + attentionOutput  # Residual connection
        attentionOutput = self.layer_norm1(attentionOutput)

        if self.verbose:
            print(f'===============================================================')
            print('attentionOutput shape after residual connection and layer norm:', attentionOutput.shape)

        # Feedforward Network
        if self.verbose:
            print(f'===============================================================')
            print('Applying Feedforward Network (FFN)')
        ffnOutput = self.ffn(attentionOutput)
        if self.verbose:
            print(f'FFN output shape: {ffnOutput.shape}')

        # Residual Connection and Layer Norm (Post-FFN)
        transformerOutput = attentionOutput + ffnOutput  # Residual connection
        transformerOutput = self.layer_norm2(transformerOutput)
        if self.verbose:
            print(f'===============================================================')
            print('Final transformerOutput shape after residual connection and layer norm:', transformerOutput.shape)

        return transformerOutput        

