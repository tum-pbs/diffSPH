This is a message passing layer that is part of the transformer architecture, however, it is expanded in functionality
to match what is normally expected from a graph neural network layer.

The normal message passing inputs are:
- queryTokens: the tokens for which we want to compute the new representation (shape: [batch_size, num_nodes_current, latentSpaceSize]) [i]
- keyTokens: the tokens that provide the context (shape: [batch_size, num_nodes_neighbor, latentSpaceSize]) [j]
- edge_index: the indices of the edges in the sparse neighborhood (shape: [2, num_edges]) where the first row are the indices for queryTokens and the second row for keyTokens
- edge_attr: the features associated with each edge (shape: [num_edges, edgeFeatureSize]) (spatial information mostly)

By adding an attention Mechanism, we add an aditional input:
- edge_attention: the values to modulate the attention scores (shape: [num_edges, num_attention_heads]) (optional)

We also add a shepard like scaling value to support spatial normalization
- S_k: the shepard values to scale the attention scores (shape: [num_edges]) (optional)

The output is:
- outputTokens: the new representation of the query tokens (shape: [batch_size, num_nodes_current, latentSpaceSize])
The output tokens have the same shape as the input query tokens, but their values have been updated based on the message passing mechanism.

A vanilla message-passing layer for a transformer will work as follows:
1. project the key tokens [b,n,L] using W_V linearly to the transformer shape as node values: [b,n,L] -> [b,n,h,t]
2. gather the values for the key tokens using edge_index[1]: [b,n,h,t] -> [ne,h,t]
3. multiply the edge values with the attention values (expand the attention values to match) [ne,h,t]x[ne,h,1] -> [ne,h,t]
4. scatter sum up the result [ne,h,t]->[b,n,h,t]
5. project the resulting values linearly using W_o after flattening: [b,n,h*t] -> [b,n,L]

A vanilla message-passing GNN ( a la MP-PDE ) will work as follows:
1. gather the query and key tokens features using edge_index, i.e., q_i = query[edge_index[0]], q_j = query[edge_index[1]]: [b,n,L] -> [ne,L]
2. concatenate edge features with node features: [ne,L]x2+[ne,E] -> [ne,2*L+E]
3. feed the concatenated features into an MLP: [ne,2*L+E]->[ne,L]
4. scatter sum up the result [ne,L]->[b,n,L]

A vanilla CConv GNN will work as follows:
1. Gather up the key tokens using the edge_index q_j = query[edge_index[1]]: [b,n,L] -> [ne,L]
2. Compute the basis terms from the edge_attrs with b basis terms per dim: [ne,d]-> [ne,b,b]
3. Construct a weight matrix that maps L to L features, i.e., [L,L], conditioned on [b,b]: [ne,b,b]->[ne,L,L]
4. Apply the weight matrix on the incoming features: [ne,L,L].[ne,L] -> [ne,L]
5. scatter sum up the result [ne,L]->[b,n,L]

Conceptually they appear different to each other but some of the steps can be collected together. If we add an additional linear input encoder from the latent space L to some transformer space T (described using h multi-heads and t features per head) we can change every process to have the same start (crucially, we can also assume that L is divisble by h and skip the linear encoding by reshaping [b,n,L] to [b,n,h,L//h]):
1. Project the neighboring/key token values [b,n,L] using a linear mapping W_V to the internal shape state as node values
2. gather the query and key tokens features using edge_index, i.e., q_i = query[edge_index[0]], q_j = query[edge_index[1]]: [b,n,L] -> [ne,L] (q_j might be unused)

Analogously we can unify the last few steps as well:
4. scatter sum up the result [ne,h,t]->[b,n,h,t]
5. project the resulting values linearly using W_o after flattening: [b,n,h*t] -> [b,n,L]

This leaves the central steps, which we can handle via branching and also add edge gating steps to all networks:
1. Project to  [b,n,h,t]
2. Gather to q_i, q_j [ne,h,t]
3. Apply message generation logic:
    - Transformer:
        - Nothing to do
    - GNN:
        - gather the relevant features as inputs, e.g., concatenate [q_i, q_j, edge_attr, edge_attention]
        - feed the concatenated features into an MLP: [ne,C]->[ne,h*t]
        - Reshape to match transformer style [ne,h,t]
    - CConv:
        - Construct a weight matrix that maps L to L features, i.e., [h*t,h*t], conditioned on [b,b]: [ne,b,b]->[ne,h*t,h*t]
        - Apply the weight matrix on the incoming features: [ne,h*t,h*t].[ne,h*t] -> [ne,h*t]
        - Reshape to match transformer style [ne,h,t]
4. Apply edge_attention if given (expand the attention values to match) [ne,h,t]x[ne,h,1] -> [ne,h,t]
5. Apply edge_weighting using s_k [ne,h,t]x[ne] -> [ne,h,t]
6. Compute the window function if required [ne,d] -> [ne] and apply it [ne,h,t]x[ne] -> [ne,h,t]
7. Apply edge gating (with optional basis encodes): [ne,d]->[ne,b*b], apply linear projection [ne,b*b]->[ne,h] and apply to messages [ne,h,t]x[ne,h]->[ne,h,t]
8. scatter sum up the result [ne,h,t]->[b,n,h,t]
9. project the resulting values linearly using W_o after flattening: [b,n,h*t] -> [b,n,L]