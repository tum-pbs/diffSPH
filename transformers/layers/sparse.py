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
from typing import Optional

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