import torch
from .activation import getActivationLayer
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