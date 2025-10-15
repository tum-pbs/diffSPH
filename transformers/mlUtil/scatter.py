import torch
import torch.nn as nn
try:
    import torch_geometric
except ImportError:
    torch_geometric = None

from .networkUtil import verbosePrint, verboseBannerPrint


def scatter_sum(input, edge_indices, dim=0, dim_size=None):

    if torch_geometric is not None:
        return torch_geometric.utils.scatter(input, edge_indices[1], dim=dim, dim_size=dim_size, reduce='sum')
    else:
        raise ImportError('torch_geometric is not installed, cannot use scatter_sum')