import torch
from typing import Optional, Union, Tuple
from diffSPH.kernels import SPHKernel

# @torch.jit.script
# def mod(x, min : float, max : float):
#     return torch.where(torch.abs(x) > (max - min) / 2, torch.sgn(x) * ((torch.abs(x) + min) % (max - min) + min), x)

from diffSPH.util import mod

def moduloDistance(xij, periodicity, min, max):
    return torch.stack([xij[:,i] if not periodic else mod(xij[:,i], min[i], max[i]) for i, periodic in enumerate(periodicity)], dim = -1)

def mod_distance(xi, xj, 
                 periodicity : Union[bool, torch.Tensor], 
                 minExtent : torch.Tensor, maxExtent : torch.Tensor):
    xij = xi - xj
    periodic = periodicity if isinstance(periodicity, torch.Tensor) else torch.tensor([periodicity for _ in range(xij.shape[1])]).to(xij.device)
    return moduloDistance(xij, periodic, minExtent, maxExtent)

def product(a,  b):
    if len(a.shape) == 1 and len(b.shape) == 1:
        return a * b
    elif len(a.shape) == 1 and len(b.shape) != 1:
        return torch.einsum('n, n... -> n...', a, b)
    elif len(a.shape) != 1 and len(b.shape) == 1:
        return torch.einsum('n..., n -> n...', a, b)
    else:
        if a.shape == b.shape:
            return a * b
        raise ValueError(f"Invalid shapes {a.shape} and {b.shape}")
    
    
# ------ Beginning of scatter functionality ------ #
# Scatter summation functionality based on pytorch geometric scatter functionality
# This is included here to make the code independent of pytorch geometric for portability
# Note that pytorch geometric is licensed under an MIT licenses for the PyG Team <team@pyg.org>
@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

from torch.profiler import record_function

@torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    with record_function("scatter_sum"):
        index = broadcast(index, src, dim)
        if out is None:
            size = list(src.size())
            if dim_size is not None:
                size[dim] = dim_size
            elif index.numel() == 0:
                size[dim] = 0
            else:
                size[dim] = int(index.max()) + 1
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
            return out.scatter_add_(dim, index, src)
        else:
            return out.scatter_add_(dim, index, src)
# ------ End of scatter functionality ------ #

