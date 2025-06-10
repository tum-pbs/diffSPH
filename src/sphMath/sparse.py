from torch.profiler import profile, record_function, ProfilerActivity
from sphMath.neighborhood import filterNeighborhood, coo_to_csr
import torch
from sphMath.modules.renorm import computeCovarianceMatrices
from sphMath.neighborhood import computeDistanceTensor
from sphMath.sphOperations.shared import scatter_sum
from sphMath.operations import sph_op
# Maronne surface detection
from sphMath.modules.surfaceDetection import computeNormalsMaronne, detectFreeSurfaceMaronne, computeColorField, detectFreeSurfaceColorFieldGradient, detectFreeSurfaceBarecasco, expandFreeSurfaceMask, computeLambdaGrad, detectFreeSurfaceColorField
import numpy as np
from sphMath.modules.density import computeDensity


from typing import Tuple
@torch.jit.script
def matvec_sparse_coo(A: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int], x):
    return scatter_sum(A[0] * x[A[1][1]], A[1][0], dim = 0, dim_size = x.shape[0]) 

@torch.jit.script
def rmatvec_sparse_coo(A: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int], x):
    return scatter_sum(A[0] * x[A[1][0]], A[1][1], dim = 0, dim_size = x.shape[0]) 

@torch.jit.script
def make_id(A : Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int]):
    M_precond = A[0].new_ones(A[2])
    M_i = torch.arange(A[2]).to(M_precond.device).to(torch.int64)
    M_j = torch.arange(A[2]).to(M_precond.device).to(torch.int64)

    return (M_precond, (M_i, M_j), A[2])

@torch.jit.script
def _get_atol_rtol(name:str, b_norm:float, atol:float=0., rtol:float=1e-5):
    """
    A helper function to handle tolerance normalization
    """
    # if atol == 'legacy' or atol is None or atol < 0:
    #     msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
    #            "if set, `atol` must be a real, non-negative number.")
    #     raise ValueError(msg)

    atol = max(float(atol), float(rtol) * float(b_norm))

    return atol, rtol
@torch.jit.script
def _get_tensor_eps(
    x: torch.Tensor,
    eps16: float = torch.finfo(torch.float16).eps,
    eps32: float = torch.finfo(torch.float32).eps,
    eps64: float = torch.finfo(torch.float64).eps,
) -> float:
    if x.dtype == torch.float16:
        return eps16
    elif x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
    else:
        raise RuntimeError(f"Expected x to be floating-point, got {x.dtype}")
    
    