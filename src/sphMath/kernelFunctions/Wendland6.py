import numpy as np
import torch
@torch.jit.script
def cpow(q, p : int):
    return torch.clamp(q, 0, 1)**p


@torch.jit.script
def k(q, dim: int = 2):        
    if dim == 1:
        return cpow(1 - q, 7) * (1 + 7 * q + 19 * q**2 + 21 * q**3)
    else:
        return cpow(1 - q, 8) * (1 + 8 * q + 25 * q**2 + 32 * q**3)
@torch.jit.script
def dkdq(q, dim: int = 2):        
    if dim == 1:
        return -6 * q * (35 * q**2 + 18 * q + 3) * cpow(1 - q,6)
    else:
        return -22 * q * (16 * q**2 + 7 *q + 1) * cpow(1 - q,7)
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    if dim == 1:
        return 18 * (105 * q**3 + 13 * q**2 - 5 *q -1) * cpow(1-q,5)
    else:
        return 22 * (160 *q**3 + 15 * q**2 - 6*q -1) * cpow(1-q,6)
@torch.jit.script
def d3kdq3(q, dim: int = 2):
    if dim == 1:
        return -1008 * q * (15 * q**2 - 4 * q - 1) * cpow(1 - q,4)
    else:
        return -1584 * q * (20 * q**2 - 5 * q - 1) * cpow(1 - q,5)
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 55/32
    elif dim == 2: return 78 / (7 * np.pi)
    else: return 1365/ (64 * np.pi)

@torch.jit.script
def kernel(rij, hij, dim : int = 2):
    return k(rij, dim) * C_d(dim) / hij**dim
    
@torch.jit.script
def kernelGradient(rij, xij, hij, dim : int = 2):
    return xij * (dkdq(rij, dim) * C_d(dim) / hij**(dim + 1))[:,None]

@torch.jit.script
def kernelLaplacian(rij, hij, dim : int = 2):
    return ((torch.where(rij > -1e-7, ((dim - 1) / (rij *hij + 1e-7 * hij)) * dkdq(rij + 1e-7, dim) * hij, 0) + d2kdq2(rij, dim)) * C_d(dim)) / hij**(dim + 2)

@torch.jit.script # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def kernelScale(dim: int = 2):
    if dim == 1: return 2.207940
    elif dim == 2: return 2.415230
    else: return 2.449490


@torch.jit.script
def packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.866