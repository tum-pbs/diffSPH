import numpy as np
import torch
@torch.jit.script
def cpow(q, p : int):
    return torch.clamp(q, 0, 1)**p

@torch.jit.script
def k(q, dim: int = 2):   
    return cpow(1-q, 4) - 5 * cpow(3/5 - q, 4) + 10 * cpow(1/5 - q, 4)
@torch.jit.script
def dkdq(q, dim: int = 2):    
    return -4 * cpow(1-q, 3) + 20 * cpow(3/5 - q, 3) - 40 * cpow(1/5 - q, 3)
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    return 12 * cpow(1-q,2) -60 * cpow(3/5 - q,2) + 120 * cpow(1/5 - q,2)
@torch.jit.script
def d3kdq3(q, dim: int = 2):
    return -24 * (1-q) + 120 * cpow(3/5 - q, 1) - 240 * cpow(1/5 - q, 1)
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 5**5/768
    elif dim == 2: return 5**6 * 3 / (2398 * np.pi)
    else: return 5**6/ (512 * np.pi)

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
    if dim == 1: return 1.936492
    elif dim == 2: return 1.977173
    else: return 2.018932

@torch.jit.script
def packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.203 