import numpy as np
import torch
@torch.jit.script
def cpow(q, p : int):
    return torch.clamp(q, 0, 1)**p


@torch.jit.script
def k(q, dim: int = 2):  
    return cpow(1-q, 3) - 4 * cpow(1/2 - q,3)
@torch.jit.script
def dkdq(q, dim: int = 2):    
    return -3 * cpow(1-q, 2) + 12 * cpow(1/2 - q,2)
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    return  6 * (1-q) + torch.where(q >= 0.5,0, - 24 * (1/2 - q))
@torch.jit.script
def d3kdq3(q, dim: int = 2):
    return -6 + torch.where(q >= 0.5,0, 24)
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 8/3
    elif dim == 2: return 80 / (7 * np.pi)
    else: return 16/ (np.pi)

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
    if dim == 1: return 1.732051
    elif dim == 2: return 1.778002
    else: return 1.825742

@torch.jit.script
def packingRatio(): # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations Table 2
    return 1.292 * 1.0175 # Correction to match DJ Price 2012 with 57.9 neighbors in 3D see page 776
    return 1.292 # 1.181 for Astrophysics