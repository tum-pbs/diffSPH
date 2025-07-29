import numpy as np
import torch
@torch.jit.script
def cpow(q, p : int):
    return torch.clamp(q, 0, 1)**p

@torch.jit.script
def k(q, dim: int = 2):  
    return (1-q**2)**3
@torch.jit.script
def dkdq(q, dim: int = 2):    
    return -6 * q * (1-q**2)**2
def d2kdq2(q, dim: int = 2):        
    return -6 * (5 * q**4 - 6 * q**2 + 1)
    
def d3kdq3(q, dim: int = 2):        
    return -6 * (20 * q**3 - 12 * q)

@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 35 / 16
    elif dim == 2: return 35 / (32 * np.pi)
    else: return 315 / (64 * np.pi)

@torch.jit.script
def kernel(rij, hij, dim : int = 2):
    return k(rij, dim) * C_d(dim) / hij**dim
    
@torch.jit.script
def kernelGradient(rij, xij, hij, dim : int = 2):
    return xij * (dkdq(rij, dim) * C_d(dim) / hij**(dim + 1))[:,None]

@torch.jit.script
def kernelLaplacian(rij, hij, dim : int = 2):
    return ((torch.where(rij > -1e-7, ((dim - 1) / (rij *hij + 1e-7 * hij)) * dkdq(rij + 1e-7, dim) * hij, 0) + d2kdq2(rij, dim)) * C_d(dim)) / hij**(dim + 2)