import numpy as np
import torch
@torch.jit.script
def cpow(q, p : int):
    return torch.clamp(q, 0, 1)**p


@torch.jit.script
def k(q, dim: int = 2):  
    return -1/2 * q**3 + q**2 +1/2 / (q) - 1
def dkdq(q, dim: int = 2):    
    return -3/2 *q**2 - 1 /(2 * q**2) + 2 * q
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    return q**(-3) - 3* q + 2
@torch.jit.script
def d3kdq3(q, dim: int = 2):
    return -3 * q**(-4) - 3

@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 15 / (2 * np.pi)
    elif dim == 2: return 10 / (9 * np.pi)
    else: return 15 / (2 * np.pi)

@torch.jit.script
def kernel(rij, hij, dim : int = 2):
    return torch.where(rij > 1e-5, k(rij, dim) * C_d(dim) / hij**dim,0)
    
@torch.jit.script
def kernelGradient(rij, xij, hij, dim : int = 2):
    return torch.where(rij.view(-1,1) > 1e-5, xij * (dkdq(rij, dim) * C_d(dim) / hij**(dim + 1))[:,None],torch.zeros_like(xij))

@torch.jit.script
def kernelLaplacian(rij, hij, dim : int = 2):
    return ((torch.where(rij > -1e-7, ((dim - 1) / (rij *hij + 1e-7 * hij)) * dkdq(rij + 1e-7, dim) * hij, 0) + d2kdq2(rij, dim)) * C_d(dim)) / hij**(dim + 2)