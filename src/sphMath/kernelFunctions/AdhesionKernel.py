import numpy as np
import torch
@torch.jit.script
def cpow(q, p : int):
    return torch.clamp(q, 0, 1)**p


@torch.jit.script
def k(q, dim: int = 2):  
    return torch.where(q > 0.5, ((-4 * q**2 + 6 * q - 2))**(1/4), 0)
@torch.jit.script
def dkdq(q, dim: int = 2):    
    return torch.where(q > 0.5, (6 - 8 * q) / (4 * (-4 * q**2 + 6 *q -2)** (3/4)),0)
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    return torch.where(q > 0.5, (16 *q**2 - 24 *q + 11) / (8 *(-4 * q**2 + 6 *q -2)**(3/4) * (2 *q**2 - 3*q + 1)),0)
@torch.jit.script
def d3kdq3(q, dim: int = 2):
    return torch.where(q > 0.5, (48 - 32 * q) / (4 * (-4 * q**2 + 6 *q -2)**(3/4) * (2 *q**2 - 3*q + 1)) ,0)
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return 0.007
    elif dim == 2: return 0.007
    else: return 0.007

@torch.jit.script
def kernel(rij, hij, dim : int = 2):
    return k(rij, dim) * C_d(dim) / hij**dim
    
@torch.jit.script
def kernelGradient(rij, xij, hij, dim : int = 2):
    return xij * (dkdq(rij, dim) * C_d(dim) / hij**(dim + 1))[:,None]

@torch.jit.script
def kernelLaplacian(rij, hij, dim : int = 2):
    return ((torch.where(rij > -1e-7, ((dim - 1) / (rij *hij + 1e-7 * hij)) * dkdq(rij + 1e-7, dim) * hij, 0) + d2kdq2(rij, dim)) * C_d(dim)) / hij**(dim + 2)