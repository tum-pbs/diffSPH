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
from sphMath.kernels import KernelType, Kernel, Kernel_Gradient, Kernel_Scale


def computeDeltaShifting(particles, domain, kernel: KernelType, sparseNeighborhood, config):
    rho0 = config.get('fluid', {}).get('rho0', 998)
    dx = config.get('particle', {}).get('dx', torch.pow(particles.masses / rho0, 1/particles.positions.shape[1]).mean().cpu().item())
    
    dx_tensor = torch.zeros_like(particles.positions)
    dx_tensor[:,0] = dx  / Kernel_Scale(kernel, particles.positions.shape[1])
    # dx_tensor[:,0] = dx  #/ Kernel_Scale(kernel, particles.positions.shape[1])
    
    
    rij, xij = computeDistanceTensor(sparseNeighborhood, False, mode = 'gather')
    
    i,j = sparseNeighborhood.row, sparseNeighborhood.col
    W_0 = Kernel(kernel, dx_tensor, particles.supports)
    k = Kernel(kernel, xij, particles.supports[i]) / W_0[i]
    gradK = Kernel_Gradient(kernel, xij, particles.supports[i])
    
    # print(W_0, k.max())

    R = config.get('shifting', {}).get('R', 0.25)
    n = config.get('shifting', {}).get('n', 4)
    term = (1 + R * torch.pow(k, n))
    densityTerm = particles.masses[j] / (particles.densities[i] + particles.densities[j])
    phi_ij = 1
    
    scalarTerm = term * densityTerm * phi_ij
    shiftAmount = scatter_sum(scalarTerm.view(-1,1) * gradK, i, dim = 0, dim_size = particles.positions.shape[0])
    
    CFL = config.get('shifting', {}).get('CFL', 0.3)
    if config.get('shifting', {}).get('computeMach', True) == False:
        Ma = 0.1
    else:
        Ma = torch.amax(torch.linalg.norm(particles.velocities, dim = -1)) / particles.soundspeeds
    shiftScaling = -CFL * Ma * (particles.supports / Kernel_Scale(kernel, particles.positions.shape[1]) * 2)**2
    
    # print(f'W_0: {W_0.min()}, {W_0.max()}, {W_0.mean()}, has nan: {torch.isnan(W_0).any()}, has inf: {torch.isinf(W_0).any()}')
    # print(f'k: {k.min()}, {k.max()}, {k.mean()}, has nan: {torch.isnan(k).any()}, has inf: {torch.isinf(k).any()}')
    # print(f'gradK: {gradK.min()}, {gradK.max()}, {gradK.mean()}, has nan: {torch.isnan(gradK).any()}, has inf: {torch.isinf(gradK).any()}')
    # print(f'densityTerm: {densityTerm.min()}, {densityTerm.max()}, {densityTerm.mean()}, has nan: {torch.isnan(densityTerm).any()}, has inf: {torch.isinf(densityTerm).any()}')
    # print(f'scalarTerm: {scalarTerm.min()}, {scalarTerm.max()}, {scalarTerm.mean()}, has nan: {torch.isnan(scalarTerm).any()}, has inf: {torch.isinf(scalarTerm).any()}')
    # print(f'shiftAmount: {shiftAmount.min()}, {shiftAmount.max()}, {shiftAmount.mean()}, has nan: {torch.isnan(shiftAmount).any()}, has inf: {torch.isinf(shiftAmount).any()}')
    # print(f'shiftScaling: {shiftScaling.min()}, {shiftScaling.max()}, {shiftScaling.mean()}, has nan: {torch.isnan(shiftScaling).any()}, has inf: {torch.isinf(shiftScaling).any()}')
    
    return shiftScaling.view(-1,1) * shiftAmount
    