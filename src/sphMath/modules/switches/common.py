from typing import Union, Optional
from dataclasses import dataclass
import torch

from sphMath.neighborhood import DomainDescription, SparseCOO
from sphMath.kernels import SPHKernel
from sphMath.operations import sph_op
from sphMath.modules.compressible import CompressibleState
from sphMath.sphOperations.shared import getTerms, compute_xij, scatter_sum
# from sphMath.modules.switches.cullenDehnenSwitch import computeShearTensor, computeM
from sphMath.modules.compressible import verbosePrint
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme, evalKernel, evalKernelGradient
from sphMath.operations import SPHOperation
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple
from sphMath.kernels import KernelType


# Correction Term from CRKSPH
def computeM(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    kernelValues = neighborhood[1]
    # CHECK THIS, does it really use the term without densities?
    return -SPHOperation(particles, kernelValues.x_ij * particles.densities[neighborhood[0].col].view(-1,1), kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Naive)
    
    # i, j        = neighborhood.row, neighborhood.col
    # h_i         = particles_a.supports[i]
    # m_j         = particles_b.masses[j]
    # x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)
    
    # gradW_i = kernel.jacobian(x_ij, h_i)   

    # dyadicProduct = m_j.view(-1,1,1) * torch.einsum('ij,ik->ijk', x_ij, gradW_i)
    
    # return -scatter_sum(dyadicProduct, i, dim = 0, dim_size = particles_a.positions.shape[0])

def computeShearTensor(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict,
        correctionMatrix):
    Vs = SPHOperation(particles, particles.velocities, kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Difference)
    
    # i, j        = neighborhood.row, neighborhood.col
    # v_i, v_j    = particles_a.velocities[i], particles_b.velocities[j]
    # m_j         = particles_b.masses[j]
    # rho_j       = particles_b.densities[j]
    # h_i         = particles_a.supports[i]
    # x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)

    # gradW_i = kernel.jacobian(x_ij, h_i)
    # v_ij = v_i - v_j
    
    # dyadicProduct = -(m_j / rho_j).view(-1,1,1) * torch.einsum('ij,ik->ijk', v_ij, gradW_i) # don't forget the minus sign! (B2) computes D_i in the Cullen paper but that is not the quantity we are interested in. Instead we care about v\otimes\nabla W_ij which requires a minus sign.
    
    # Vs = scatter_sum(dyadicProduct, i, dim = 0, dim_size = particles_a.positions.shape[0])
    if config.get('correctVelocityGradient', False):
        if correctionMatrix is not None:
            Vs = torch.einsum('ijk, ikl -> ijl', correctionMatrix, Vs)
    trace = torch.einsum('...ii', Vs)
    
    traces = torch.eye(Vs.shape[1], device=Vs.device) * trace.view(-1, 1, 1) / Vs.shape[1]
    
    Shear = (Vs + Vs.transpose(1,2))/2 - traces
    Rotation = (Vs - Vs.transpose(1,2)) / 2
    
    return trace, Shear, Rotation

def computeDivergence(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    viscousDivergenceScheme = config.get('viscousDivergenceScheme', 'naive')
    if viscousDivergenceScheme == 'naive':
        return SPHOperation(particles, particles.velocities, kernel, neighborhood[0], neighborhood[1], operation = Operation.Divergence, supportScheme=supportScheme, divergenceMode=DivergenceMode.div, gradientMode = GradientMode.Difference)
        
        # return sph_op(psphState, psphState, domain, wrappedKernel, actualNeighbors, 'superSymmetric', 'divergence', 'difference', quantity=(psphState.velocities, psphState.velocities))
    elif viscousDivergenceScheme == 'cullen':
        if config.get('correctVelocityGradient', False):
            M = computeM(particles, kernel, neighborhood, supportScheme, config)
            M_inv = torch.linalg.pinv(M)        
        else:
            M = None
            M_inv = None
        div, S, Rot = computeShearTensor(particles, kernel, neighborhood, supportScheme, config, M_inv)
        return div
    