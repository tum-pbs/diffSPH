from typing import Union
from diffSPH.util import ParticleSet, ParticleSetWithQuantity
from diffSPH.neighborhood import DomainDescription, SparseCOO
from diffSPH.kernels import SPHKernel
from diffSPH.operations import sph_op
from diffSPH.operations import sph_op, SPHOperation
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState
from diffSPH.schemes.gasDynamics import CompressibleState
from diffSPH.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from diffSPH.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple

#divide by rho0
from diffSPH.util import getSetConfig 
from diffSPH.modules.viscosity import compute_Pi_v2
from diffSPH.modules.adaptiveSmoothingASPH import evaluateOptimalSupportOwen
from diffSPH.schemes.gasDynamics import CompressibleSystem, CompressibleUpdate, CompressibleState
import copy
from diffSPH.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product
import torch
from diffSPH.kernels import evalW, evalGradW, evalDerivativeW
from diffSPH.sphOperations.shared import getTerms, compute_xij
from diffSPH.modules.viscosity import compute_Pi
from diffSPH.neighborhood import evalKernelGradient
from diffSPH.kernels import getSPHKernelv2
from diffSPH.schemes.states.common import BasicState


def computeViscosity_Monaghan1997(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather,
        config: Dict = {}):
    
    i, j        = neighborhood[0].row, neighborhood[0].col
    gradW       = evalKernelGradient(neighborhood[1], supportScheme, True)

    x_ij = neighborhood[1].x_ij
    h_i = particles.supports[neighborhood[0].row]
    h_j = particles.supports[neighborhood[0].col]

    kernel_ = getSPHKernelv2(kernel)
    gradW = evalGradW(kernel_, x_ij, h_i, h_j, 'symmetric')

    Pi_ij       = compute_Pi(particles, particles, config['domain'], neighborhood[0], config, useJ = False, thermalConductivity=False)

    term        = (particles.masses[j] * Pi_ij).view(-1,1) * gradW

    return -scatter_sum(term, i, dim_size = particles.positions.shape[0], dim = 0)


def computeViscosity_Monaghan1997(
        particles: Union[BasicState, CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather,
        config: Dict = {}):
    
    i, j        = neighborhood[0].row, neighborhood[0].col
    Pi_ij       = compute_Pi_v2(particles, particles, config['domain'], neighborhood[0], config, useJ = False, thermalConductivity=False)

    v_ij = particles.velocities[j] - particles.velocities[i]

    return SPHOperation(
        particles,
        quantity = (Pi_ij * particles.densities[j]).view(-1,1) * v_ij,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation=Operation.Laplacian,
        supportScheme = supportScheme,
        gradientMode = GradientMode.Difference,
        laplacianMode= LaplacianMode.default,
        positiveDivergence=False
    )

from torch.utils.checkpoint import checkpoint

def computeViscosity_deltaSPH_inviscid(
        particles: Union[BasicState, CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather,
        config: Dict = {}):
    # with record_function("[SPH] - [Viscosity (deltaSPH)]"):
    i, j = neighborhood[0].row, neighborhood[0].col
    densities = particles.densities if isinstance(particles, CompressibleState) else particles.densities
    alpha = 0.01
    
    Pi_delta = checkpoint(compute_Pi, particles, particles, neighborhood[0].domain, neighborhood[0], {
        'diffusion':{
            'correctXi': True,
            'viscosityFormulation': 'Monaghan1992', # due to *h multiplication in delta SPH term
            'C_l': alpha,
        },
        'kernel': kernel,
        
        }, False, False, use_reentrant=True) # Comes divided by rho_i
    
    return -SPHOperation(
        particles,
        quantity = Pi_delta,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation=Operation.Gradient,
        supportScheme = supportScheme,
        gradientMode = GradientMode.Naive,
        # gradientRenormalizationMatrix = particles.gradCorrectionMatrices
    ) * config['fluid']['rho0']  # Multiply with rho0 for alpha scaling, see delta SPH paper



    i, j = neighborhood[0].row, neighborhood[0].col
    densities = particles.densities if isinstance(particles, CompressibleState) else particles.densities
    alpha = getSetConfig(config, 'diffusion', 'alpha', 0.01)
    
    Pi_delta = compute_Pi_v2(particles, particles, neighborhood[0].domain, neighborhood[0], {
        'diffusion':{
            'correctXi': True,
            'viscosityFormulation': 'Monaghan1992', # due to *h multiplication in delta SPH term
            'C_l': alpha,
        },
        'kernel': kernel,
        
        }, useJ = False, thermalConductivity=False) # Comes divided by rho_i
    v_ij = particles.velocities[j] - particles.velocities[i]

    return SPHOperation(
        particles,
        quantity = (Pi_delta * particles.densities[j]).view(-1,1) * v_ij,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation=Operation.Laplacian,
        supportScheme = SupportScheme.Symmetric,
        gradientMode = GradientMode.Difference,
        laplacianMode= LaplacianMode.default,
        positiveDivergence=False
    ) * config['fluid']['rho0'] 
