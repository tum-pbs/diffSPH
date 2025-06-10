
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product
import torch
from typing import Union
from sphMath.util import ParticleSet, ParticleSetWithQuantity
from sphMath.neighborhood import DomainDescription, SparseCOO
from sphMath.kernels import SPHKernel
from sphMath.operations import sph_op
from sphMath.kernels import evalW, evalGradW, evalDerivativeW
from sphMath.sphOperations.shared import getTerms, compute_xij
from sphMath.modules.viscosity import compute_Pi
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.operations import sph_op, SPHOperation
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme, evalKernelGradient
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple
from sphMath.util import getSetConfig
from sphMath.kernels import getSPHKernelv2




# def computeMonaghan1997Dissipation(
#         particles: Union[CompressibleState, WeaklyCompressibleState],
#         kernel: SPHKernel,
#         neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
#         supportScheme: SupportScheme = SupportScheme.Gather,
#         config: Dict = {}):
#     
#     f = getSetConfig(config, 'diffusion', 'thermalConductivity', 1)
#     i, j                        = neighborhood[0].row, neighborhood[0].col


#     x_ij, r_ij = neighborhood[1].x_ij, neighborhood[1].r_ij
#     h_i = particles.supports[neighborhood[0].row]
#     h_j = particles.supports[neighborhood[0].col]

#     kernel_ = getSPHKernelv2(kernel)
#     W_symm = evalDerivativeW(kernel_, x_ij, h_i, h_j, 'symmetric')
#     # W_symm = torch.linalg.norm(evalKernelGradient(neighborhood[1], supportScheme, True), dim=-1)

#     Pi_v       = compute_Pi(particles, particles, config['domain'], neighborhood[0], config, useJ = False, thermalConductivity=False)
#     Pi_u       = compute_Pi(particles, particles, config['domain'], neighborhood[0], config, useJ = False, thermalConductivity=True)

#     u_ij        = particles.velocities[neighborhood[0].row] - particles.velocities[neighborhood[0].col]
#     ux_ij       = torch.einsum('ij,ij->i', u_ij, x_ij)
#     mu_ij = ux_ij / (r_ij + 1e-14 * h_i)
#     mu_ij[ux_ij > 0] = 0

#     Omega_ij = -f * Pi_u + 1/2 * Pi_v * mu_ij
#     # SPHOperation multiplies with m[j]/rho[j] so need to premultiply rho[j]
#     term = particles.masses[j] * Omega_ij * W_symm

#     # print('--------------------------------------')
#     # print('Pi_v', Pi_v)
#     # print('Pi_u', Pi_u)
#     # print('mu_ij', mu_ij)
#     # print('Omega_ij', Omega_ij)

#     return scatter_sum(term, i, dim_size = particles.positions.shape[0], dim = 0)
    
from sphMath.modules.viscosity import compute_Pi_v2

def computeThermalConductivity_Monaghan1997(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather,
        config: Dict = {}):
    
    f = getSetConfig(config, 'diffusion', 'thermalConductivity', 1)
    i, j        = neighborhood[0].row, neighborhood[0].col

    #includes u_i - u_j
    Pi_u       = compute_Pi_v2(particles, particles, config['domain'], neighborhood[0], config, useJ = False, thermalConductivity=True)
    # minus here as Pi includes a - sign, i.e., it computes the correct u_j - u_i for a difference gradient
    # multiply with rho[j] to counteract the SPHOperation with m[j]/rho[j]
    term = -f * Pi_u

    u_ij = particles.internalEnergies[j] - particles.internalEnergies[i]

    result = -SPHOperation(
        particles,
        quantity = term * particles.densities[j] * u_ij,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation=Operation.Laplacian,
        supportScheme = supportScheme,
        gradientMode = GradientMode.Difference,
        laplacianMode= LaplacianMode.default,
        positiveDivergence=False
    )
    return result

def computeThermalDissipation_Monaghan1997(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather,
        config: Dict = {}):
    

    Pi_v       = compute_Pi(particles, particles, config['domain'], neighborhood[0], config, useJ = False, thermalConductivity=False)

    x_ij = neighborhood[1].x_ij

    # we have to be very careful here with the signs!
    # the gradient computes (if done mathematically correctly) u_j - u_i
    # the equations in most papers read u_i - u_j!
    # so the condition of v_ab \dot x_ab <  0 then 0 for a 'positive' divergence term
    # becomes v_j - v_i \dot x_ab > 0 then 0!
    # the SPH Operation computes the 'correct' term so we need to provide it with a term with flipped sign

    i, j = neighborhood[0].row, neighborhood[0].col
    u_ij        = particles.velocities[j] - particles.velocities[i]
    ux_ij       = torch.einsum('ij,ij->i', u_ij, x_ij)
    term = 1/2 *  Pi_v

    result = SPHOperation(
        particles,
        quantity = particles.densities[j] * term * ux_ij,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        supportScheme = supportScheme,
        operation=Operation.Laplacian,
        gradientMode = GradientMode.Difference,
        laplacianMode= LaplacianMode.default,
        positiveDivergence=False
    )
    return result



def computeMonaghan1997Dissipation(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather,
        config: Dict = {}):
    return computeThermalConductivity_Monaghan1997(particles, kernel, neighborhood, supportScheme, config) + computeThermalDissipation_Monaghan1997(particles, kernel, neighborhood, supportScheme, config)