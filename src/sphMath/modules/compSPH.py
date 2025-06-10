import torch
from sphMath.util import ParticleSet, DomainDescription
from sphMath.sampling import buildDomainDescription, sampleRegularParticles
from sphMath.neighborhood import buildNeighborhood, coo_to_csr, SparseCOO
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product
from sphMath.kernels import SPHKernel, evalW, evalGradW, evalDerivativeW
# from sphMath.modules.compressible import CompressibleState

from sphMath.schemes.gasDynamics import CompressibleState, CompressibleUpdate
from typing import Union
from typing import List
# from sphMath.modules.compressible import CompressibleState, CompressibleUpdate, verbosePrint
import numpy as np
from typing import Tuple

from sphMath.sphOperations.shared import getTerms, compute_xij
from sphMath.modules.viscosity import compute_Pi
from sphMath.schemes.baseScheme import verbosePrint
from sphMath.kernels import Kernel_xi
from sphMath.kernels import KernelType
from sphMath.neighborhood import evalKernel, evalKernelGradient
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.operations import SPHOperation
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple


def compute_aij(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config):
    
    i, j = neighborhood[0].row, neighborhood[0].col
    gradW_i, gradW_j = evalKernelGradient(neighborhood[1], SupportScheme.Symmetric, False)
    
    P = (particles.pressures / particles.densities**2 / particles.omega).view(-1,1)
    Q_i = compute_Pi(particles, particles, neighborhood[0].domain, neighborhood[0], config, useJ = False).view(-1,1)# * particles.densities[j]
    Q_j = compute_Pi(particles, particles, neighborhood[0].domain, neighborhood[0], config, useJ = True).view(-1,1)# * particles.densities[j]
        
    viscosityTerm = 1/2 * (Q_i * gradW_i + Q_j * gradW_j) * particles.masses[j].view(-1,1)
    pressureTerm =      (P[i] * gradW_i + P[j] * gradW_j) * particles.masses[j].view(-1,1)
    
    
    # i, j                        = neighborhood.row, neighborhood.col
    # rho_i, rho_j, rho_bar       = getTerms(i, j, (particles_a.densities,    particles_b.densities))
    # c_i, c_j, c_bar             = getTerms(i, j, (particles_a.soundspeeds,  particles_b.soundspeeds))
    # h_i, h_j, h_bar             = getTerms(i, j, (particles_a.supports,     particles_b.supports))
    # m_i, m_j                    =                 particles_a.masses[i],    particles_b.masses[j]
    # omega_i, omega_j, omega_bar = getTerms(i, j, (particles_a.omega,        particles_b.omega))

    # x_ij, r_ij                  = compute_xij(particles_a, particles_b, neighborhood, domain)
    # gradW_i = evalGradW(kernel, x_ij, h_i, h_j, 'symmetric')
    # gradW_j = evalGradW(kernel, x_ij, h_i, h_j, 'symmetric')
    
    # Q_i = compute_Pi(particles_a, particles_b, domain, neighborhood, solverConfig, useJ = False).view(-1,1)
    # Q_j = compute_Pi(particles_a, particles_b, domain, neighborhood, solverConfig, useJ = True).view(-1,1)
    # P_i = (particles_a.pressures[i] / rho_i**2 / omega_i).view(-1,1)
    # P_j = (particles_b.pressures[j] / rho_j**2 / omega_j).view(-1,1)
    
    # viscosityTerm   = 1/2 * (Q_i * gradW_i + Q_j * gradW_j) * m_j.view(-1,1)
    # pressureTerm    =       (P_i * gradW_i + P_j * gradW_j) * m_j.view(-1,1)
        
    return -viscosityTerm, -pressureTerm

def compSPH_acceleration(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config):
    i               = neighborhood[0].row    
    av_ij, ap_ij    = compute_aij(particles, kernel, neighborhood, supportScheme, config)
    av              = scatter_sum(av_ij, i, dim = 0, dim_size = particles.positions.shape[0])
    ap              = scatter_sum(ap_ij, i, dim = 0, dim_size = particles.positions.shape[0])

    return ap + av, ap_ij, av_ij

def compSPH_dudt(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config):
    i, j                        = neighborhood[0].row, neighborhood[0].col 

    gradW_i = evalKernelGradient(neighborhood[1], SupportScheme.Symmetric, True)
    v_ij    = particles.velocities[i] - particles.velocities[j]

    dot     = torch.einsum('ij, ij -> i', v_ij, gradW_i) 

    Pi_i    = compute_Pi(particles, particles, config['domain'], neighborhood[0], config, useJ = False)

    P     = particles.pressures / (particles.densities**2 * particles.omega)
    # Does not use the symmetrized variant, see E.3 in the CRKSPH paper
    term    = particles.masses[j] * (P[i] + 1/2 * Pi_i) * dot 
    # SPHOperation multiplies with m[j]/rho[j] so need to premultiply rho[j]
    # term = particles.densities[j] * (P[i] + 1/2 * Pi_i) * dot 

    return scatter_sum(term, i, dim = 0, dim_size = particles.positions.shape[0])
    


def computeDeltaU(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict,
        dt: float,
        v_halfstep: torch.Tensor,
        ap_ij: torch.Tensor, 
        av_ij: torch.Tensor, 
        f_ij: torch.Tensor):
    f_term          = config.get('energyScheme', 'equalWork')
    i, j            = neighborhood[0].row, neighborhood[0].col
    v_i, v_j        = v_halfstep[i], v_halfstep[j]
    m_i             = particles.masses[i]

    v_ji            = v_j - v_i    
    a_ij            = ap_ij + av_ij
    
    deltaE_thermal      = m_i * torch.einsum('ij, ij -> i', v_ji, a_ij ) * dt
    deltaE_pressure     = m_i * torch.einsum('ij, ij -> i', v_ji, ap_ij) * dt
    deltaE_viscosity    = m_i * torch.einsum('ij, ij -> i', v_ji, av_ij) * dt
    
    u_ij = torch.zeros_like(deltaE_thermal)
    if f_term == 'PdV':
        u_ij = f_ij * deltaE_pressure / m_i + 0.5 * deltaE_viscosity / m_i
    else:
        u_ij = f_ij * deltaE_thermal / m_i        
    
    return scatter_sum(u_ij, i, dim = 0, dim_size = particles.positions.shape[0]) / dt


from enum import Enum
from sphMath.enums import EnergyScheme
def compute_fij(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict,
        dt: float,
        v_halfstep: torch.Tensor,
        ap_ij: torch.Tensor, 
        av_ij: torch.Tensor):        
    f_term          = config.get('energyScheme', EnergyScheme.equalWork)
    gamma           = config.get('gamma', 1.4)
    i, j            = neighborhood[0].row, neighborhood[0].col
    v_i, v_j        = v_halfstep[i], v_halfstep[j]
    u_i, u_j        = particles.internalEnergies[i], particles.internalEnergies[j]
    P_i, P_j        = particles.pressures[i], particles.pressures[j]
    rho_i, rho_j    = particles.densities[i], particles.densities[j]
    m_i, m_j        = particles.masses[i], particles.masses[j]

    v_ji            = v_j - v_i    
    u_ji            = u_j - u_i
    a_ij            = ap_ij + av_ij 

    deltaE_thermal  = m_i * torch.einsum('ij, ij -> i', v_ji, a_ij ) * dt
    
    f_ij = None
    sgn = lambda x: torch.where(x == 0, torch.zeros_like(x), torch.sign(x))
        
    if f_term == EnergyScheme.equalWork:
        f_ij = torch.ones_like(rho_i) * 1/2
    elif f_term == EnergyScheme.PdV:
        # raise NotImplementedError # Doesnt work with the current multistep code        
        f_ij = (P_i / rho_i**2) / (P_i / rho_i**2 + P_j / rho_j**2)
    elif f_term == EnergyScheme.diminishing:
        u_ji_norm = torch.abs(u_ji)        
        f_ij = 0.5 * (1 + (u_ji * sgn(deltaE_thermal)) / (u_ji_norm + 1 / (1 + u_ji_norm)))
    elif f_term == EnergyScheme.monotonic:
        A = deltaE_thermal / (u_ji + 1e-14)
        B = torch.where(A >= 0, A / m_i, A / m_j)
        
        term_1 = torch.maximum(torch.zeros_like(B), sgn(B))
        term_2 = m_i / (deltaE_thermal + 1e-14) * ( (deltaE_thermal + m_i * u_i + m_j * u_j) / (m_i + m_j) - u_i)
        
        f_ij = torch.where(torch.abs(B) <= 1, term_1, term_2)
    elif f_term == EnergyScheme.hybrid:
        u_ji_norm = torch.abs(u_ji)
        A = deltaE_thermal / (u_ji + 1e-14)
        B = torch.where(A >= 0, A / m_i, A / m_j)
        
        term_1 = torch.maximum(torch.zeros_like(B), sgn(B))
        term_2 = m_i / (deltaE_thermal + 1e-14) * ( (deltaE_thermal + m_i * u_i + m_j * u_j) / (m_i + m_j) - u_i)
        
        f_ij_mono = torch.where(torch.abs(B) <= 1, term_1, term_2)
        f_ij_sm = 0.5 * (1 + (u_ji * sgn(deltaE_thermal)) / (u_ji_norm + 1 / (1 + u_ji_norm)))
        
        chi = torch.abs(u_ji) / (torch.abs(u_i) + torch.abs(u_j) + 1e-14)
        # chi = 1-chi
        f_ij = chi * f_ij_mono + (1 - chi) * f_ij_sm
        
        
    elif f_term == EnergyScheme.CRK:
        s_i = P_i / rho_i**gamma
        s_j = P_j / rho_j**gamma
        s_min = torch.minimum(s_i.abs(), s_j.abs())
        s_max = torch.maximum(s_i.abs(), s_j.abs())
        
        f_ij = torch.ones_like(s_i) * 0.5
        
        maskA = torch.abs(s_i - s_j) == 0
        maskB = torch.logical_or(torch.logical_and(deltaE_thermal >= 0, s_i >= s_j), torch.logical_and(deltaE_thermal < 0, s_i < s_j))
        maskC = torch.logical_or(torch.logical_and(deltaE_thermal >= 0, s_i < s_j), torch.logical_and(deltaE_thermal < 0, s_i >= s_j))
        
        
        f_ijA = 0.5
        f_ijB = s_min / (s_min + s_max)
        f_ijC = s_max / (s_min + s_max)
            
        f_ij = torch.where(maskA, f_ijA, f_ij)
        f_ij = torch.where(torch.logical_and(torch.logical_not(maskA), maskB), f_ijB, f_ij)
        f_ij = torch.where(torch.logical_and(torch.logical_and(torch.logical_not(maskA), torch.logical_not(maskB)), maskC), f_ijC, f_ij)

    return f_ij



def compSPH_deltaU_multistep(
        dt : float,
        initialState: CompressibleState,
        particleStates: List[CompressibleState],
        updates : List[CompressibleUpdate],
        butcherTerms: np.ndarray,
        solverConfig: dict,
        verbose: bool = False
):
    verbosePrint(verbose, '[DeltaU] Computing Accelerations')
    fullAcceleration = torch.zeros_like(initialState.velocities)
    for i, update in enumerate(updates):
        fullAcceleration += butcherTerms[i] * update.velocities
    halfStepVelocity = initialState.velocities + 0.5 * fullAcceleration * dt

    deltaU = torch.zeros_like(initialState.internalEnergies)
    verbosePrint(verbose, '[DeltaU] Computing DeltaU')
    for ii, cState in enumerate(particleStates):
        verbosePrint(verbose, f'[DeltaU]\tStep {ii:2d}/{len(particleStates):2d}\tk = {butcherTerms[ii]:.2f}')
        i = cState.adjacency.row
        j = cState.adjacency.col

        v_i = halfStepVelocity[i]
        v_j = halfStepVelocity[j]
        v_ji = v_j - v_i

        f_ij = cState.f_ij
        av_ij = cState.ap_ij
        ap_ij = cState.av_ij
        
        k = butcherTerms[ii]
        if solverConfig.get('energyScheme', EnergyScheme.equalWork) == EnergyScheme.PdV:
            term = k * f_ij * torch.einsum('ij, ij -> i', v_ji, ap_ij)
            term += k * 1/2 * torch.einsum('ij, ij -> i', v_ji, av_ij)
        else:
            term = k * f_ij * torch.einsum('ij, ij -> i', v_ji, ap_ij + av_ij)
            
        deltaU += scatter_sum(term, i, dim = 0, dim_size = initialState.positions.shape[0])
    verbosePrint(verbose, '[DeltaU] Done')
    return deltaU


