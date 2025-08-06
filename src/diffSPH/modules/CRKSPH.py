from typing import Union, Optional
from dataclasses import dataclass
import dataclasses
# from diffSPH.modules.compressible import CompressibleParticleSet, BaseSPHSystem
from diffSPH.neighborhood import DomainDescription, SparseCOO
from diffSPH.kernels import SPHKernel
import torch
from diffSPH.operations import sph_op
from diffSPH.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product
from diffSPH.modules.eos import idealGasEOS
from diffSPH.neighborhood import NeighborhoodInformation
from diffSPH.integrationSchemes.util import integrateQ

from diffSPH.modules.compressible import CompressibleState
from diffSPH.operations import sph_op, SPHOperation
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState
from diffSPH.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme, evalKernel, evalKernelGradient
from diffSPH.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple

# from diffSPH.modules.viscosity import compute_Pi

@dataclass
class KernelMoments:
    m_0: torch.Tensor
    m_1: torch.Tensor
    m_2: torch.Tensor
    dm_0dgamma: torch.Tensor
    dm_1dgamma: torch.Tensor
    dm_2dgamma: torch.Tensor

def computeGeometricMoments(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):

    i = neighborhood[0].row
    j = neighborhood[0].col

    x_ij = neighborhood[1].x_ij    
    V_j = particles.apparentArea[j]
    
    W_ij = evalKernel(neighborhood[1], supportScheme=supportScheme, combined=True)
    gradW_ij = evalKernelGradient(neighborhood[1], supportScheme=supportScheme, combined=True)

    dyadicProduct = lambda a, b: torch.einsum('i...j,i...k->i...jk', a, b)
    delta = lambda i, j: 1 if i == j else 0
    nu = x_ij.shape[1]
    dtype = x_ij.dtype
    device = x_ij.device
    eye = torch.eye(nu, dtype = dtype, device = device)
    
    m_0 = V_j * W_ij
    m_1 = x_ij * (V_j * W_ij).view(-1, 1)
    m_2 = dyadicProduct(x_ij, x_ij) * (V_j * W_ij).view(-1, 1, 1)
        
    dm_0dgamma = (V_j).view(-1, 1) * gradW_ij
    dm_1dgamma = (V_j).view(-1, 1, 1) * (dyadicProduct(x_ij, gradW_ij) + W_ij.view(-1,1,1) * eye)
    # x_ij^alpha delta^{beta, gamma} + delta^{alpha, gamma} x_ij^beta
    
    # dm_2dgamma = (V_j).view(-1, 1, 1, 1) * (dyadicProduct(dyadicProduct(x_ij, x_ij), gradW_ij))
    dm_2dgamma = torch.zeros(x_ij.shape[0], nu, nu, nu, dtype = dtype, device = device)
    for alpha in range(nu):
        for beta in range(nu):
            for gamma in range(nu):
                gradTerm = x_ij[:, alpha] * x_ij[:, beta] * gradW_ij[:, gamma]
                deltaA = x_ij[:,alpha] * delta(beta, gamma)
                deltaB = delta(alpha, gamma) * x_ij[:, beta]
                kernelTerm = W_ij * (deltaA + deltaB)
                dm_2dgamma[..., gamma, alpha, beta] = V_j * (gradTerm + kernelTerm)
    
    # print(diagonalTerm.shape)
    
    # print(m_0.shape, m_1.shape, m_2.shape, dm_0dgamma.shape, dm_1dgamma.shape, dm_2dgamma.shape)
    
    
    m_0 = scatter_sum(m_0, i, dim = 0, dim_size = particles.positions.shape[0])
    m_1 = scatter_sum(m_1, i, dim = 0, dim_size = particles.positions.shape[0])
    m_2 = scatter_sum(m_2, i, dim = 0, dim_size = particles.positions.shape[0])
    
    # print(f'm_0: {m_0.min():8.3g}, {m_0.max():8.3g}, {m_0.mean():8.3g} has nan: {torch.isnan(m_0).any()} has inf: {torch.isinf(m_0).any()}')
    # print(f'm_1: {m_1.min():8.3g}, {m_1.max():8.3g}, {m_1.mean():8.3g} has nan: {torch.isnan(m_1).any()} has inf: {torch.isinf(m_1).any()}')
    # print(f'm_2: {m_2.min():8.3g}, {m_2.max():8.3g}, {m_2.mean():8.3g} has nan: {torch.isnan(m_2).any()} has inf: {torch.isinf(m_2).any()}')
    
    dm_0dgamma = scatter_sum(dm_0dgamma, i, dim = 0, dim_size = particles.positions.shape[0])
    dm_1dgamma = scatter_sum(dm_1dgamma, i, dim = 0, dim_size = particles.positions.shape[0])
    dm_2dgamma = scatter_sum(dm_2dgamma, i, dim = 0, dim_size = particles.positions.shape[0])
    
    return KernelMoments(
        m_0 = m_0,
        m_1 = m_1,
        m_2 = m_2,
        dm_0dgamma = dm_0dgamma,
        dm_1dgamma = dm_1dgamma,
        dm_2dgamma = dm_2dgamma
    )


def computeCRKTerms(moments: KernelMoments, num_nbrs: torch.Tensor, supports: torch.Tensor):
    m_0 = moments.m_0
    m_1 = moments.m_1
    m_2 = moments.m_2
    dm_0dgamma = moments.dm_0dgamma
    dm_1dgamma = moments.dm_1dgamma
    dm_2dgamma = moments.dm_2dgamma


    m_2_det = torch.det(m_2).abs()
    m_2_inv = torch.linalg.pinv(m_2)
    
    # print(f'm_2_det: {m_2_det.min():8.3g}, {m_2_det.max():8.3g}, {m_2_det.mean():8.3g} has nan: {torch.isnan(m_2_det).any()} has inf: {torch.isinf(m_2_det).any()}')
    # 
    is_singular = torch.where(m_2_det < 1e-10, 1.0, 0.0)
    # print(f'Number of singular matrices: {is_singular.sum()}')
    #     # Eq. 12.
    # ai = 1.0/(m0 - dot(temp_vec, m1, d))
    A = 1 / (m_0 - torch.einsum('nij, ni, nj -> n', m_2_inv, m_1, m_1))
    # # Eq. 13.
    # mat_vec_mult(m2inv, m1, d, bi)
    # for gam in range(d):
    #     bi[gam] = -bi[gam]
    B = - torch.einsum('nij, nj -> ni', m_2_inv, m_1)

    gradA = torch.zeros_like(dm_0dgamma)
    gradB = torch.zeros_like(dm_1dgamma)

    # print(gradA.shape, gradB.shape)
    nu = gradA.shape[1]
    gradATerm1 = dm_0dgamma
    gradATerm2 = torch.einsum('nij, nj, nki -> nk', m_2_inv, m_1, dm_1dgamma)
    # gradATerm3 = torch.einsum('nij, ncj, ni -> nc', m_2_inv, dm_1dgamma, m_1)
    gradATerm4 = torch.einsum('nil, nklm, nmj, nj, ni -> nk', m_2_inv, dm_2dgamma, m_2_inv, m_1, m_1)
    gradA = - (A **2).view(-1,1) * ( gradATerm1 - 2 * gradATerm2 + gradATerm4)

    gradBTerm1 = torch.einsum('nij, nkj -> nki', m_2_inv, dm_1dgamma)
    gradBTerm2 = torch.einsum('nil, nklm, nmj, nj -> nki', m_2_inv, dm_2dgamma, m_2_inv, m_1)
    gradB = -gradBTerm1 + gradBTerm2

    # num_nbrs = coo_to_csr(actualNeighbors).rowEntries

    mask = (num_nbrs < 2) | (is_singular > 0.0)

    # if N_NBRS < 2 or is_singular > 0.0:
    # d_ai[d_idx] = 1.0
    # for i in range(d):
    #     d_gradai[d * d_idx + i] = 0.0
    #     d_bi[d * d_idx + i] = 0.0
    #     for j in range(d):
    #         d_gradbi[d2 * d_idx + d * i + j] = 0.0
    A[mask] = 1.0
    for i in range(nu):
        gradA[mask, i] = 0.0
        B[mask, i] = 0.0
        for j in range(nu):
            gradB[mask, i, j] = 0.0
            
    return A, B, gradA, gradB

from diffSPH.neighborhood import correctedKernel, correctedGradientKernel

from diffSPH.modules.CRKSPH import computeGeometricMoments, computeCRKTerms, correctedKernel, correctedGradientKernel

def computeCRKVolume(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    

    i = neighborhood[0].row
    j = neighborhood[0].col
    
    W_ij = evalKernel(neighborhood[1], supportScheme=supportScheme, combined=True)
    
    return 1 / scatter_sum(W_ij, i, dim = 0, dim_size=particles.positions.shape[0])

def computeCRKDensity(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    
    i = neighborhood[0].row
    j = neighborhood[0].col  
    V_j = particles.apparentArea[j]
    m_i = particles.masses[i]
    m_j = particles.masses[j]

    # W_ij = evalKernel(neighborhood[1], supportScheme=supportScheme, combined=True)
    W_ij = correctedKernel(particles.A[i], particles.B[i], neighborhood[1])
    # should be m_j for particles from the same material
    # however, this would lead to problems on the hydrostatic case
    # but that shouldnt happen?
    termA = scatter_sum(m_i * V_j * W_ij, i, dim = 0, dim_size=particles.positions.shape[0])
    termB = scatter_sum(V_j * V_j * W_ij, i, dim = 0, dim_size=particles.positions.shape[0])
    
    return termA / termB

def computeVelocityTensor(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    
    i = neighborhood[0].row
    j = neighborhood[0].col  
    
    v_i = particles.velocities[i]
    v_j = particles.velocities[j]
    v_ij = v_i - v_j
    
    V_j = particles.masses[j]
    
    gradW_ij = evalKernelGradient(neighborhood[1], supportScheme=supportScheme, combined=True)
    # gradW_ij = correctedGradientKernel(particles.A[i], particles.B[i], particles.gradA[i], particles.gradB[i], neighborhood[1])
    # gradW_ji = -correctedGradientKernel(particles.A[j], particles.B[j], particles.gradA[j], particles.gradB[j], neighborhood[1], xji=True)
    
    # gradW = (gradW_ij - gradW_ji) / 2
    gradW = gradW_ij

    # gradW_ij = correctedGradientKernel(x_ij, h_i, h_j, kernel, particles_a.A[i], particles_a.B[i], particles_a.gradA[i], particles_a.gradB[i])
    term = V_j.view(-1,1,1) * torch.einsum('na, nb -> nab', v_ij, gradW)
    return - scatter_sum(term, i, dim = 0, dim_size=particles.positions.shape[0])

from diffSPH.util import scatter_max

def limiterVL(x):
    vL = 2 / (1 + x)
    return torch.where(x > 0, x * vL**2, 0.0)

def computeVanLeer(xij_, DvDxi, DvDxj):
    xij = 0.5 * (xij_)
    gradi = torch.einsum('na, na -> n', torch.einsum('nab, nb -> na', DvDxi, xij), xij)
    gradj = torch.einsum('na, na -> n', torch.einsum('nab, nb -> na', DvDxj, xij), xij)

    rif = gradj.sgn() * gradj.abs().clamp(min = 1e-30)
    rjf = gradi.sgn() * gradi.abs().clamp(min = 1e-30)
    ri = gradi / (rif + 1e-30)
    rj = gradj / (rjf + 1e-30)
    rij = torch.minimum(ri, rj)
    phi = limiterVL(rij)
    # print(f'Van Leer Limiter:' )
    # print(f'gradi: {gradi.min()}, {gradi.max()}, {gradi.mean()} - shape: {gradi.shape} has nan: {torch.isnan(gradi).any()} has inf: {torch.isinf(gradi).any()}')
    # print(f'gradj: {gradj.min()}, {gradj.max()}, {gradj.mean()} - shape: {gradj.shape} has nan: {torch.isnan(gradj).any()} has inf: {torch.isinf(gradj).any()}')
    # print('')
    # print(f'ri: {ri.min()}, {ri.max()}, {ri.mean()} - shape: {ri.shape} has nan: {torch.isnan(ri).any()} has inf: {torch.isinf(ri).any()}')
    # print(f'rj: {rj.min()}, {rj.max()}, {rj.mean()} - shape: {rj.shape} has nan: {torch.isnan(rj).any()} has inf: {torch.isinf(rj).any()}')
    # print('')
    # print(f'rij: {rij.min()}, {rij.max()}, {rij.mean()} - shape: {rij.shape} has nan: {torch.isnan(rij).any()} has inf: {torch.isinf(rij).any()}')
    # print(f'phi: {phi.min()}, {phi.max()}, {phi.mean()} - shape: {phi.shape} has nan: {torch.isnan(phi).any()} has inf: {torch.isinf(phi).any()}')

    return phi

from diffSPH.util import getSetConfig
from diffSPH.kernels import Kernel_xi
from diffSPH.modules.viscosity import compute_Pi

def computeCRKAccel(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict,
        velocityTensor: torch.Tensor):
    xi = Kernel_xi(config['kernel'], particles.positions.shape[0])
    # eta_max = getSetConfig(config, 'CRKSPH', 'eta_max', 4.0)
    eta_max = xi
    eta_crit = getSetConfig(config, 'CRKSPH', 'eta_crit', 1 / 4)  * eta_max
    eta_fold = getSetConfig(config, 'CRKSPH', 'eta_fold', 0.2)  * eta_max

    i = neighborhood[0].row
    j = neighborhood[0].col

    x_ij = neighborhood[1].x_ij
    
    h_i = particles.supports[i]
    h_j = particles.supports[j]


    H_i = h_i #/ eta_max
    H_j = h_j #/ eta_max
    eta_i = x_ij / H_i.view(-1,1) * eta_max
    eta_j = x_ij / H_j.view(-1,1) * eta_max

    # print(f'x_ij: {torch.linalg.norm(x_ij, dim=-1).min():8.3g}, {torch.linalg.norm(x_ij, dim=-1).max():8.3g}, {torch.linalg.norm(x_ij, dim=-1).mean():8.3g} has nan: {torch.isnan(x_ij).any()} has inf: {torch.isinf(x_ij).any()}')
    
    v_i = particles.velocities[i]
    v_j = particles.velocities[j]
    
    V_i = particles.apparentArea[i]
    V_j = particles.apparentArea[j]
    rho_i = particles.densities[i]
    rho_j = particles.densities[j]
    c_i = particles.soundspeeds[i]
    c_j = particles.soundspeeds[j]
    
    P_i = particles.pressures[i]
    P_j = particles.pressures[j]
    
    # gradW_ij = evalKernelGradient(neighborhood[1], supportScheme=supportScheme, combined=True)
    # gradW_ji = evalKernelGradient(neighborhood[1], supportScheme=supportScheme, combined=True)
    # gradW = gradW_ij + gradW_ji
    
    gradW_ij = correctedGradientKernel(particles.A[i], particles.B[i], particles.gradA[i], particles.gradB[i], neighborhood[1])
    gradW_ji = -correctedGradientKernel(particles.A[j], particles.B[j], particles.gradA[j], particles.gradB[j], neighborhood[1], xji=True)
    # gradW = gradW_ij - gradW_ji
    
    # gradW = gradW_ij
    # gradW_ji = gradW_ji
    
    # print('-'*80)
    # print(f'gradW_ij: {gradW_ij.min():8.3g}, {gradW_ij.max():8.3g}, {gradW_ij.mean():8.3g} has nan: {torch.isnan(gradW_ij).any()} has inf: {torch.isinf(gradW_ij).any()}')
    # print(f'gradW_ji: {gradW_ji.min():8.3g}, {gradW_ji.max():8.3g}, {gradW_ji.mean():8.3g} has nan: {torch.isnan(gradW_ji).any()} has inf: {torch.isinf(gradW_ji).any()}')
    # print(f'gradW: {gradW.min():8.3g}, {gradW.max():8.3g}, {gradW.mean():8.3g} has nan: {torch.isnan(gradW).any()} has inf: {torch.isinf(gradW).any()}')    
    
    # gradW_i, gradW_j = evalKernelGradient(neighborhood[1], SupportScheme.Symmetric, False)
    # gradW_ij = gradW_i
    # gradW_ji = gradW_j
    # gradW = gradW_i + gradW_j

    # print(f'gradW_ij: {gradW_i.min():8.3g}, {gradW_i.max():8.3g}, {gradW_i.mean():8.3g} has nan: {torch.isnan(gradW_i).any()} has inf: {torch.isinf(gradW_i).any()}')
    # print(f'gradW_ji: {gradW_j.min():8.3g}, {gradW_j.max():8.3g}, {gradW_j.mean():8.3g} has nan: {torch.isnan(gradW_j).any()} has inf: {torch.isinf(gradW_j).any()}')
    # print(f'gradW: {gradW.min():8.3g}, {gradW.max():8.3g}, {gradW.mean():8.3g} has nan: {torch.isnan(gradW).any()} has inf: {torch.isinf(gradW).any()}')

    # W_ij = correctedKernel(particles.A[i], particles.B[i], neighborhood[1])
    # W_ji = correctedKernel(particles.A[j], particles.B[j], neighborhood[1], xji=True)
    # W = W_ij #+ W_ji

    # W = evalKernel(neighborhood[1], supportScheme=supportScheme, combined=True)
    # gradW = evalKernelGradient(neighborhood[1], supportScheme=supportScheme, combined=True)
    # gradW = gradW_ij
    # print(f'gradW_ij: {gradW_ij.shape} has nan: {torch.isnan(gradW_ij).any()} has inf: {torch.isinf(gradW_ij).any()}')
    


    # r_ij_i = torch.einsum('nba, nab -> n', tensorA[i], torch.einsum('na, nb -> nab', x_ij, x_ij)) #+ 1e-6 * h_i
    # r_ij_j = torch.einsum('nba, nab -> n', tensorB[j], torch.einsum('na, nb -> nab', x_ij, x_ij))

    # safe_div = lambda a, b: torch.where(b.abs() > 1e-6, a / b, torch.zeros_like(a))

    # r_ij_hat = r_ij_i / (r_ij_j + 1e-6 * h_i)
    # r_ij_hat = safe_div(r_ij_i, r_ij_j)
    eta_i_norm = torch.linalg.norm(eta_i, dim = -1)
    eta_j_norm = torch.linalg.norm(eta_j, dim = -1)
    eta_ij = torch.minimum(eta_i_norm, eta_j_norm)
    
    # print(f'r_ij_hat: {r_ij_hat.shape} has nan: {torch.isnan(r_ij_hat).any()} has inf: {torch.isinf(r_ij_hat).any()}')
    # print(f'eta_ij: {eta_ij.shape} has nan: {torch.isnan(eta_ij).any()} has inf: {torch.isinf(eta_ij).any()}')
    
    # van leer flux limiter
    # eta_crit = solverConfig.get('eta_crit', 1 / 4)
    # eta_fold = solverConfig.get('eta_fold', 0.2)
    
    # xij_2 = 0.5 * x_ij # midpoint distance
    # gradi  = torch.einsum('nab, na, nb -> n', tensorA[i], xij_2, xij_2)
    # gradj  = torch.einsum('nab, na, nb -> n', tensorB[j], xij_2, xij_2)
    # ri = gradi.abs() / h_i
    # rj = gradj.abs() / h_j
    # r_ij_hat = torch.minimum(ri, rj)


    factor = torch.where(eta_ij < eta_crit, torch.exp(- ((eta_ij - eta_crit)/eta_fold)**2), torch.ones_like(eta_ij))

    # phi_ij = (4 * r_ij_hat / (1 + r_ij_hat)**2).clamp(0, 1) * factor
    phi_ij = computeVanLeer(x_ij / particles.supports[i].view(-1,1), velocityTensor[i], velocityTensor[j]) 
    # print(f'phi_ij: min: {phi_ij.min():8.3g}, max: {phi_ij.max():8.3g}, mean {phi_ij.mean():8.3g} has nan: {torch.isnan(phi_ij).any()} has inf: {torch.isinf(phi_ij).any()}')
    # phi_ij[:] = 0
    
    if torch.any(phi_ij < -0.01) or torch.any(phi_ij > 1.01):
        print(f'phi_ij: min: {phi_ij.min():8.3g}, max: {phi_ij.max():8.3g}, mean {phi_ij.mean():8.3g} has nan: {torch.isnan(phi_ij).any()} has inf: {torch.isinf(phi_ij).any()}')
    # print(f'factor: min: {factor.min():8.3g}, max: {factor.max():8.3g}, mean {factor.mean():8.3g} has nan: {torch.isnan(factor).any()} has inf: {torch.isinf(factor).any()}')
    phi_ij = phi_ij * factor

    phi_ij = phi_ij.clamp(0, 1)
    # print(f'phi_ij: min: {phi_ij.min():8.3g}, max: {phi_ij.max():8.3g}, mean {phi_ij.mean():8.3g} has nan: {torch.isnan(phi_ij).any()} has inf: {torch.isinf(phi_ij).any()}')
    # phi_ij[:] = 0.50

    # "Mike Method", see Spheral, LimitedMonaghanGingoldViscosity
    v_i_hat = v_i - phi_ij.view(-1,1) / 2 * torch.einsum('nba, na -> nb', velocityTensor[i], x_ij)
    v_j_hat = v_j + phi_ij.view(-1,1) / 2 * torch.einsum('nba, na -> nb', velocityTensor[j], x_ij)
    v_ij_hat = v_i_hat - v_j_hat


    # print(f'eta_i: min: {eta_i.min():8.3g}, max: {eta_i.max():8.3g}, mean {eta_i.mean():8.3g} has nan: {torch.isnan(eta_i).any()} has inf: {torch.isinf(eta_i).any()}')
    # print(f'eta_j: min: {eta_j.min():8.3g}, max: {eta_j.max():8.3g}, mean {eta_j.mean():8.3g} has nan: {torch.isnan(eta_j).any()} has inf: {torch.isinf(eta_j).any()}')
    # print(f'eta_ij: min: {eta_ij.min():8.3g}, max: {eta_ij.max():8.3g}, mean {eta_ij.mean():8.3g} has nan: {torch.isnan(eta_ij).any()} has inf: {torch.isinf(eta_ij).any()}')
    # print(f'eta_crit: {eta_crit:8.3g}, eta_fold: {eta_fold:8.3g}, eta_max: {eta_max:8.3g}')

    # prod = torch.einsum('nab, na -> nb', tensorA[i] + tensorB[j], x_ij) 
    # v_ij_hat = v_i - v_j# - phi_ij.view(-1,1) / 2 * prod   
    
    # print(f'v_ij: min: {v_ij.min():8.3g}, max: {v_ij.max():8.3g} has nan: {torch.isnan(v_ij).any()} has inf: {torch.isinf(v_ij).any()}, shape: {v_ij.shape}')
    # print(f'v_ij_hat: min: {v_ij_hat.min():8.3g}, max: {v_ij_hat.max():8.3g} has nan: {torch.isnan(v_ij_hat).any()} has inf: {torch.isinf(v_ij_hat).any()}, shape: {v_ij_hat.shape}')
    # print(f'prod: min: {prod.min():8.3g}, max: {prod.max():8.3g} has nan: {torch.isnan(prod).any()} has inf: {torch.isinf(prod).any()}, shape: {prod.shape}')
    
    # Normal formulation is: mu_ij = ux_ij / (r_ij**2 + 1e-14 * h**2) * h / xi
    # u_ij = v_ij_hat # If phi_ij = 0
    # r_ij = torch.linalg.norm(x_ij, dim=-1)
    # ux_ij = torch.einsum('ij, ij -> i', v_ij_hat, x_ij) # dot product of v_ij_hat and x_ij
    
    # eta_i = x_ij / h_i.view(-1,1) * eta_max
    # eta_j = x_ij / h_j.view(-1,1) * eta_max
    # vx_ij_hat_i = ux_ij / h_i * eta_max
    # vx_ij_hat_j = ux_ij / h_j * eta_max    
    
    vx_ij_hat_i = torch.einsum('ij, ij -> i', v_ij_hat, eta_i)
    vx_ij_hat_j = torch.einsum('ij, ij -> i', v_ij_hat, eta_j)#.clamp(max = 0)
    # vx_ij_hat = torch.einsum('ij, ij -> i', v_ij_hat, x_ij / particles.supports[i].view(-1,1) * eta_max)

    # mu_i = (vx_ij_hat_i / (torch.einsum('ni, ni -> n', eta_i, eta_i) + 1e-14 * h_i ** 2))
    # Remove Epsilon
    # mu_i = (vx_ij_hat_j / (torch.einsum('ni, ni -> n', eta_j, eta_j)))
    # Refactor with _i:
    # mu_i = (ux_ij / (torch.einsum('ni, ni -> n', eta_j, eta_j))) / h_i * eta_max
    # torch.einsum('ni, ni -> n', eta_j, eta_j) = r_ij**2 / h_i**2 * eta_max**2
    # mu_i = (ux_ij / r_ij**2) / h_i * eta_max * h_i**2 / eta_max**2
    # mu_i = (ux_ij / r_ij**2) * h_i / eta_max
    
    mu_i = (vx_ij_hat_i / (torch.einsum('ni, ni -> n', eta_i, eta_i) + 1e-14 * h_i ** 2))# / eta_max
    mu_j = (vx_ij_hat_j / (torch.einsum('ni, ni -> n', eta_j, eta_j) + 1e-14 * h_j ** 2))# / eta_max

    mu_i = mu_i.clamp(max = 0)
    mu_j = mu_j.clamp(max = 0)



    # print(f'mu_i: min: {mu_i.min():8.3g}, max: {mu_i.max():8.3g} has nan: {torch.isnan(mu_i).any()} has inf: {torch.isinf(mu_i).any()}, shape: {mu_i.shape}')
    # print(f'mu_j: min: {mu_j.min():8.3g}, max: {mu_j.max():8.3g} has nan: {torch.isnan(mu_j).any()} has inf: {torch.isinf(mu_j).any()}, shape: {mu_j.shape}')

    # mu_i = mu_j = -0.1
    correctXi = getSetConfig(config, 'diffusion', 'correctXi', True)

    xi = Kernel_xi(kernel, x_ij.shape[1]) if correctXi else 1.0
    # mu_i = (vx_ij_hat / (torch.linalg.norm(x_ij, dim = -1) ** 2 + 1e-14 * h_i ** 2) * h_i).clamp(max = 0) / xi
    # mu_j = (vx_ij_hat / (torch.linalg.norm(x_ij, dim = -1) ** 2 + 1e-14 * h_j ** 2) * h_j).clamp(max = 0) / xi    

    # mu_i = mu_j = (torch.einsum('ij, ij -> i', v_ij, x_ij) / (torch.linalg.norm(x_ij, dim = -1) ** 2 + 0.01 * h_i ** 2)).clamp(max = 0) * h_i / kernel.xi(x_ij.shape[1])
    
    C_l = getSetConfig(config, 'diffusion', 'C_l', 2)
    C_q = getSetConfig(config, 'diffusion', 'C_q', 1)

    C_l = 1
    C_q = 2
    
    K = 1
    v_sig_i = C_l * c_i - C_q * mu_i
    v_sig_j = C_l * c_j - C_q * mu_j

    Pi_i = -K / rho_i * v_sig_i * mu_i
    Pi_j = -K / rho_j * v_sig_j * mu_j

    # Q_i = Pi_i * rho_i**2
    # Q_i = rho_i * (-C_l * c_i * mu_i + C_q * mu_i ** 2)
    # Q_j = rho_j * (-C_l * c_j * mu_j + C_q * mu_j ** 2)
    Q_i = Pi_i * rho_i * rho_i
    Q_j = Pi_j * rho_j * rho_j
    
    term = V_i * V_j * (P_i + P_j + Q_i + Q_j)
    # term = V_i * V_j * (P_i + P_j)
    # term = term.view(-1,1) * gradW

    # if torch.any(term[[W == 0]] != 0):
        # raise ValueError('term has non-zero values where W_ij = 0')
    
    
    # gradW_i = evalKernelGradient(neighborhood[1], supportScheme=supportScheme, combined=True)
    # gradW_j = evalKernelGradient(neighborhood[1], supportScheme=supportScheme, combined=True)
    
    
    # print(gradW_ij, gradW_ji)
    # gradW_i, gradW_j = evalKernelGradient(neighborhood[1], SupportScheme.Symmetric, False)
    # print(gradW_i, gradW_j)
    
    # gradW_i = gradW_ij
    # gradW_j = -gradW_ji
    # gradW_j = gradW_ij
    
    # print('------------------------------------------------------------------------------------------')
    
    # print(f'gradW_i: min: {gradW_i.min():8.3g}, max: {gradW_i.max():8.3g}, mean {gradW_i.mean():8.3g} has nan: {torch.isnan(gradW_i).any()} has inf: {torch.isinf(gradW_i).any()}')
    # print(f'gradW_ij: min: {gradW_ij.min():8.3g}, max: {gradW_ij.max():8.3g}, mean {gradW_ij.mean():8.3g} has nan: {torch.isnan(gradW_ij).any()} has inf: {torch.isinf(gradW_ij).any()}')

    
    # print(f'V: min: {particles.apparentArea.min():8.3g}, max: {particles.apparentArea.max():8.3g}, mean {particles.apparentArea.mean():8.3g} has nan: {torch.isnan(particles.apparentArea).any()} has inf: {torch.isinf(particles.apparentArea).any()}')
    # V = particles.masses / particles.densities
    # print(f'V_: min: {V.min():8.3g}, max: {V.max():8.3g}, mean {V.mean():8.3g} has nan: {torch.isnan(V).any()} has inf: {torch.isinf(V).any()}')
    
    
    # termA = scatter_sum(m_i * V_j * W_ij, i, dim = 0, dim_size=particles.positions.shape[0])
    # termB = scatter_sum(V_j * V_j * W_ij, i, dim = 0, dim_size=particles.positions.shape[0])
    
    # rho = termA / termB
    # V = m_i / sum_j m_i V_j W_ij * sum V_j V_j W_ij
    # V = sum V_j V_j W_ij / sum V_j W_ijz
    # V_i = particles.masses[i] / particles.densities[i]
    # V_j = particles.masses[j] / particles.densities[j]
    # P_i = P[i]
    # P_j = P[j]
    
    # Q_i = rho_i * (-C_l * c_i * mu_i + C_q * mu_i ** 2)
    # Q_j = rho_j * (-C_l * c_j * mu_j + C_q * mu_j ** 2)
    ap_i_crk =  - (1 / (1 * particles.masses[i])).view(-1,1) * (V_i * V_j * (P_i)).view(-1,1) * gradW_ij
    ap_j_crk =  - (1 / (1 * particles.masses[i])).view(-1,1) * (V_i * V_j * (P_j)).view(-1,1) * gradW_ji
    
    
    av_i_crk = - (1 / (2 * particles.masses[i])).view(-1,1) * (V_i * V_j * (Q_i)).view(-1,1) * gradW_ij
    av_j_crk = - (1 / (2 * particles.masses[i])).view(-1,1) * (V_i * V_j * (Q_j)).view(-1,1) * gradW_ji
    # av_i_crk = - (1 / (2 )) * (V_j * (Q_i)).view(-1,1) * gradW_i
    # av_j_crk = - (1 / (2 ))* (V_j * (Q_j)).view(-1,1) * gradW_j
    
    # print(f'Q_i: min: {Q_i.min():8.3g}, max: {Q_i.max():8.3g}, mean {Q_i.mean():8.3g} has nan: {torch.isnan(Q_i).any()} has inf: {torch.isinf(Q_i).any()}, {Q_i}')
    
    
    # Q_i = compute_Pi(particles, particles, neighborhood[0].domain, neighborhood[0], config, useJ = False)# * particles.densities[j]
    # print(f'Q_i: min: {Q_i.min():8.3g}, max: {Q_i.max():8.3g}, mean {Q_i.mean():8.3g} has nan: {torch.isnan(Q_i).any()} has inf: {torch.isinf(Q_i).any()}, {Q_i}')
    # Q_j = compute_Pi(particles, particles, neighborhood[0].domain, neighborhood[0], config, useJ = True)# * particles.densities[j]
    # av_i_crk =  -1/2 * (Q_i.view(-1,1) * gradW_i) * particles.masses[j].view(-1,1)
    # av_j_crk =  -1/2 * (Q_j.view(-1,1) * gradW_j) * particles.masses[j].view(-1,1)
    
    # print(f'av_i_crk: min: {av_i_crk.min():8.3g}, max: {av_i_crk.max():8.3g}, mean {av_i_crk.mean():8.3g} has nan: {torch.isnan(av_i_crk).any()} has inf: {torch.isinf(av_i_crk).any()}')
    # print(f'av_j_crk: min: {av_j_crk.min():8.3g}, max: {av_j_crk.max():8.3g}, mean {av_j_crk.mean():8.3g} has nan: {torch.isnan(av_j_crk).any()} has inf: {torch.isinf(av_j_crk).any()}')
    
    # print(f'ap_i: min: {ap_i.min():8.3g}, max: {ap_i.max():8.3g}, mean {ap_i.mean():8.3g} has nan: {torch.isnan(ap_i).any()} has inf: {torch.isinf(ap_i).any()}')
    # print(f'ap_j: min: {ap_j.min():8.3g}, max: {ap_j.max():8.3g}, mean {ap_j.mean():8.3g} has nan: {torch.isnan(ap_j).any()} has inf: {torch.isinf(ap_j).any()}')
    # print(f'ap_i_crk: min: {ap_i_crk.min():8.3g}, max: {ap_i_crk.max():8.3g}, mean {ap_i_crk.mean():8.3g} has nan: {torch.isnan(ap_i_crk).any()} has inf: {torch.isinf(ap_i_crk).any()}')
    # print(f'ap_j_crk: min: {ap_j_crk.min():8.3g}, max: {ap_j_crk.max():8.3g}, mean {ap_j_crk.mean():8.3g} has nan: {torch.isnan(ap_j_crk).any()} has inf: {torch.isinf(ap_j_crk).any()}')
    
    P = (particles.pressures / particles.densities**2 / particles.omega).view(-1,1)
    Q_i = compute_Pi(particles, particles, neighborhood[0].domain, neighborhood[0], config, useJ = False) * particles.densities[i]
    Q_j = compute_Pi(particles, particles, neighborhood[0].domain, neighborhood[0], config, useJ = True) * particles.densities[j]
    # This returns -K / rho[i/j] * v_sig * mu_ij
    # v_sig = C_l * c - C_q * mu_ij
        
    # viscosityTerm = 1/2 * (Q_i * gradW_i + Q_j * gradW_j) * particles.masses[j].view(-1,1)
    # pressureTerm =      (P[i] * gradW_i + P[j] * gradW_j) * particles.masses[j].view(-1,1)
    
    # av_i = -1/2 * (Q_i.view(-1,1) * gradW_i) * V_j.view(-1,1)
    # av_j = -1/2 * (Q_j.view(-1,1) * gradW_j) * V_j.view(-1,1)
    # ap_i = -     (P[i] * gradW_i) * particles.masses[j].view(-1,1)
    # ap_j = -     (P[j] * gradW_j) * particles.masses[j].view(-1,1)
    # print(f'av_i: min: {av_i.min():8.3g}, max: {av_i.max():8.3g}, mean {av_i.mean():8.3g} has nan: {torch.isnan(av_i).any()} has inf: {torch.isinf(av_i).any()}')
    # print(f'av_j: min: {av_j.min():8.3g}, max: {av_j.max():8.3g}, mean {av_j.mean():8.3g} has nan: {torch.isnan(av_j).any()} has inf: {torch.isinf(av_j).any()}')  
    # P_i = V_i * V_j * P_i / (particles.masses[i] **2)
    # P_j = V_i * V_j * P_j / (particles.masses[i] **2)

    # print(f'P_i: min: {P_i.min():8.3g}, max: {P_i.max():8.3g}, mean {P_i.mean():8.3g} has nan: {torch.isnan(P_i).any()} has inf: {torch.isinf(P_i).any()}')
    # print(f'P_j: min: {P_j.min():8.3g}, max: {P_j.max():8.3g}, mean {P_j.mean():8.3g} has nan: {torch.isnan(P_j).any()} has inf: {torch.isinf(P_j).any()}')
    # print(f'P[i]: min: {P[i].min():8.3g}, max: {P[i].max():8.3g}, mean {P[i].mean():8.3g} has nan: {torch.isnan(P[i]).any()} has inf: {torch.isinf(P[i]).any()}')
    # print(f'P[j]: min: {P[j].min():8.3g}, max: {P[j].max():8.3g}, mean {P[j].mean():8.3g} has nan: {torch.isnan(P[j]).any()} has inf: {torch.isinf(P[j]).any()}')
    
    
    # a_ij = - (1 / (2 * particles.masses[i])).view(-1,1) * term
    ap_ij = ap_i_crk + ap_j_crk
    av_ij = av_i_crk + av_j_crk
    a_ij = ap_ij + av_ij


    term[torch.linalg.norm(x_ij, dim = -1) / torch.maximum(h_i, h_j) > 1] = 0.0

    return a_ij, scatter_max(phi_ij, i, dim = 0, dim_size=particles.positions.shape[0])[0], ap_i_crk, ap_j_crk, av_i_crk, av_j_crk

def computeCRKdvdt(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict,
        velocityTensor: torch.Tensor):
    
    
    dvdt_ij, phi, ap_i, ap_j, av_i, av_j = computeCRKAccel(particles, kernel, neighborhood, supportScheme, config, velocityTensor)
    i = neighborhood[0].row
    return scatter_sum(dvdt_ij, i, dim = 0, dim_size=particles.positions.shape[0]), phi, ap_i + ap_j, av_i + av_j

def computeCRKdudt(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict,
        velocityTensor: torch.Tensor,
        dvdt: torch.Tensor,
        dt: float):
    
    i = neighborhood[0].row
    j = neighborhood[0].col
    
    v_i = particles.velocities[i]
    v_j = particles.velocities[j]
    v_i_pred = v_i + dvdt[i] * dt
    v_j_pred = v_j + dvdt[j] * dt
    
    dvdt_ij, phi, ap_i, ap_j, av_i, av_j = computeCRKAccel(particles, kernel, neighborhood, supportScheme, config, velocityTensor)
    
    v_ij = v_i - v_j
    # return -scatter_sum(torch.einsum('ij, ij -> i', v_ij, ap_j + av_j), i, dim = 0, dim_size=particles.positions.shape[0])
    return -scatter_sum(torch.einsum('ij, ij -> i', v_ij, ap_i + av_i), i, dim = 0, dim_size=particles.positions.shape[0])
    
    s_i = particles_a.pressures[i] / particles_a.densities[i] ** solverConfig['fluid']['gamma']
    s_j = particles_b.pressures[j] / particles_b.densities[j] ** solverConfig['fluid']['gamma']
    s_min = torch.minimum(s_i.abs(), s_j.abs())
    s_max = torch.maximum(s_i.abs(), s_j.abs())
    
    delta_u_ij = 1/2 * torch.einsum('ij, ij -> i', v_j + v_j_pred - v_i - v_i_pred, dvdt_ij)
    
    f_ij = torch.ones_like(s_i) * 1/2
    f_ij_2 = s_min / (s_min + s_max)
    f_ij_3 = s_max / (s_min + s_max)
    
    mask = (s_i - s_j).abs() < 1e-5
    # maskA = ((delta_u_ij >= 0 & s_i >= s_j) | (delta_u_ij < 0 & s_i < s_j)) & ~mask
    # maskB = ((delta_u_ij >= 0 & s_i < s_j) | (delta_u_ij < 0 & s_i >= s_j)) & ~mask
    
    mask_a = torch.logical_and(torch.logical_not(mask), torch.logical_or(
        torch.logical_and(delta_u_ij >= 0, s_i >= s_j),
        torch.logical_and(delta_u_ij < 0, s_i < s_j)
    ))
    mask_b = torch.logical_and(torch.logical_not(mask), torch.logical_or(
        torch.logical_and(delta_u_ij >= 0, s_i < s_j),
        torch.logical_and(delta_u_ij < 0, s_i >= s_j)
    ))
    
    f_ij = torch.where(mask_a, f_ij_2, f_ij)
    f_ij = torch.where(mask_b, f_ij_3, f_ij)
    
    return scatter_sum(f_ij * delta_u_ij, i, dim = 0, dim_size=particles_a.positions.shape[0])
    