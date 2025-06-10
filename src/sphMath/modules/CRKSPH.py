from typing import Union, Optional
from dataclasses import dataclass
import dataclasses
# from sphMath.modules.compressible import CompressibleParticleSet, BaseSPHSystem
from sphMath.neighborhood import DomainDescription, SparseCOO
from sphMath.kernels import SPHKernel
import torch
from sphMath.operations import sph_op
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product
from sphMath.modules.eos import idealGasEOS
from sphMath.neighborhood import NeighborhoodInformation
from sphMath.integrationSchemes.util import integrateQ
from sphMath.util import scatter_max

from sphMath.modules.compressible import CompressibleState
from sphMath.operations import sph_op, SPHOperation
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme, evalKernel, evalKernelGradient
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple

# from sphMath.modules.viscosity import compute_Pi

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


def computeCRKTerms(moments: KernelMoments, num_nbrs: torch.Tensor):
    m_0 = moments.m_0
    m_1 = moments.m_1
    m_2 = moments.m_2
    dm_0dgamma = moments.dm_0dgamma
    dm_1dgamma = moments.dm_1dgamma
    dm_2dgamma = moments.dm_2dgamma


    m_2_det = torch.det(m_2).abs()
    m_2_inv = torch.linalg.pinv(m_2)
    is_singular = torch.where(m_2_det < 1e-6, 1.0, 0.0)
    #     # Eq. 12.
    # ai = 1.0/(m0 - dot(temp_vec, m1, d))
    A = 1 / (m_0 - torch.einsum('nij, nj, ni -> n', m_2_inv, m_1, m_1))
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
    gradATerm2 = torch.einsum('nij, nj, nci -> nc', m_2_inv, m_1, dm_1dgamma)
    gradATerm3 = torch.einsum('nij, ncj, ni -> nc', m_2_inv, dm_1dgamma, m_1)
    gradATerm4 = torch.einsum('nad, ncde, neb, nb, na -> nc', m_2_inv, dm_2dgamma, m_2_inv, m_1, m_1)
    gradA = - (A **2).view(-1,1) * ( gradATerm1 - gradATerm2 - gradATerm3 + gradATerm4)

    gradBTerm1 = torch.einsum('nab, ncb -> nca', m_2_inv, dm_1dgamma)
    gradBTerm2 = torch.einsum('nad, ncde, neb, nb -> nca', m_2_inv, dm_2dgamma, m_2_inv, m_1)
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

from sphMath.neighborhood import correctedKernel, correctedGradientKernel

from sphMath.modules.CRKSPH import computeGeometricMoments, computeCRKTerms, correctedKernel, correctedGradientKernel

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
    
    V_j = particles.apparentArea[j]
    
    gradW_ij = evalKernelGradient(neighborhood[1], supportScheme=supportScheme, combined=True)

    # gradW_ij = correctedGradientKernel(x_ij, h_i, h_j, kernel, particles_a.A[i], particles_a.B[i], particles_a.gradA[i], particles_a.gradB[i])
    term = V_j.view(-1,1,1) * torch.einsum('na, nb -> nab', v_ij, gradW_ij)
    return - scatter_sum(term, i, dim = 0, dim_size=particles.positions.shape[0])

from sphMath.util import scatter_max

def limiterVL(x):
    vL = 2 / (1 + x)
    return torch.where(x > 0, x * vL**2, 0.0)

def computeVanLeer(xij, DvDxi, DvDxj):
    xij = 0.5 * (xij)
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

from sphMath.util import getSetConfig
from sphMath.kernels import Kernel_xi

def computeCRKAccel(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict,
        velocityTensor: torch.Tensor):
    eta_max = getSetConfig(config, 'CRKSPH', 'eta_max', 4.0)
    eta_crit = getSetConfig(config, 'CRKSPH', 'eta_crit', 1 / 4)
    eta_fold = getSetConfig(config, 'CRKSPH', 'eta_fold', 0.2)    

    i = neighborhood[0].row
    j = neighborhood[0].col

    x_ij = neighborhood[1].x_ij
    
    h_i = particles.supports[i]
    h_j = particles.supports[j]


    H_i = h_i / eta_max
    H_j = h_j / eta_max
    eta_i = x_ij / H_i.view(-1,1)
    eta_j = x_ij / H_j.view(-1,1)
    
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
    
    gradW_ij = correctedGradientKernel(particles.A[i], particles.B[i], particles.gradA[i], particles.gradB[i], neighborhood[1])
    gradW_ji = correctedGradientKernel(particles.A[j], particles.B[j], particles.gradA[j], particles.gradB[j], neighborhood[1], xji=True)
    gradW = gradW_ij - gradW_ji
    W_ij = correctedKernel(particles.A[i], particles.B[i], neighborhood[1])
    W_ji = correctedKernel(particles.A[j], particles.B[j], neighborhood[1], xji=True)
    W = W_ij + W_ji

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
    phi_ij = computeVanLeer(x_ij, velocityTensor[i], velocityTensor[j]) * factor
    # phi_ij[:] = 0
    
    if torch.any(phi_ij < -0.01) or torch.any(phi_ij > 1.01):
        print(f'phi_ij: min: {phi_ij.min():8.3g}, max: {phi_ij.max():8.3g}, mean {phi_ij.mean():8.3g} has nan: {torch.isnan(phi_ij).any()} has inf: {torch.isinf(phi_ij).any()}')

    phi_ij = phi_ij.clamp(0, 1)
    # phi_ij[:] = 1

    # "Mike Method", see Spheral, LimitedMonaghanGingoldViscosity
    v_i_hat = v_i - phi_ij.view(-1,1) / 2 * torch.einsum('nba, na -> nb', velocityTensor[i], x_ij)
    v_j_hat = v_j + phi_ij.view(-1,1) / 2 * torch.einsum('nba, na -> nb', velocityTensor[j], x_ij)
    v_ij_hat = v_i_hat - v_j_hat


    # prod = torch.einsum('nab, na -> nb', tensorA[i] + tensorB[j], x_ij) 
    # v_ij_hat = v_i - v_j# - phi_ij.view(-1,1) / 2 * prod   
    
    # print(f'v_ij: min: {v_ij.min():8.3g}, max: {v_ij.max():8.3g} has nan: {torch.isnan(v_ij).any()} has inf: {torch.isinf(v_ij).any()}, shape: {v_ij.shape}')
    # print(f'v_ij_hat: min: {v_ij_hat.min():8.3g}, max: {v_ij_hat.max():8.3g} has nan: {torch.isnan(v_ij_hat).any()} has inf: {torch.isinf(v_ij_hat).any()}, shape: {v_ij_hat.shape}')
    # print(f'prod: min: {prod.min():8.3g}, max: {prod.max():8.3g} has nan: {torch.isnan(prod).any()} has inf: {torch.isinf(prod).any()}, shape: {prod.shape}')
    
    vx_ij_hat_i = torch.einsum('ij, ij -> i', v_ij_hat, eta_i)
    vx_ij_hat_j = torch.einsum('ij, ij -> i', v_ij_hat, eta_j)#.clamp(max = 0)
    vx_ij_hat = torch.einsum('ij, ij -> i', v_ij_hat, x_ij)
    
    # print(f'vx_ij: min: {vx_ij.min():8.3g}, max: {vx_ij.max():8.3g} has nan: {torch.isnan(vx_ij).any()} has inf: {torch.isinf(vx_ij).any()}, shape: {vx_ij.shape}')
    # print(f'vx_ij_hat: min: {vx_ij_hat.min():8.3g}, max: {vx_ij_hat.max():8.3g} has nan: {torch.isnan(vx_ij_hat).any()} has inf: {torch.isinf(vx_ij_hat).any()}, shape: {vx_ij_hat.shape}')
    

    mu_i = (vx_ij_hat_i / (torch.einsum('ni, ni -> n', eta_i, eta_i) + 1e-14 * h_i ** 2)).clamp(max = 0)
    mu_j = (vx_ij_hat_j / (torch.einsum('ni, ni -> n', eta_j, eta_j) + 1e-14 * h_j ** 2)).clamp(max = 0)

    correctXi = getSetConfig(config, 'diffusion', 'correctXi', True)

    xi = Kernel_xi(kernel, x_ij.shape[1]) if correctXi else 1.0
    # mu_i = (vx_ij_hat / (torch.linalg.norm(x_ij, dim = -1) ** 2 + 1e-14 * h_i ** 2) * h_i).clamp(max = 0) / xi
    # mu_j = (vx_ij_hat / (torch.linalg.norm(x_ij, dim = -1) ** 2 + 1e-14 * h_j ** 2) * h_j).clamp(max = 0) / xi    

    # mu_i = mu_j = (torch.einsum('ij, ij -> i', v_ij, x_ij) / (torch.linalg.norm(x_ij, dim = -1) ** 2 + 0.01 * h_i ** 2)).clamp(max = 0) * h_i / kernel.xi(x_ij.shape[1])
    
    C_l = getSetConfig(config, 'diffusion', 'C_l', 2)
    C_q = getSetConfig(config, 'diffusion', 'C_q', 1)
    Q_i = rho_i * (-C_l * c_i * mu_i + C_q * mu_i ** 2)
    Q_j = rho_j * (-C_l * c_j * mu_j + C_q * mu_j ** 2)
    
    term = V_i * V_j * (P_i + P_j + Q_i + Q_j)
    # term = V_i * V_j * (P_i + P_j)
    term = term.view(-1,1) * gradW

    # if torch.any(term[[W == 0]] != 0):
        # raise ValueError('term has non-zero values where W_ij = 0')
    
    ap_i =  - (1 / (2 * particles.masses[i])).view(-1,1) * (V_i * V_j * (P_i)).view(-1,1) * gradW
    av_i = - (1 / (2 * particles.masses[i])).view(-1,1) * (V_i * V_j * (Q_i)).view(-1,1) * gradW
    ap_j =  - (1 / (2 * particles.masses[i])).view(-1,1) * (V_i * V_j * (P_j)).view(-1,1) * gradW
    av_j = - (1 / (2 * particles.masses[i])).view(-1,1) * (V_i * V_j * (Q_j)).view(-1,1) * gradW
    
    return - (1 / (2 * particles.masses[i])).view(-1,1) * term, scatter_max(phi_ij, i, dim = 0, dim_size=particles.positions.shape[0])[0], ap_i, ap_j, av_i, av_j

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
    