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

from sphMath.modules.compressible import CompressibleState
from sphMath.modules.viscosity import compute_Pi
from sphMath.modules.compressible import CompressibleState
from sphMath.operations import sph_op, SPHOperation
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme, evalKernel, evalKernelGradient, evalKernelDkDh
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple



# $$P_i = \sum_j (\gamma -1) m_j u_j W_{ij}(h_i)$$
# $$\rho_i = \sum_j m_j W_{ij}(h_i)$$
# $$n_i = \sum_j W_{ij}(h_i)$$

# $$\frac{D v_i^\alpha}{Dt} = -\sum_j m_j \left[ (\gamma - 1)^2 u_i u_j \left(\frac{f_{ij}}{P_i}\partial_\alpha W_{ij}(h_i) + \frac{f_{ij}}{P_j}\partial_\alpha W_{ij}(h_i)\right) + q_{\text{acc}ij}^\alpha\right]$$

# $$\frac{DE_i}{Dt} = m_i v_i^\alpha + \sum_j m_i m_j \left[(\gamma - 1)^2 u_i u_j \frac{f_{ij}}{P_i} v_{ij}^\alpha \partial_\alpha W_{ij}(h_i) + v_{ij}^\alpha q_{\text{acc}ij}^\alpha\right]$$

# $$q_{\text{acc}ij}^\alpha = \frac{1}{2} (\rho_i \Pi_i + \rho_j + \Pi_j) \frac{\partial_\alpha W_{ij}(h_i) + \partial_\alpha W_{ij}(h_j)}{\rho_i + \rho_j}$$

# $$f_{ij} = 1 - \left(\frac{h_i}{\nu(\gamma -1)n_i m_j u_j}\frac{\partial P_i}{\partial h_i}\right)\left(1 + \frac{h_i}{\nu n_i}\frac{\partial n_i}{\partial h_i}\right)^{-1}$$

# $$\frac{\partial n_i}{\partial h_i} = - \sum_j h_i^{-1} \left(\nu W_{ij}(h_i) + \eta_i \frac{\partial W}{\partial\eta}(\eta_i)\right)$$

# $$\frac{\partial P_i}{\partial h_i} = - \sum_j (\gamma -1) m_j u_j h_i^{-1}\left(\nu W_{ij}(h_i) + \eta_i \frac{\partial W}{\partial\eta}(\eta_i)\right)$$


# $$P_i = \sum_j (\gamma -1) m_j u_j W_{ij}(h_i)$$
def computePressure_PSH(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    

    i = neighborhood[0].row
    j = neighborhood[0].col
    gamma = config['fluid']['gamma']    
    u_i = particles.internalEnergies[i]
    u_j = particles.internalEnergies[j]    

    term = (gamma - 1) * particles.masses[j] * u_j * evalKernel(neighborhood[1], supportScheme = supportScheme, combined = True) #kernel.eval(x_ij, supports[0][i])

    return scatter_sum(term, i, dim = 0, dim_size = particles.positions.shape[0])


def computeKernelMoment(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    

    i = neighborhood[0].row
    j = neighborhood[0].col

    W_ij = evalKernel(neighborhood[1], supportScheme = supportScheme, combined = True)

    return scatter_sum(W_ij, i, dim = 0, dim_size = particles.positions.shape[0])

def compute_dndh(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    

    i = neighborhood[0].row
    j = neighborhood[0].col

    W_ij = evalKernelDkDh(neighborhood[1], supportScheme = supportScheme, combined = True)

    return scatter_sum(W_ij, i, dim = 0, dim_size = particles.positions.shape[0])
def compute_dPdh(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    

    i = neighborhood[0].row
    j = neighborhood[0].col
    gamma = config['fluid']['gamma']
    
    u_j = particles.internalEnergies[j]

    W_ij = evalKernelDkDh(neighborhood[1], supportScheme = supportScheme, combined = True)
    term = (gamma - 1) * particles.masses[j] * u_j * W_ij

    return scatter_sum(term, i, dim = 0, dim_size = particles.positions.shape[0])

def compute_fij(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}
):
    

    i = neighborhood[0].row
    j = neighborhood[0].col
    
    h_i = particles.supports[i]
    n_i = particles.n[i]
    m_j = particles.masses[j]
    u_j = particles.internalEnergies[j]
    dPdh_i = particles.dPdh[i]
    dndh_i = particles.dndh[i]
    nu = particles.positions.shape[1]
    gamma = config['fluid']['gamma']

    leftTerm = (h_i / (nu * (gamma - 1) * n_i * m_j * u_j)) * dPdh_i
    rightTerm = 1 + h_i / (nu * n_i) * dndh_i

    return 1 - leftTerm / rightTerm

def compute_fji(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}
):
    

    i = neighborhood[0].row
    j = neighborhood[0].col

    h_j = particles.supports[j]
    n_j = particles.n[j]
    m_i = particles.masses[i]
    u_i = particles.internalEnergies[i]
    dPdh_j = particles.dPdh[j]
    dndh_j = particles.dndh[j]
    nu = particles.positions.shape[1]
    gamma = config['fluid']['gamma']
    
    leftTerm = (h_j / (nu * (gamma - 1) * n_j * m_i * u_i)) * dPdh_j
    rightTerm = 1 + h_j / (nu * n_j) * dndh_j
    
    return 1 - leftTerm / rightTerm


def compute_qij_acc(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    

    i = neighborhood[0].row
    j = neighborhood[0].col
    
    gamma = config['fluid']['gamma']
    v_ij = particles.velocities[i] - particles.velocities[j]
    
    rho_i = particles.densities[i]
    rho_j = particles.densities[j]

    gradW_i, gradW_j = evalKernelGradient(neighborhood[1], supportScheme = supportScheme, combined = False)

    P_i = particles.P[i]
    c_i = torch.sqrt(gamma * P_i.abs() / rho_i)
    P_j = particles.P[j]
    c_j = torch.sqrt(gamma * P_j.abs() / rho_j)

    Pi_i = compute_Pi(particles, particles, config['domain'], neighborhood[0], config, useJ = False)
    Pi_j = compute_Pi(particles, particles, config['domain'], neighborhood[0], config, useJ = True)
    QPiij = Pi_i #/ (rho_i)
    QPiji = Pi_j #/ (rho_j)
    
    Qacci = QPiij.view(-1,1) * gradW_i * 1/2
    Qaccj = QPiji.view(-1,1) * gradW_j * 1/2
    
    workQi = torch.einsum('ij,ij->i', v_ij, Qacci)
    workQj = torch.einsum('ij,ij->i', v_ij, Qaccj)

    return Qacci, Qaccj, workQi, workQj

    leftTerm = rho_i * Pi_i + rho_j * Pi_j
    rightTerm = (gradW_i + gradW_j) / (rho_i + rho_j).view(-1,1)
    return 1/2 * leftTerm.view(-1,1) * rightTerm

def compute_dvdt(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    

    i = neighborhood[0].row
    j = neighborhood[0].col
    
    gamma = config['fluid']['gamma']

    u_i = particles.internalEnergies[i]
    u_j = particles.internalEnergies[j]
    m_j = particles.masses[j]
    P_i = particles.P[i]
    P_j = particles.P[j]


    gradW_i, gradW_j = evalKernelGradient(neighborhood[1], supportScheme = supportScheme, combined = False)

    fij = compute_fij(particles, kernel, neighborhood, supportScheme, config)
    fji = compute_fji(particles, kernel, neighborhood, supportScheme, config)
    
    Qacci, Qaccj, workQi, workQj = compute_qij_acc(particles, kernel, neighborhood, supportScheme, config)

    fac = ((gamma - 1) **2 * u_i * u_j).view(-1,1)
    pressureTerm = fac * ((fij / P_i).view(-1,1) * gradW_i + (fji / P_j).view(-1,1) * gradW_j)

    term = m_j.view(-1,1) * (pressureTerm + Qacci + Qaccj)

    # print(f'fij: {fij.shape}, fji: {fji.shape}, qij: {q_ij.shape}')
    # print(f'gradW_i: {gradW_i.shape}, gradW_j: {gradW_j.shape}')
    # print(f'fac: {fac.shape}, pressureTerm: {pressureTerm.shape}, term: {term.shape}')

    return -scatter_sum(term, i, dim = 0, dim_size = particles.positions.shape[0])

def compute_dEdt(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict,
        dvdt):
    

    i = neighborhood[0].row
    j = neighborhood[0].col
    
    gamma = config['fluid']['gamma']

    u_i = particles.internalEnergies[i]
    u_j = particles.internalEnergies[j]
    m_j = particles.masses[j]
    P_i = particles.P[i]
    m_i = particles.masses[i]
    v_ij = particles.velocities[i] - particles.velocities[j]

    gradW_i, gradW_j = evalKernelGradient(neighborhood[1], supportScheme = supportScheme, combined = False)

    fij = compute_fij(particles, kernel, neighborhood, supportScheme, config)
    # fji = compute_fji(particles, kernel, neighborhood, supportScheme, config)

    Qacci, Qaccj, workQi, workQj = compute_qij_acc(particles, kernel, neighborhood, supportScheme, config)

    fac = (gamma - 1) **2 * u_i * u_j
    pressureRatio = fij / P_i
    
    pressureTerm = fac * pressureRatio * torch.einsum('ij,ij->i', v_ij, gradW_i)
    viscosityTerm = workQi

    term = m_i * m_j * (pressureTerm + viscosityTerm)
    # return dEdt + scatter_sum(term, i, dim = 0, dim_size = particles_a.positions.shape[0])
    return scatter_sum(term, i, dim = 0, dim_size = particles.positions.shape[0])

# artificial conductivity
def compute_ArtificialConductivity(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    

    i = neighborhood[0].row
    j = neighborhood[0].col
    
    gamma = config['fluid']['gamma']

    u_i = particles.internalEnergies[i]
    u_j = particles.internalEnergies[j]
    m_j = particles.masses[j]
    P_i = particles.P[i]
    P_j = particles.P[j]
    m_i = particles.masses[i]
    v_ij = particles.velocities[i] - particles.velocities[j]

    x_ij , r_ij = neighborhood[1].x_ij, neighborhood[1].r_ij
    rho_i = particles.densities[i]
    rho_j = particles.densities[j]
    rho_bar = 1/2 * (rho_i + rho_j)

    P_i = particles.P[i]
    c_i = torch.sqrt(gamma * P_i.abs() / rho_i)
    P_j = particles.P[j]
    c_j = torch.sqrt(gamma * P_j.abs() / rho_j)
    
    h_i = particles.supports[i]
    # h_j = supports[1][j]
    # h_bar = 1/2 * (supports[0][i] + supports[1][j])

    # gradW_i = kernel.jacobian(x_ij, supports[0][i])
    # gradW_j = kernel.jacobian(x_ij, supports[1][j])

    # q_i = r_ij / get_i(supports, i)
    # q_j = r_ij / get_j(supports, j)
    # dim = particles.positions.shape[1]
    K_ij = evalKernelDkDh(neighborhood[1], supportScheme = supportScheme, combined = True)

    # W_i = kernel.module.dkdq(q_i, dim) * kernel.module.C_d(dim) / get_i(supports, i)**(dim + 1)
    # W_j = kernel.module.dkdq(q_j, dim) * kernel.module.C_d(dim) / get_j(supports, j)**(dim + 1)
    # K_ij = (W_i + W_j) / 2

    alpha_c = 0.25
    w_ij = torch.einsum('ij, ij -> i', v_ij, x_ij) / (r_ij + 1e-6 * h_i)
    v_s = c_i + c_j - 3 * w_ij
    # Read & Hayfield SPHS modification
    v_s = torch.where(3 * w_ij < c_i + c_j, v_s, 0) # could also just check if v_s < 0
    # Monaghan switch
    # v_s = torch.where(w_ij > 0, torch.zeros_like(v_s), v_s)

    if particles.alphas is not None:
        alpha_i = particles.alphas[i]
        alpha_j = particles.alphas[j]
        alpha_ij = 1/2 * (alpha_i + alpha_j)
    else:
        alpha_ij = 1

    # Read & Hayfield SPHS  pressure limiter 
    L_ij = torch.abs(P_i - P_j) / (P_i + P_j + 1e-6)

    termA = m_i * m_j * alpha_ij * v_s
    termB = K_ij / rho_bar

    term = termA * termB * L_ij * (u_i - u_j)

    return alpha_c * scatter_sum(term, i, dim = 0, dim_size = particles.positions.shape[0])
