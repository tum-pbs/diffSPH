from typing import Union, Optional
from dataclasses import dataclass
import torch

from sphMath.neighborhood import DomainDescription, SparseCOO
from sphMath.kernels import SPHKernel
from sphMath.operations import sph_op
from sphMath.modules.compressible import CompressibleState
from sphMath.sphOperations.shared import getTerms, compute_xij, scatter_sum

from sphMath.modules.switches.common import computeM, computeShearTensor, computeDivergence
    
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.operations import SPHOperation
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple
from sphMath.kernels import KernelType
from sphMath.util import getSetConfig


def computeSecondOrderV(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict, 
        dvdt : torch.Tensor, 
        correctionMatrix : torch.Tensor):
    
    # The signs here should have been wrong, double check!
    V = SPHOperation(particles, particles.velocities, kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Difference)
    Vdot = SPHOperation(particles, dvdt, kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Difference)
    
    # i, j        = neighborhood.row, neighborhood.col
    # v_i, v_j    = particles_a.velocities[i], particles_b.velocities[j]
    # dv_i, dv_j  = dvdt_a[i], dvdt_b[j]
    # h_i         = particles_a.supports[i]
    # m_j         = particles_b.masses[j]
    # rho_j       = particles_b.densities[j]
    # x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)

    # gradW_i = kernel.jacobian(x_ij, h_i)
    # v_ij = v_i - v_j
    # dv_ij = dv_i - dv_j
    
    # dyadicV = (m_j / rho_j).view(-1,1,1) * torch.einsum('ij,ik->ijk', v_ij, gradW_i)
    # dyadicVdot = (m_j / rho_j).view(-1,1,1) * torch.einsum('ij,ik->ijk', dv_ij, gradW_i)
    
    # V = scatter_sum(dyadicV, i, dim = 0, dim_size = particles_a.positions.shape[0])
    correctVelocityGradient = getSetConfig(config, 'diffusionSwitch', 'correctGradient', False)
    if correctVelocityGradient:
        if correctionMatrix is not None:
            V = torch.einsum('ijk, ikl -> ijl', correctionMatrix, V)
        else:
            raise ValueError('Correction matrix is None, but correctVelocityGradient is True')
    V2 = torch.einsum('ijk, ikl -> ijl', V, V)
    # Vdot_ = scatter_sum(dyadicVdot, i, dim = 0, dim_size = particles_a.positions.shape[0])
    # From Cullen & Dehnen 2010 : We can estimate ∇˙ ·υ either from the change in the estimated ∇·υ over 
    # the last time step or as the trace of V [This is not implemented here as this would require 
    # storing the previous divergence estimate. Spheral (the code implementing CRKSPH by LLNL) does this.]    
    # Note that, by virtue of equation (B11), we could estimate ∇˙·υ also as ∇· υ˙ - tr(V^2) with the 
    # acceleration divergence ∇·υ˙ estimated using the standard divergence estimator, in the hope that 
    # its O(h0) error term is small since the acceleration is hardly sheared.
    divdotdvdt = SPHOperation(particles, dvdt, kernel, neighborhood[0], neighborhood[1], operation = Operation.Divergence, supportScheme=supportScheme, gradientMode=GradientMode.Difference, consistentDivergence=False)
    
    # divdotdvdt = sph_op(particles_a, particles_b, domain, kernel, neighborhood, 'gather', 'divergence', gradientMode = 'difference' , consistentDivergence = False, quantity=(dvdt_a, dvdt_b))
    return divdotdvdt - torch.einsum('...ii', V2)

    # Naïve alternative solution as described above
    A = SPHOperation(particles, dvdt, kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Difference)
    # A = sph_op(particles_a, particles_b, domain, kernel, neighborhood, 'gather', 'gradient', gradientMode = 'difference' , consistentDivergence = False, quantity=(dvdt_a, dvdt_b))
    correctVelocityGradient = getSetConfig(config, 'diffusionSwitch', 'correctGradient', False)
    if correctVelocityGradient:
        if correctionMatrix is not None: # This is an assumption, not described in the paper
            A = torch.einsum('ijk, ikl -> ijl', correctionMatrix, A)
        else:
            raise ValueError('Correction matrix is None, but correctVelocityGradient is True')
    return torch.einsum('...ii', A + V2)

    
from sphMath.util import scatter_max
from sphMath.neighborhood import evalKernel, evalKernelGradient
def computeR(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict, 
        div:torch.Tensor): # Referred to as R in eq 17 in Cullen and Dehnen 2010, referred to as Xi in CRKSPH and Hopkins
    i, j = neighborhood[0].row, neighborhood[0].col
    
    # i, j        = neighborhood.row, neighborhood.col
    # h_i         = particles_a.supports[i]
    # m_j         = particles_b.masses[j]
    # x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)
    
    # W_i         = kernel.eval(x_ij, h_i)
    # term        = m_j * torch.sign(div[j]) * W_i

    term = particles.densities[j] * torch.sign(div[j]) #* evalKernel(kernelValues, supportScheme, True)
    # The term is m[j] * |div[j]|, interpolate multiplies the given term with m[j]/rho[j] so need to premultiply by rho[j]

    summedTerm = SPHOperation(
        particles,
        quantity = term,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation= Operation.Interpolate        
    )

    # CRKSPH and Hopkins compute 1-R as Xi, we follow the convention of Cullen and Dehnen 2010
    # This also means that our 'Xi' term is the limiter term that is not given an explicit name
    # in CRKSPH and Hopkins notations. Also mind the minus sign.
    return 1 / particles.densities * summedTerm
    
def computeXi(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict,
        div, 
        S, 
        R): # See Cullen and Dehnen 2010, eq 18
    beta_xi = getSetConfig(config, 'diffusionSwitch', 'beta_xi', 2) # from the original Cullen and Dehnen paper Eq. 18
    limitXi = getSetConfig(config, 'diffusionSwitch', 'limitXi', False) # Workaround for testing for cases with 0 divergence
    
    nominator       = beta_xi * (1 - R)**4 * div
    denominatorLeft = nominator
    # This agrees with Spheral and Cullen and Dehnen. F.3 in the CRKSPH code is inconclusive
    # trace           = torch.einsum('...ii', torch.einsum("...ij, ...kj -> ...ki", S, S))
    trace = 0.0 if S is None else torch.einsum('...ii', torch.einsum("...ij, ...kj -> ...ki", S, S))
    
    if limitXi:
        return (nominator**2 + 1e-14 * particles.supports) / (denominatorLeft**2 + trace + 1e-14 * particles.supports)
    return (nominator**2) / (denominatorLeft**2 + trace + 1e-14 * particles.supports)
    
def compute_vsig(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict,):
    i, j = neighborhood[0].row, neighborhood[0].col
    kernelValues = neighborhood[1]
    c_i, c_j, c_bar = getTerms(i, j, (particles.soundspeeds, particles.soundspeeds))
    # v_i, v_j        = particles_a.velocities[i], particles_b.velocities[j]
    # h_i             = particles_a.supports[i]
    # x_ij, r_ij      = compute_xij(particles_a, particles_b, neighborhood, domain)

    # See Cullen and Dehnen 2010, eq 15
    v_ij            = particles.velocities[i] - particles.velocities[j]
    mu_ij           = torch.einsum('ij,ij->i', v_ij, kernelValues.x_ij) / (torch.linalg.norm(kernelValues.r_ij, dim = -1) + 1e-6 * particles.supports[i])

    vsigs           = c_bar - mu_ij
    vsigs[mu_ij > 0]= 0
    
    # This is an annoying dependency on torch_scatter, should implement something different
    return scatter_max(vsigs, i, dim = 0, dim_size = particles.positions.shape[0])[0]
    
from sphMath.modules.compressible import verbosePrint
from sphMath.kernels import KernelType

@dataclass(slots = True)
class CullenDehnenState:
    alpha0s: torch.Tensor            # Alpha0 values that get integrated over time
    alphas:  torch.Tensor            # The alpha values for the Cullen-Dehnen viscosity model
    M:       Optional[torch.Tensor]  # Gradient Correction matrix from CRKSPH
    M_inv:   Optional[torch.Tensor]  # Correction is performed using the inverse matrix
    div:     Optional[torch.Tensor]  # Divergence of the velocity field
    ddivdt:  Optional[torch.Tensor]  # Second order divergence of the velocity field
    Shear:   Optional[torch.Tensor]  # Shear tensor
    Rot:     Optional[torch.Tensor]  # Rotation tensor
    R:       Optional[torch.Tensor]  # R term from Cullen and Dehnen 2010
    Xi:      Optional[torch.Tensor]  # Limiter Term, unnamed in CRKSPH
    v_sig:   Optional[torch.Tensor]  # Signal Velocity Term

from sphMath.kernels import Kernel_xi

def computeCullenTerms(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {},
        dt: float = 0.0,
        verbose = False):
    
    alpha_min = getSetConfig(config, 'diffusionSwitch', 'alpha_min', 0.02)

    # verbosePrint(verbose, '[Cullen]\t\tComputing Cullen Terms')
    correctVelocityGradient = getSetConfig(config, 'diffusionSwitch', 'correctGradient', False)
    if correctVelocityGradient:
        verbosePrint(verbose, '[Cullen]\t\tComputing M')
        M = computeM(particles, kernel, neighborhood, supportScheme, config)
        M_inv = torch.linalg.pinv(M)        
    else:
        M = None
        M_inv = None

    # verbosePrint(verbose, '[Cullen]\t\tComputing Shear Tensor')
    # div, S, Rot = computeShearTensor(psphState, psphState, domain, wrappedKernel, actualNeighbors, solverConfig, M_inv)
    div = SPHOperation(particles, particles.velocities, kernel, neighborhood[0], neighborhood[1], operation = Operation.Divergence, supportScheme=supportScheme, divergenceMode=DivergenceMode.div, gradientMode = GradientMode.Difference)
    # div = sph_op(psphState, psphState, domain, wrappedKernel, actualNeighbors, 'superSymmetric', 'divergence', 'difference', quantity=(psphState.velocities, psphState.velocities))
    verbosePrint(verbose, '[Cullen]\t\tComputing Limiter')
    R = computeR(particles, kernel, neighborhood, supportScheme, config, div)
    verbosePrint(verbose, '[Cullen]\t\tComputing Xi')
    S = None
    Rot = None
    Xi = computeXi(particles, kernel, neighborhood, supportScheme, config, div, S, R)
    
    # The original Cullen and Dehnen paper already uses ddiv_dt for the local alpha
    # The hopkins modified form only uses ddiv_dt for the temporal evolution of alpha
    verbosePrint(verbose, '[Cullen]\t\tSecond order Divergence')
    # ddiv_dt = computeSecondOrderV(psphState, psphState, domain, wrappedKernel, actualNeighbors, solverConfig, dvdt, dvdt, correctionMatrix=CDState.M_inv)

    # ddiv_dt = sph_op(psphState, psphState, domain, wrappedKernel, actualNeighbors, 'superSymmetric', 'divergence', 'difference', quantity=(dvdt, dvdt))
    ddiv_dt = (particles.divergence - div) / dt
    
    alpha_max = getSetConfig(config, 'diffusionSwitch', 'alpha_max', 2)# from CRKSPH
    # The 1/xi is based on Hopkins' ATHENA paper after eq. F17
    f_kern      = 1/Kernel_xi(kernel, particles.positions.shape[1])# wrappedKernel.xi(domain.dim)
    
    # Eq. 13 in Cullen and Dehnen 2010
    A_i = Xi * (-ddiv_dt).clamp(min = 0)
    v_sig = compute_vsig(particles, kernel, neighborhood, supportScheme, config)
    
    h_i = particles.supports #* f_kern
    scaling = h_i**2 * A_i / (v_sig**2 + h_i **2 * A_i + 1e-14 * h_i)
    alphas = alpha_max * scaling
    
    alpha0s = particles.alpha0s.clone()
    l = 0.05 # See Cullen and Dehnen 2010, eq 16
    tau_i = h_i * f_kern / (2 * l * v_sig)
    alpha_dot = (alphas - alpha0s) / tau_i
    
    # print(f'alphas: {alphas.min()}, {alphas.max()}, {alphas.mean()}')
    # print(f'alpha0s: {alpha0s.min()}, {alpha0s.max()}, {alpha0s.mean()}')
    # print(f'alpha_dot: {alpha_dot.min()}, {alpha_dot.max()}, {alpha_dot.mean()}')
    
    alpha0s = torch.where(alphas > alpha0s, alphas, alpha0s)
    
    alpha0s = torch.where(alphas < alpha0s, alpha0s + alpha_dot * dt, alpha0s)
    
    alpha0s = alpha0s.clamp(min = alpha_min, max = alpha_max)
    
    
    verbosePrint(verbose, '[Cullen]\t\tComputing Alphas')
    # alphas = (Xi * psphState.alpha0s).clamp(min = alpha_min)
    return alphas, CullenDehnenState(
        alpha0s = alpha0s,
        alphas = alphas,
        M = M,
        M_inv = M_inv,
        div = div,
        ddivdt = ddiv_dt,
        Shear = S,
        Rot = Rot,
        R = R,
        Xi = Xi,
        v_sig = v_sig
    )
    
def computeCullenUpdate(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict,
        dt: float,
        dvdt: torch.Tensor,
        CDState: CullenDehnenState,
        verbose = False):#psphState, domain, wrappedKernel, actualNeighbors, solverConfig, dvdt, CDState, dt, verbose = False):
    # The original Cullen and Dehnen paper already performs the update in the computeCullenTerms function
    # So we simply return the state here (for now)

    
    return CDState.alpha0s, CullenDehnenState(
        alpha0s = CDState.alpha0s,
        alphas = CDState.alphas,
        M = CDState.M,
        M_inv = CDState.M_inv,
        div = CDState.div,
        ddivdt = CDState.ddivdt,
        Shear = CDState.Shear,
        Rot = CDState.Rot,
        R = CDState.R,
        Xi = CDState.Xi,
        v_sig = CDState.v_sig
    )

    