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

from sphMath.modules.switches.CullenDehnen2010 import computeSecondOrderV, computeXi, computeR, compute_vsig
    
from sphMath.kernels import KernelType


    
from sphMath.util import scatter_max, getSetConfig
from sphMath.neighborhood import evalKernel, evalKernelGradient

from sphMath.modules.compressible import verbosePrint

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


def computeHopkinsTerms(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
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
    div, S, Rot = computeShearTensor(particles, kernel, neighborhood, supportScheme, config, M_inv)
    # div = sph_op(psphState, psphState, domain, wrappedKernel, actualNeighbors, 'superSymmetric', 'divergence', 'difference', quantity=(psphState.velocities, psphState.velocities))
    verbosePrint(verbose, '[Cullen]\t\tComputing Limiter')
    R = computeR(particles, kernel, neighborhood, supportScheme, config, div)
    verbosePrint(verbose, '[Cullen]\t\tComputing Xi')
    # S = None
    # Rot = None
    Xi = computeXi(particles, kernel, neighborhood, supportScheme, config, div, S, R)
    
    verbosePrint(verbose, '[Cullen]\t\tComputing Alphas')
    alphas = (Xi * particles.alpha0s).clamp(min = alpha_min)
    return alphas, CullenDehnenState(
        alpha0s = particles.alpha0s,
        alphas = alphas,
        M = M,
        M_inv = M_inv,
        div = div,
        ddivdt = None,
        Shear = S,
        Rot = Rot,
        R = R,
        Xi = Xi,
        v_sig = None
    )
    
from sphMath.kernels import Kernel_xi
def computeHopkinsUpdate(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme,
        config: Dict,
        dt: float,
        dvdt : torch.Tensor, 
        CDState : CullenDehnenState, 
        verbose = False):
    # This implements the Hopkins modified form
    
    alpha_max = getSetConfig(config, 'diffusionSwitch', 'alpha_max', 2) # from CRKSPH
    # The 1/xi is based on Hopkins' ATHENA paper after eq. F17
    f_kern      = 1/Kernel_xi(kernel, particles.positions.shape[1])
    # f_kern = 1/3 # hard coded result for cubic spline from Hopkins 2015, Hopkins also uses h=1 as the cutoff so these terms _should_ be equivalent?
    # f_kern = 1
    beta_c = getSetConfig(config, 'diffusionSwitch', 'beta_c', 0.7)
    beta_d = getSetConfig(config, 'diffusionSwitch', 'beta_d', 0.05)

    # verbosePrint(verbose, '[Cullen]\t\tComputing Cullen Update')
    verbosePrint(verbose, '[Cullen]\t\tSecond order Divergence')
    # ddiv_dt = computeSecondOrderV(psphState, psphState, domain, wrappedKernel, actualNeighbors, solverConfig, dvdt, dvdt, correctionMatrix=CDState.M_inv)

    # ddiv_dt = sph_op(psphState, psphState, domain, wrappedKernel, actualNeighbors, 'superSymmetric', 'divergence', 'difference', quantity=(dvdt, dvdt))
    ddiv_dt = (CDState.div - particles.divergence) / dt

    verbosePrint(verbose, '[Cullen]\t\tComputing Vsig')
    # There is an issue here!
    # Hopkins and CRKSPH both use the same offset term offset = beta_c * c_s^2 / (f_kern * h)^2
    # In CRKSPH F.5 for some reason the equation reads as alpha_max * |ddiv_dt| / (alpha_max * |ddiv_dt| + offset)
    # Whereas Hopkins uses alpha_max * |ddiv_dt| / (|ddiv_dt| + offset)
    
    # CRKSPH form
    # alpha_div = alpha_max * ddiv_dt.abs()
    # alpha_tmp = alpha_div / (alpha_div + beta_c * psphState.soundspeeds**2 / (f_kern * psphState.supports)**2)
    # Hopkins form
    alpha_div = ddiv_dt.abs()
    alpha_tmp = alpha_max * alpha_div / (alpha_div + beta_c * particles.soundspeeds**2 / (f_kern * particles.supports)**2 + 1e-14)
    
    # print(f'alpha_max: {alpha_max}')
    # print(f'alpha_div: {ddiv_dt.min()}, {ddiv_dt.max()}, {ddiv_dt.mean()}')
    # print(f'alpha_tmp: {alpha_tmp.min()}, {alpha_tmp.max()}, {alpha_tmp.mean()}')
    term = beta_c * particles.soundspeeds**2 / (f_kern * particles.supports)**2
    # print(f'term: {term.min()}, {term.max()}, {term.mean()}')
    # print(f'c_s: {psphState.soundspeeds.min()}, {psphState.soundspeeds.max()}, {psphState.soundspeeds.mean()}')
    
    
    alpha_tmp = torch.where(torch.logical_or(ddiv_dt > 0, CDState.div > 0), 0, alpha_tmp)
    # print(f'alpha_tmp(c): {alpha_tmp.min()}, {alpha_tmp.max()}, {alpha_tmp.mean()}')

    v_sig = compute_vsig(particles, kernel, neighborhood, supportScheme, config)

    verbosePrint(verbose, '[Cullen]\t\tComputing Alpha0s')
    alpha_0 = particles.alpha0s
    alpha_0_next = alpha_tmp + (alpha_0 - alpha_tmp) * torch.exp(- beta_d * dt * v_sig / (2 * f_kern * particles.supports))
    alpha_0_next = torch.where(alpha_tmp >= alpha_0, alpha_tmp, alpha_0_next)
    alpha0s = alpha_0_next
    
    return alpha0s, CullenDehnenState(
        alpha0s = alpha0s,
        alphas = CDState.alphas,
        M = CDState.M,
        M_inv = CDState.M_inv,
        div = CDState.div,
        ddivdt = ddiv_dt,
        Shear = CDState.Shear,
        Rot = CDState.Rot,
        R = CDState.R,
        Xi = CDState.Xi,
        v_sig = v_sig
    )

    