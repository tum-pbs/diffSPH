


# from sphMath.modules.compressible import 
from sphMath.modules.adaptiveSmoothingASPH import evaluateOptimalSupportOwen
from sphMath.schemes.gasDynamics import CompressibleSystem, CompressibleUpdate, CompressibleState
import copy
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
from torch.profiler import record_function
from sphMath.schemes.weaklyCompressible import checkTensor
from sphMath.neighborhood import coo_to_csr
from sphMath.operations import SPHOperationCompiled
from sphMath.operations import sph_op, SPHOperation
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode, access_optional
from typing import Dict, Tuple
from sphMath.util import getSetConfig

@torch.jit.script
def computeDensityDeltaTerm_(
        particles: WeaklyCompressibleState,
        delta: float,
        xi: float,
        scheme : str,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
):
    
    i, j                        = neighborhood[0].row, neighborhood[0].col
    h_i, h_j, h_bar             = getTerms(i, j, (particles.supports, particles.supports))
    r_ij, x_ij                  = neighborhood[1].r_ij, neighborhood[1].x_ij        
    n_ij = torch.nn.functional.normalize(x_ij, dim=1)

    if scheme == 'deltaSPH':
        grad_ij = access_optional(particles.gradRhoL, i) + access_optional(particles.gradRhoL, j)
        rho_ij = 2 * (particles.densities[j] - particles.densities[i]) / (r_ij + 1e-6 * h_bar)
        psi_ij = -rho_ij.view(-1,1) * n_ij + grad_ij
    elif scheme == 'denormalized':
        grad_ij = access_optional(particles.gradRho, i) + access_optional(particles.gradRho, j)
        rho_ij = 2 * (particles.densities[j] - particles.densities[i]) / (r_ij + 1e-6 * h_bar)
        psi_ij = -rho_ij.view(-1,1) * n_ij - grad_ij # check this sign!
    elif scheme == 'densityOnly':
        rho_ij = 2 * (particles.densities[j] - particles.densities[i]) / (r_ij + 1e-6 * h_bar)
        psi_ij = -rho_ij.view(-1,1) * n_ij
    elif scheme == 'deltaOnly':
        grad_ij = access_optional(particles.gradRhoL, i) + access_optional(particles.gradRhoL, j)
        psi_ij = -grad_ij
    elif scheme == 'denormalizedOnly':
        grad_ij = access_optional(particles.gradRho, i) + access_optional(particles.gradRho, j)
        psi_ij = -grad_ij
    else:
        raise ValueError("Invalid scheme. Available schemes are: deltaSPH, denormalized, densityOnly, deltaOnly, denormalizedOnly.")
        
    div = SPHOperationCompiled(
        particles,
        quantity = psi_ij,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation=Operation.Divergence,
        gradientMode = GradientMode.Naive,
        supportScheme = supportScheme,
        consistentDivergence= False,
    )
    
    return delta * particles.supports / xi * particles.soundspeeds * div


from sphMath.kernels import KernelType, Kernel_Scale, Kernel_xi
def computeDensityDeltaTerm(
        particles: WeaklyCompressibleState,
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    
    with record_function("[SPH] - [Density Diffusion (deltaSPH)]"):
        neighborhood = (neighborhood[0](neighborhood[0], kernel)) if neighborhood[1] is None else neighborhood
        delta = getSetConfig(config, 'diffusion', 'delta', 0.1)
        scheme = getSetConfig(config, 'diffusion', 'scheme', 'deltaSPH')
        
        return computeDensityDeltaTerm_(
            particles,
            delta,
            Kernel_xi(kernel, particles.positions.shape[1]),
            scheme,
            neighborhood,
            supportScheme
        )