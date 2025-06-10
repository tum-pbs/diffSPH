from typing import Union, Optional
from dataclasses import dataclass
import torch

from sphMath.neighborhood import DomainDescription, SparseCOO
from sphMath.kernels import SPHKernel
from sphMath.operations import sph_op
from sphMath.modules.compressible import CompressibleState
from sphMath.sphOperations.shared import getTerms, compute_xij, scatter_sum
from sphMath.modules.switches.CullenDehnen2010 import computeShearTensor, computeM
from sphMath.modules.compressible import verbosePrint

from sphMath.modules.switches.common import computeDivergence
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.operations import SPHOperation
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple
from sphMath.kernels import KernelType


def computeMorrisMonaghan1997Switch(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {},
        dt: float = 0.0,
        verbose = False):
    div     = computeDivergence(particles, kernel, neighborhood, supportScheme, config)
    # Computation of tau based on Morris and Monaghan1997 eq 11
    # Delta sets the decay rate, based on 11 this is a free parameter in the range of 2 to 5
    # delta = 3
    # M2 = torch.sqrt((solverConfig['gamma'] - 1) / (2 * solverConfig['gamma']))    
    # l1 = 1 / delta * M2
    # However, Morris and Monaghan 1997 use a fixed value of l1 = 0.2
    l1 = 0.2
    tau = particles.supports / l1 * particles.soundspeeds
       
    S = (-div).clamp(min = 0)
    
    alpha0 = 0.1
    alphas = particles.alpha0s
    
    dalphadt = -(alphas - alpha0) / tau + S
    
    return alphas, dalphadt
    