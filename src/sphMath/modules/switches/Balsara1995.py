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


def computeBalsara1995Switch(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {},
        dt: float = 0.0,
        verbose = False):
    div     = computeDivergence(particles, kernel, neighborhood, supportScheme, config)
    curl    = SPHOperation(particles, particles.velocities, kernel, neighborhood[0], neighborhood[1], operation = Operation.Curl, supportScheme=supportScheme, gradientMode=GradientMode.Difference)
    
    # curl    = sph_op(psphState, psphState, domain, wrappedKernel, actualNeighbors, 'superSymmetric', 'curl', 'difference', quantity=(psphState.velocities, psphState.velocities))
    
    balsara = div.abs() / (div.abs() + torch.linalg.norm(curl, dim = -1) + 1e-14 * particles.supports)
    
    return balsara, None

    