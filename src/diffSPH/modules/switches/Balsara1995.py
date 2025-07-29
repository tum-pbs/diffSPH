from typing import Union, Optional
from dataclasses import dataclass
import torch


from diffSPH.neighborhood import DomainDescription, SparseCOO
from diffSPH.kernels import SPHKernel
from diffSPH.operations import sph_op
from diffSPH.modules.compressible import CompressibleState
from diffSPH.sphOperations.shared import getTerms, compute_xij, scatter_sum
from diffSPH.modules.switches.CullenDehnen2010 import computeShearTensor, computeM
from diffSPH.modules.compressible import verbosePrint

from diffSPH.modules.switches.common import computeDivergence
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState
from diffSPH.schemes.gasDynamics import CompressibleState
from diffSPH.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from diffSPH.operations import SPHOperation
from diffSPH.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple
from diffSPH.kernels import KernelType


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

    