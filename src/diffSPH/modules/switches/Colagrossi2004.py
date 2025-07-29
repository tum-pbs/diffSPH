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
from diffSPH.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from diffSPH.operations import SPHOperation
from typing import Dict, Tuple
from diffSPH.kernels import KernelType


def computeColagrossi2004Switch(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {},
        dt: float = 0.0,
        verbose = False):
    verbosePrint(verbose, '[Cullen]\t\tComputing Cullen Terms')
    if config.get('correctVelocityGradient', False):
        verbosePrint(verbose, '[Cullen]\t\tComputing M')
        M = computeM(particles, kernel, neighborhood, supportScheme, config)
        M_inv = torch.linalg.pinv(M)        
    else:
        M = None
        M_inv = None
    div, S, Rot = computeShearTensor(particles, kernel, neighborhood, supportScheme, config, M_inv)
    # The notation is confusing at best, Colagrossi 2004 uses the notation sqrt(E:E), Monaghan2005 uses sqrt(E^ij E^ij)
    # In line with the Cullen and Dehnen 2010 scheme we use the trace of the product of the shear tensor with itself
    # This is equivalent to the Frobenius norm of the shear tensor. Note that Colagrossi and Monaghan use the actual norm
    # Cullen and Dehnen only use the squared term
    trace          = torch.einsum('...ii', torch.einsum("...ij, ...kj -> ...ki", S, S))
    norm           = torch.sqrt(trace)
    
    colagrossi = div.abs() / (div.abs() + norm + 1e-14 * particles.supports)

    return colagrossi, None
