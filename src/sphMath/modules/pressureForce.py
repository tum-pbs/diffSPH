import torch
from sphMath.util import ParticleSet, DomainDescription
from sphMath.sampling import buildDomainDescription, sampleRegularParticles
from sphMath.neighborhood import buildNeighborhood, coo_to_csr, SparseCOO
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product
from sphMath.kernels import SPHKernel, evalW, evalGradW, evalDerivativeW
# from sphMath.modules.compressible import CompressibleState

from sphMath.schemes.gasDynamics import CompressibleState, CompressibleUpdate
from typing import Union
from typing import List
# from sphMath.modules.compressible import CompressibleState, CompressibleUpdate, verbosePrint
import numpy as np
from typing import Tuple

from sphMath.sphOperations.shared import getTerms, compute_xij
from sphMath.modules.viscosity import compute_Pi
from sphMath.schemes.baseScheme import verbosePrint
from sphMath.operations import sph_op
from torch.profiler import record_function
from sphMath.operations import sph_op, SPHOperation
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple
from sphMath.util import getSetConfig
from sphMath.enums import KernelCorrectionScheme

def computePressureForce(
        particles: WeaklyCompressibleState,
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather,
        config: Dict = {}):
    
    with record_function("[SPH] - [Pressure Gradient]"):
        i, j = neighborhood[0].row, neighborhood[0].col
        pressureTerm = getSetConfig(config, 'pressure', 'term', 'Antuono')
        p_i = particles.pressures[i]
        p_j = particles.pressures[j]

        gradH = particles.omega if hasattr(particles, 'omega') else None

        p_ij = None
        if pressureTerm == 'conservative':
            p_ij = p_j - p_i
        elif pressureTerm == 'nonConservative':
            p_ij = p_i + p_j
        elif pressureTerm == 'Antuono':
            switch = p_i >= 0.0
            if hasattr(particles, 'surfaceMask') and particles.surfaceMask is not None:
                surfaceMask = particles.surfaceMask[i]
                switch = torch.logical_or(switch, surfaceMask > 0.5)
            # switch = torch.logical_or(switch, particles.kinds[j] != 0)
            p_ij = torch.where(switch, p_j + p_i, p_j - p_i)
        elif pressureTerm == 'i':
            p_ij = p_i
        elif pressureTerm == 'j':
            p_ij = p_j
        elif pressureTerm == 'symmetric':
            return -SPHOperation(
                particles,
                particles.pressures,
                kernel,
                neighborhood = neighborhood[0],
                kernelValues = neighborhood[1],
                operation=Operation.Gradient,
                gradientMode = GradientMode.Symmetric,
                supportScheme = supportScheme,
                correctionTerms=[KernelCorrectionScheme.gradH] if particles.omega is not None else None
                ) / particles.densities.view(-1,1)
        else:
            raise ValueError(f'Unknown pressure term: {pressureTerm}')
        
        return -SPHOperation(
            particles,
            quantity = p_ij,
            kernel = kernel,
            neighborhood = neighborhood[0],
            kernelValues = neighborhood[1],
            operation=Operation.Gradient,
            gradientMode = GradientMode.Naive,
            supportScheme = supportScheme,
            correctionTerms=[KernelCorrectionScheme.gradH] if particles.omega is not None else None
            ) / particles.densities.view(-1,1)


        

