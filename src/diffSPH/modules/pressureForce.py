import torch
from diffSPH.util import ParticleSet, DomainDescription
from diffSPH.sampling import buildDomainDescription, sampleRegularParticles
from diffSPH.neighborhood import buildNeighborhood, coo_to_csr, SparseCOO
from diffSPH.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product
from diffSPH.kernels import SPHKernel, evalW, evalGradW, evalDerivativeW
# from diffSPH.modules.compressible import CompressibleState

from diffSPH.schemes.gasDynamics import CompressibleState, CompressibleUpdate
from typing import Union
from typing import List
# from diffSPH.modules.compressible import CompressibleState, CompressibleUpdate, verbosePrint
import numpy as np
from typing import Tuple

from diffSPH.sphOperations.shared import getTerms, compute_xij
from diffSPH.modules.viscosity import compute_Pi
from diffSPH.schemes.baseScheme import verbosePrint
from diffSPH.operations import sph_op
from torch.profiler import record_function
from diffSPH.operations import sph_op, SPHOperation
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState
from diffSPH.schemes.gasDynamics import CompressibleState
from diffSPH.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from diffSPH.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple
from diffSPH.util import getSetConfig
from diffSPH.enums import KernelCorrectionScheme

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
                # correctionTerms=[KernelCorrectionScheme.gradH] if particles.omega is not None else None
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
            # correctionTerms=[KernelCorrectionScheme.gradH] if particles.omega is not None else None
            ) / particles.densities.view(-1,1)


        

