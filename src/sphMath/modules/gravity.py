

from torch.profiler import profile, record_function, ProfilerActivity
from sphMath.neighborhood import filterNeighborhood, coo_to_csr
import torch
from sphMath.modules.renorm import computeCovarianceMatrices
from sphMath.neighborhood import computeDistanceTensor
from sphMath.sphOperations.shared import scatter_sum
from sphMath.operations import sph_op
# Maronne surface detection
from sphMath.modules.surfaceDetection import computeNormalsMaronne, detectFreeSurfaceMaronne, computeColorField, detectFreeSurfaceColorFieldGradient, detectFreeSurfaceBarecasco, expandFreeSurfaceMask, computeLambdaGrad, detectFreeSurfaceColorField
import numpy as np
from sphMath.modules.density import computeDensity
from sphMath.modules.shifting.deltaPlusShifting import computeDeltaShifting
from sphMath.modules.shifting.implicitShifting import computeShifting
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
import copy


def computeGravity(fluidState : WeaklyCompressibleState, config):
    if not config.get('gravity', {}).get('active', False):
        return torch.zeros_like(fluidState.velocities)
    with record_function("[SPH] - External Gravity Field (g)"):
        if config.get('gravity', {}).get('mode', 'directional') == 'potential':
            x = fluidState.positions
            minD = config['domain'].min
            maxD = config['domain'].max
            periodic = config['domain'].periodic
            periodicity = torch.tensor([periodic] * config['domain'].dim, dtype = x.dtype, device = x.device) if isinstance(periodic, bool) else periodic

            dtype = minD.dtype
            device = minD.device
            dim = config['domain'].dim


            origin = config.get('gravity', {}).get('potentialOrigin', [0.0] * dim)
            if not isinstance(origin, torch.Tensor):
                origin = torch.tensor(origin, dtype = dtype, device = device)
            center = origin

            xij = x - center
            rij = torch.linalg.norm(xij, dim = -1)
            xij[rij > 1e-7] = xij[rij > 1e-7] / rij[rij > 1e-7, None]

            magnitude = config.get('gravity', {}).get('magnitude', 1)
            return - magnitude**2 * xij * (rij)[:,None] #/ fluidState['fluidDensities'][:,None]

        else:
            v = fluidState.velocities
            direction = config.get('gravity', {}).get('direction', [-1.0] * config['domain'].dim)
            if not isinstance(direction, torch.Tensor):
                direction = torch.tensor(direction, dtype = fluidState['positions'].dtype, device = fluidState['positions'].device)
            return (direction[:v.shape[1]] * config['gravity']['magnitude']).repeat(v.shape[0], 1)

