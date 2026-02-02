from diffSPH.noise import generateOctaveNoise
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from util import *

def sampleVoronoi(positions, nGrid, octaves = 2, baseFrequency = 1, kind = 'perlin', tileable=True, seed = 12365, vmin = 0.0, vmax = 1.0, config=None):
    positions = getPeriodicPositions(positions, config['domain'])
    xx, yy , noise = generateOctaveNoise(n = nGrid * 4, dim = 2, octaves = octaves, baseFrequency = baseFrequency, kind = kind, tileable=tileable, seed = seed)
    cTarget = noise / 2 + 0.5
    cTarget = vmin + (vmax - vmin) * cTarget
    cInterp = RegularGridInterpolator((np.linspace(-1,1,cTarget.shape[0]), np.linspace(-1,1,cTarget.shape[1])), cTarget.numpy())
    cGrid = torch.tensor(cInterp(positions.cpu())).to('cuda')
    return cGrid.to(positions.device, positions.dtype)


def smoothValues(quantity, particleState, nIters, neighbors, config):
    sampled  = quantity.clone()
    for _ in range(nIters):
        sampled = SPHOperation(
            particleState,
            sampled,
            config['kernel'],
            neighbors.get('noghost')[0],
            neighbors.get('noghost')[1],
            Operation.Interpolate,
            supportScheme = SupportScheme.Gather)
    return sampled


def smoothState(state, particleState, smoothIters, neighbors, config):
    smoothState = WaveEquationState(
        u = smoothValues(state.u, particleState, smoothIters, neighbors, config),
        v = smoothValues(state.v, particleState, smoothIters, neighbors, config),
        c = smoothValues(state.c, particleState, smoothIters, neighbors, config),
        damping = smoothValues(state.damping, particleState, smoothIters, neighbors, config)
    )    
    return smoothState