from diffSPH.noise import generateOctaveNoise
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from util import *
from typing import List, Union

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

from sample import sampleVoronoi, smoothValues
import math

def addNoise(
    particleState, config, neighbors,
    grid, noiseAmplitude = 0.1, uMagnitude = 10,
    noiseType: str = 'perlin',
    smoothIter: int = 4,
    seed: int = 42,
):
    u_min = torch.min(grid).cpu().item()
    u_max = torch.max(grid).cpu().item()

    nx = int(math.sqrt(particleState.positions.shape[0]))
    if u_min == u_max:
        u_min = -uMagnitude
        u_max = uMagnitude

    generator = torch.Generator(device=particleState.positions.device)
    generator.manual_seed(seed)

    if noiseType == 'perlin':
        uNoise = sampleVoronoi(particleState.positions, nx * 2, octaves = 2, baseFrequency = 2, seed = seed, config = config)
    elif noiseType == 'uniform':
        # uNoise = torch.rand_like(grid, generator=generator)
        uNoise = torch.rand_like(grid)
    elif noiseType == 'normal':
        # uNoise = torch.randn_like(grid, generator=generator)
        uNoise = torch.randn_like(grid)
    else:
        raise ValueError(f"Unsupported noise type: {noiseType}")

    uNoise = smoothValues(
        uNoise,
        particleState,
        smoothIter, neighbors,
        config
    )
    uNoiseNormalized = (uNoise - torch.min(uNoise)) / (torch.max(uNoise) - torch.min(uNoise))
    uNoise = uNoiseNormalized * (u_max - u_min) + u_min

    return torch.lerp(grid, uNoise, noiseAmplitude)


def populateCGrid(cGrid, cSourceGrid, 
        boundaryC = 0.01, obstacleC = 0.5, defaultC = 1.0,
        randomObstacleC = False, obstacleCRange = (0.3, 0.7)):
    cGrid = torch.ones_like(cGrid) * defaultC

    boundaryIds = torch.unique(cSourceGrid)
    boundaryIds = boundaryIds[boundaryIds != 0]  # Exclude background (0)

    for bid in boundaryIds:
        mask = (cSourceGrid == bid)
        if bid == -1:
            cGrid[mask] = boundaryC
        else:
            if randomObstacleC:
                cGrid[mask] = torch.empty_like(cGrid[mask]).uniform_(*obstacleCRange)
            else:
                cGrid[mask] = obstacleC
    return cGrid

def populateUGrid(uGrid, uSourceGrid, sourceMagnitudes : Union[float, int, List[float]], randomMagnitude = False, magnitudeRange = (-10.0, 10.0)):
    sourceIds = torch.unique(uSourceGrid)
    sourceIds = sourceIds[sourceIds != 0]  # Exclude background (0)

    for sid in sourceIds:
        # print('setting source id', sid)
        mask = (uSourceGrid == sid)
        sourceMagnitude = 0.0
        if randomMagnitude:
            sourceMagnitude = torch.empty(1).uniform_(*magnitudeRange).item()
        elif isinstance(sourceMagnitudes, float) or isinstance(sourceMagnitudes, int):
            sourceMagnitude = sourceMagnitudes
        else:
            sourceMagnitude = sourceMagnitudes[int(sid.item())-1]
        uGrid[mask] = sourceMagnitude
    return uGrid