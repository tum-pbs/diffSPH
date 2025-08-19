import torch
from diffSPH.sdf import operatorDict
from diffSPH.sampling import generateNoiseInterpolator, computeDensity

def rampDivergenceFree(positions, noise, sdf_func, offset, d0 = 0.25):
    sdf = sdf_func(positions)
    r = (sdf - offset) / d0  -0.5
    # r = (sdf - offset) / d0 / 0.5 - 1
#     ramped = r * r * (3 - 2 * r)
    ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
    # ramped = torch.tanh(r)
    # ramped = torch.sigmoid(r * 6)
    # ramped = r
    ramped[ramped >= 1] = 1
    ramped[ramped <= -1] = -1
#     ramped[r <= 0] = 0
#     ramped[r <= -1] = -1
    
    # return sdf
    # return ramped * (noise)
    return (ramped /2 + 0.5) * (noise)


def generateRamp(perennialState, config):
    regions = config['regions']
    boundary_sdfs = [region['sdf'] for region in regions if region['type'] == 'boundary']
    # print(boundary_sdfs)
    combined_sdf = lambda x: boundary_sdfs[0](x)[0]
    for sdf in boundary_sdfs[1:]:
        combined_sdf = operatorDict['union'](combined_sdf, lambda x, sdf = sdf: sdf(x)[0])


    buffer = config.get('boundary', {}).get('potentialBuffer', 4)
    buffer = 4
    # print(f'Using buffer {buffer} for ramp generation')
    dx = config['particle']['dx']
    offset = config['particle']['dx'] * 6
    offset = perennialState.supports
    # d0 = buffer * perennialState.supports
    # print(f'Using buffer {d0} for ramp generation')
    d0 = 0.25

    ramp = rampDivergenceFree(perennialState.positions, torch.ones_like(perennialState.densities), combined_sdf,
                              offset=offset,
                              d0=d0)
    return ramp


from diffSPH.neighborhood import filterNeighborhood, filterNeighborhoodByKind, coo_to_csr, buildNeighborhood, computeDistanceTensor, evaluateNeighborhood, SupportScheme
from diffSPH.operations import SPHOperation, Operation, GradientMode

def sampleRamp(particleState, potentialField, domain, config, neighborhood, neighbors, smoothingSteps  = 4):
    ramp = generateRamp(particleState, config) if len([r for r in config['regions'] if r['type'] == 'boundary']) > 0 else torch.ones_like(particleState.densities)

    neighs = neighbors.get('noghost')

    rho = computeDensity(particleState, config['kernel'], neighs, SupportScheme.Gather, config) 
    priorDensity = particleState.densities.clone() 
    particleState.densities = rho
    
    smoothedRamp = ramp.clone()
    for i in range(smoothingSteps):
        smoothedRamp = SPHOperation(
            particleState,
            quantity = smoothedRamp,
            kernel = config['kernel'],
            neighborhood = neighs[0],
            kernelValues = neighs[1],
            operation=Operation.Interpolate,
            supportScheme = SupportScheme.Gather,
        )
    potential = potentialField * smoothedRamp
    gradTerm = SPHOperation(particleState, potential, config['kernel'], neighs[0], neighs[1], Operation.Gradient, SupportScheme.Gather, GradientMode.Difference)
    particleState.densities = priorDensity

    velocities = torch.stack([gradTerm[:,1], -gradTerm[:,0]], dim = 1)
    velocities = velocities / torch.linalg.norm(velocities, dim = 1, keepdim = True).max()

    velocities[particleState.kinds != 0, :] = 0

    return velocities, smoothedRamp


def sampleDivergenceFreeNoise(particleState, domain, config, nxGrid, octaves = 3, lacunarity = 2, persistence = 0.5, baseFrequency = 1, tileable = True, kind = 'perlin', seed = 45906734, smoothingSteps = 4):
    neighborhood, neighbors = evaluateNeighborhood(particleState, domain, config['kernel'], verletScale = config['neighborhood']['verletScale'], mode =  SupportScheme.SuperSymmetric, priorNeighborhood=None)
    
    noiseGen = generateNoiseInterpolator(nxGrid, nxGrid, domain, dim = domain.dim, octaves = octaves, lacunarity = lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = seed)
    potential = noiseGen(particleState.positions).to(particleState.positions.dtype)

    velocities, smoothedRamp = sampleRamp(particleState, potential, domain, config, neighborhood, neighbors, smoothingSteps)
    return velocities, potential, smoothedRamp

import numpy as np

def sampleTGV(particleState, domain, config, k_, smoothingSteps = 4):
    neighborhood, neighbors = evaluateNeighborhood(particleState, domain, config['kernel'], verletScale = config['neighborhood']['verletScale'], mode =  SupportScheme.SuperSymmetric, priorNeighborhood=None)

    ktgv = k_ / 2
    if k_ % 2 == 0:
        phaseShift_x = np.pi / 2# / k
        phaseShift_y = np.pi / 2# / k
    else:
        phaseShift_x = 0
        phaseShift_y = 0
    k = k_/2
    if k % 2 == 0:
        potential = torch.sin(np.pi * k * particleState.positions[:,0]) * torch.sin(np.pi * k * particleState.positions[:,1]) / 6
    else:
        potential = torch.cos(np.pi * k * particleState.positions[:,0] + phaseShift_x) * torch.cos(np.pi * k * particleState.positions[:,1] + phaseShift_y) / 6


    velocities, smoothedRamp = sampleRamp(particleState, potential, domain, config, neighborhood, neighbors, smoothingSteps)
    return velocities, potential, smoothedRamp




def plotScalar(fig, axis, label, fluidParticles, quantity, domain, solverConfig, s, gridVisualization=True, gridResolution=256, mapping = None, cmap = 'viridis', vmin=None, vmax=None, operator = None):
    return visualizeParticles(fig, axis,
                     particles = fluidParticles, 
                     domain = domain, 
                     quantity = quantity, 
                     which = 'fluid',
                     visualizeBoth=False,
                     kernel = solverConfig['kernel'],
                     plotDomain = False,
                     operation= operator,
                    #  scaling = 'sym',
                     cmap = cmap,
                     midPoint=1.5,
                     mapping = mapping,
                     title=label,
                     markerSize = s,
                     gridVisualization=gridVisualization, gridResolution=gridResolution, vmin=vmin, vmax=vmax)


import matplotlib.pyplot as plt
from diffSPH.plotting import visualizeParticles, updatePlot

def plotSampling(particleState, domain, config, velocities, ramp, potential, s = 4):
    fig, axis = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False, sharex=True, sharey=True)

    # s = 4
    fluidParticles = particleState

    visualizeParticles(fig, axis[0,0],
                        particles = fluidParticles, 
                        domain = domain, 
                        quantity = velocities, 
                        which = 'both',
                        visualizeBoth=True,
                        kernel = config['kernel'],
                        plotDomain = False,
                        # operation= 'divergence',
                        #  scaling = 'sym',
                        cmap = 'viridis',
                        midPoint=1.5,
                        mapping = 'L2',
                        title=None,
                        markerSize = s,
                        gridVisualization=False, gridResolution=255, vmin=None, vmax=None, streamLines=False)

    # internalEnergyPlotState = plotScalar(fig, axis[0,1], 'Density $\\rho$', fluidParticles, fluidParticles.densities ,     domain, config, s, gridVisualization=True, gridResolution=256, cmap = 'RdBu_r', mapping = None)
    velocityXPlotState      = plotScalar(fig, axis[0,1], 'Velocity X',          fluidParticles, potential,          domain, config, s, gridVisualization=False, gridResolution=255, mapping = 'L2', cmap = 'viridis')
    velocityYPlotState      = plotScalar(fig, axis[0,2], 'Velocity Y',          fluidParticles, ramp,          domain, config, s, gridVisualization=False, gridResolution=255, mapping = '.x', cmap = 'RdBu_r')
    
    fig.tight_layout()

    return fig, axis

