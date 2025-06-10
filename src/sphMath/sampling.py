from sphMath.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr
from sphMath.util import volumeToSupport
from sphMath.util import ParticleSet
import torch
import numpy as np


def buildDomainDescription(l, dim, periodic = False, device = 'cpu', dtype = torch.float32):
    minDomain = [-l/2] * dim
    maxDomain = [l/2] * dim
    return DomainDescription(torch.tensor(minDomain, device = device, dtype = dtype), torch.tensor(maxDomain, device = device, dtype = dtype), torch.tensor([periodic] * dim, dtype = torch.bool, device = device) if isinstance(periodic,bool) else periodic, dim)


def buildPointCloud(nx, domain: DomainDescription = None, targetNeighbors = 16, jitter = 0.0, band = 0, shortEdge = True):
    periodicity = domain.periodic
    dxs = []
    for d in range(domain.dim):
        l = domain.max[d] - domain.min[d]
        dx = l/(nx if periodicity[d] else nx-1)
        # x = torch.linspace(domain.min[d] + offset, domain.max[d] - offset, nx + band * 2, device = domain.min.device, dtype = domain.min.dtype)
        # spaces.append(x)
        dxs.append(dx)

    spaces = []
    if shortEdge:
        dx = torch.min(torch.tensor(dxs))
    else:
        dx = torch.max(torch.tensor(dxs))
    # print(dxs, dx, nx)
    ns = []
    for d in range(domain.dim):
        l = domain.max[d] - domain.min[d]
        nd = (torch.ceil(l/dx)).to(torch.int32)
        dn = l / (nd if periodicity[d] else nd-1)
        offset = dx/2 if periodicity[d] else 0
        offset -= dx * band
        x = torch.linspace(domain.min[d] + offset, domain.max[d] - offset, nd + band * 2, device = domain.min.device, dtype = domain.min.dtype)
        spaces.append(x)
        ns.append(nd + band * 2)

    print(f'{shortEdge}: dxs: {dxs}, ns: {ns}, nx: {nx}')




        # print(f'dim: {d}, nx: {nx}, dx: {dx}, min: {x.min()}, max: {x.max()}, periodic: {periodicity[d]}, dxActual: {x[1] - x[0]}')
    # print(dxs)
    grid = torch.meshgrid(*spaces, indexing='ij')
    pos = torch.stack([g.flatten() for g in grid], dim=1)
    # mean_dx = torch.mean(torch.tensor(dxs))
    if jitter > 0:
        pos += torch.rand_like(pos) * dx * jitter
    area = dx ** domain.dim
    support = volumeToSupport(area, targetNeighbors, domain.dim)
    supports = torch.ones_like(pos[:, 0]) * support
    return PointCloud(positions = pos, supports = supports), area, support

def getSpacing(nx, domain: DomainDescription):
    periodicity = domain.periodic
    dxs = []
    for d in range(domain.dim):
        l = domain.max[d] - domain.min[d]
        dx = l/(nx if periodicity[d] else nx-1)
        dxs.append(dx.item())
    return np.min(dxs)

def sampleRegularParticles(nx : int, domain : DomainDescription, targetNeighbors: int, jitter = 0.0, band = 0, shortEdge=True):
    pc, area, support = buildPointCloud(nx, domain, targetNeighbors, jitter = jitter, band = band, shortEdge = shortEdge)
    return ParticleSet(positions = pc.positions, supports = pc.supports, masses = torch.ones_like(pc.positions[:, 0]) * area, 
    densities = torch.ones_like(pc.positions[:, 0]))

def filterParticles_wSDF(particles: ParticleSet, sdf):
    d, n = sdf(particles.positions)
    mask = d > 0
    return ParticleSet(
        positions = particles.positions[mask],
        supports = particles.supports[mask],
        masses = particles.masses[mask],
        densities = particles.densities[mask],
    )

from sphMath.noise import generateNoise 
from scipy.interpolate import RegularGridInterpolator

def generateNoiseInterpolator(fluidResolution, noiseResolution, domain: DomainDescription, dim = 2, octaves = 3, lacunarity=2, persistence = 0.5, baseFrequency = 1, tileable = False, kind = 'perlin', seed = 1248097):
    _, _, noiseField = generateNoise(noiseResolution, dim = dim, octaves = octaves, lacunarity=lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = seed)
    grid = []
    dx = 1/2 * getSpacing(fluidResolution, domain)
    for d in range(domain.dim):
        grid.append(np.linspace(domain.min[d].detach().cpu() + dx , domain.max[d].detach().cpu() - dx, noiseResolution))
    interpolator = RegularGridInterpolator(grid, noiseField.cpu().numpy(), bounds_error=False, fill_value=None)
    return lambda x: torch.tensor(interpolator(x.cpu().numpy()).reshape(x.shape[0])).to(x.device)



from typing import Union
from sphMath.operations import sph_op
from sphMath.kernels import SPHKernel, getSPHKernelv2, KernelType, Kernel_N_H
from sphMath.neighborhood import buildNeighborhood
from sphMath.util import ParticleSetWithQuantity

def mapToGrid(
    fluidParticles      : Union[ParticleSet, ParticleSetWithQuantity],
    quantity            : torch.Tensor, 
    domain              : DomainDescription,
    kernel              : KernelType = getSPHKernelv2('Wendland4'),
    gridResolution      : int = 128,
    ):
    dx = getSpacing(gridResolution, domain)

    longestDomainSide = torch.max(domain.max - domain.min)
    shortestDomainSide = torch.min(domain.max - domain.min)
    dxGrid = shortestDomainSide / gridResolution
    targetNeighbors = Kernel_N_H(kernel, domain.dim)

    gridPos = []
    gridResolution = []
    for i in range(domain.dim):
        gridLength = domain.max[i] - domain.min[i]
        currentResolution = int(gridLength / dxGrid)

        gridPos.append(torch.linspace(domain.min[i], domain.max[i], currentResolution, device = domain.min.device, dtype = domain.min.dtype))
        gridResolution.append(currentResolution)
    
    grid = torch.meshgrid(gridPos, indexing='ij')
    gridPositions = torch.stack(grid, dim = -1).view(-1, domain.dim)
    gridCloud = ParticleSet(positions = gridPositions, supports = torch.ones_like(gridPositions[:, 0]) * volumeToSupport(dx ** domain.dim,targetNeighbors, domain.dim ), masses = None, densities = None)

    gridNeighborhood, sparseNeighborhood = buildNeighborhood(gridCloud, fluidParticles, domain, verletScale= 1.0)

    quantizedParticles = ParticleSetWithQuantity(fluidParticles.positions, fluidParticles.supports, fluidParticles.masses, fluidParticles.densities, quantity)

    interpolated = sph_op(gridCloud, quantizedParticles, domain, getSPHKernelv2(kernel), sparseNeighborhood, operation = 'interpolate', supportScheme='scatter')

    return grid, interpolated.view(gridResolution)


import random
from sphMath.sampling import getSpacing
from sphMath.operations import sph_op
from sphMath.modules.density import computeDensity

def generateTestData(nx, l, dim, kernel, jitter, field = 'constant:0', device = 'cpu', dtype = torch.float32, periodic = True):
    wrappedKernel = getSPHKernelv2(kernel)
    targetNeighbors = wrappedKernel.N_H(dim)

    domain = buildDomainDescription(l, dim, periodic = periodic, device = device, dtype = dtype)
    particles_a = sampleRegularParticles(nx, domain, targetNeighbors, jitter = jitter)

    neighborhood = buildNeighborhood(particles_a, particles_a, domain, verletScale= 1.5)
    actualNeighbors = filterNeighborhood(neighborhood)

    CSR, CSC = coo_to_csrsc(actualNeighbors)

    sampledField = torch.zeros_like(particles_a.positions[:, 0])

    if field.startswith('constant'):
        value = float(field.split(':')[1])
        sampledField = torch.ones_like(sampledField) * value
    elif field.startswith('noise'):
        if field == 'noise':
            seed = random.randint(0, 1000000)
            noiseGen = generateNoiseInterpolator(nx, nx, domain, dim = 2, octaves = 3, lacunarity=2, persistence = 0.5,             baseFrequency = 1, tileable = True, kind = 'perlin', seed = 1248097)
            sampledField = noiseGen(particles_a.positions)
        elif 'x' not in field:
            vectorDim = int(field.split(':')[1])
            seeds = [random.randint(0, 1000000) for _ in range(vectorDim)]
            sampled = []
            for seed in seeds:
                noiseGen = generateNoiseInterpolator(nx, nx, domain, dim = 2, octaves = 3, lacunarity=2, persistence = 0.5,             baseFrequency = 1, tileable = True, kind = 'perlin', seed = seed)
                sampled.append(noiseGen(particles_a.positions))
                
            sampledField = torch.stack(sampled, dim = -1)
    elif field.startswith('rand'):
        if field == 'rand':
            sampledField = torch.rand_like(sampledField)
        elif 'x' not in field:
            vectorDim = int(field.split(':')[1])
            sampledField = torch.rand(particles_a.positions.shape[0], vectorDim, device = device, dtype = dtype)
    elif field == 'sin.x':
        sampledField = torch.sin(particles_a.positions[:, 0] * np.pi)
    elif field == 'sin.y':
        sampledField = torch.sin(particles_a.positions[:, 1] * np.pi)
    elif field == 'sin.xy':
        sampledField = torch.sin(particles_a.positions[:, 0] * np.pi) * torch.sin(particles_a.positions[:, 1] * np.pi)
    elif field == 'exp':
        sampledField = particles_a.positions * torch.exp(-torch.linalg.norm(particles_a.positions, dim = 1)**2 * 4).view(-1,1)
    elif field.startswith('tgv'):
        q = torch.zeros_like(particles_a.positions)
        kf = 2
        if ':' in field:
            kf = float(field.split(':')[1])

        k = kf * np.pi

        pos = particles_a.positions
        if domain.dim == 2:
            if kf % 2 == 0:
                print('k is even')
                q[:,0] =  1 * torch.cos(k/2 * pos[:,0] + k/4) * torch.sin(k/2 * pos[:,1] + k/4)
                q[:,1] = -1 * torch.sin(k/2 * pos[:,0] + k/4) * torch.cos(k/2 * pos[:,1] + k/4)
            else:
                q[:,0] =  1 * torch.cos(k/2 * pos[:,0]) * torch.sin(k/2 * pos[:,1])
                q[:,1] = -1 * torch.sin(k/2 * pos[:,0]) * torch.cos(k/2 * pos[:,1])
        elif domain.dim == 3:
            q[:,0] =  1 * torch.cos(k/2 * pos[:,0]) * torch.sin(k * pos[:,1]) * torch.sin(k/2 * pos[:,2])
            q[:,1] = -1 * torch.sin(k/2 * pos[:,0]) * torch.cos(k * pos[:,1]) * torch.sin(k/2 * pos[:,2])
            q[:,2] =  1 * torch.sin(k/2 * pos[:,0]) * torch.sin(k * pos[:,1]) * torch.cos(k/2 * pos[:,2])
        sampledField = q
        # pos.clone() * torch.exp(-torch.linalg.norm(pos, dim = 1)**2 * 4).view(-1,1)
    elif field.startswith('divFree'):
        octaves = 3
        baseFrequency = 1
        if ':' in field and 'x' not in field:
            octaves = int(field.split(':')[1])
        elif 'x' in field:
            octaves = int(field.split(':')[1].split('x')[1])
            baseFrequency = int(field.split(':')[1].split('x')[0])
        noise_gen = generateNoiseInterpolator(nx, nx, domain, dim = 2, octaves = octaves, lacunarity=2, persistence = 0.5, baseFrequency = baseFrequency, tileable = True, kind = 'perlin', seed = 1248097)
        
        potential = noise_gen(particles_a.positions).to(device).to(dtype)

        rho = computeDensity(particles_a, particles_a, domain, wrappedKernel, actualNeighbors, 'symmetric')
        particles_a = particles_a._replace(densities = rho)
        gradTerm = sph_op(particles_a, particles_a, domain, wrappedKernel, actualNeighbors, operation = 'gradient', supportScheme='symmetric', quantity= potential, gradientMode = 'difference')

        velocities = torch.stack([gradTerm[:,1], -gradTerm[:,0]], dim = -1)

        velocities = velocities / torch.max(torch.linalg.norm(velocities, dim = 1)).view(-1,1)

        sampledField = velocities


    



    return ParticleSetWithQuantity(particles_a.positions, particles_a.supports, particles_a.masses, particles_a.densities, sampledField), domain, wrappedKernel, actualNeighbors, CSR, CSC, getSpacing(nx, domain)


def sampleShell(nx, domain, targetNeighbors, circle = True, extraRings = 0):
    dx = (domain.max[0] - domain.min[0]) / nx
    # dx = dx.to(torch.float64)
    area = dx ** 2

    # print(area)
    # 
    # pi dr² = dx^2
    # dr = sqrt(dx^2/pi)
    dr_init = torch.sqrt(dx ** 2 / np.pi)

    shells = []

    shells.append((dr_init, torch.tensor([[0,0]], device = dx.device, dtype = dx.dtype), torch.tensor([dx**2], device = dx.device, dtype = dx.dtype)))

    # print(shells)
    remaining = extraRings
    for i in range(1, nx * 2):
        dr_hat = dr_init
        r_beg = torch.sum(torch.hstack([s[0] for s in shells]))
        r_end = r_beg + dr_hat

        shellArea = (np.pi * r_end ** 2 - np.pi * r_beg ** 2)
        # print(f'{i} : n_hat = {shellArea / area}')
        n_hat = torch.ceil(shellArea / area).to(torch.int32)

        optimalArea = n_hat * dx**2

        # pi (r_beg + dr)^2 - pi r_beg^2 = n_hat dx^2

        dr = torch.sqrt(r_beg**2 + n_hat * dx**2 / np.pi) - r_beg

        shellAreaOptimized = (np.pi * (r_beg + dr) ** 2 - np.pi * r_beg ** 2)
        # print(f'{i} : {shellAreaOptimized / optimalArea}, optimal Area: {optimalArea}, shell Area: {shellAreaOptimized}, {dr/ dr_hat}')

        dTheta = 2 * np.pi / n_hat
        theta = torch.linspace(0, 2 * np.pi, n_hat + 1, dtype = dx.dtype, device = dx.device)[:-1]

        theta += random.random() * dTheta

        x = (r_beg + dr_hat / 2) * torch.cos(theta)
        y = (r_beg + dr_hat / 2) * torch.sin(theta)
        pts = torch.stack([x,y], dim = 1)
        # print(pts.shape)
        areas = torch.ones(n_hat, device = dx.device, dtype = dx.dtype) * shellAreaOptimized / n_hat
        if circle:
            if r_beg + dr / 2> (domain.max[0] - domain.min[0]) / 2:
                remaining -= 1
                if (remaining < 0):
                    break
        else:
            insidePtcls = (pts[:, 0] > domain.min[0]) & (pts[:, 0] < domain.max[0]) & (pts[:, 1] > domain.min[1]) & (pts[:, 1] < domain.max[1])
            if not insidePtcls.any():
                remaining -= 1
                if (remaining < 0):
                    break
                # break
            pts = pts[insidePtcls]
            areas = areas[insidePtcls]

        shells.append((dr, pts, areas))


    positions = torch.vstack([s[1] for s in shells])
    areas = torch.hstack([s[2] for s in shells])

    # print(shellArea / optimalArea)

    
    supports = volumeToSupport(areas, targetNeighbors, domain.dim)
    # supports = torch.ones_like(positions[:, 0]) * support

    return ParticleSet(
        positions = positions,
        supports = supports,
        masses = areas,
        densities = torch.ones_like(positions[:, 0])
    )


from sphMath.schemes.states.wcsph import WeaklyCompressibleUpdate, WeaklyCompressibleState
from sphMath.modules.particleShifting import solveShifting
from sphMath.schemes.deltaSPH import DeltaPlusSPHSystem
from sphMath.sampling import ParticleSet

import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm



def sampleOptimal(nx, domain, targetNeighbors, kernel, jitter = 0.1, shiftIters = 128, shiftScheme = 'IPS'):
    band = 0
    dim = domain.dim
    particles_full = sampleRegularParticles(nx, domain, targetNeighbors, jitter = 0.00, band = band)
    particleDx = particles_full.masses.pow(1/dim).mean().item()
    device = particles_full.positions.device

    particles = WeaklyCompressibleState(
        particles_full.positions,
        supports = particles_full.supports,
        masses = particles_full.masses,
        
        densities = torch.zeros_like(particles_full.masses),
        velocities = torch.zeros_like(particles_full.positions),
        
        pressures = torch.zeros_like(particles_full.masses),
        soundspeeds = torch.ones_like(particles_full.masses),
        
        kinds = torch.zeros_like(particles_full.masses),
        materials = torch.zeros_like(particles_full.masses),
        UIDs = torch.arange(particles_full.positions.shape[0], device = device, dtype = torch.int64),
    )

    config = {
        'domain': domain,
        'kernel': kernel,
        'verletScale': 1.4,
        'particle': {'dx': particleDx, 'support': particles.supports.max().item()},
        'shifting': {'scheme': shiftScheme,
                     'computeMach': False,
                     'summationDensity': True,
                    'solverThreshold': 0.5 * particleDx},
        'neighborhood': {'verletScale': 1.4, 'targetNeighbors': targetNeighbors},
    }

    neighborhood = None
    particleSystem = DeltaPlusSPHSystem(domain, neighborhood, 0., particles, 'momentum', None, [], [], config)
    particleSystem.systemState.positions += jitter * torch.randn_like(particleSystem.systemState.positions) * particleDx


    for i in tqdm(range(shiftIters), leave=False):
        dx, neighborhood, neighbors, overallStates, densities, velocities, fs, n, lMin = solveShifting(particleSystem, 0.1, config, verbose = False)
        particleSystem.neighborhoodInfo = neighborhood
        particleSystem.systemState.positions += dx

    return ParticleSet(particleSystem.systemState.positions, particleSystem.systemState.supports, particleSystem.systemState.masses, torch.ones_like(particleSystem.systemState.masses))


from sphMath.sdf import operatorDict

def rampDivergenceFree(positions, noise, sdf_func, offset, d0 = 0.25):
    sdf = sdf_func(positions)
#     r = sdf / d0 /2  + 0.5
    r = (sdf - offset) / d0 / 0.5 - 1
#     ramped = r * r * (3 - 2 * r)
    ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
#     ramped = r
    ramped[r >= 1] = 1
    ramped[r <= -1] = -1
#     ramped[r <= 0] = 0
#     ramped[r <= -1] = -1
    
    # return sdf
    return (ramped /2 + 0.5) * (noise)


def generateRamp(perennialState, config):
    regions = config['regions']
    boundary_sdfs = [region['sdf'] for region in regions if region['type'] == 'boundary']
    combined_sdf = lambda x: boundary_sdfs[0](x)[0]
    for sdf in boundary_sdfs[1:]:
        combined_sdf = operatorDict['union'](combined_sdf, lambda x: sdf(x)[0])


    buffer = config.get('boundary', {}).get('potentialBuffer', 4)
    ramp = rampDivergenceFree(perennialState.positions, torch.ones_like(perennialState.densities), combined_sdf, 
                              offset = config['particle']['dx']/2, 
                              d0 = buffer * perennialState.supports)
    return ramp


from sphMath.neighborhood import filterNeighborhood, filterNeighborhoodByKind, coo_to_csr, buildNeighborhood, computeDistanceTensor, evaluateNeighborhood, SupportScheme
from sphMath.operations import SPHOperation, Operation, GradientMode

def sampleDivergenceFreeNoise(particleState, domain, config, nxGrid, octaves = 3, lacunarity = 2, persistence = 0.5, baseFrequency = 1, tileable = True, kind = 'perlin', seed = 45906734):
    neighborhood, neighbors = evaluateNeighborhood(particleState, domain, config['kernel'], verletScale = config['neighborhood']['verletScale'], mode =  SupportScheme.SuperSymmetric, priorNeighborhood=None)
    
    noiseGen = generateNoiseInterpolator(nxGrid, nxGrid, domain, dim = domain.dim, octaves = octaves, lacunarity = lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = seed)

    dtype = particleState.positions.dtype
    ramp = generateRamp(particleState, config) if len([r for r in config['regions'] if r['type'] == 'boundary']) > 0 else 1
    potential = noiseGen(particleState.positions).to(dtype)* ramp

    rho = computeDensity(particleState, config['kernel'], neighbors.get('noghost'), SupportScheme.Scatter, config) 
    priorDensity = particleState.densities.clone() 
    particleState.densities = rho

    gradTerm = SPHOperation(particleState, potential, config['kernel'], neighbors.get('noghost')[0], neighbors.get('noghost')[1], Operation.Gradient, SupportScheme.Scatter, GradientMode.Difference)
    
    # sph_op(particleState, particleState, domain, config['kernel'], sparseNeighborhood, operation = 'gradient', supportScheme = 'symmetric', quantity = potential, gradientMode = 'difference')

    velocities = torch.stack([gradTerm[:,1], -gradTerm[:,0]], dim = 1)
    velocities = velocities / torch.linalg.norm(velocities, dim = 1, keepdim = True).max()

    velocities[particleState.kinds != 0, :] = 0

    particleState.densities = priorDensity
    return velocities