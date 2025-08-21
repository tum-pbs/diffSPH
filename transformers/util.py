from diffSPH.schemes.states.wcsph import WeaklyCompressibleUpdate, WeaklyCompressibleState
from diffSPH.sampling import ParticleSet
from diffSPH.simple import *
import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm
from diffSPH.modules.particleShifting import solveShifting

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
        'neighborhood': {'verletScale': 1.4, 'targetNeighbors': targetNeighbors,
        'computeHessian': True, 'computeDkDh': True},
    }

    neighborhood = None
    particleSystem = DeltaPlusSPHSystem(domain, neighborhood, 0., particles, 'momentum', None, [], [], config)
    particleSystem.systemState.positions += jitter * torch.randn_like(particleSystem.systemState.positions) * particleDx


    for i in tqdm(range(shiftIters), leave=False):
        dx, neighborhood, neighbors, overallStates, densities, velocities, fs, n, lMin = solveShifting(particleSystem, 0.1, config, verbose = False)
        particleSystem.neighborhoodInfo = neighborhood
        particleSystem.systemState.positions += dx

    return ParticleSet(particleSystem.systemState.positions, particleSystem.systemState.supports, particleSystem.systemState.masses, torch.ones_like(particleSystem.systemState.masses))

from diffSPH.dataLoaderUtils.state import WeaklyCompressibleSPHState

def wcstateToState(state):
    return WeaklyCompressibleSPHState(
        positions = state.positions,
        supports = state.supports,
        masses = state.masses,
        densities = state.densities,
        velocities = state.velocities if hasattr(state, 'velocities') else torch.zeros_like(state.positions),

        kinds = state.kinds if hasattr(state, 'kinds') else torch.zeros_like(state.masses),
        materials = state.materials if hasattr(state, 'materials') else torch.zeros_like(state.masses),
        UIDs = state.UIDs if hasattr(state, 'UIDs') else torch.arange(state.positions.shape[0], device=state.positions.device, dtype=torch.int64),

        numParticles = [state.positions.shape[0]],
        time = [0.],
        dt = [1e-3],
        timestep = [0],
        key = [''],

        boundaryNormals = None,
        boundaryDistances = None,
        boundaryIndices = None,

        rigidBodies = None,
        batches = None
    )

import torch
from diffSPH.sphOperations.shared import scatter_sum
from matplotlib.colors import LogNorm

def plotDistribution(fig, axis, particleState, neighborhood, logNorm = True, nnx = 63):
    ddx = 2 / (nnx)
    hij = (particleState.supports[neighborhood[0].row] + particleState.supports[neighborhood[0].col]) / 2
    # print(hij.shape)
    # print(neighborhood[0].row, neighborhood[0].col)

    positions = neighborhood[1].x_ij / hij.view(-1,1)
    # positions = positions[neighborhood['indices'][0] != neighborhood['indices'][1]]

    index = ((positions + 1) / ddx).to(torch.int64)
    linIdx = index[:,0] * nnx + index[:,1]
    # print(linIdx, linIdx.min(), linIdx.max())

    counter = scatter_sum(torch.ones_like(linIdx), dim = 0, index = linIdx, dim_size = nnx**2).reshape(nnx,nnx).cpu().numpy()
    if logNorm:
        sc = axis.imshow(counter, norm=LogNorm(), extent=(-1, 1, -1, 1))
    else:
        sc = axis.imshow(counter, extent=(-1, 1, -1, 1))

    # print(counter.min(), counter.max()) 

    # neighGrid = torch.zeros(64,64)
    cbar = fig.colorbar(sc, ax=axis)
    axis.set_aspect('equal')
    axis.set_title('Neighbor Distribution')
    return sc, cbar