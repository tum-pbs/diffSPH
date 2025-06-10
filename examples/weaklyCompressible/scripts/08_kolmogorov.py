import torch
import warnings
import os
import copy
import subprocess
import shlex
import matplotlib.pyplot as plt
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm
from torch.profiler import profile,  ProfilerActivity

if torch.cuda.is_available():
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'

from sphMath.sampling import buildDomainDescription, sampleDivergenceFreeNoise
from sphMath.modules.adaptiveSmoothingASPH import n_h_to_nH
from sphMath.plotting import visualizeParticles, updatePlot
from sphMath.integration import getIntegrator, getIntegrationEnum
from sphMath.util import volumeToSupport
from sphMath.boundary import sampleDomainSDF
from sphMath.kernels import Kernel_Scale, getKernelEnum
from sphMath.sdf import getSDF, sdfFunctions, operatorDict, sampleSDF
from sphMath.regions import buildRegion, filterRegion, plotRegions
from sphMath.modules.timestep import computeTimestep
from sphMath.schemes.initializers import initializeSimulation, updateBodyParticles
from sphMath.schemes.deltaSPH import deltaPlusSPHScheme, DeltaPlusSPHSystem
from sphMath.schema import getSimulationScheme
from sphMath.enums import *
import math
import argparse
from exampleUtil import setupExampleSimulation, runSimulation, postProcess
import numpy as np
from sphMath.modules.particleShifting import shuffleParticles
import h5py
from sphMath.io import writeAttributesWCSPH, getState, saveState


parser = argparse.ArgumentParser(description='SPH Simulation Parameters')
parser.add_argument('--L', '-L', type=float, default=2, help='Length of the domain')
parser.add_argument('--nx', '-nx', type=int, default=32, help='Number of particles along one dimension')
parser.add_argument('--n_h', '-n_h', type=int, default=4, help='Number of neighbors for smoothing length')
parser.add_argument('--rho0', '-rho0', type=float, default=1, help='Density of the fluid')
parser.add_argument('--band', '-b', type=int, default=0, help='Domain boundary width')
parser.add_argument('--targetDt', '-dt', type=float, default=0.0005, help='Target time step')
parser.add_argument('--fps', '-fps', type=int, default=50, help='Frames per second')
parser.add_argument('--timeLimit', '-t', type=float, default=10, help='Total simulation time')
parser.add_argument('--device', '-d', type=str, default='cuda', help='Device to run the simulation on (e.g., cuda:0, cpu)')
parser.add_argument('--precision', '-p', type=str, default='float32', help='Precision of the simulation (e.g., float32, float64)')
parser.add_argument('--kernel', type=str, default='Wendland4', help='Kernel type (e.g., Wendland4, Gaussian)')
parser.add_argument('--integrationScheme', '-i', type=str, default='symplecticEuler', help='Integration scheme (e.g., symplecticEuler, leapfrog)')
parser.add_argument('--octaves', type=int, default = 4)
parser.add_argument('--lacunarity', type=int, default =2)
parser.add_argument('--persistence', type=float, default =0.5)
parser.add_argument('--baseFrequency', type =int, default=2)
parser.add_argument('--kind', type=str, default='perlin')
parser.add_argument('--seed', type=int, default=45906734 )
parser.add_argument('--obstacle', type=bool, default=False)
parser.add_argument('--k', type=int, default=4, help='wavenumber')
parser.add_argument('--export', '-e', type=bool, default=False, help='Export simulation data')
parser.add_argument('--exportInterval', '-ei', type=int, default=1, help='Export interval in timesteps')

args = parser.parse_args()

## Setup simulation parameters that are different
simulationName = 'Kolmogorov Random Flow'

octaves = args.octaves
lacunarity = args.lacunarity
persistence = args.persistence
baseFrequency = args.baseFrequency
kind = args.kind
seed = args.seed
obstacle = args.obstacle
tileable = True
k = args.k

exportName = f'06_kolmogorov_{k}'
freeSurface = False

print(f"Running simulation {simulationName}\n")
## Setup consistent parameters
L = args.L
nx = args.nx * L
dx = L / nx
targetDt = args.targetDt
rho0 = args.rho0
band = args.band
fps = args.fps
timeLimit = args.timeLimit
print(f"Simulation parameters:\n"
      f"  Domain length: {L}\n"
      f"  Number of particles: {nx}\n"
      f"  Smoothing length: {dx}\n"
      f"  Target time step: {targetDt}\n"
      f"  Density: {rho0}\n"
      f"  Bandwidth: {band}\n"
      f"  Frames per second: {fps}\n"
      f"  Time limit: {timeLimit}\n"
      f"  Device: {args.device}\n"
      f"  Precision: {args.precision}\n"
      f"  Kernel: {args.kernel}\n"
      f"  Integration scheme: {args.integrationScheme}")

kernel = getKernelEnum(args.kernel)
scheme = SimulationScheme.DeltaSPH
integrationScheme = getIntegrationEnum(args.integrationScheme)

device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32 if args.precision == 'float32' else torch.float64

targetNeighbors = n_h_to_nH(args.n_h, 2)
c_s = 0.3 * volumeToSupport(dx**2, targetNeighbors, 2) / Kernel_Scale(kernel, 2) / targetDt
exportInterval = 1 / fps
plotInterval = int(math.ceil(exportInterval / targetDt))
timesteps = int(timeLimit / targetDt)

dim = 2
CFL = 0.3

domain = buildDomainDescription(l = L + dx * (band) * 2, dim = dim, periodic = True, device = device, dtype = dtype)
interiorDomain = buildDomainDescription(l = L, dim = dim, periodic = False, device = device, dtype = dtype)
wrappedKernel = kernel

simulator, SimulationSystem, config, integrator = getSimulationScheme(
     scheme, kernel, integrationScheme, 
     1.0, targetNeighbors, domain)
integrationScheme = getIntegrator(integrationScheme)

config['particle'] = {
    'nx': nx + 2 * band,
    'dx': L/nx,
    'targetNeighbors': targetNeighbors,
    'band': band
}
config['fluid'] = {
    'rho0': rho0,
    'c_s': c_s
}
config['surfaceDetection']['active'] = freeSurface
config['shifting']['freeSurface'] = freeSurface

# Setup forcing term
from sphMath.sampling import generateNoiseInterpolator
from sphMath.util import getPeriodicPositions
import numpy as np
nxGrid = nx * 2

domain_cpu = buildDomainDescription(l = L, dim = dim, periodic = True, device = torch.device('cpu'), dtype = dtype)
domain_cpu.min = domain.min.cpu()
domain_cpu.max = domain.max.cpu()
noiseGen = generateNoiseInterpolator(nxGrid, nxGrid, domain_cpu, dim = domain.dim, octaves = octaves, lacunarity = lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = seed)

# k = 4
xi = 1
noiseLevel = 0.01

# Kolmogorov turbulence
def forcing(x, mask, state, t, dt):
    pos = getPeriodicPositions(x, domain)
    noiseOffset = torch.rand(pos.shape[1], device = device, dtype = dtype) * L - L/2
    noisePos = getPeriodicPositions(x + noiseOffset, domain)
    u_x = xi * torch.sin(k * np.pi * pos[:,1])
    u_y = noiseGen(pos.detach().cpu()).to(dtype = x.dtype, device = x.device) * noiseLevel
    return torch.stack([u_x, u_y], dim = 1)



print('Finished setting up simulation parameters')
fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert = True)
# domain_sdf = lambda x: sampleDomainSDF(x, interiorDomain, invert = False)
obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('hexagon')['function'](x, torch.tensor(1/4).to(points.device)), invert = False)

# box_sdf = lambda points: sampleSDF(points, lambda x: getSDF('box')['function'](x, torch.tensor([0.5,0.5]).to(points.device)))

# inlet_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([L/16,L/2]).to(points.device)), torch.tensor([domain.min[0]+L/16,0]).to(points.device)), invert = False)
# outlet_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([L/12,L]).to(points.device)), torch.tensor([domain.max[0]-L/12,0]).to(points.device)), invert = False)
# outletBuffer_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([L/8,L]).to(points.device)), torch.tensor([domain.max[0]-L/8,0]).to(points.device)), invert = False)


regions = []

# regions.append(buildRegion(sdf = domain_sdf, config = config, type = 'boundary', kind = 'constant'))
# regions.append(buildRegion(sdf = obstacle_sdf, config = config, type = 'boundary', kind = 'constant'))
regions.append(buildRegion(sdf = fluid_sdf, config = config, type = 'fluid'))
regions.append(buildRegion(sdf = fluid_sdf, config = config, type = 'forcing', dirichletValues={'velocities': forcing}))

# regions.append(buildRegion(sdf = inlet_sdf, config = config, type = 'inlet', dirichletValues={'densities': config['fluid']['rho0'], 'velocities': torch.tensor([1,0], device = device, dtype = dtype)}, updateValues = {'densities': 0, 'velocities': torch.tensor([0,0], device = device, dtype = dtype)}))

# regions.append(buildRegion(sdf = outlet_sdf, config = config, type = 'outlet'))
# regions.append(buildRegion(sdf = outletBuffer_sdf, config = config, type = 'buffer', bufferValues = ['densities', 'velocities', 'pressures']))

# regions.append(buildRegion(sdf = box_sdf, config = config, type = 'dirichlet', dirichletValues={'densities': 2.0, 'velocities': torch.tensor([1,2], device = device, dtype = dtype), 'pressures': lambda x: torch.where(x[:,0] > 0, 0.0, 1.0)}, updateValues = {'densities': 2.0}))


for region in regions:
    region = filterRegion(region, regions)

print('Finished setting up regions')
## Setup initial fluid state
particleState, config, rigidBodies = initializeSimulation(scheme,config, regions)

# particleState.positions = shuffleParticles(particleState, config, 4)

velocities = sampleDivergenceFreeNoise(particleState, domain, config, nx * 2, 
                                    octaves = octaves, lacunarity = lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = seed)
# particleState.velocities[:] = velocities


## Setup simulation, everything after this should be the same for all examples
fig, axis, velocityPlot, densityPlot, uidPlot, particleSystem, dt, initialKineticEnergy, initialPotentialEnergy, initialEnergy = setupExampleSimulation(simulationName, scheme, particleState, config, regions, stacked = 'horizontal', figsize = (12, 4), markerSize=2)
print('Ready to run simulation')
## Run simulation

imagePrefix = f'./images/{exportName}/'
os.makedirs(imagePrefix, exist_ok = True)
fig.savefig(f'{imagePrefix}frame_{0:05d}.png', dpi = 100)

if args.export:
    outputPrefix = f'./output/'
    os.makedirs(outputPrefix, exist_ok = True)
    outFile = h5py.File(f'{outputPrefix}{exportName}.hdf5', 'w')
    print(f'Exporting simulation data to {outFile.filename}')
    writeAttributesWCSPH(outFile, scheme, kernel, integrationScheme, targetNeighbors, targetDt, L, nx, band, rho0, c_s, freeSurface, dim, timeLimit, config, particleState)
    fluidMask = particleState.kinds == 0
    boundaryMask = particleState.kinds == 1
    nonGhostMask = particleState.kinds != 2

    initialState = outFile.create_group('initialState')
    initialFluidState = initialState.create_group('fluid')
    initialBoundaryState = initialState.create_group('boundary')
    saveState(particleState, fluidMask, initialFluidState, config)
    saveState(particleState, boundaryMask, initialBoundaryState, config)
    saveState(particleState, nonGhostMask, initialState, config)
else:
    outFile = None

runSimulation(simulationName, particleSystem, integrationScheme, simulator, timesteps, dt, config, fig, axis, velocityPlot, densityPlot, uidPlot, initialEnergy, imagePrefix, plotInterval, outFile = outFile, exportInterval = args.exportInterval)
if args.export:
    outFile.close()
    print(f'Exported simulation data to {outputPrefix}{exportName}.hdf5')

print('Finished simulation')
## Post process simulation
postProcess(imagePrefix, fps, exportName)
## Cleanup