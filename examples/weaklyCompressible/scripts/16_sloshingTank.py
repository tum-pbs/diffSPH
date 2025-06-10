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
parser.add_argument('--band', '-b', type=int, default=5, help='Domain boundary width')
parser.add_argument('--targetDt', '-dt', type=float, default=0.0005, help='Target time step')
parser.add_argument('--fps', '-fps', type=int, default=50, help='Frames per second')
parser.add_argument('--timeLimit', '-t', type=float, default=4, help='Total simulation time')
parser.add_argument('--device', '-d', type=str, default='cuda', help='Device to run the simulation on (e.g., cuda:0, cpu)')
parser.add_argument('--precision', '-p', type=str, default='float32', help='Precision of the simulation (e.g., float32, float64)')
parser.add_argument('--kernel', type=str, default='Wendland4', help='Kernel type (e.g., Wendland4, Gaussian)')
parser.add_argument('--integrationScheme', '-i', type=str, default='symplecticEuler', help='Integration scheme (e.g., symplecticEuler, leapfrog)')
parser.add_argument('--export', '-e', type=bool, default=False, help='Export simulation data')
parser.add_argument('--exportInterval', '-ei', type=int, default=1, help='Export interval in timesteps')

args = parser.parse_args()

## Setup simulation parameters that are different
simulationName = 'Sloshing Tank'
exportName = f'16_sloshingTank'

freeSurface = True

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

aspect = 1/2
domain = buildDomainDescription(l = L + dx * (band) * 2, dim = dim, periodic = True, device = device, dtype = dtype)
domain.min = torch.tensor([-L/2/ aspect - dx*band, -L/2 - dx * band], device = device, dtype = dtype)
domain.max = torch.tensor([L/2/ aspect + dx*band, L/2 + dx * band], device = device, dtype = dtype)

interiorDomain = buildDomainDescription(l = L, dim = dim, periodic = True, device = device, dtype = dtype)
interiorDomain.min = torch.tensor([-L/2 / aspect, -L/2], device = device, dtype = dtype)
interiorDomain.max = torch.tensor([L/2/ aspect, L/2], device = device, dtype = dtype)
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
config['gravity'] = {
    'active': True,
    'magnitude': 10,
    'mode': 'directional',
    'direction': torch.tensor([0, -1], device = device, dtype = dtype)
}

print('Finished setting up simulation parameters')
fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert = True)
domain_sdf = lambda x: sampleDomainSDF(x, interiorDomain, invert = False)
obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('circle')['function'](x, torch.tensor(1/4).to(points.device)), invert = False)

W = 1.5
H = 1/2
F = 1/4 * H

fluid = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([W,F]).to(points.device)), torch.tensor([0,-H/2-F]).to(points.device)), invert = False)

box_outer = operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([W + dx * band,H + dx * band]).to(device)), torch.tensor([0, 0]).to(device))
box_inner = operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([W,H]).to(device)), torch.tensor([0, 0]).to(device))

box_inner_ = operatorDict['invert'](box_inner)
box_outer_ = operatorDict['invert'](box_outer)

combined_sdf = lambda points: sampleSDF(points, operatorDict['union'](box_outer_, box_inner), invert = True)
# combined_sdf = lambda points: sampleSDF(points, box_inner_, False)

regions = []

regions.append(buildRegion(sdf = fluid, config = config, type = 'fluid'))
regions.append(buildRegion(sdf = combined_sdf, config = config, type = 'boundary', kind = 'driven'))
regions[-1]['sdf'] = lambda points: sampleSDF(points, box_inner, invert = True)

for region in regions:
    region = filterRegion(region, regions)

for region in regions:
    print(region['particles'].masses.shape)

print('Finished setting up regions')
## Setup initial fluid state
particleState, config, rigidBodies = initializeSimulation(scheme,config, regions)

import numpy as np
t = torch.tensor(0, device = device, dtype = dtype)
rigidBodies[-1].angularVelocity = np.pi/16 * torch.cos(t * np.pi)

## Setup simulation, everything after this should be the same for all examples
fig, axis, velocityPlot, densityPlot, uidPlot, particleSystem, dt, initialKineticEnergy, initialPotentialEnergy, initialEnergy = setupExampleSimulation(simulationName, scheme, particleState, config, regions, stacked = 'horizontal', figsize = (12, 4))
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

def callBackFn(particleSystem):
    particleSystem.rigidBodies[-1].angularVelocity = np.pi/16 * torch.cos(particleSystem.t * np.pi)
    return particleSystem
def plotCallbackFn(fig, axis, particleSystem, i):
    for ax in axis.flatten():
        ax.set_xlim(-2.05, 2.05)
        ax.set_ylim(-1.5, 1.5)

runSimulation(simulationName, particleSystem, integrationScheme, simulator, timesteps, dt, config, fig, axis, velocityPlot, densityPlot, uidPlot, initialEnergy, imagePrefix, plotInterval, callBackFn=callBackFn, plotCallbackFn = plotCallbackFn, outFile = outFile, exportInterval = args.exportInterval)
if args.export:
    outFile.close()
    print(f'Exported simulation data to {outputPrefix}{exportName}.hdf5')
print('Finished simulation')
## Post process simulation
postProcess(imagePrefix, fps, exportName)
## Cleanup