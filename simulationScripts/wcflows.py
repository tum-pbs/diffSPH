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

from diffSPH.sampling import buildDomainDescription
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH
from diffSPH.plotting import visualizeParticles, updatePlot
from diffSPH.integration import getIntegrator
from diffSPH.util import volumeToSupport
from diffSPH.boundary import sampleDomainSDF
from diffSPH.kernels import Kernel_Scale
from diffSPH.sdf import getSDF, sdfFunctions, operatorDict, sampleSDF
from diffSPH.regions import buildRegion, filterRegion, plotRegions
from diffSPH.modules.timestep import computeTimestep
from diffSPH.schemes.initializers import initializeSimulation, updateBodyParticles
from diffSPH.schemes.deltaSPH import deltaPlusSPHScheme, DeltaPlusSPHSystem
from diffSPH.schema import getSimulationScheme
from diffSPH.enums import *
from exampleUtil import setupExampleSimulation, runSimulation, postProcess
from diffSPH.sampling import sampleDivergenceFreeNoise
import math
from diffSPH.sampling import sampleDivergenceFreeNoise
from diffSPH.modules.particleShifting import solveShifting, shuffleParticles
import numpy as np
import datetime
import argparse


parser = argparse.ArgumentParser(description='Run a regional shock simulation with adaptive smoothing.')

parser.add_argument('--nx', type=int, default=128, help='Number of particles in each dimension.')
parser.add_argument('--gamma', type=float, default=5/3, help='Adiabatic index for the ideal gas EOS.')
parser.add_argument('--timeLimit', type=float, default=4.096, help='Time limit for the simulation in seconds.')
parser.add_argument('--fps', type=int, default=50, help='Frames per second for the output video.')
parser.add_argument('--caseName', type=str, default='wcsph', help='Name of the simulation case.')

parser.add_argument('--velocityNoise', action='store_true', help='Whether to add noise to the initial velocity field.')
parser.add_argument('--octaves', type=int, default=2, help='Number of octaves for the noise field.')
parser.add_argument('--lacunarity', type=int, default=2, help='Lacunarity for the noise field.')
parser.add_argument('--persistence', type=float, default=0.5, help='Persistence for the noise field.')
parser.add_argument('--baseFrequency', type=float, default=4.0, help='Base frequency for the noise field.')
parser.add_argument('--tileable', type=bool, default=True, help='Whether the noise field should be tileable.')
parser.add_argument('--kind', type=str, default='perlin', help='Type of noise to generate.')
parser.add_argument('--seed', type=int, default=4235, help='Seed for the random number generator.')

parser.add_argument('--TGV', type=int, default=0, help='Turbulent kinetic energy for the noise field.')
parser.add_argument('--normalizeEnergy', action='store_true', help='Whether to normalize the energy of the noise field.')
parser.add_argument('--initialEnergyTarget', type=float, default=1.0, help='Target energy for the initial noise field.')

parser.add_argument('--obstacle', action='store_true', help='Whether to include an obstacle in the simulation.')
parser.add_argument('--domainBoundary', action='store_true', help='Whether to use a domain boundary for the simulation.')
parser.add_argument('--boundaryViscosity', type=float, default=0.01, help='Viscosity for the boundary particles.')

parser.add_argument('--plot', dest='plot', action='store_true', default=True, help='Whether to plot the regions in the simulation.')
parser.add_argument('--no-plot', dest='plot', action='store_false', help='Disable plotting of the regions in the simulation.')
parser.add_argument('--export', action='store_true', help='Whether to export the simulation data to a file.')
parser.add_argument('--exportDir', type=str, default='./out/', help='Directory to export the simulation data to.')

parser.add_argument('--rho0', type=float, default=1.0, help='Initial density of the particles.')
parser.add_argument('--Pinitial', type=float, default=1.0, help='Initial pressure of the particles.')
parser.add_argument('--dt', type=float, default=1e-3, help='Time step for the simulation.')

parser.add_argument('--verbose', action='store_true', help='Whether to print verbose output during the simulation.')

parser.add_argument('--integrationScheme', type=str, default='symplecticEuler', choices=['rungeKutta2', 'Euler'], help='Integration scheme to use.')
parser.add_argument('--kernelType', type=str, default='B7', choices=['CubicSpline', 'CubicSpline'], help='Kernel type to use.')

parser.add_argument('--gpu', type=int, default=0, help='GPU index to use for the simulation.')
parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs.')
parser.add_argument('--cs', type=float, default=20, help='Speed of sound for the simulation.')

args = parser.parse_args()
verbose = args.verbose

simulationName = 'Bounded Random Flow'

octaves = int(args.octaves)
lacunarity = int(args.lacunarity)
persistence = args.persistence
baseFrequency = int(args.baseFrequency)
tileable = args.tileable
kind = args.kind
seed = args.seed
obstacle = args.obstacle
domainBoundary = args.domainBoundary


nx = args.nx
currentTime = datetime.datetime.now()
timestamp = currentTime.strftime("%Y-%m-%d_%H-%M-%S")
fileName = f'{args.caseName}_{args.domainBoundary}_{args.obstacle}_{args.TGV}_{nx**2}_{timestamp}_{args.seed}'
imagePrefix = f'{args.exportDir}/images/{fileName}/'
exportName = f'{args.exportDir}/data/{fileName}.h5'




# exportName = f'07_boundedRandomFlow_{octaves}_{lacunarity}_{persistence}_{baseFrequency}_{tileable}_{kind}_{seed}_{"with" if obstacle else "without"}'

L = 2
dx = L / nx
targetDt = args.dt
dt = args.dt
rho0 = args.rho0
freeSurface = False
band = 4 if domainBoundary else 0
fps = args.fps
timeLimit = args.timeLimit

kernel = None

for k in KernelType:
    if k.name == args.kernelType:
        kernel = k
        break
if kernel is None:
    raise ValueError(f'Unknown kernel type: {args.kernelType}. Available options are: {[k.name for k in KernelType]}')


scheme = SimulationScheme.DeltaSPH

integrationScheme = None
for i in IntegrationSchemeType:
    if i.name == args.integrationScheme:
        integrationScheme = i
        break
if integrationScheme is None:
    raise ValueError(f'Unknown integration scheme: {args.integrationScheme}. Available options are: {[i.name for i in IntegrationSchemeType]}')

integrationScheme = IntegrationSchemeType.symplecticEuler

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32
targetNeighbors = n_h_to_nH(4, 2)
c_s_CFL = 0.35 * volumeToSupport(dx**2, targetNeighbors, 2) / Kernel_Scale(kernel, 2) / targetDt
exportInterval = 1 / args.fps
plotInterval = int(math.ceil(exportInterval / targetDt))
timesteps = int(args.timeLimit / targetDt)

if args.verbose:
    print(f'Running simulation {args.caseName} with {timesteps} timesteps, targetDt = {targetDt}, exportInterval = {exportInterval}, plotInterval = {plotInterval}')
    print(f'Using {device} with dtype {dtype}, kernel = {kernel.name}, integration scheme = {integrationScheme.name}, targetNeighbors = {targetNeighbors:.2g}, c_s = {c_s_CFL:.1f}')

dim = 2
CFL = 0.3

domain = buildDomainDescription(l = L + dx * (band) * 2, dim = dim, periodic = True, device = device, dtype = dtype)
interiorDomain = buildDomainDescription(l = L, dim = dim, periodic = not domainBoundary, device = device, dtype = dtype)
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
    'c_s': args.cs
}
config['surfaceDetection']['active'] = freeSurface
config['shifting']['freeSurface'] = freeSurface

config['diffusion']['boundary'] = args.boundaryViscosity

fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert = True)
domain_sdf = lambda x: sampleDomainSDF(x, interiorDomain, invert = False)
obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('circle')['function'](x, torch.tensor(1/4).to(points.device)), invert = False)

box_sdf = lambda points: sampleSDF(points, lambda x: getSDF('box')['function'](x, torch.tensor([0.5,0.5]).to(points.device)))

inlet_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([0.125,0.125]).to(points.device)), torch.tensor([-0.5,0.5]).to(points.device)), invert = False)
outlet_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([0.125,0.125]).to(points.device)), torch.tensor([0.5,-0.5]).to(points.device)), invert = False)
outletBuffer_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([0.25,0.25]).to(points.device)), torch.tensor([0.5,-0.5]).to(points.device)), invert = False)


regions = []

if domainBoundary:
    regions.append(buildRegion(sdf = domain_sdf, config = config, type = 'boundary', kind = 'constant'))
if obstacle:
    regions.append(buildRegion(sdf = obstacle_sdf, config = config, type = 'boundary', kind = 'zero'))
regions.append(buildRegion(sdf = fluid_sdf, config = config, type = 'fluid'))


# regions.append(buildRegion(sdf = inlet_sdf, config = config, type = 'inlet', dirichletValues={'densities': config['fluid']['rho0'], 'velocities': torch.tensor([1,0], device = device, dtype = dtype)}, updateValues = {'densities': 0, 'velocities': torch.tensor([0,0], device = device, dtype = dtype)}))

# regions.append(buildRegion(sdf = outlet_sdf, config = config, type = 'outlet'))
# regions.append(buildRegion(sdf = outletBuffer_sdf, config = config, type = 'buffer', bufferValues = ['densities', 'velocities', 'pressures']))

# regions.append(buildRegion(sdf = box_sdf, config = config, type = 'dirichlet', dirichletValues={'densities': 2.0, 'velocities': torch.tensor([1,2], device = device, dtype = dtype), 'pressures': lambda x: torch.where(x[:,0] > 0, 0.0, 1.0)}, updateValues = {'densities': 2.0}))


for region in regions:
    region = filterRegion(region, regions)

from diffSPH.sampling import sampleDivergenceFreeNoise, generateRamp

from diffSPH.neighborhood import filterNeighborhood, filterNeighborhoodByKind, coo_to_csr, buildNeighborhood, computeDistanceTensor, evaluateNeighborhood, SupportScheme
from diffSPH.operations import SPHOperation, Operation, GradientMode
from diffSPH.modules.density import computeDensity

from samplingUtils import plotSampling, sampleDivergenceFreeNoise, sampleTGV




particleState, config, rigidBodies = initializeSimulation(scheme, config, regions)
particleState.positions = shuffleParticles(particleState, config, 4)

if args.velocityNoise:
    velocities, potential, ramp = sampleDivergenceFreeNoise(particleState, domain, config, nx * 2, smoothingSteps = 4, octaves = octaves, lacunarity = lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = seed)
    # fig, axis = plotSampling(particleState, domain, config, velocities, ramp, potential, s = s)
else:
    velocities, potential, ramp = sampleTGV(particleState, domain, config, k_ = args.TGV, smoothingSteps = 4)
    # fig, axis = plotSampling(particleState, domain, config, velocities, ramp, potential, s = s)

particleState.velocities[:, :] = velocities

E_k0 = 0.5 * particleState.masses * torch.linalg.norm(particleState.velocities, dim = -1)**2
totalInitialEnergy = E_k0.sum()

if args.verbose:
    print(f'Velocity Magnitudes: min: {particleState.velocities.norm(dim = -1).min().item()}, max: {particleState.velocities.norm(dim = -1).max().item()}, mean: {particleState.velocities.norm(dim = -1).mean().item()}')

    print(f'Total initial energy: {totalInitialEnergy.item()}')

if args.normalizeEnergy:
    # particleState.velocities /= E_k0.sum().sqrt()
    particleState.velocities *= args.initialEnergyTarget / E_k0.sum().sqrt()
    totalInitialEnergy = E_k0.sum()

    if args.verbose:
        print(f'Velocity Magnitudes: min: {particleState.velocities.norm(dim = -1).min().item()}, max: {particleState.velocities.norm(dim = -1).max().item()}, mean: {particleState.velocities.norm(dim = -1).mean().item()}')

        print(f'Total initial energy: {totalInitialEnergy.item()}')

maxVelocity = particleState.velocities.norm(dim = -1).max().item()
if args.verbose:
    print(f'Maximum velocity: {maxVelocity}')
    print(f'Speed of sound: {config["fluid"]["c_s"]}')
    print(f'CFL timestep: {c_s_CFL}')


if config['fluid']['c_s'] < maxVelocity * 10:
    print(f'Warning: Speed of sound {config["fluid"]["c_s"]} is too low for maximum velocity {maxVelocity}. Increase c_s or decrease max velocity.')
    # raise ValueError(f'Speed of sound {config["fluid"]["c_s"]} is too low for maximum velocity {maxVelocity}. Increase c_s or decrease max velocity.')
if config['fluid']['c_s'] >= c_s_CFL:
    print(f'Warning: Speed of sound {config["fluid"]["c_s"]} is too high for CFL condition {c_s_CFL}. Decrease c_s or increase targetDt.')
    # raise ValueError(f'Speed of sound {config["fluid"]["c_s"]} is too high for CFL condition {c_s_CFL}. Decrease c_s or increase targetDt.')

if args.export:
    os.makedirs(os.path.dirname(exportName), exist_ok = True)

if args.plot or args.export:
    os.makedirs(imagePrefix, exist_ok = True)

from diffSPH.neighborhood import evaluateNeighborhood, filterNeighborhoodByKind, coo_to_csr

# dt = computeTimestep(scheme, 1e-2, particleState, config, None)
particles = copy.deepcopy(particleState)
particleSystem = DeltaPlusSPHSystem(config['domain'], None, 0., copy.deepcopy(particleState), 'momentum', None, rigidBodies = config['rigidBodies'], regions = config['regions'], config = config)

for rigidBody in config['rigidBodies']:
    particleState = updateBodyParticles(scheme, particleState, rigidBody)

## Setup values for plotting 

kineticEnergy = 0.5 * particleState.densities * (particleState.velocities ** 2).sum(1)

t = 0
E_k0 = 0.5 * particleState.masses * torch.linalg.norm(particleState.velocities, dim = -1)**2
E_k = 0.5 * particleState.masses * torch.linalg.norm(particleState.velocities, dim = -1)**2

rhoMin = particleState.densities.min().detach().cpu().item() / config['fluid']['rho0']
rhoMean = particleState.densities.mean().detach().cpu().item() / config['fluid']['rho0']
rhoMax = particleState.densities.max().detach().cpu().item() / config['fluid']['rho0']

initialVelocity = particleState.velocities.clone()
initialDensities = particleState.densities.clone()
initialPotentialEnergy = None

if config['gravity']['active']:
    if 'mode' in config['gravity'] and config['gravity']['mode'] == 'potential':
        B = config['gravity']['magnitude']
        initialPotentialEnergy = 0.5 * B**2 * particleState.masses / config['fluid']['rho0'] * particleState.densities * torch.linalg.norm(particleState.positions, dim = -1)**2

initialKineticEnergy = 0.5 * particleState.masses / config['fluid']['rho0'] * particleState.densities * torch.linalg.norm(initialVelocity, dim = -1)**2
initialEnergy = initialKineticEnergy + initialPotentialEnergy if initialPotentialEnergy is not None else initialKineticEnergy
totalInitialEnergy = (initialEnergy).sum().detach().cpu().item()
totalEnergy = (initialEnergy).sum().detach().cpu().item()

actualState = particleSystem

neighborhood, neighbors = evaluateNeighborhood(particles, domain, kernel, verletScale = 1.0, mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
particles.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries
actualState.systemState.numNeighbors = particles.numNeighbors


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

if args.plot:

    fig, axis = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False, sharex=True, sharey=True)

    s = 4
    if nx == 32:
        s = 16
    if nx == 64:
        s = 4
    if nx == 128:
        s = 1
    if nx == 256:
        s = 0.5


    fluidParticles = actualState.systemState


    densityPlotState        = plotScalar(fig, axis[0,0], 'Velocity magnitude',     fluidParticles, fluidParticles.velocities,           domain, config, s, gridVisualization=True, gridResolution=256, cmap = 'viridis', mapping = 'L2')
    internalEnergyPlotState = plotScalar(fig, axis[0,1], 'Density $\\rho$', fluidParticles, fluidParticles.densities ,     domain, config, s, gridVisualization=True, gridResolution=256, cmap = 'RdBu_r', mapping = None)
    supportPlotState        = plotScalar(fig, axis[0,2], 'Particle Index',         fluidParticles, fluidParticles.UIDs,            domain, config, s, gridVisualization=False, gridResolution=256, cmap = 'magma', mapping = None)
    numNeighborsPlotState   = plotScalar(fig, axis[1,2], 'Number of Neighbors', fluidParticles, fluidParticles.numNeighbors,        domain, config, s, gridVisualization=True, gridResolution=256, mapping = '.y', cmap = 'viridis')
    velocityXPlotState      = plotScalar(fig, axis[1,0], 'Velocity X',          fluidParticles, fluidParticles.velocities,          domain, config, s, gridVisualization=True, gridResolution=256, mapping = '.x', cmap = 'RdBu_r')
    velocityYPlotState      = plotScalar(fig, axis[1,1], 'Velocity Y',          fluidParticles, fluidParticles.velocities,          domain, config, s, gridVisualization=True, gridResolution=256, mapping = '.y', cmap = 'RdBu_r')

    # for ax in axis.flatten():
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1,1)
    # kineticEnergy = 0.5 * (torch.linalg.norm(actualState.systemState.velocities, dim = -1) **2 * actualState.systemState.masses).sum()
    # thermalEnergy = (actualState.systemState.internalEnergies * actualState.systemState.masses).sum()
    # totalEnergy = kineticEnergy + thermalEnergy


    fig.suptitle(f'{args.caseName}, ptcls = {particleState.positions.shape[0]}, kernel = {config["kernel"].name}, neighbors = {config["targetNeighbors"]:.2g}, $c_s$ = {config["fluid"]["c_s"]:.1f}\n$\\rho$ = [{rhoMin:.4g} | {rhoMean:.4g} | {rhoMax:.4g}], $E_0$ = {totalInitialEnergy:.4g}, $E$ = {totalEnergy:.4g}, $\\Delta E$ = {totalEnergy - totalInitialEnergy:.4g}, $t$ = {particleSystem.t:.4g}, $\\Delta t$ = {dt:.2e}')


    fig.tight_layout()

    fig.savefig(f'{args.exportDir}/{imagePrefix}frame_{0:05d}.png', dpi = 200)
# fig.savefig(f'{fileName}.png', dpi = 200)

# exit()


from diffSPH.io import initializeOutputFile, writeParticleData, writeParticleDataMinimal

if args.export:
    outFile = initializeOutputFile(exportName, actualState, config,simulationName='testData')
    outGroup = outFile['simulationData']
    writeParticleData(outGroup, actualState, step = 0, dt = dt)

    outFile['caseSpecificData'].attrs['caseName'] = args.caseName
    outFile['caseSpecificData'].attrs['nx'] = nx
    outFile['caseSpecificData'].attrs['rho0'] = rho0

    outFile['caseSpecificData'].attrs['velocityNoise'] = args.velocityNoise
    outFile['caseSpecificData'].attrs['octaves'] = octaves
    outFile['caseSpecificData'].attrs['lacunarity'] = lacunarity
    outFile['caseSpecificData'].attrs['persistence'] = persistence
    outFile['caseSpecificData'].attrs['baseFrequency'] = baseFrequency
    outFile['caseSpecificData'].attrs['tileable'] = tileable
    outFile['caseSpecificData'].attrs['kind'] = kind
    outFile['caseSpecificData'].attrs['seed'] = seed
    outFile['caseSpecificData'].attrs['obstacle'] = obstacle
    outFile['caseSpecificData'].attrs['domainBoundary'] = domainBoundary

    outFile['caseSpecificData'].attrs['TGV'] = args.TGV
    outFile['caseSpecificData'].attrs['normalizeEnergy'] = args.normalizeEnergy
    outFile['caseSpecificData'].attrs['initialEnergyTarget'] = args.initialEnergyTarget

    outFile['caseSpecificData'].attrs['cs'] = args.cs

    outFile['caseSpecificData'].attrs['rho0'] = args.rho0
    outFile['caseSpecificData'].attrs['dt'] = dt

    outFile['caseSpecificData'].attrs['verbose'] = args.verbose
    outFile['caseSpecificData'].attrs['integrationScheme'] = args.integrationScheme
    outFile['caseSpecificData'].attrs['kernelType'] = args.kernelType

    outFile['caseSpecificData'].attrs['CFL'] = CFL
    outFile['caseSpecificData'].attrs['targetNeighbors'] = targetNeighbors
    outFile['caseSpecificData'].attrs['exportInterval'] = exportInterval
    outFile['caseSpecificData'].attrs['plotInterval'] = plotInterval
    outFile['caseSpecificData'].attrs['timestamp'] = timestamp



gtqdms = []
import portalocker
with portalocker.Lock('README.md', flags = 0x2, timeout = None):
    for g in range(args.gpus):
        gtqdms.append(tqdm(range(timesteps), position = g, leave = True))

tq = gtqdms[args.gpu]
tq.reset()
tq.total = timesteps

states = []
# dt = newDt
# actualState.systemState.divergence = torch.zeros_like(actualState.systemState.densities)
for i in (range(timesteps)):
# while(True):
    actualState, currentState, updates = integrator.function(actualState, dt, simulator, config, priorStep = actualState.priorStep)
    actualState.priorStep = [updates[-1], currentState[-1]]
    # if i%100 == 0:
        # states.append(copy.deepcopy(simulationState).to(dtype = torch.float32, device = 'cpu'))

    kineticEnergy = 0.5 * (torch.linalg.norm(actualState.systemState.velocities, dim = -1) **2 * actualState.systemState.masses).sum()
    thermalEnergy = 0# (actualState.systemState.internalEnergies * actualState.systemState.masses).sum()
    totalEnergy = kineticEnergy + thermalEnergy

    rhoMin = actualState.systemState.densities.min().detach().cpu().item() / config['fluid']['rho0']
    rhoMean = actualState.systemState.densities.mean().detach().cpu().item() / config['fluid']['rho0']
    rhoMax = actualState.systemState.densities.max().detach().cpu().item() / config['fluid']['rho0']

    rhoMaxFromMean = rhoMax - rhoMean
    rhoMinFromMean = rhoMean - rhoMin
    rhoMaxVis = max(rhoMaxFromMean, abs(rhoMinFromMean))
    rhovmin = rhoMean - rhoMaxVis
    rhovmax = rhoMean + rhoMaxVis


    tq.set_postfix({
        'Kinetic Energy': kineticEnergy.item(),
        'Total Energy': totalEnergy.item(),
        'Time': actualState.t.item() if torch.is_tensor(actualState.t) else actualState.t,
        'nx': args.nx,
        'boundary': args.domainBoundary,
        'obstacle': args.obstacle,
        'viscosity': config['diffusion']['boundary'],
    })
    tq.update(1)
    t = actualState.t.item() if torch.is_tensor(actualState.t) else actualState.t

    if args.export:
        frameGroup = writeParticleDataMinimal(outGroup, actualState, step = i+1, dt = dt)

        # frameGroup.attrs['timestepCFL'] = timestepCFL
        # frameGroup.attrs['CFLNumber'] = CFLNumber.item() if torch.is_tensor(CFLNumber) else CFLNumber
        # frameGroup.attrs['c_s_max'] = c_s_max.item() if torch.is_tensor(c_s_max) else c_s_max
        # frameGroup.attrs['h_min'] = h_min.item() if torch.is_tensor(h_min) else h_min
        # frameGroup.attrs['xi'] = xi.item() if torch.is_tensor(xi) else xi
        # frameGroup.attrs['dt_cfl'] = dt_cfl.item() if torch.is_tensor(dt_cfl) else dt_cfl
        frameGroup.attrs['kineticEnergy'] = kineticEnergy.item() if torch.is_tensor(kineticEnergy) else kineticEnergy
        frameGroup.attrs['thermalEnergy'] = thermalEnergy.item() if torch.is_tensor(thermalEnergy) else thermalEnergy
        frameGroup.attrs['totalEnergy'] = totalEnergy.item() if torch.is_tensor(totalEnergy) else totalEnergy

    if args.plot:
        if (i % plotInterval == 0 and i > 0) or i == timesteps - 1:
            updatePlot(densityPlotState, actualState.systemState, actualState.systemState.velocities)
            updatePlot(internalEnergyPlotState, actualState.systemState, actualState.systemState.densities)
            updatePlot(supportPlotState, actualState.systemState, actualState.systemState.UIDs)


            neighborhood, neighbors = evaluateNeighborhood(actualState.systemState, config['domain'], config['kernel'], verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=actualState.neighborhoodInfo)

            numNeighbors = coo_to_csr(filterNeighborhoodByKind(actualState.systemState, neighbors.neighbors, which = 'noghost')).rowEntries
            updatePlot(numNeighborsPlotState, actualState.systemState, numNeighbors)

            # numNeighbors = coo_to_csr(filterNeighborhoodByKind(actualState.systemState, currentState[-1][1].fullAdjacency, which = 'noghost')).rowEntries
            # numNeighbors = coo_to_csr(currentState[-1][1].fullAdjacency).rowEntries

            updatePlot(velocityXPlotState, actualState.systemState, actualState.systemState.velocities)
            # for patch in axis[0,2].patches[:]:
            #     print(patch)
            #     if isinstance(patch, mpl.patches.FancyArrowPatch):
            #         patch.remove()
            updatePlot(velocityYPlotState, actualState.systemState, actualState.systemState.velocities)

            # for ax in axis.flatten():
                # ax.set_xlim(-1, 1)
                # ax.set_ylim(-1,1)

            fig.suptitle(f'{simulationName}, ptcls = {particleSystem.systemState.positions.shape[0]}, kernel = {config["kernel"].name}, neighbors = {config["targetNeighbors"]:.2g}, $c_s$ = {config["fluid"]["c_s"]:.1f}\n$\\rho$ = [{rhoMin:.4g} | {rhoMean:.4g} | {rhoMax:.4g}], $E_0$ = {totalInitialEnergy:.4g}, $E$ = {totalEnergy:.4g}, $\\Delta E$ = {totalEnergy - totalInitialEnergy:.4g}, $t$ = {t:.4g}, $\\Delta t$ = {dt:.2e}')


            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.savefig(f'{args.exportDir}/{imagePrefix}frame_{i:05d}.png', dpi = 200)

if args.export:
    outFile.close()


import os
import subprocess
import shlex
import imageio.v2 as imageio
from skimage.transform import resize, rescale
from skimage.io import imread


def postProcess(imagePrefix, fps, exportName, targetLongEdge = 600):
    fileList = os.listdir(imagePrefix)
    fileList = [f for f in fileList if f.endswith('.png') and f.startswith('frame_')]
    fileList = sorted(fileList)

    writer = imageio.get_writer(f'{imagePrefix}/output.mp4', fps=fps, bitrate='30M')
    for image in fileList:
        writer.append_data(imageio.imread(imagePrefix + image))
    writer.close()

    images = []
    for image in fileList:
        images.append(imread(imagePrefix + image))

    currentImageSize = images[0].shape
    currentLongEdge = max(images[0].shape[0], images[0].shape[1])
    ratio = targetLongEdge / currentLongEdge

    images = [(rescale(image, ratio, anti_aliasing=True, channel_axis=-1)*255).astype('uint8') for image in images]

    # for image in images:
        # print(image.shape)
        # break

    # images = [resizeImageLongEdge(i, 600) for i in images]

    imageio.mimsave(f'{imagePrefix}/output.gif', images, fps=fps, loop = 0)
    
    print('Copying video to videos folder')
    os.makedirs(f'./videos/', exist_ok= True)
    # subprocess.run(shlex.split(f'cp {imagePrefix}/output.mp4 ./videos/{exportName}.mp4'))
    subprocess.run(shlex.split(f'cp {imagePrefix}/output.gif ./videos/{exportName}.gif'))
    lastFrameFile = imagePrefix + fileList[-1]
    subprocess.run(shlex.split(f'cp {lastFrameFile} ./videos/{exportName}.png'))
    print('Done!')


if args.plot:
    postProcess(
        imagePrefix = imagePrefix,
        fps = 50,
        exportName = fileName,
        targetLongEdge = 1200
    )
# def postProcess(imagePrefix, fps, exportName, targetLongEdge = 600):