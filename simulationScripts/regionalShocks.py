import torch
import os
import copy
if torch.cuda.is_available():
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'

import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt

from diffSPH.operations import sph_operation, mod
from diffSPH.sampling import buildDomainDescription, sampleRegularParticles
from diffSPH.modules.eos import idealGasEOS
from diffSPH.modules.timestep import computeTimestep
from diffSPH.schema import getSimulationScheme
from diffSPH.reference.sod import buildSod_reference, sodInitialState, generateSod1D
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH
from diffSPH.reference.sod import plotSod
# from diffSPH.reference.linear import buildLinearWaveSimulation, runLinearWaveTest
from diffSPH.enums import *
from diffSPH.reference.sod import plotSod

from scriptUtils import sampleRegionsSymmetric, mergeParticles, plotDomain, recreateUIDs
from diffSPH.schemes.states.compressiblesph import CompressibleState
from diffSPH.plotting import visualizeParticles, updatePlot
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH, evaluateOptimalSupportOwen

from diffSPH.plotting import visualizeParticles, updatePlot
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH, evaluateOptimalSupportOwen
from diffSPH.sampling import generateNoiseInterpolator, sampleDivergenceFreeNoise


from diffSPH.modules.eos import idealGasEOS
from diffSPH.modules.compressible import CompressibleState
from diffSPH.modules.density import computeDensity
from diffSPH.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr

from diffSPH.neighborhood import evaluateNeighborhood, filterNeighborhoodByKind, SupportScheme

from diffSPH.plotting import visualizeParticles, updatePlot
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH, evaluateOptimalSupportOwen
from diffSPH.io import initializeOutputFile, writeParticleData, writeParticleDataMinimal
import datetime
import matplotlib as mpl
from diffSPH.kernels import Kernel_xi
from diffSPH.util import postProcess
import argparse

parser = argparse.ArgumentParser(description='Run a regional shock simulation with adaptive smoothing.')

parser.add_argument('--nx', type=int, default=128, help='Number of particles in each dimension.')
parser.add_argument('--gamma', type=float, default=5/3, help='Adiabatic index for the ideal gas EOS.')
parser.add_argument('--timeLimit', type=float, default=4.0, help='Time limit for the simulation in seconds.')
parser.add_argument('--fps', type=int, default=50, help='Frames per second for the output video.')
parser.add_argument('--caseName', type=str, default='regional', help='Name of the simulation case.')

parser.add_argument('--splitLineX', type=float, default=0.0, help='X-coordinate of the split line.')
parser.add_argument('--splitLineY', type=float, default=0.0, help='Y-coordinate of the split line.')
parser.add_argument('--regions', type=int, default=4, help='Number of regions to simulate.')
parser.add_argument('--sdf', type=str, default=None, help='SDF function to use for masking particles.')

parser.add_argument('--densityNoise', action='store_true', help='Whether to add noise to the initial density field.')
parser.add_argument('--pressureNoise', action='store_true', help='Whether to add noise to the initial pressure field.')
parser.add_argument('--densityNoiseSeed', type=int, default=42, help='Seed for the random number generator for density noise.')
parser.add_argument('--pressureNoiseSeed', type=int, default=42, help='Seed for the random number generator for pressure noise.')

parser.add_argument('--velocityNoise', action='store_true', help='Whether to add noise to the initial velocity field.')
parser.add_argument('--octaves', type=int, default=2, help='Number of octaves for the noise field.')
parser.add_argument('--lacunarity', type=int, default=2, help='Lacunarity for the noise field.')
parser.add_argument('--persistence', type=float, default=0.5, help='Persistence for the noise field.')
parser.add_argument('--baseFrequency', type=float, default=4.0, help='Base frequency for the noise field.')
parser.add_argument('--tileable', type=bool, default=True, help='Whether the noise field should be tileable.')
parser.add_argument('--kind', type=str, default='perlin', help='Type of noise to generate.')
parser.add_argument('--seed', type=int, default=4235, help='Seed for the random number generator.')

parser.add_argument('--rho0', type=float, default=1.0, help='Initial density of the particles.')
parser.add_argument('--Pinitial', type=float, default=1.0, help='Initial pressure of the particles.')
parser.add_argument('--dt', type=float, default=1e-3, help='Time step for the simulation.')

parser.add_argument('--minRho', type=float, default=0.1, help='Minimum density for the particles.')
parser.add_argument('--maxRho', type=float, default=4.0, help='Maximum density for the particles.')
parser.add_argument('--minP', type=float, default=0.1, help='Minimum pressure for the particles.')
parser.add_argument('--maxP', type=float, default=4.0, help='Maximum pressure for the particles.')
parser.add_argument('--verbose', action='store_true', help='Whether to print verbose output during the simulation.')

parser.add_argument('--adaptiveHScheme', type=str, default='Owen', choices=['Owen', 'None'], help='Adaptive smoothing scheme to use.')
parser.add_argument('--simulationScheme', type=str, default='CRKSPH', choices=['CRKSPH', 'CompSPH'], help='Simulation scheme to use.')
parser.add_argument('--integrationScheme', type=str, default='rungeKutta2', choices=['rungeKutta2', 'Euler'], help='Integration scheme to use.')
parser.add_argument('--kernelType', type=str, default='B7', choices=['B7', 'CubicSpline'], help='Kernel type to use.')

parser.add_argument('--gpu', type=int, default=0, help='GPU index to use for the simulation.')
parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs.')

parser.add_argument('--plot', dest='plot', action='store_true', default=True, help='Whether to plot the regions in the simulation.')
parser.add_argument('--no-plot', dest='plot', action='store_false', help='Disable plotting of the regions in the simulation.')
parser.add_argument('--export', action='store_true', help='Whether to export the simulation data to a file.')
parser.add_argument('--exportDir', type=str, default='./out/', help='Directory to export the simulation data to.')

args = parser.parse_args()
verbose = args.verbose


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32

dim = 2
# kernel = KernelType.B7
kernel = None

for k in KernelType:
    if k.name == args.kernelType:
        kernel = k
        break
if kernel is None:
    raise ValueError(f'Unknown kernel type: {args.kernelType}. Available options are: {[k.name for k in KernelType]}')

targetNeighbors = n_h_to_nH(4, dim)
CFL = 0.3
if verbose:
    print(f'Using device: {device}, dtype: {dtype}, kernel: {kernel}, targetNeighbors: {targetNeighbors}, CFL: {CFL}')

nx = args.nx
rho0 = args.rho0
gamma = args.gamma
caseName = args.caseName
if verbose:
    print(f'Using gamma: {gamma}, caseName: {caseName}')

scheme = None
for s in SimulationScheme:
    if s.name == args.simulationScheme:
        scheme = s
        break
if scheme is None:
    raise ValueError(f'Unknown simulation scheme: {args.simulationScheme}. Available options are: {[s.name for s in SimulationScheme]}')

integrationScheme = None
for i in IntegrationSchemeType:
    if i.name == args.integrationScheme:
        integrationScheme = i
        break
if integrationScheme is None:
    raise ValueError(f'Unknown integration scheme: {args.integrationScheme}. Available options are: {[i.name for i in IntegrationSchemeType]}')

viscositySwitch = ViscositySwitch.NoneSwitch
supportScheme = AdaptiveSupportScheme.OwenScheme

if args.adaptiveHScheme == 'Owen':
    supportScheme = AdaptiveSupportScheme.OwenScheme
elif args.adaptiveHScheme == 'None':
    supportScheme = AdaptiveSupportScheme.NoScheme
elif args.adaptiveHScheme == 'Monaghan':
    supportScheme = AdaptiveSupportScheme.MonaghanScheme

if verbose:
    print(f'Using scheme: {scheme}, integrationScheme: {integrationScheme}, viscositySwitch: {viscositySwitch}, supportScheme: {supportScheme}')

domain = buildDomainDescription(l = 1, dim = dim, periodic = True, device = device, dtype = dtype)
domain.min = torch.tensor([-1, -1], device = device, dtype = dtype)
domain.max = torch.tensor([1, 1], device = device, dtype = dtype)

simulator, SimulationSystem, solverConfig, integrator = getSimulationScheme(
     scheme, kernel, integrationScheme, 
     gamma, targetNeighbors, domain, 
     viscositySwitch=viscositySwitch, supportScheme = supportScheme)

L = 2
band = 0
solverConfig['particle'] = {
    'nx': nx + 2 * band,
    'dx': L/nx,
    'targetNeighbors': targetNeighbors,
    'band': band
}

splitLineX = args.splitLineX
splitLineY = args.splitLineY

particlesA, particlesB, particlesC, particlesD, splitx, splity = sampleRegionsSymmetric(domain, [nx, nx, nx, nx], targetNeighbors, splitLineX, splitLineY)
particles, setIndex = mergeParticles([particlesA, particlesB, particlesC, particlesD])


from scriptUtils import maskParticles_2


setIndex, sdf_ = maskParticles_2(particles, args.regions, domain, nx, sdf = args.sdf, sdfParameters = None, split_x = splitLineX, split_y = splitLineY)

if verbose:
    print(f'Set index shape: {setIndex.shape}, unique set indices: {setIndex.unique()}')

Pinitial = torch.ones_like(particles.positions[:, 0], device=device, dtype=dtype) * args.Pinitial
rho = torch.ones_like(particles.positions[:, 0], device=device, dtype=dtype) * rho0

masses = particles.masses.clone()
m0 = torch.sum(masses) / particles.positions.shape[0]

import numpy as np

np.random.seed(args.seed)  # For reproducibility




numberOfRegions = int(setIndex.max().item() + 1)
Pinitials = np.random.uniform(args.minP, args.maxP, size = (numberOfRegions,))
rhoInitials = np.random.uniform(args.minRho, args.maxRho, size = (numberOfRegions,))
if numberOfRegions > 1:
    for i in range(numberOfRegions):
        Pinitial[setIndex == i] = Pinitials[i]
        rho[setIndex == i] = rhoInitials[i] * rho0
        masses[setIndex == i] = rhoInitials[i] * m0
        if verbose:
            print(f'Setting region {i}: Pinitial = {Pinitials[i]}, rho = {rhoInitials[i] * rho0}, mass = {rhoInitials[i] * m0}, particles in region: {setIndex[setIndex == i].shape[0]}')


# Pinitial[setIndex == 3] = 0.1795
# rho[setIndex == 3] = 0.25
# particles = particles._replace(masses = torch.where(setIndex == 3, particles.masses / 4.0, particles.masses))

A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = Pinitial, rho = rho, gamma = gamma)
# v_initial = torch.zeros_like(particles_l.positions)

# if verbose:
#     print(f'Initial pressure range: {P_.min().item()} - {P_.max().item()}')
#     print(f'Initial density range: {rho.min().item()} - {rho.max().item()}')
#     print(f'Initial internal energy range: {u_.min().item()} - {u_.max().item()}')
#     print(f'Initial sound speed range: {c_s.min().item()} - {c_s.max().item()}')

simulationState = CompressibleState(
    positions = particles.positions,
    supports = particles.supports,
    masses = masses,
    densities = rho,        
    velocities = torch.zeros_like(particles.positions, device = device, dtype = dtype),
    
    kinds = torch.zeros_like(particles.positions[:,0], dtype = torch.int32),
    materials = torch.zeros_like(particles.positions[:,0], dtype = torch.int32),
    UIDs = recreateUIDs(particles, domain, verbose=False),

    internalEnergies = u_,
    totalEnergies = None,
    entropies = A_,
    pressures = P_,
    soundspeeds = c_s,

    alphas = torch.ones_like(rho),
    alpha0s = torch.ones_like(rho)
)

# simulationState.velocities[setIndex == 3,0] = 1


octaves = int(args.octaves)
lacunarity = int(args.lacunarity)
persistence = float(args.persistence)
baseFrequency = int(args.baseFrequency)
tileable = args.tileable
kind = args.kind
seed = args.seed

# octaves = 2
# lacunarity = 2
# persistence = 0.5
# baseFrequency = 4
# tileable = True
# kind = 'perlin'
# seed = 4235

noiseGen = generateNoiseInterpolator(nx, nx, domain, dim = domain.dim, octaves = octaves, lacunarity = lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = seed)
velocityField = sampleDivergenceFreeNoise(simulationState, domain, solverConfig, nx, octaves, lacunarity, persistence, baseFrequency, tileable, kind, seed)

if args.velocityNoise:
    simulationState.velocities = velocityField
    if verbose:
        print(f'Added noise to initial velocities with shape: {velocityField.shape} and range: {velocityField.min().item()} - {velocityField.max().item()}')

if args.densityNoise:
    noiseGen = generateNoiseInterpolator(nx, nx, domain, dim = domain.dim, octaves = octaves, lacunarity = lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = args.densityNoiseSeed)
    noiseField = noiseGen(simulationState.positions).to(dtype)
    simulationState.densities = args.minRho + (args.maxRho - args.minRho) * (noiseField + 1) / 2

    simulationState.masses = simulationState.densities * m0
    # Pinitial[setIndex == i] = Pinitials[i]
    # rho[setIndex == i] = rhoInitials[i] * rho0
    # masses[setIndex == i] = rhoInitials[i] * m0
    if verbose:
        print(f'Added noise to initial densities with shape: {simulationState.densities.shape} and range: {simulationState.densities.min().item()} - {simulationState.densities.max().item()}')
if args.pressureNoise:
    noiseGen = generateNoiseInterpolator(nx, nx, domain, dim = domain.dim, octaves = octaves, lacunarity = lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = args.pressureNoiseSeed)
    noiseField = noiseGen(simulationState.positions).to(dtype)
    Pinitial = args.minP + (args.maxP - args.minP) * (noiseField + 1) / 2

# noiseField = noiseGen(simulationState.positions).to(dtype)

# simulationState.masses = m0 + m0 * 0.75 * noiseField
# Pinitial = Pinitial + Pinitial * 0.75 * noiseField

# fig, axis = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False, sharex=True, sharey=True)
# _ = visualizeParticles(fig, axis[0,0], simulationState, domain, kernel = solverConfig['kernel'], cmap = 'viridis', mapping = '.x', gridVisualization = True, quantity = velocityField[:,0])
# _ = visualizeParticles(fig, axis[0,1], simulationState, domain, kernel = solverConfig['kernel'], cmap = 'viridis', mapping = '.y', gridVisualization = True, quantity = velocityField[:,1])
# _ = visualizeParticles(fig, axis[0,2], simulationState, domain, kernel = solverConfig['kernel'], cmap = 'viridis', mapping = 'L2', gridVisualization = True, quantity = velocityField, streamLines = True)

# fig.tight_layout()

neighborhood, neighbors = evaluateNeighborhood(simulationState, domain, kernel, verletScale = 1.4, mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
# numNeighbors = coo_to_csr(neighborhood.fullAdjacency).rowEntries
numNeighbors = coo_to_csr(filterNeighborhoodByKind(simulationState, neighbors.neighbors, which = 'noghost')).rowEntries
densities = computeDensity(simulationState, kernel, neighbors.get('noghost'), SupportScheme.Gather, solverConfig)
# print(densities.min(), densities.max(), densities.mean())

simulationState.densities = densities

A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = Pinitial, rho = simulationState.densities, gamma = gamma)

if verbose:
    print(f'Initial pressure range: {P_.min().item()} - {P_.max().item()}')
    print(f'Initial density range: {simulationState.densities.min().item()} - {simulationState.densities.max().item()}')
    print(f'Initial internal energy range: {u_.min().item()} - {u_.max().item()}')
    print(f'Initial sound speed range: {c_s.min().item()} - {c_s.max().item()}')

simulationState.internalEnergies = u_


# exit()

particleSystem = SimulationSystem(
        systemState = simulationState,
        domain = domain,
        neighborhoodInfo = neighborhood,
        t = 0
    )

timeLimit = args.timeLimit
dt = computeTimestep(scheme, 1e-3, particleSystem.systemState, solverConfig, None)
# dt = torch.tensor(1e-4, dtype = dtype, device = device)
timesteps = int(timeLimit / dt)
actualState = copy.deepcopy(particleSystem)

import math

dt = args.dt
fps = args.fps
exportInterval = 1 / fps
exportSteps = int(math.ceil(exportInterval / dt))
newDt = exportInterval / exportSteps
plotInterval = int(math.floor(exportInterval / newDt))
plotInterval = min(max(plotInterval,1), 1000)
if verbose:
    print(f'Export Interval: {exportInterval}, Export Steps: {exportSteps}')
    print(f'Current dt: {dt}')
    print(f'Plot Interval: {plotInterval}')
    print(f'export Steps: {exportSteps}')
    print(f'New dt: {newDt}')

timesteps = int(timeLimit / newDt)
currentTime = datetime.datetime.now()
timestamp = currentTime.strftime("%Y-%m-%d_%H-%M-%S")

imagePrefix = f'{args.exportDir}/images/{caseName}_{nx**2}_{args.seed}_{timestamp}_{args.gpu}/'
exportName = f'{args.exportDir}/data/{caseName}_{nx**2}_{args.seed}_{timestamp}_{args.gpu}.h5'
if args.export:
    os.makedirs(os.path.dirname(exportName), exist_ok = True)
    os.makedirs(imagePrefix, exist_ok = True)


def plotScalar(fig, axis, label, fluidParticles, quantity, domain, solverConfig, s, gridVisualization=True, gridResolution=256, mapping = None, cmap = 'viridis'):
    return visualizeParticles(fig, axis,
                     particles = fluidParticles, 
                     domain = domain, 
                     quantity = quantity, 
                     which = 'fluid',
                     visualizeBoth=False,
                     kernel = solverConfig['kernel'],
                     plotDomain = False,
                    #  scaling = 'sym',
                     cmap = cmap,
                     midPoint=1.5,
                     mapping = mapping,
                     title=label,
                     markerSize = s,
                     gridVisualization=gridVisualization, gridResolution=gridResolution)


if args.plot:
    fig, axis = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False, sharex=True, sharey=True)

    s = 0.25
    fluidParticles = actualState.systemState


    densityPlotState        = plotScalar(fig, axis[0,0], 'Density $\\rho$',     fluidParticles, densities,           domain, solverConfig, s, gridVisualization=False, gridResolution=256, cmap = 'viridis', mapping = None)
    internalEnergyPlotState = plotScalar(fig, axis[0,1], 'Internal Energy $u$', fluidParticles, fluidParticles.internalEnergies,    domain, solverConfig, s, gridVisualization=False, gridResolution=256, cmap = 'magma', mapping = None)
    supportPlotState        = plotScalar(fig, axis[0,2], 'Support $h$',         fluidParticles, fluidParticles.supports,            domain, solverConfig, s, gridVisualization=True, gridResolution=256, cmap = 'magma', mapping = None)
    numNeighborsPlotState   = plotScalar(fig, axis[1,2], 'Number of Neighbors', fluidParticles, numNeighbors,                       domain, solverConfig, s, gridVisualization=True, gridResolution=256, mapping = '.y', cmap = 'viridis')
    velocityXPlotState      = plotScalar(fig, axis[1,0], 'Velocity X',          fluidParticles, fluidParticles.velocities,          domain, solverConfig, s, gridVisualization=True, gridResolution=256, mapping = '.x', cmap = 'RdBu_r')
    velocityYPlotState      = plotScalar(fig, axis[1,1], 'Velocity Y',          fluidParticles, fluidParticles.velocities,          domain, solverConfig, s, gridVisualization=True, gridResolution=256, mapping = '.y', cmap = 'RdBu_r')

    # for ax in axis.flatten():
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1,1)
    kineticEnergy = 0.5 * (torch.linalg.norm(actualState.systemState.velocities, dim = -1) **2 * actualState.systemState.masses).sum()
    thermalEnergy = (actualState.systemState.internalEnergies * actualState.systemState.masses).sum()
    totalEnergy = kineticEnergy + thermalEnergy



    fig.suptitle(f'{solverConfig["schemeName"]}\n{caseName}, t = {actualState.t:2f} [step: {0:4d}], dt = {dt:.3g}, ptcls = {len(actualState.systemState.positions)}\nTotal Energy: {totalEnergy:.3g}, Kinetic Energy: {kineticEnergy:.3g}, Thermal Energy: {thermalEnergy:.3g}')
    # fig.suptitle(f'{solverConfig["schemeName"]}\nSedov-Taylor Explosion, t = {simulationState.t:2f} [step: {0:4d}], dt = {dt:.3g}, ptcls = {len(simulationState.systemState.positions)}\nTotal Energy: {totalEnergy:.3g}, Kinetic Energy: {kineticEnergy:.3g}, Thermal Energy: {thermalEnergy:.3g}')

    fig.tight_layout()

    fig.savefig(f'{imagePrefix}frame_{0:05d}.png', dpi = 200)
    
if verbose:
    print(f'Done preparing simulation')

if args.export:
    outFile = initializeOutputFile(exportName, actualState, solverConfig,simulationName='testData')
    outGroup = outFile['simulationData']
    writeParticleData(outGroup, actualState, step = 0, dt = newDt)

    outFile['caseSpecificData'].attrs['caseName'] = caseName
    outFile['caseSpecificData'].attrs['nx'] = nx
    outFile['caseSpecificData'].attrs['gamma'] = gamma
    outFile['caseSpecificData'].attrs['rho0'] = rho0

    outFile['caseSpecificData'].attrs['splitLineX'] = splitLineX
    outFile['caseSpecificData'].attrs['splitLineY'] = splitLineY
    outFile['caseSpecificData'].attrs['regions'] = args.regions
    if args.sdf is None:
        outFile['caseSpecificData'].attrs['sdf'] = 'None'
    else:
        outFile['caseSpecificData'].attrs['sdf'] = args.sdf

    outFile['caseSpecificData'].attrs['densityNoise'] = args.densityNoise
    outFile['caseSpecificData'].attrs['pressureNoise'] = args.pressureNoise
    outFile['caseSpecificData'].attrs['densityNoiseSeed'] = args.densityNoiseSeed
    outFile['caseSpecificData'].attrs['pressureNoiseSeed'] = args.pressureNoiseSeed

    outFile['caseSpecificData'].attrs['velocityNoise'] = args.velocityNoise
    outFile['caseSpecificData'].attrs['octaves'] = octaves
    outFile['caseSpecificData'].attrs['lacunarity'] = lacunarity
    outFile['caseSpecificData'].attrs['persistence'] = persistence
    outFile['caseSpecificData'].attrs['baseFrequency'] = baseFrequency
    outFile['caseSpecificData'].attrs['tileable'] = tileable
    outFile['caseSpecificData'].attrs['kind'] = kind
    outFile['caseSpecificData'].attrs['seed'] = seed

    outFile['caseSpecificData'].attrs['rho0'] = args.rho0
    outFile['caseSpecificData'].attrs['Pinitial'] = args.Pinitial
    outFile['caseSpecificData'].attrs['dt'] = dt
    outFile['caseSpecificData'].attrs['minRho'] = args.minRho
    outFile['caseSpecificData'].attrs['maxRho'] = args.maxRho
    outFile['caseSpecificData'].attrs['minP'] = args.minP
    outFile['caseSpecificData'].attrs['maxP'] = args.maxP

    outFile['caseSpecificData'].attrs['verbose'] = args.verbose
    outFile['caseSpecificData'].attrs['adaptiveHScheme'] = args.adaptiveHScheme
    outFile['caseSpecificData'].attrs['simulationScheme'] = args.simulationScheme
    outFile['caseSpecificData'].attrs['integrationScheme'] = args.integrationScheme
    outFile['caseSpecificData'].attrs['kernelType'] = args.kernelType

    outFile['caseSpecificData'].attrs['CFL'] = CFL
    outFile['caseSpecificData'].attrs['targetNeighbors'] = targetNeighbors
    outFile['caseSpecificData'].attrs['exportInterval'] = exportInterval
    outFile['caseSpecificData'].attrs['exportSteps'] = exportSteps
    outFile['caseSpecificData'].attrs['newDt'] = newDt
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
dt = newDt
actualState.systemState.divergence = torch.zeros_like(actualState.systemState.densities)
for i in (range(timesteps)):
# while(True):
    actualState, currentState, updates = integrator.function(actualState, dt, simulator, solverConfig, priorStep = actualState.priorStep)
    actualState.priorStep = [updates[-1], currentState[-1]]
    actualState.priorStep = None
    # if i%100 == 0:
        # states.append(copy.deepcopy(simulationState).to(dtype = torch.float32, device = 'cpu'))

    kineticEnergy = 0.5 * (torch.linalg.norm(actualState.systemState.velocities, dim = -1) **2 * actualState.systemState.masses).sum()
    thermalEnergy = (actualState.systemState.internalEnergies * actualState.systemState.masses).sum()
    totalEnergy = kineticEnergy + thermalEnergy

    # tq.set_postfix({
    #     'Kinetic Energy': kineticEnergy.item(),
    #     'Thermal Energy': thermalEnergy.item(),
    #     'Total Energy': totalEnergy.item(),
    #     'Time': actualState.t.item() if torch.is_tensor(actualState.t) else actualState.t,
    # })

    c_s = idealGasEOS(A=None, P = None, u = actualState.systemState.internalEnergies, rho = actualState.systemState.densities, gamma = solverConfig['fluid']['gamma'])[-1]
    c_s_max = c_s.max()
    h_min = actualState.systemState.supports.min()
    xi = Kernel_xi(solverConfig['kernel'], actualState.systemState.positions.shape[1])

    # xi = solverConfig['kernel'].xi(particles.positions.shape[1])
    # CFL = 0.3
    timestepCFL = 0.3
    dt_cfl = timestepCFL * h_min / (c_s_max * xi)
    CFLNumber = (c_s_max * xi * dt) / h_min
    # if verbose:
        # print(f'Current dt: {dt:.3g}, CFL dt: {dt_cfl:.3g}, h_min: {h_min:.3g}, c_s_max: {c_s_max:.3g}, xi: {xi:.3g}, EK: {kineticEnergy:.3g}, ET: {thermalEnergy:.3g}, ETot: {totalEnergy:.3g}, CFL: {CFLNumber:.3g}')

    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        tq.set_postfix({
            'dt': dt.item() if torch.is_tensor(dt) else dt,
            'CFL dt': dt_cfl.item() if torch.is_tensor(dt_cfl) else dt_cfl,
            'h_min': h_min.item() if torch.is_tensor(h_min) else h_min,
            'c_s_max': c_s_max.item() if torch.is_tensor(c_s_max) else c_s_max,
            'xi': xi.item() if torch.is_tensor(xi) else xi,
            'CFL': CFLNumber.item() if torch.is_tensor(CFLNumber) else CFLNumber,
            'KE': kineticEnergy.item() if torch.is_tensor(kineticEnergy) else kineticEnergy,
            'TE': thermalEnergy.item() if torch.is_tensor(thermalEnergy) else thermalEnergy,
            'E': totalEnergy.item() if torch.is_tensor(totalEnergy) else totalEnergy,
        })
        tq.update()

    if args.export:
        frameGroup = writeParticleDataMinimal(outGroup, actualState, step = i+1, dt = dt)

        frameGroup.attrs['timestepCFL'] = timestepCFL
        frameGroup.attrs['CFLNumber'] = CFLNumber.item() if torch.is_tensor(CFLNumber) else CFLNumber
        frameGroup.attrs['c_s_max'] = c_s_max.item() if torch.is_tensor(c_s_max) else c_s_max
        frameGroup.attrs['h_min'] = h_min.item() if torch.is_tensor(h_min) else h_min
        frameGroup.attrs['xi'] = xi.item() if torch.is_tensor(xi) else xi
        frameGroup.attrs['dt_cfl'] = dt_cfl.item() if torch.is_tensor(dt_cfl) else dt_cfl
        frameGroup.attrs['kineticEnergy'] = kineticEnergy.item() if torch.is_tensor(kineticEnergy) else kineticEnergy
        frameGroup.attrs['thermalEnergy'] = thermalEnergy.item() if torch.is_tensor(thermalEnergy) else thermalEnergy
        frameGroup.attrs['totalEnergy'] = totalEnergy.item() if torch.is_tensor(totalEnergy) else totalEnergy

    if args.plot:
        if (i % plotInterval == 0 and i > 0) or i == timesteps - 1:
            updatePlot(densityPlotState, actualState.systemState, actualState.systemState.densities)
            updatePlot(internalEnergyPlotState, actualState.systemState, actualState.systemState.internalEnergies)
            updatePlot(supportPlotState, actualState.systemState, actualState.systemState.supports)


            neighborhood, neighbors = evaluateNeighborhood(actualState.systemState, solverConfig['domain'], solverConfig['kernel'], verletScale = solverConfig['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=actualState.neighborhoodInfo)
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

            fig.suptitle(f'{solverConfig["schemeName"]}\n{caseName}, t = {actualState.t:2f} [step: {i:4d}], dt = {dt:.3g}, ptcls = {len(actualState.systemState.positions)}\nTotal Energy: {totalEnergy:.3g}, Kinetic Energy: {kineticEnergy:.3g}, Thermal Energy: {thermalEnergy:.3g}')


            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.savefig(f'{imagePrefix}frame_{i:05d}.png', dpi = 200)
# outFile.close()

import os
import subprocess
import shlex
import imageio.v2 as imageio
from skimage.transform import resize, rescale
from skimage.io import imread

def postProcess(imagePrefix, fps, exportName, targetLongEdge = 600):
    output = 'timestamp'
    scale = 1280

    command = f'/usr/bin/ffmpeg -loglevel warning -hide_banner -y -framerate {fps} -f image2 -pattern_type glob -i '+ imagePrefix + f'/frame_*.png -c:v libx264 -b:v 20M -r {fps} ' + imagePrefix + '/output.mp4'
    commandB = f'/usr/bin/ffmpeg -loglevel warning -hide_banner -y -i {imagePrefix}/output.mp4 -vf "fps={fps},scale={scale}:-1:flags=lanczos,palettegen" {imagePrefix}/palette.png'
    commandC = f'/usr/bin/ffmpeg -loglevel warning -hide_banner -y -i {imagePrefix}/output.mp4 -i {imagePrefix}/palette.png -filter_complex "fps={fps},scale={scale}:-1:flags=lanczos[x];[x][1:v]paletteuse" {imagePrefix}/output.gif'

    # print('Creating video from  frames (frame count: {})'.format(len(os.listdir(imagePrefix))))
    # print(f'Creating video from frames (frame count: {timesteps})')
    subprocess.run(shlex.split(command))
    # print('Creating palette from video')
    subprocess.run(shlex.split(commandB))
    # print('Creating gif from video')
    subprocess.run(shlex.split(commandC))

    # print('Copying video to videos folder')
    os.makedirs(f'./videos/', exist_ok= True)
    os.makedirs(f'./lastFrames/', exist_ok= True)

    subprocess.run(shlex.split(f'cp {imagePrefix}/output.mp4 ./videos/{exportName}.mp4'))
    subprocess.run(shlex.split(f'cp {imagePrefix}/output.gif ./videos/{exportName}.gif'))
    lastFrameFile = f'{imagePrefix}/frame_{timesteps - 1:05d}.png'
    subprocess.run(shlex.split(f'cp {lastFrameFile} ./lastFrames/{exportName}.png'))
    # print('Done!')


# def postProcess(imagePrefix, fps, exportName, targetLongEdge = 600):
#     fileList = os.listdir(imagePrefix)
#     fileList = [f for f in fileList if f.endswith('.png') and f.startswith('frame_')]
#     fileList = sorted(fileList)

#     writer = imageio.get_writer(f'{imagePrefix}/output.mp4', fps=fps, bitrate='30M')
#     for image in fileList:
#         writer.append_data(imageio.imread(imagePrefix + image))
#     writer.close()

#     images = []
#     for image in fileList:
#         images.append(imread(imagePrefix + image))

#     currentImageSize = images[0].shape
#     currentLongEdge = max(images[0].shape[0], images[0].shape[1])
#     ratio = targetLongEdge / currentLongEdge

#     images = [(rescale(image, ratio, anti_aliasing=True, channel_axis=-1)*255).astype('uint8') for image in images]

#     # for image in images:
#         # print(image.shape)
#         # break

#     # images = [resizeImageLongEdge(i, 600) for i in images]

#     imageio.mimsave(f'{imagePrefix}/output.gif', images, fps=fps, loop = 0)
    
#     print('Copying video to videos folder')
#     os.makedirs(f'./videos/', exist_ok= True)
#     # subprocess.run(shlex.split(f'cp {imagePrefix}/output.mp4 ./videos/{exportName}.mp4'))
#     subprocess.run(shlex.split(f'cp {imagePrefix}/output.gif ./videos/{exportName}.gif'))
#     lastFrameFile = imagePrefix + fileList[-1]
#     subprocess.run(shlex.split(f'cp {lastFrameFile} ./videos/{exportName}.png'))
#     print('Done!')


if args.plot:
    postProcess(
        imagePrefix = imagePrefix,
        fps = 50,
        exportName = f'{caseName}_{nx**2}_{timestamp}_{args.gpu}',
        targetLongEdge = 1200
    )
# def postProcess(imagePrefix, fps, exportName, targetLongEdge = 600):