
import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

import matplotlib.pyplot as plt
import os
import torch
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'

import torch
from gencase import *
from util import visualize, sampleParticles, generateInitialVariables, SamplingScheme
from sample import smoothState, addNoise, populateCGrid, populateUGrid, smoothValues
from util import plotState, plotInitialState
from simulation import runSimulation
from util import getCurrentTimestamp
from argparse import ArgumentParser
from dataset_util import *

parser = ArgumentParser()

parser.add_argument('--nx', type=int, default=128, help='Number of grid points in one dimension')
parser.add_argument('--sampling', type=str, default='regular', help='Particle sampling scheme: regular, regular_jittered, glass, optimal, random')

parser.add_argument('--dt', type=float, default=0.02, help='Time step size')
parser.add_argument('--nIter', type=int, default=200, help='Number of iterations')

parser.add_argument('--uMagnitudes', type=float, nargs='*', default=[10], help='List of possible wave speeds for obstacles')
parser.add_argument('--uRandomMagnitude', action='store_true', help='Enable random magnitudes for sources')
parser.add_argument('--uRandomMin', type=float, default=-10.0, help='Minimum magnitude for sources when uRandomMagnitude is enabled')
parser.add_argument('--uRandomMax', type=float, default=10.0, help='Maximum magnitude for sources when uRandomMagnitude is enabled')

parser.add_argument('--smoothICs', action='store_true', help='Enable smoothing of initial conditions')
parser.add_argument('--smoothIters', type=int, default=4, help='Number of smoothing iterations for initial conditions')

parser.add_argument('--plotInterval', type=int, default=10, help='Plotting interval')
parser.add_argument('--export', action='store_true', help='Whether to export simulation data')
parser.add_argument('--exportImages', action='store_true', help='Whether to export simulation images')

parser.add_argument('--filePrefix', type=str, default='waveEqn', help='Prefix for output files')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

parser.add_argument('--domainBox', action='store_true', help='Enable domain boundary box')
parser.add_argument('--domainDamping', action='store_true', help='Enable domain boundary damping')

parser.add_argument('--enableNoise', action='store_true', help='Enable noise addition to initial conditions')
parser.add_argument('--noiseType', type=str, default='perlin', help='Type of noise to add: perlin, uniform, normal')
parser.add_argument('--noiseAmplitude', type=float, default=0.02, help='Amplitude of noise to add to initial conditions')
parser.add_argument('--noiseSmoothIter', type=int, default=4, help='Number of smoothing iterations for noise')
parser.add_argument('--noiseSeed', type=int, default=42, help='Random seed for noise generation')

parser.add_argument('--boundarySpeed', type=float, default=0.01, help='Wave speed at boundaries')
parser.add_argument('--obstacleSpeeds', type=float, nargs='*', default=[0.5], help='List of possible wave speeds for obstacles')


parser.add_argument('--defaultSpeed', type=float, default=1.0, help='Default wave speed in the medium')
parser.add_argument('--randomObstacleSpeed', action='store_true', help='Enable random wave speeds for obstacles')
parser.add_argument('--obstacleSpeedMin', type=float, default=0.3, help='Minimum wave speed for obstacles when randomObstacleSpeed is enabled')
parser.add_argument('--obstacleSpeedMax', type=float, default=0.7, help='Maximum wave speed for obstacles when randomObstacleSpeed is enabled')

parser.add_argument('--figureDpi', type=int, default=200, help='DPI for saved figures')
parser.add_argument('--caseIndex', type=int, default=1, help='Index for the simulation case (used in file naming)')


args = parser.parse_args()

folder = f'dataset_{args.nx}_{args.sampling}'


verbose = args.verbose
if verbose:
    print("Simulation Configuration:")
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')

# exit()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if verbose:
    print(f'Using device: {device}')

plotInterval = args.plotInterval
nIter = args.nIter
dt = args.dt
nx = args.nx
# uMagnitude = args.uMagnitude
sampling = args.sampling
samplingScheme = None
export = args.export
timestamp = getCurrentTimestamp()

setup = args.caseIndex


for scheme in SamplingScheme:
    if scheme.name == sampling:
        samplingScheme = scheme
        if verbose:
            print(f'Using sampling scheme: {samplingScheme.name}')
        break
if samplingScheme is None:
    raise ValueError(f'Unknown sampling scheme: {sampling}')

prefix = f'{args.filePrefix}_case_{setup:02d}_{nx}_{sampling}'
os.makedirs(f'output', exist_ok=True)
folderName = f'{folder}/{prefix}_{timestamp}'
os.makedirs(f'output/{folderName}', exist_ok=True)
if verbose:
    print(f'Output folder: output/{folderName}')

config, domain, device, dtype, kernel = generateInitialVariables(
    nx, device = device
)
config['integrationScheme'] = IntegrationSchemeType.rungeKutta4
integrator = getIntegrator(config['integrationScheme'])
if verbose:
    print(f'Using integrator: {config["integrationScheme"].name}')


'''
For our dataset we want to have a few different initial conditions.
These are:
- A single Gaussian bell in the center of the domain
- Two Gaussian bells, one in the left half of the domain and one in the right half
- A Gaussian bell in the center with a random noise pattern superimposed on top
- A random noise pattern with no Gaussian bell
- A vertical line source in the center of the domain
- A horizontal line source in the center of the domain
- A circular ring source in the center of the domain
- A square source in the center of the domain
- Two vertical line sources, one in the left half of the domain and one in the right half
- Two horizontal line sources, one in the top half of the domain and one in the bottom half
- Two circular ring sources, one in the left half of the domain and one in the right half
- Two square sources, one in the top half of the domain and one in the bottom half

All of them have no boundaries (case 01) and constant wave speed c = 1 everywhere.
''';

# uGrid, vGrid, cGrid, dampGrid, uSourceGrid, cSourceGrid = genInitial(
#     particleState, config,
#     nx,
#     domainBox = False,
#     domainDamping = False,
# )

backgroundC = 1
gaussianSourceRadius = 0.2
circularSourceInnerRadius = 0.15
circularSourceOuterRadius = 0.25
squareSourceHalfSize = 0.15
lineSourceWidth = 0.05
uMagnitude = 10

L = domain.max - domain.min 
domainCenter = domain.min + 0.5 * L
# print(domainCenter)
topOffset = domainCenter + torch.tensor([0, 0.25], device = device, dtype = dtype) * L
bottomOffset = domainCenter + torch.tensor([0, -0.25], device = device, dtype = dtype) * L
leftOffset = domainCenter + torch.tensor([-0.25, 0], device = device, dtype = dtype) * L
rightOffset = domainCenter + torch.tensor([0.25, 0], device = device, dtype = dtype) * L
# print(topOffset, bottomOffset, leftOffset, rightOffset)



particles, particleState, neighbors, counter, uGrid, vGrid, cGrid, dampGrid, uSourceGrid, cSourceGrid, waveSystem = genInitialConditions(
    args,
    case = setup,
    domainCenter = domainCenter,
    topOffset = topOffset,
    bottomOffset = bottomOffset,
    leftOffset = leftOffset,
    rightOffset = rightOffset,
    gaussianSourceRadius = gaussianSourceRadius,
    circularSourceInnerRadius = circularSourceInnerRadius,
    circularSourceOuterRadius = circularSourceOuterRadius,
    squareSourceHalfSize = squareSourceHalfSize,
    lineSourceWidth = lineSourceWidth,
    uMagnitude = uMagnitude,
    device = device,
    dtype = dtype
)

eulerIntegrator = getIntegrator(IntegrationSchemeType.explicitEuler)
RK4Integrator = getIntegrator(IntegrationSchemeType.rungeKutta4)

waveSystem_ = copyWaveSystem(waveSystem)

n = waveSystem.waveState.u.shape[0]
nx = int(n**0.5)
# print(f'Assuming square grid with nx={nx}, ny={nx}')
domainArea = (config['domain'].max[0] - config['domain'].min[0]) * (config['domain'].max[1] - config['domain'].min[1])
# print(f'Particle area: {domainArea/n}, particle spacing: {(domainArea/n)**0.5}')
dx = (domainArea/n)**0.5

dt = 0.02
cflNumber = max(args.obstacleSpeeds + [args.defaultSpeed]) * dt / dx

hMax = waveSystem_.systemState.supports.max().item()
if args.verbose:
    print(f'Max wave speed: {waveSystem_.waveState.c.max().item():.4f}')
    print(f'Max support radius: {hMax:.4f}')
    print(f'CFL number (based on dx): {cflNumber:.4f} [dx={dx:.4f}, dt={dt:.4f}]')
    print(f'CFL number (based on hMax): {max(args.obstacleSpeeds + [args.defaultSpeed]) * dt / hMax:.4f} [hMax={hMax:.4f}, dt={dt:.4f}]')


integrator = RK4Integrator
# integrator = eulerIntegrator




################################################################################
#                              Visualization Setup                             #
################################################################################

fig, axis = visualize(particles, particleState.numNeighbors, counter)
fig.savefig(f'output/{folderName}/initial_particles.png', dpi = args.figureDpi)

fig, axis = plotInitialState(
    particleState, config,
    uGrid, vGrid, cGrid, dampGrid,
    uSourceGrid, cSourceGrid
)
fig.savefig(f'output/{folderName}/initial_fields.png', dpi = args.figureDpi)

fig, axis, uPlot, vPlot, cPlot, dampPlot = plotState(
    particleState,
    waveSystem.waveState,
    config, kernel,
    markerSize = 0.5,
    plotGrid = True,
    plotCD = False)
fig.savefig(f'output/{folderName}/initial_state.png', dpi = args.figureDpi)



export = True

################################################################################
#                                Run Simulation                                #
################################################################################

if verbose:
    print("Starting simulation...")

runSimulation(fig, particleState, uPlot, vPlot, waveSystem_, waveSystemFunction, integrator, nx, dt, args.nIter, kernel, config, args.export, plotInterval = 1, exportImages = args.exportImages, umin = -5, umax = 5, vmin = -50, vmax = 50, prefix = prefix, timestamp = timestamp, folderName = folderName)

if verbose:
    print("Simulation complete.")