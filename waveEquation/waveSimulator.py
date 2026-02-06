
import os
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

from dataclasses import dataclass
@dataclass
class Arguments:
    nx: int = 50
    sampling: str = 'regular'

    dt: float = 0.0025
    nIter: int = 1024

    uMagnitudes: list = (10,)
    uRandomMagnitude: bool = False
    uRandomMin: float = -10.0
    uRandomMax: float = 10.0

    smoothICs: bool = False
    smoothIters: int = 4

    plotInterval: int = 50
    export: bool = False
    exportImages: bool = False

    filePrefix: str = 'waveEqn'
    verbose: bool = False   

    domainBox: bool = False
    domainDamping: bool = False

    enableNoise: bool = False
    noiseType: str = 'perlin'
    noiseAmplitude: float = 0.02
    noiseSmoothIter: int = 4
    noiseSeed: int = 42

    boundarySpeed: float = 0.01
    obstacleSpeeds: list = (0.5,)

    defaultSpeed: float = 1.0
    randomObstacleSpeed: bool = False
    obstacleSpeedMin: float = 0.3
    obstacleSpeedMax: float = 0.7

    figureDpi: int = 200
    displayImages: bool = False

    waveCase: int = 1
    boundaryCase: int = 1

    sourceRadii: list = (0.15,)
    sourceShapes: list = ('circle',)

    sourceRandomRadius: bool = False
    sourceRandomRotation: bool = False

    boundaryRadii: list = (0.25, 0.25)
    boundaryShapes: list = ('square', 'circle')
    boundaryOffsets: list = ((0.0, 0.0), (0.0, 0.0))
    boundaryRotations: list = (0.0, 0.0)

    boundaryRandomRadius: bool = False
    boundaryRandomRotation: bool = False
    boundaryRandomOffset: bool = False

    radiusRange: tuple = (0.05, 0.15)
    offsetRange: tuple = ((-0.5, 0.5), (-0.5, 0.5))
    
    uMaskLeft: bool = False
    uMaskRight: bool = False
    uMaskTop: bool = False
    uMaskBottom: bool = False

def runCase(
        args: Arguments,
):
    ################################################################################
    #                                Setup Simulation                             #
    ################################################################################

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


    for scheme in SamplingScheme:
        if scheme.name == sampling:
            samplingScheme = scheme
            if verbose:
                print(f'Using sampling scheme: {samplingScheme.name}')
            break
    if samplingScheme is None:
        raise ValueError(f'Unknown sampling scheme: {sampling}')

    prefix = f'{args.filePrefix}_{nx}_{sampling}'
    os.makedirs('output', exist_ok=True)
    folderName = f'{prefix}_{timestamp}'
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

    ################################################################################
    #                             Particle Generation                              #
    ################################################################################

    particles, numNeighbors, counter = sampleParticles(nx, scheme=samplingScheme)
    particleState = BasicState(particles.positions, particles.supports, particles.masses, particles.densities, torch.zeros_like(particles.positions), torch.zeros(particles.positions.shape[0], device = device, dtype = torch.int64), torch.zeros(particles.positions.shape[0], device = device, dtype = torch.int64), torch.arange(particles.positions.shape[0], device = device), particles.positions.shape[0])
    neighborhood, neighbors = evaluateNeighborhood(particleState, config['domain'], kernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
    particleState.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particleState, neighbors.neighbors, which = 'noghost')).rowEntries
    particleState.densities = computeDensity(particleState, kernel, neighbors.get('noghost'), SupportScheme.Gather, config)

    ################################################################################
    #                                Boundary Setup                                #
    ################################################################################

    uGrid, vGrid, cGrid, dampGrid, uSourceGrid, cSourceGrid = genInitial(
        particleState, config,
        nx,
        domainBox = args.domainBox,
        domainDamping = args.domainDamping,
    )
    ################################################################################
    #                                Source Setup                                  #
    ################################################################################

    if verbose:
        print("Setting up sources...")


    uGrid, vGrid, cGrid, dampGrid, uSourceGrid, cSourceGrid, sourceCounter = genCase_07(
        particleState, config,
        nx,
        radii = args.sourceRadii,
        rotations = 0,
        shapes = args.sourceShapes,
        domainBox = args.domainBox,
        domainDamping = args.domainDamping,
        randomRadius = args.sourceRandomRadius,
        randomRotation = args.sourceRandomRotation,
        radiusRange = args.radiusRange,
    )

    if verbose:
        print("Setting up obstacles...")

    cSourceGrid = genBoundaryCase_01(
        particleState, config, nx,
        cSourceGrid,
        radii = args.boundaryRadii,
        rotations = args.boundaryRotations,
        offsets = args.boundaryOffsets,
        shapes = args.boundaryShapes,

        randomRadius = args.boundaryRandomRadius,
        randomRotation = args.boundaryRandomRotation,
        randomOffset = args.boundaryRandomOffset,

        radiusRange = args.radiusRange,
        rotationRange = (0, 2*np.pi),
        offsetRange = args.offsetRange
    )

    # sourceCounter = 0
    # uSourceGrid, sourceCounter = setupQuadrantSources(particleState, config, nx, uSourceGrid = uSourceGrid, sourceCounter = sourceCounter, radius = 0.15, topLeft = True, topRight = True, bottomLeft = True, bottomRight = True, sourceShape = 'box')
    # uSourceGrid, sourceCounter = setupCrossSources(particleState, config, nx, uSourceGrid = uSourceGrid, sourceCounter = sourceCounter, radius = 0.15, sourceTop = False, sourceBottom = False, sourceLeft = True, sourceRight = True, sourceShape = 'circle')
    # uSourceGrid, sourceCounter = setupCenterSource(particleState, config, nx, uSourceGrid = uSourceGrid, sourceCounter = sourceCounter, radius = 0.15, sourceShape = 'circle')

    # for i in range(4):
    #     uSourceGrid, sourceCounter = addRandomCircle(particleState, config, nx, uSourceGrid = uSourceGrid, sourceCounter = sourceCounter, radiusRange = (0.05, 0.15), magnitude = 1.0)

    ################################################################################
    #                              Obstacle Setup                                  #
    ################################################################################


    # cSourceGrid = setupSingleSlit(particleState, config , nx, cSourceGrid, slotWidth = 0.05, slotHeight = 0.2)
    # cSourceGrid = setupDoubleSlit(particleState, config, nx, cSourceGrid, slotWidths = [0.05, 0.05], slotHeights = [0.2, -0.2])
    # cSourceGrid = setupPrism(particleState, config, nx, cSourceGrid, prismSideLength = 0.5, prismOffset = (0.0, 0.0), prismPreRotation = np.pi/2, prismPostRotation = np.pi/6)
    # cSourceGrid = setupRectangle(particleState, config, nx, cSourceGrid, halfExtents = (0.2, 0.1), offset = (0.0, 0.0), preRotation = np.pi/6, postRotation = np.pi/6)
    # cSourceGrid = setupSphere(particleState, config, nx, cSourceGrid, radius = 0.2, offset = (0.0, 0.0))

    ################################################################################
    #                           Initial Condition Setup                            #
    ################################################################################

    if verbose:
        print("Setting up initial conditions...")

    uGrid = torch.zeros_like(uGrid)
    vGrid = torch.zeros_like(vGrid)

    uGrid = populateUGrid(uGrid, uSourceGrid,
        sourceMagnitudes = args.uMagnitudes,
        randomMagnitude = args.uRandomMagnitude,
        magnitudeRange = (args.uRandomMin, args.uRandomMax)
    )

    if args.smoothICs:
        if verbose:
            print("Smoothing initial conditions...")
        uGrid = smoothValues(
            uGrid,
            particleState,
            args.smoothIters, neighbors,
            config
        )

    cGrid = populateCGrid(cGrid, cSourceGrid,
        boundaryC = args.boundarySpeed,
        obstacleC = args.obstacleSpeeds[0],
        defaultC = args.defaultSpeed,
        randomObstacleC = args.randomObstacleSpeed,
        obstacleCRange = (args.obstacleSpeedMin, args.obstacleSpeedMax)
    )

    uGrid = addNoise(
        particleState, config, neighbors,
        uGrid, cSourceGrid,
        noiseAmplitude = args.noiseAmplitude, uMagnitude = max(args.uMagnitudes),
        noiseType = args.noiseType,
        smoothIter = args.noiseSmoothIter,
        seed = args.noiseSeed,
    )
    if args.uMaskLeft:
        uGrid[particleState.positions[:,0] < 0] = 0.0
    if args.uMaskRight:
        uGrid[particleState.positions[:,0] > 0] = 0.0
    if args.uMaskTop:
        uGrid[particleState.positions[:,1] > 0] = 0.0
    if args.uMaskBottom:
        uGrid[particleState.positions[:,1] < 0] = 0.0

    waveState = WaveEquationState(
        u = uGrid,
        v = vGrid,
        c = cGrid,
        damping = dampGrid,
    )
    smoothedState = smoothState(waveState, particleState, 0, neighbors, config)

    waveSystem = WaveSystem(
        systemState = particleState,
        waveState = smoothedState,
        neighborhood = neighbors.get('noghost'),
        t = 0.0
    )

    ################################################################################
    #                              Visualization Setup                             #
    ################################################################################

    fig, axis = visualize(particles, numNeighbors, counter)
    fig.savefig(f'output/{folderName}/initial_particles.png', dpi = args.figureDpi)

    fig, axis = plotInitialState(
        particleState, config,
        uGrid, vGrid, cGrid, dampGrid,
        uSourceGrid, cSourceGrid
    )
    # if args.displayImages:
    #     plt.show()
    fig.savefig(f'output/{folderName}/initial_fields.png', dpi = args.figureDpi)

    fig, axis, uPlot, vPlot, cPlot, dampPlot = plotState(
        particleState,
        smoothedState,
        config, kernel,
        markerSize = 0.5,
        plotGrid = True,
        plotCD = False)
    # if args.displayImages:
    #     plt.show()
    fig.savefig(f'output/{folderName}/initial_state.png', dpi = args.figureDpi)

    ################################################################################
    #                                Run Simulation                                #
    ################################################################################

    if verbose:
        print("Starting simulation...")

    runSimulation(fig, particleState, uPlot, vPlot, waveSystem, waveSystemFunction, integrator, nx, dt, nIter, kernel, config, export, plotInterval = args.plotInterval, exportImages = args.exportImages, umin = -7.5, umax = 7.5, vmin = -50, vmax = 50, prefix = prefix, timestamp = timestamp)

    if verbose:
        print("Simulation complete.")