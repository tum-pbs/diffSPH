import ipywidgets as widgets
# from IPython.display import clear_output
import warnings
from tqdm import TqdmExperimentalWarning

from util import SamplingScheme
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
from tqdm.autonotebook import tqdm
from scipy.ndimage import gaussian_filter1d
# import h5py
# import copy
import os
import torch
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'

# import shlex
import torch

from diffSPH.sampling import sampleRegularParticles, sampleOptimal
from diffSPH.operations import sph_operation, mod
from diffSPH.sampling import buildDomainDescription, sampleRegularParticles
from diffSPH.modules.eos import idealGasEOS
from diffSPH.schema import getSimulationScheme
from diffSPH.reference.sod import buildSod_reference, sodInitialState, generateSod1D
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH
from diffSPH.reference.sod import plotSod
from diffSPH.operations import GradientMode, LaplacianMode, SupportScheme
from diffSPH.enums import *
from diffSPH.schemes.states.common import BasicState
from diffSPH.neighborhood import SupportScheme, evaluateNeighborhood
from diffSPH.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr, filterNeighborhoodByKind
from diffSPH.modules.density import computeDensity

from diffSPH.sdf import operatorDict, getSDF
from diffSPH.sphOperations.shared import scatter_sum
from matplotlib.colors import LogNorm

from waveEqn import WaveEquationState, waveEquation2
from typing import Union, Tuple
from diffSPH.neighborhood import SparseNeighborhood, PrecomputedNeighborhood
from diffSPH.operations import GradientMode, LaplacianMode
from diffSPH.kernels import SPHKernel
from diffSPH.enums import *
from diffSPH.operations import SPHOperation, Operation
from diffSPH.schemes.states.common import BasicState
from diffSPH.neighborhood import SupportScheme, evaluateNeighborhood
from diffSPH.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr, filterNeighborhoodByKind
from diffSPH.operations import ParticleSetWithQuantity, sph_op
from diffSPH.kernels import getSPHKernelv2
from diffSPH.plotting import visualizeParticles, updatePlot, plotDistribution
from diffSPH.operations import SPHOperation, Operation
from diffSPH.util import getPeriodicPositions
from diffSPH.integrationSchemes.util import integrateQ
from diffSPH.integration import getIntegrator
import copy
from waveEqn import WaveSystem, waveSystemFunction

from enum import Enum, auto
from typing import List
from sample import sampleVoronoi

from diffSPH.noise import generateOctaveNoise
from scipy.interpolate import RegularGridInterpolator
import numpy as np

def genGaussianBell(positions, center, radius, magnitude):
    # Compute the distance from each particle to the center
    distances = torch.norm(positions - center, dim=1)
    # Compute the Gaussian bell values
    bell = magnitude * torch.exp(-0.5 * (distances / radius) ** 2)
    return bell

def genCircle(positions, center, radius, magnitude):
    distances = torch.norm(positions - center, dim=1)
    circle = torch.where(distances <= radius, magnitude, torch.zeros_like(distances))
    return circle
def copyWaveSystem(waveSystem):
    waveStateCopy = WaveEquationState(
        u = waveSystem.waveState.u.clone().detach(),
        v = waveSystem.waveState.v.clone().detach(),
        c = waveSystem.waveState.c.clone().detach(),
        damping = waveSystem.waveState.damping.clone().detach(),
    )
    return WaveSystem(
        systemState = waveSystem.systemState,
        waveState = waveStateCopy,
        neighborhood = waveSystem.neighborhood,
        t = waveSystem.t
    )

from gencase import *
from util import generateInitialVariables, sampleParticles
from sample import *

def genInitialConditions(args, case, domainCenter, topOffset, bottomOffset, leftOffset, rightOffset, gaussianSourceRadius, circularSourceInnerRadius, circularSourceOuterRadius, squareSourceHalfSize, lineSourceWidth, uMagnitude, device, dtype):

    plotInterval = args.plotInterval
    nIter = args.nIter
    dt = args.dt
    nx = args.nx
    # uMagnitude = args.uMagnitude
    sampling = args.sampling
    samplingScheme = None
    export = args.export
    for scheme in SamplingScheme:
        if scheme.name == sampling:
            samplingScheme = scheme
            if args.verbose:
                print(f'Using sampling scheme: {samplingScheme.name}')
            break
    if samplingScheme is None:
        raise ValueError(f'Unknown sampling scheme: {sampling}')

    config, domain, device, dtype, kernel = generateInitialVariables(
        nx, device = device
    )
    config['integrationScheme'] = IntegrationSchemeType.rungeKutta4
    integrator = getIntegrator(config['integrationScheme'])
    if args.verbose:
        print(f'Using integrator: {config["integrationScheme"].name}')

    ################################################################################
    #                             Particle Generation                              #
    ################################################################################

    particles, numNeighbors, counter = sampleParticles(args.nx, scheme=samplingScheme)
    particleState = BasicState(particles.positions, particles.supports, particles.masses, particles.densities, torch.zeros_like(particles.positions), torch.zeros(particles.positions.shape[0], device = device, dtype = torch.int64), torch.zeros(particles.positions.shape[0], device = device, dtype = torch.int64), torch.arange(particles.positions.shape[0], device = device), particles.positions.shape[0])
    neighborhood, neighbors = evaluateNeighborhood(particleState, config['domain'], kernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
    particleState.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particleState, neighbors.neighbors, which = 'noghost')).rowEntries
    particleState.densities = computeDensity(particleState, kernel, neighbors.get('noghost'), SupportScheme.Gather, config)

    ################################################################################
    #                                Boundary Setup                                #
    ################################################################################

    uGrid, vGrid, cGrid, dampGrid, uSourceGrid, cSourceGrid = genInitial(
        particleState, config,
        args.nx,
        domainBox = args.domainBox,
        domainDamping = args.domainDamping,
    )

    if case == 1:
        # Case 01: Single Gaussian bell in the center of the domain
        uGrid = genCircle(particleState.positions, domainCenter, gaussianSourceRadius, uMagnitude)
    elif case == 2:
        # Case 02: Two Gaussian bells, one in the left half of the domain and one in the right half
        uGrid = genCircle(particleState.positions, leftOffset, gaussianSourceRadius, uMagnitude) - genCircle(particleState.positions, rightOffset, gaussianSourceRadius, uMagnitude)
    elif case == 3:
        # Case 03: A Gaussian bell in the center with a random noise pattern superimposed on top
        uGrid = genCircle(particleState.positions, domainCenter, gaussianSourceRadius, uMagnitude)

        uGrid = addNoise(
            particleState, config, neighbors,
            uGrid, cSourceGrid,
            noiseAmplitude = 0.25, uMagnitude = uMagnitude,
            noiseType = 'uniform',
            smoothIter = 1,
            seed = args.noiseSeed,
            octaves = 4,
            baseFrequency = 4,
        )
    elif case == 4:
        # Case 04: A random noise pattern with no Gaussian bell
        uGrid = addNoise(
            particleState, config, neighbors,
            torch.zeros_like(particleState.positions[:,0]), cSourceGrid,
            noiseAmplitude = 1, uMagnitude = uMagnitude,
            noiseType = 'uniform',
            smoothIter = 0,
            seed = args.noiseSeed,
            octaves = 4,
            baseFrequency = 4,
        )
    elif case == 5:
        # Case 05: A vertical line source in the center of the domain
        uGrid = torch.where(torch.abs(particleState.positions[:,0] - domainCenter[0]) < lineSourceWidth, uMagnitude, torch.zeros_like(particleState.positions[:,0]))
    elif case == 6:
        # Case 06: A horizontal line source in the center of the domain
        uGrid = torch.where(torch.abs(particleState.positions[:,1] - domainCenter[1]) < lineSourceWidth, uMagnitude, torch.zeros_like(particleState.positions[:,0]))
    elif case == 7:
        # Case 07: A circular ring source in the center of the domain
        uGrid = genCircle(particleState.positions, domainCenter, circularSourceOuterRadius, uMagnitude) - genCircle(particleState.positions, domainCenter, circularSourceInnerRadius, uMagnitude)
    elif case == 8:
        # Case 08: A square source in the center of the domain
        uGrid = torch.where(
            (torch.abs(particleState.positions[:,0] - domainCenter[0]) < squareSourceHalfSize) &
            (torch.abs(particleState.positions[:,1] - domainCenter[1]) < squareSourceHalfSize),
            uMagnitude, torch.zeros_like(particleState.positions[:,0])
        )
    elif case == 9:
        # Case 09: Two vertical line sources, one in the left half of the domain and one in the right half
        uGrid = torch.where(
            (torch.abs(particleState.positions[:,0] - leftOffset[0]) < lineSourceWidth) |
            (torch.abs(particleState.positions[:,0] - rightOffset[0]) < lineSourceWidth),
            uMagnitude, torch.zeros_like(particleState.positions[:,0])
        )
        uGrid[particleState.positions[:,0] > domainCenter[0]] *= -1
    elif case == 10:
        # Case 10: Two horizontal line sources, one in the top half of the domain and one in the bottom half
        uGrid = torch.where(
            (torch.abs(particleState.positions[:,1] - topOffset[1]) < lineSourceWidth) |
            (torch.abs(particleState.positions[:,1] - bottomOffset[1]) < lineSourceWidth),
            uMagnitude, torch.zeros_like(particleState.positions[:,0])
        )
        uGrid[particleState.positions[:,1] < domainCenter[1]] *= -1
    elif case == 11:
        # Case 11: Two circular ring sources, one in the left half of the domain and one in the right half
        uGrid = genCircle(particleState.positions, leftOffset, circularSourceOuterRadius, uMagnitude) - genCircle(particleState.positions, leftOffset, circularSourceInnerRadius, uMagnitude) + genCircle(particleState.positions, rightOffset, circularSourceOuterRadius, uMagnitude) - genCircle(particleState.positions, rightOffset, circularSourceInnerRadius, uMagnitude) 
        uGrid[particleState.positions[:,0] >= domainCenter[0]] *= -1
    elif case == 12:
        # Case 12: Two square sources, one in the top half of the domain and one in the bottom half
        uGrid = torch.where(
            ((torch.abs(particleState.positions[:,0] - topOffset[0]) < squareSourceHalfSize) &
            (torch.abs(particleState.positions[:,1] - topOffset[1]) < squareSourceHalfSize)) |
            ((torch.abs(particleState.positions[:,0] - bottomOffset[0]) < squareSourceHalfSize) &
            (torch.abs(particleState.positions[:,1] - bottomOffset[1]) < squareSourceHalfSize)),
            uMagnitude, torch.zeros_like(particleState.positions[:,0])
        )
        uGrid[particleState.positions[:,1] < domainCenter[1]] *= -1
    else:
        raise ValueError(f'Unknown initial condition case: {case}')

    # normalize uGrid to be in the range [-1, 1]
    uGrid = uGrid / torch.max(torch.abs(uGrid)) * uMagnitude
    vGrid = torch.zeros_like(uGrid)

    uPreSmoothMax = torch.max(uGrid).cpu().detach().item()
    uPreSmoothMin = torch.min(uGrid).cpu().detach().item()

    uGrid = smoothValues(
        uGrid,
        particleState,
        2, neighbors,
        config
    )
    # rescale uGrid back to original range after smoothing
    # first scale to [0, 1]
    # print(uPreSmoothMax, uPreSmoothMin, '---', torch.max(uGrid).cpu().detach().item(), torch.min(uGrid).cpu().detach().item())
    # uGrid = uGrid / (torch.max(uGrid) - torch.min(uGrid)) + 0.5
    # print(torch.max(uGrid), torch.min(uGrid))
    # uGrid = uGrid * (uPreSmoothMax - uPreSmoothMin) + uPreSmoothMin
    # print(torch.max(uGrid), torch.min(uGrid))


    cSourceGrid = genBoundaryCase_01(
        particleState, config, nx,
        cSourceGrid,
        radii = [0.25, 0.25],
        rotations = [0.0, 0.0],
        offsets = [(0.0, 0.0), (0.0, 0.)],
        shapes = ['square', 'circle'],

        randomRadius = False,
        randomRotation = False,
        randomOffset = False,

        radiusRange = (0.03, 0.1),
        rotationRange = (0, 2*np.pi),
        offsetRange = ((-0.5, 0.5), (-0.5, 0.5))
    )

    # return uGrid, vGrid

    ################################################################################
    #                           Initial Condition Setup                            #
    ################################################################################

    if args.verbose:
        print("Setting up initial conditions...")

    # uGrid = torch.zeros_like(uGrid)
    # vGrid = torch.zeros_like(vGrid)

    # uGrid = populateUGrid(uGrid, uSourceGrid,
    #     sourceMagnitudes = args.uMagnitudes * sourceCounter if len(args.uMagnitudes) == 1 else args.uMagnitudes,
    #     randomMagnitude = args.uRandomMagnitude,
    #     magnitudeRange = (args.uRandomMin, args.uRandomMax)
    # )

    if args.smoothICs:
        if args.verbose:
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
    # uGrid[particleState.positions[:,0] > 0] = 0.0

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
    return particles, particleState, neighbors, counter, uGrid, vGrid, cGrid, dampGrid, uSourceGrid, cSourceGrid, waveSystem