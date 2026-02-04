import ipywidgets as widgets
# from IPython.display import clear_output
import warnings
from tqdm import TqdmExperimentalWarning
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

def sampleC(particleState, config):
    positions = getPeriodicPositions(particleState.positions, config['domain'])
    cGrid = torch.ones(positions.shape[0], device = 'cuda') * 0.5

    # xx, yy , noise = generateOctaveNoise(n = nx * 4, dim = 2, octaves = 2, baseFrequency = 1, kind = 'perlin', tileable=True, seed = 12365)

    # cTarget = noise / 2 + 0.5
    # # cGrid = noise / 2 + 0.5

    # # # cTarget = torch.where(yy < 0, 1, 0.5)
    # # # cTarget[:] = 0.5
    # # cTarget = torch.where(torch.sqrt(xx**2 + yy**2) < 0.25, 0, 1)
    # cInterp = RegularGridInterpolator((np.linspace(-1,1,cTarget.shape[0]), np.linspace(-1,1,cTarget.shape[1])), cTarget.numpy())
    # cGrid = torch.tensor(cInterp(particleState.positions.cpu())).to('cuda')

    # cGrid = sampleVoronoi(positions, nx, octaves = 3, baseFrequency = 2, kind = 'perlin', tileable=True, seed = 12365, vmin = 0.5, vmax = 1.0, config=config)

    # cGrid[:] *= 2 - 1 * torch.abs(grid[:,1])

    # cGrid = sampleVoronoiC(positions)
    cGrid[:] = torch.where(torch.logical_or(
        torch.logical_and(positions[:,0] > 0.5, positions[:,1] < 0.5), 
        torch.logical_and(positions[:,0] < -0.5, positions[:,1] < 0.5)), 0., 1)
    # cGrid[:] = torch.where(torch.logical_and(positions[:,0].abs() > 0.125, positions[:,1].abs() < 0.025), 0.0, 1)
    cGrid[:] = 1
    return cGrid

class InitialConditionType(Enum):
    OneCircle = auto() # One circle source in center
    TwoCircles = auto() # Two circle sources at top and bottom
    ThreeCircles = auto() # Three circle sources in a regular triangle
    RandomCircles = auto() # Multiple random circle sources
    LineSourceTop = auto() # Line source at top
    LineSourceBottom = auto() # Line source at bottom
    LineSources = auto() # Line source at top and bottom
    RandomField = auto()
    SourceLeft = auto() # Source on the left side
    SourceRight = auto() # Source on the right side
    SourceLeftRight = auto() # Sources on left and right sides
    SourceLeftTopBottom = auto() # Sources on left side, top and bottom
    SourceRightTopBottom = auto() # Sources on right side, top and bottom
    SourceQuadrants = auto() # Sources in all four quadrants
    
def sampleInitialConditions(particleState, neighbors, config, 
    conditionType: InitialConditionType, 
    sourceRadii: Union[float, List[float]] = 0.125,
    sourceMagnitudes: Union[float, List[float]] = 10.0):
    positions = getPeriodicPositions(particleState.positions, config['domain'])
    sampled = torch.zeros(positions.shape[0], device = positions.device)
    
    radii = [sourceRadii] if isinstance(sourceRadii, float) else sourceRadii
    magnitudes = [sourceMagnitudes] if isinstance(sourceMagnitudes, float) else sourceMagnitudes
    
    
    if conditionType == InitialConditionType.OneCircle:
        source = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, sourceRadii), torch.tensor([0.0,-0.0], device = positions.device))(positions) < 0, 1, 0).float()
        source = source * magnitudes[0 % len(magnitudes)]
    elif conditionType == InitialConditionType.TwoCircles:
        source1 = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[0]), torch.tensor([0.0,0.5], device = positions.device))(positions) < 0, 1, 0).float()
        source2 = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[1]), torch.tensor([0.0,-0.5], device = positions.device))(positions) < 0, 1, 0).float()
        source = source1 * magnitudes[0 % len(magnitudes)] + source2 * magnitudes[1 % len(magnitudes)]
    elif conditionType == InitialConditionType.ThreeCircles:
        angles = [0, 2*3.14159/3, 4*3.14159/3]
        source = torch.zeros(positions.shape[0], device = positions.device)
        if len(radii) == 1:
            radii = radii * 3
        if len(magnitudes) == 1:
            magnitudes = magnitudes * 3
        for i in range(3):
            location = torch.tensor([0.5 * torch.cos(torch.tensor(angles[i])), 0.5 * torch.sin(torch.tensor(angles[i]))], device = positions.device)
            curSource = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[i]), location)(positions) < 0, 1, 0).float()
            source += curSource * magnitudes[i]
    elif conditionType == InitialConditionType.RandomCircles:
        numSources = 5
        source = torch.zeros(positions.shape[0], device = positions.device)
        for i in range(numSources):
            location = (torch.rand(2, device = positions.device) * 2 - 1) * 0.75
            curSource = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[i % len(radii)]), location)(positions) < 0, 1, 0).float()
            source += curSource * magnitudes[i % len(magnitudes)]
    elif conditionType == InitialConditionType.LineSourceTop:
        source = torch.where(operatorDict['translate'](lambda points: getSDF('box')['function'](points, torch.tensor([1.0,0.015], device = positions.device)), torch.tensor([0.0,0.85], device = positions.device))(positions) < 0, 1, 0).float()
        source = source * magnitudes[0 % len(magnitudes)]
    elif conditionType == InitialConditionType.LineSourceBottom:
        source = torch.where(operatorDict['translate'](lambda points: getSDF('box')['function'](points, torch.tensor([1.0,0.015], device = positions.device)), torch.tensor([0.0,-0.85], device = positions.device))(positions) < 0, 1, 0).float()
        source = source * magnitudes[0 % len(magnitudes)]
    elif conditionType == InitialConditionType.LineSources:
        source1 = torch.where(operatorDict['translate'](lambda points: getSDF('box')['function'](points, torch.tensor([1.0,0.015], device = positions.device)), torch.tensor([0.0,0.85], device = positions.device))(positions) < 0, 1, 0).float()
        source2 = torch.where(operatorDict['translate'](lambda points: getSDF('box')['function'](points, torch.tensor([1.0,0.015], device = positions.device)), torch.tensor([0.0,-0.85], device = positions.device))(positions) < 0, 1, 0).float()
        source = source1 * magnitudes[0 % len(magnitudes)] + source2 * magnitudes[1 % len(magnitudes)]
    elif conditionType == InitialConditionType.RandomField:
        source = sampleVoronoi(positions, nx = int(math.sqrt(positions.shape[0])), octaves = 3, baseFrequency = 2, kind = 'perlin', tileable=True, seed = 12365, vmin = 0.0, vmax = magnitudes[0], config=config)
    elif conditionType == InitialConditionType.SourceLeft:
        source = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[0]), torch.tensor([-0.75,0.0], device = positions.device))(positions) < 0, 1, 0).float()
        source = source * magnitudes[0 % len(magnitudes)]
    elif conditionType == InitialConditionType.SourceRight:
        source = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[0]), torch.tensor([0.75,0.0], device = positions.device))(positions) < 0, 1, 0).float()
        source = source * magnitudes[0 % len(magnitudes)]
    elif conditionType == InitialConditionType.SourceLeftRight:
        sourceLeft = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[0]), torch.tensor([-0.75,0.0], device = positions.device))(positions) < 0, 1, 0).float()
        sourceRight = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[1]), torch.tensor([0.75,0.0], device = positions.device))(positions) < 0, 1, 0).float()
        source = sourceLeft * magnitudes[0 % len(magnitudes)] + sourceRight * magnitudes[1 % len(magnitudes)]
    elif conditionType == InitialConditionType.SourceLeftTopBottom:
        sourceTL = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[0]), torch.tensor([-0.75,0.5], device = positions.device))(positions) < 0, 1, 0).float()
        sourceBL = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[1]), torch.tensor([-0.75,-0.5], device = positions.device))(positions) < 0, 1, 0).float()
        source = sourceTL * magnitudes[0 % len(magnitudes)] + sourceBL * magnitudes[1 % len(magnitudes)]
    elif conditionType == InitialConditionType.SourceRightTopBottom:
        sourceTR = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[0]), torch.tensor([0.75,0.5], device = positions.device))(positions) < 0, 1, 0).float()
        sourceBR = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[1]), torch.tensor([0.75,-0.5], device = positions.device))(positions) < 0, 1, 0).float()
        source = sourceTR * magnitudes[0 % len(magnitudes)] + sourceBR * magnitudes[1 % len(magnitudes)]
    elif conditionType == InitialConditionType.SourceQuadrants:
        sourceTL = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[0]), torch.tensor([-0.5,0.5], device = positions.device))(positions) < 0, 1, 0).float()
        sourceTR = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[1]), torch.tensor([0.5,0.5], device = positions.device))(positions) < 0, 1, 0).float()
        sourceBL = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[2]), torch.tensor([-0.5,-0.5], device = positions.device))(positions) < 0, 1, 0).float()
        sourceBR = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, radii[3]), torch.tensor([0.5,-0.5], device = positions.device))(positions) < 0, 1, 0).float()
        source = (sourceTL * magnitudes[0 % len(magnitudes)] + 
                  sourceTR * magnitudes[1 % len(magnitudes)] +
                  sourceBL * magnitudes[2 % len(magnitudes)] +
                  sourceBR * magnitudes[3 % len(magnitudes)])   
    else:
        raise ValueError(f'Unknown InitialConditionType: {conditionType}')
    sampled = source
    return sampled

def sampleU(particleState, neighbors, config, uMag = 10):
    positions = getPeriodicPositions(particleState.positions, config['domain'])
    sampled = torch.zeros(positions.shape[0], device = positions.device)
    

    # numWaves = torch.randint(2, 8, (1,)).item()
    # for _ in range(numWaves):
    #     location = torch.rand(2, device = positions.device) * 2 - 1
    #     amplitude = torch.rand(1, device = positions.device) * 2 - 1
    #     radius = torch.rand(1, device = positions.device) * 0.15 + 0.05

    #     sphereEmitter = lambda points: getSDF('circle')['function'](points, radius)
    #     translated = operatorDict['translate'](sphereEmitter, torch.tensor([location[0],location[1]], device = positions.device))
    #     curSample = amplitude * torch.where(translated(positions) < 0, 1, 0).float()[:,0]
    #     # print(curSample.shape)
    #     sampled += curSample

    # positions = getModPosition(positions_, config)
    # sphereEmitter = lambda points: getSDF('circle')['function'](points, 0.025)
    # translated = operatorDict['translate'](sphereEmitter, torch.tensor([0.0,0.5], device = positions.device))
    # sampled = torch.where(translated(positions) < 0, 1, 0).float()

    # translated = operatorDict['translate'](sphereEmitter, torch.tensor([0.0,-0.5], device = positions.device))
    # sampled -= torch.where(translated(positions) < 0, 1, 0).float()
    # sampled = smoothGrid(gridState, sampled) * 10

    # lineEmitter = operatorDict['translate'](lambda points: getSDF('box')['function'](points, torch.tensor([1.0,0.015], device = positions.device)), torch.tensor([0.0,0.85], device = positions.device))
    # sampled = torch.where(lineEmitter(positions) < 0, 1, 0).float()
    # lineEmitter2 = operatorDict['translate'](lambda points: getSDF('box')['function'](points, torch.tensor([1.0,0.015], device = positions.device)), torch.tensor([0.0,-0.85], device = positions.device))
    # sampled -= torch.where(lineEmitter2(positions) < 0, 1, 0).float()

    sampled = torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, 0.125), torch.tensor([0.0,-0.5], device = positions.device))(positions) < 0, 1, 0).float()

    # sampled -= torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, 0.125), torch.tensor([0.0,0.5], device = positions.device))(positions) < 0, 1, 0).float()

    
    # sampled = torch.where(translated(positions) < 0, 1, 0).float()

    # sampled = SPHOperation(
    #     particleState,
    #     sampled,
    #     config['kernel'],
    #     neighbors.get('noghost')[0],
    #     neighbors.get('noghost')[1],
    #     Operation.Interpolate,
    #     supportScheme = SupportScheme.Gather)

    # for _ in range(8):
        # sampled = smoothGrid(gridState, sampled)


    sampled = sampled * uMag
    return sampled



# def sampleU(particleState, neighbors, config, uMag = 10):
#     positions = getPeriodicPositions(particleState.positions, config['domain'])
#     sampled = torch.zeros(positions.shape[0], device = positions.device)

#     sampled =torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, 0.125), torch.tensor([0.0,-0.5], device = positions.device))(positions) < 0, 1, 0).float()
#     sampled = sampled - torch.where(operatorDict['translate'](lambda points: getSDF('circle')['function'](points, 0.125), torch.tensor([0.0,0.5], device = positions.device))(positions) < 0, 1, 0).float()

#     sampled = sampled * uMag
#     return sampled


def translate(points, offset):
    return points + offset
def rotate(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    rotationMatrix = torch.tensor([[c, -s], [s, c]], device = points.device, dtype = points.dtype)
    return points @ rotationMatrix.T

def sampleSphere(particleState, config, 
    radius: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    preRotation: float = 0.0,
    postRotation: float = 0.0):
    positions = getPeriodicPositions(particleState.positions, config['domain'])
    
    preRotated = rotate(positions, postRotation)
    translated = translate(preRotated, torch.tensor([-offset[0], -offset[1]], device = positions.device))
    postRotated = rotate(translated, preRotation)
    
    baseSDF = lambda points: getSDF('circle')['function'](points, radius)
    
    sampled = torch.where(baseSDF(postRotated) < 0, 1, 0).float()
    return sampled

def sampleBox(particleState, config, 
    halfExtents: Tuple[float, float],
    offset: Tuple[float, float] = (0.0, 0.0),
    preRotation: float = 0.0,
    postRotation: float = 0.0):
    positions = getPeriodicPositions(particleState.positions, config['domain'])
    
    preRotated = rotate(positions, postRotation)
    translated = translate(preRotated, torch.tensor([-offset[0], -offset[1]], device = positions.device))
    postRotated = rotate(translated, preRotation)
    
    baseSDF = lambda points: getSDF('box')['function'](points, torch.tensor(halfExtents, device = points.device))
    sampled = torch.where(baseSDF(postRotated) < 0, 1, 0).float()
    return sampled

def sampleHorizontalLine(particleState, config, 
    thickness: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    preRotation: float = 0.0,
    postRotation: float = 0.0):
    positions = getPeriodicPositions(particleState.positions, config['domain'])

    preRotated = rotate(positions, postRotation)
    translated = translate(preRotated, torch.tensor([-offset[0], -offset[1]], device = positions.device))
    postRotated = rotate(translated, preRotation)
    
    length = config['domain'].max[0] - config['domain'].min[0]
    
    baseSDF = lambda points: getSDF('box')['function'](points, torch.tensor([length/2, thickness/2], device = points.device))
    
    sampled = torch.where(baseSDF(postRotated) < 0, 1, 0).float()
    return sampled

def sampleVerticalLine(particleState, config, 
    thickness: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    preRotation: float = 0.0,
    postRotation: float = 0.0):
    positions = getPeriodicPositions(particleState.positions, config['domain'])

    preRotated = rotate(positions, postRotation)
    translated = translate(preRotated, torch.tensor([-offset[0], -offset[1]], device = positions.device))
    postRotated = rotate(translated, preRotation)
    
    length = config['domain'].max[1] - config['domain'].min[1]
    
    baseSDF = lambda points: getSDF('box')['function'](points, torch.tensor([thickness/2, length/2], device = points.device))
    
    sampled = torch.where(baseSDF(postRotated) < 0, 1, 0).float()
    return sampled

def sampleVesica(particleState, config, 
    radius: float, width: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    preRotation: float = 0.0,
    postRotation: float = 0.0):
    positions = getPeriodicPositions(particleState.positions, config['domain'])
    
    preRotated = rotate(positions, postRotation)
    translated = translate(preRotated, torch.tensor([-offset[0], -offset[1]], device = positions.device))
    postRotated = rotate(translated, preRotation)
    
    baseSDF = lambda points: getSDF('vesica')['function'](points, radius, width)

    sampled = torch.where(baseSDF(postRotated) < 0, 1, 0).float()
    return sampled


def sampleTriangle(particleState, config, 
    v0: Tuple[float, float], v1: Tuple[float, float], v2: Tuple[float, float],
    offset: Tuple[float, float] = (0.0, 0.0),
    preRotation: float = 0.0,
    postRotation: float = 0.0):
    positions = getPeriodicPositions(particleState.positions, config['domain'])
    
    preRotated = rotate(positions, postRotation)
    translated = translate(preRotated, torch.tensor([-offset[0], -offset[1]], device = positions.device))
    postRotated = rotate(translated, preRotation)
    
    baseSDF = lambda points: getSDF('triangle')['function'](points, torch.tensor(v0, device = points.device), torch.tensor(v1, device = points.device), torch.tensor(v2, device = points.device))
    sampled = torch.where(baseSDF(postRotated) < 0, 1, 0).float()
    return sampled

def sampleEquilateralTriangle(particleState, config, 
    sideLength: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    preRotation: float = 0.0,
    postRotation: float = 0.0):
    height = (3**0.5) / 2 * sideLength
    v0 = (-sideLength/2, -height/3)
    v1 = (sideLength/2, -height/3)
    v2 = (0.0, 2*height/3)
    return sampleTriangle(particleState, config, v0, v1, v2, offset, preRotation, postRotation)


def generateSingleSlits(particleState, config,
                       slotWidths: List[float] = [0.05], slotHeights: List[float] = [0.2]):
    wall = sampleVerticalLine(
        particleState, config,
        thickness = 0.04,
        offset = (0.0, 0.0),
        preRotation = 0.0,
        postRotation = 0.0
    )
    obstacle = wall.clone()
    for slotWidth, slotHeight in zip(slotWidths, slotHeights):
        slit = sampleBox(
            particleState, config,
            halfExtents = (0.05, slotWidth / 2),
            offset = (0.0, slotHeight),
            preRotation = 0.0,
            postRotation = 0.0
        )
        obstacle = obstacle - slit
    obstacle = torch.clamp(obstacle, min=0.0, max=1.0)
    return obstacle

def generateDomainBox(particleState, config,
                       boxHalfExtents: Tuple[float, float] = (0.5, 0.5)):
    box = sampleBox(
        particleState, config,
        halfExtents = boxHalfExtents,
        offset = (0.0, 0.0),
        preRotation = 0.0,
        postRotation = 0.0
    )
    return box


from damping import *
def genInitial(
    particleState, config,
    nx,
    domainBox: bool = True,
    domainDamping: bool = True,
):
    device = particleState.positions.device
    u, v = torch.zeros(nx**2, device = device), torch.zeros(nx**2, device = device)
    cGrid = torch.ones(nx**2, device = device)
    dampGrid = torch.zeros(nx**2, device = device)

    uSourceGrid = torch.zeros(nx**2, device = device, dtype = torch.long)
    cSourceGrid = torch.zeros(nx**2, device = device, dtype = torch.long)

    if domainBox:
        box = generateDomainBox(particleState, config,
                boxHalfExtents = (0.95, 0.95))
        cGrid = torch.where(box > 0, cGrid, 0.01)
        cSourceGrid = torch.where(box > 0, cSourceGrid, -1)
    if domainDamping:
        dampGrid = createDampingProfile(particleState, config, DampingProfiles.borderDamping_strong)


    return u, v, cGrid, dampGrid, uSourceGrid, cSourceGrid
    

def setupSingleSlit(
        particleState, config,
        nx,
        cSourceGrid,
        slotWidth: float = 0.05, slotHeight: float = 0.2
):
    obstacle = generateSingleSlits(
        particleState, config,
        slotWidths = [slotWidth], slotHeights = [slotHeight]
    )
    
    cSourceGrid = torch.where(obstacle <= 0, cSourceGrid, -1)
    return cSourceGrid

def setupDoubleSlit(
        particleState, config,
        nx,
        cSourceGrid,
        slotWidths: List[float] = [0.05, 0.05], slotHeights: List[float] = [0.2, -0.2]
):
    obstacle = generateSingleSlits(
        particleState, config,
        slotWidths = slotWidths, slotHeights = slotHeights
    )
    
    cSourceGrid = torch.where(obstacle <= 0, cSourceGrid, -1)
    return cSourceGrid

def setupPrism(
        particleState, config,
        nx,
        cSourceGrid,
        prismSideLength: float = 0.3,
        prismOffset: Tuple[float, float] = (0.0, 0.0),
        prismPreRotation: float = 0.0,
        prismPostRotation: float = 0.0,
        addWall: bool = True
):
    prism = sampleEquilateralTriangle(
        particleState, config,
        sideLength = prismSideLength,
        offset = prismOffset,
        preRotation = prismPreRotation,
        postRotation = prismPostRotation
    )
    if addWall:
        wall = sampleVerticalLine(
            particleState, config,
            thickness = 0.04,
            offset = prismOffset,
        )
        cSourceGrid = torch.where(wall <= 0, cSourceGrid, -1)

    cSourceGrid = torch.where(prism <= 0, cSourceGrid, 1)
    return cSourceGrid

def setupRectangle(
        particleState, config,
        nx,
        cSourceGrid,
        halfExtents: Tuple[float, float] = (0.2, 0.1),
        offset: Tuple[float, float] = (0.0, 0.0),
        preRotation: float = 0.0,
        postRotation: float = 0.0,
        addWall: bool = True
):
    rectangle = sampleBox(
        particleState, config,
        halfExtents = halfExtents,
        offset = offset,
        preRotation = preRotation,
        postRotation = postRotation
    )
    if addWall:
        wall = sampleVerticalLine(
            particleState, config,
            thickness = 0.04,
            offset = offset,
        )
        cSourceGrid = torch.where(wall <= 0, cSourceGrid, -1)

    cSourceGrid = torch.where(rectangle <= 0, cSourceGrid, 1)
    return cSourceGrid

def setupSphere(
        particleState, config,
        nx,
        cSourceGrid,
        radius: float = 0.2,
        offset: Tuple[float, float] = (0.0, 0.0),
        preRotation: float = 0.0,
        postRotation: float = 0.0,
        addWall: bool = True
):
    sphere = sampleSphere(
        particleState, config,
        radius = radius,
        offset = offset,
        preRotation = preRotation,
        postRotation = postRotation
    )
    if addWall:
        wall = sampleVerticalLine(
            particleState, config,
            thickness = 0.04,
            offset = offset,
        )
        cSourceGrid = torch.where(wall <= 0, cSourceGrid, -1)

    cSourceGrid = torch.where(sphere <= 0, cSourceGrid, 1)
    return cSourceGrid


def setupQuadrantSources(
        particleState, config,
        nx,
        sourceCounter: int,
        uSourceGrid,
        radius = 0.1,
        topLeft: bool = True,
        topRight: bool = True,
        bottomLeft: bool = True,
        bottomRight: bool = True,
        sourceShape: str = 'circle'
):
    if topLeft:
        sphereTL = sampleSphere(
            particleState, config,
            radius = radius,
            offset = (-0.5, 0.5),
        ) if sourceShape == 'circle' else sampleBox(
            particleState, config,
            halfExtents = (radius, radius),
            offset = (-0.5, 0.5),
        )
        uSourceGrid = torch.where(sphereTL <= 0, uSourceGrid, sourceCounter+1)
        sourceCounter += 1
    if topRight:    
        sphereTR = sampleSphere(
            particleState, config,
            radius = radius,
            offset = (0.5, 0.5),
        ) if sourceShape == 'circle' else sampleBox(
            particleState, config,
            halfExtents = (radius, radius),
            offset = (0.5, 0.5),
        )
        uSourceGrid = torch.where(sphereTR <= 0, uSourceGrid, sourceCounter+1)
        sourceCounter += 1
    if bottomLeft:    
        sphereBL = sampleSphere(
            particleState, config,
            radius = radius,
            offset = (-0.5, -0.5),
        ) if sourceShape == 'circle' else sampleBox(
            particleState, config,
            halfExtents = (radius, radius),
            offset = (-0.5, -0.5),
        )
        uSourceGrid = torch.where(sphereBL <= 0, uSourceGrid, sourceCounter+1)
        sourceCounter += 1
    if bottomRight:    
        sphereBR = sampleSphere(
            particleState, config,
            radius = radius,
            offset = (0.5, -0.5),
        ) if sourceShape == 'circle' else sampleBox(
            particleState, config,
            halfExtents = (radius, radius),
            offset = (0.5, -0.5),
        )
        uSourceGrid = torch.where(sphereBR <= 0, uSourceGrid, sourceCounter+1)
        sourceCounter += 1
    return uSourceGrid, sourceCounter

def setupCrossSources(
        particleState, config,
        nx,
        sourceCounter: int,
        uSourceGrid,
        radius: float = 0.1,
        sourceTop: bool = True,
        sourceBottom: bool = True,
        sourceLeft: bool = True,
        sourceRight: bool = True,
        sourceShape: str = 'circle'
):
    if sourceTop:
        sphereTop = sampleSphere(
            particleState, config,
            radius = radius,
            offset = (0.0, 0.5),
        ) if sourceShape == 'circle' else sampleBox(
            particleState, config,
            halfExtents = (radius, radius),
            offset = (0.0, 0.5),
        )
        uSourceGrid = torch.where(sphereTop <= 0, uSourceGrid, sourceCounter+1)
        sourceCounter += 1
    if sourceBottom:    
        sphereBottom = sampleSphere(
            particleState, config,
            radius = radius,
            offset = (0.0, -0.5),
        ) if sourceShape == 'circle' else sampleBox(
            particleState, config,
            halfExtents = (radius, radius),
            offset = (0.0, -0.5),
        )
        uSourceGrid = torch.where(sphereBottom <= 0, uSourceGrid, sourceCounter+1)
        sourceCounter += 1
    if sourceLeft:    
        sphereLeft = sampleSphere(
            particleState, config,
            radius = radius,
            offset = (-0.5, 0.0),
        ) if sourceShape == 'circle' else sampleBox(
            particleState, config,
            halfExtents = (radius, radius),
            offset = (-0.5, 0.0),
        )
        uSourceGrid = torch.where(sphereLeft <= 0, uSourceGrid, sourceCounter+1)
        sourceCounter += 1
    if sourceRight:    
        sphereRight = sampleSphere(
            particleState, config,
            radius = radius,
            offset = (0.5, 0.0),
        ) if sourceShape == 'circle' else sampleBox(
            particleState, config,
            halfExtents = (radius, radius),
            offset = (0.5, 0.0),
        )
        uSourceGrid = torch.where(sphereRight <= 0, uSourceGrid, sourceCounter+1)
        sourceCounter += 1
    return uSourceGrid, sourceCounter
    
def setupCenterSource(
        particleState, config,
        nx,
        sourceCounter: int,
        uSourceGrid,
        radius: float = 0.1,
        sourceShape: str = 'circle'
):
    centerSource = sampleSphere(
        particleState, config,
        radius = radius,
        offset = (0.0, 0.0),
    ) if sourceShape == 'circle' else sampleBox(
        particleState, config,
        halfExtents = (radius, radius),
        offset = (0.0, 0.0),
    )
    uSourceGrid = torch.where(centerSource <= 0, uSourceGrid, sourceCounter+1)
    sourceCounter += 1
    return uSourceGrid, sourceCounter

def addRandomCircle(
        particleState, config,
        nx,
        sourceCounter: int,
        uSourceGrid,
        radiusRange: Tuple[float, float] = (0.05, 0.15),
        magnitude: float = 1.0,
):
    radius = torch.rand(1).item() * (radiusRange[1] - radiusRange[0]) + radiusRange[0]
    location = (torch.rand(2, device = particleState.positions.device) * 2 - 1) * (1 - radius)
    sphere = sampleSphere(
        particleState, config,
        radius = radius,
        offset = (location[0].item(), location[1].item()),
    )
    uSourceGrid = torch.where(sphere <= 0, uSourceGrid, sourceCounter+1)
    sourceCounter += 1
    return uSourceGrid, sourceCounter