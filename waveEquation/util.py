# %matplotlib widget
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

import h5py

from diffSPH.operations import ParticleSet
from enum import Enum

class SamplingScheme(Enum):
    regular = 1
    jittered = 2
    glass = 3
    optimal = 4
    random = 5

def sampleParticles(nx: int, scheme: SamplingScheme):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    L = 2
    dim = 2

    kernel = KernelType.Wendland4
    targetNeighbors = n_h_to_nH(4, dim)

    domain = buildDomainDescription(L, dim, True, device, dtype)

    particles = sampleOptimal(nx, domain, targetNeighbors, kernel, 0., 0, shiftScheme = 'delta')

    config = {
        'domain': domain,
        'kernel': kernel,
        'targetNeighbors': targetNeighbors,
        'neighborhood':{
            'verletScale': 1.0,
            'computeDkDh': True
        }
    }

    config['gradientMode'] = GradientMode.Difference
    config['laplacianMode'] = LaplacianMode.Brookshaw
    config['supportScheme'] = SupportScheme.Gather
    config['integrationScheme'] = IntegrationSchemeType.rungeKutta4
    
    if scheme == SamplingScheme.regular:
        particles = sampleRegularParticles(
            nx = nx,
            targetNeighbors=targetNeighbors,
            domain=domain,
        )
    elif scheme == SamplingScheme.jittered:
        particles = sampleRegularParticles(
            nx = nx,
            targetNeighbors=targetNeighbors,
            domain=domain,
            jitter=0
        )
        dx = 2 * L / nx
        jitterAmount = dx * 0.25
        particles = particles._replace(
            positions = particles.positions + (torch.rand_like(particles.positions) - 0.5) * jitterAmount
        )
        
    elif scheme == SamplingScheme.random:
        particles = sampleRegularParticles(
            nx = nx,
            targetNeighbors=targetNeighbors,
            domain=domain,
            jitter=0.5
        )
        particles = particles._replace(
            positions = torch.rand_like(particles.positions) * L * 2 - L
        )
        
    elif scheme == SamplingScheme.optimal:
        particles = sampleOptimal(
            nx = nx,
            domain=domain,
            targetNeighbors=targetNeighbors,
            kernel=kernel,
            jitter = 0.5,
            shiftScheme='delta',
            shiftIters=128
        )
    elif scheme == SamplingScheme.glass:
        files = ['position_samples_1024.h5', 'position_samples_4096.h5', 'position_samples_16384.h5', 'position_samples_65536.h5']
        numParticles = nx ** dim
        selectedFile = None
        for file in files:
            if int(file.split('_')[-1].split('.')[0]) >= numParticles:
                selectedFile = file
                break
        if selectedFile is None:
            raise ValueError('No suitable glass file found for the given number of particles.')
        data = None
        with h5py.File(selectedFile, 'r') as f:
            data = f['positions']
            numSamples = data.shape[0]
            randomIndex = torch.randint(0, numSamples, (1,)).item()
            positions = torch.tensor(data[randomIndex], device=device, dtype=dtype)
            densities = f['densities'][randomIndex]
            supports = f['supports'][randomIndex]
            numNeighbors = f['numNeighbors'][randomIndex]
            counter = f['counter'][randomIndex]
            
            particles = ParticleSet(
                positions=positions,
                supports=torch.tensor(supports, device=device, dtype=dtype),
                masses=torch.ones_like(positions[:,0], device=device, dtype=dtype) * (L**dim / positions.shape[0]),
                densities=torch.tensor(densities, device=device, dtype=dtype),
            )
        return particles, torch.tensor(numNeighbors, device=device, dtype=dtype), torch.tensor(counter, device=device, dtype=dtype)

    particleState = BasicState(particles.positions, particles.supports, particles.masses, particles.densities, torch.zeros_like(particles.positions), torch.zeros(particles.positions.shape[0], device = device, dtype = torch.int64), torch.zeros(particles.positions.shape[0], device = device, dtype = torch.int64), torch.arange(particles.positions.shape[0], device = device), particles.positions.shape[0])
    neighborhood, neighbors = evaluateNeighborhood(particleState, config['domain'], kernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
    particleState.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particleState, neighbors.neighbors, which = 'noghost')).rowEntries
    particleState.densities = computeDensity(particleState, kernel, neighbors.get('noghost'), SupportScheme.Gather, config)

    neighs = neighbors.get('noghost')
    nnx = 63
    ddx = 2 / (nnx)
    hij = (particleState.supports[neighs[0].row] + particleState.supports[neighs[0].col]) / 2

    positions = neighs[1].x_ij / hij.view(-1,1)
    index = ((positions + 1) / ddx).to(torch.int64)
    linIdx = index[:,0] * nnx + index[:,1]
    counter = scatter_sum(torch.ones_like(linIdx), dim = 0, index = linIdx, dim_size = nnx**2).reshape(nnx,nnx).cpu()#.numpy()
        
    particles = particles._replace(densities = particleState.densities)    
    
    return particles, particleState.numNeighbors, counter

def visualize(
    particles: ParticleSet,
    numNeighbors: torch.Tensor,
    counter: torch.Tensor,
):    
    fig, axis = plt.subplots(1,3 , figsize=(14,4), squeeze=False)
    L =2
    dim = 2
    kernel = KernelType.Wendland4
    targetNeighbors = n_h_to_nH(4, dim)

    device  = particles.positions.device
    dtype = particles.positions.dtype
    domain = buildDomainDescription(L, dim, True, device, dtype)

    # particles = sampleOptimal(nx, domain, targetNeighbors, kernel, 0., 0, shiftScheme = 'delta')

    config = {
        'domain': domain,
        'kernel': kernel,
        'targetNeighbors': targetNeighbors,
        'neighborhood':{
            'verletScale': 1.0,
            'computeDkDh': True
        }
    }

    config['gradientMode'] = GradientMode.Difference
    config['laplacianMode'] = LaplacianMode.Brookshaw
    config['supportScheme'] = SupportScheme.Gather
    config['integrationScheme'] = IntegrationSchemeType.rungeKutta4
    
    
    particleState = BasicState(particles.positions, particles.supports, particles.masses, particles.densities, torch.zeros_like(particles.positions), torch.zeros(particles.positions.shape[0], device = device, dtype = torch.int64), torch.zeros(particles.positions.shape[0], device = device, dtype = torch.int64), torch.arange(particles.positions.shape[0], device = device), particles.positions.shape[0])

    visualizeParticles(fig, axis[0,0], particleState, config['domain'], particleState.densities,
        kernel, which = 'both', visualizeBoth = True, cbar = True, cmap = 'viridis', markerSize = 0.25, gridVisualization = False, title = 'Particle Density')

    visualizeParticles(fig, axis[0,1], particleState, config['domain'], numNeighbors,
        kernel, which = 'both', visualizeBoth = True, cbar = True, cmap = 'magma', markerSize = 0.25, gridVisualization = False, title = 'Number of Neighbors')

    sc = axis[0,2].imshow(counter.cpu().numpy().T, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis', norm=LogNorm(vmin=1, vmax=counter.max().item()))
    fig.colorbar(sc, ax=axis[0,2], label='Number of Particles')
    axis[0,2].set_title('Relative Particle Distribution')
    fig.tight_layout()
    return fig, axis


def generateInitialVariables(nx, device = None):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    L = 2
    dim = 2

    kernel = KernelType.Wendland4
    targetNeighbors = n_h_to_nH(4, dim)

    domain = buildDomainDescription(L, dim, True, device, dtype)

    particles = sampleOptimal(nx, domain, targetNeighbors, kernel, 0., 0, shiftScheme = 'delta')

    config = {
        'domain': domain,
        'kernel': kernel,
        'targetNeighbors': targetNeighbors,
        'neighborhood':{
            'verletScale': 1.0
        }
    }

    config['gradientMode'] = GradientMode.Difference
    config['laplacianMode'] = LaplacianMode.Brookshaw
    config['supportScheme'] = SupportScheme.Gather
    config['integrationScheme'] = IntegrationSchemeType.rungeKutta4
    
    return config, domain, device, dtype, kernel

import h5py
import datetime
def getCurrentTimestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import math
def plotState(
        particleState,
        waveState: WaveEquationState,
        config, kernel,
        markerSize: float = 0.5,
        plotGrid: bool = False,
        plotCD: bool = False):
    if plotCD:
        fig, axis = plt.subplots(2,2 , figsize=(10,9), squeeze=False, sharey=True)
    else:
        fig, axis = plt.subplots(1,2 , figsize=(10,5), squeeze=False, sharey=True)

    nx = int(math.sqrt(particleState.positions.shape[0]))

    uPlot = visualizeParticles(
        fig, axis[0,0], particleState, config['domain'], waveState.u,
        kernel, which = 'both', visualizeBoth = True, cbar = True, cmap = 'RdBu_r', markerSize = markerSize, gridVisualization = plotGrid, scaling ='sym', midPoint = 0.0, gridResolution = nx * 2)

    vPlot = visualizeParticles(
        fig, axis[0,1], particleState, config['domain'], waveState.v,
        kernel, which = 'both', visualizeBoth = True, cbar = True, cmap = 'RdBu_r', markerSize = markerSize, gridVisualization = plotGrid, scaling ='sym', midPoint = 0.0, gridResolution = nx * 2)
    axis[0,0].set_title('u')
    axis[0,1].set_title('v')

    if plotCD:
        cPlot = visualizeParticles(
            fig, axis[1,0], particleState, config['domain'], waveState.c,
            kernel, which = 'both', visualizeBoth = True, cbar = True, cmap = 'magma', markerSize = markerSize, gridVisualization = plotGrid, gridResolution =  nx * 2)

        dampPlot = visualizeParticles(
            fig, axis[1,1], particleState, config['domain'], waveState.damping,
            kernel, which = 'both', visualizeBoth = True, cbar = True, cmap = 'viridis', markerSize = markerSize, gridVisualization = plotGrid, gridResolution =  nx * 2)
        axis[1,0].set_title('c')
        axis[1,1].set_title('damping')
    else:
        cPlot, dampPlot = None, None

    fig.tight_layout()

    return fig, axis, uPlot, vPlot, cPlot, dampPlot



def plotInitialState(
    particleState, config,
    uGrid, vGrid, cGrid, dampGrid,
    uSourceGrid, cSourceGrid
):
    fig,axis = plt.subplots(2,3, figsize=(10,5), squeeze=False)

    sc = axis[0,0].scatter(particleState.positions[:,0].cpu(), particleState.positions[:,1].cpu(), c=uSourceGrid.cpu(), s=1, cmap='tab10')
    fig.colorbar(sc, ax=axis[0,0])
    axis[0,0].set_title('u Source Grid')

    sc = axis[1,0].scatter(particleState.positions[:,0].cpu(), particleState.positions[:,1].cpu(), c=cSourceGrid.cpu(), s=1, cmap='tab10')
    fig.colorbar(sc, ax=axis[1,0])
    axis[1,0].set_title('c Source Grid')

    sc = axis[0,1].scatter(particleState.positions[:,0].cpu(), particleState.positions[:,1].cpu(), c=uGrid.cpu(), s=1, cmap='viridis')
    fig.colorbar(sc, ax=axis[0,1])
    axis[0,1].set_title('u Initial Condition')

    sc = axis[0,2].scatter(particleState.positions[:,0].cpu(), particleState.positions[:,1].cpu(), c=vGrid.cpu(), s=1, cmap='cividis')
    fig.colorbar(sc, ax=axis[0,2])
    axis[0,2].set_title('v Initial Condition')

    sc = axis[1,1].scatter(particleState.positions[:,0].cpu(), particleState.positions[:,1].cpu(), c=cGrid.cpu(), s=1, cmap='jet')
    fig.colorbar(sc, ax=axis[1,1])
    axis[1,1].set_title('c Grid')

    sc = axis[1,2].scatter(particleState.positions[:,0].cpu(), particleState.positions[:,1].cpu(), c=dampGrid.cpu(), s=1, cmap='magma')
    fig.colorbar(sc, ax=axis[1,2])
    axis[1,2].set_title('Damping Grid')


    for ax in axis.flatten():
        ax.set_xlim(config['domain'].min[0].cpu().item(), config['domain'].max[0].cpu().item())
        ax.set_ylim(config['domain'].min[1].cpu().item(), config['domain'].max[1].cpu().item())
        ax.set_aspect('equal')
    fig.tight_layout()

    return fig, axis