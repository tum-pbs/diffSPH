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

import h5py

from dataclasses import dataclass
@dataclass
class DataSetProperties:
    skipInitialSteps: int = 0
    skipFinalSteps: int = 0
    
    temporalCoarseGrainingRate: int = 1
    unrollLength: int = 1
    historyLength: int = 0

'''
Given a sample simulation S with timesteps
[0,1,2,3,...,T-1]

We want to load a contiguous sequence of unrollLength timesteps, starting at some random time t, and use the previous historyLength timesteps as input to predict the next unrollLength timesteps.

So we want the dataloader to return

[t - h * c, t - (h-1) * c, ..., t - c] as the history with h = historyLength and c = temporalCoarseGrainingRate
[t + c, t + 2c, ..., t + unrollLength * c] as the target with u = unrollLength and c = temporalCoarseGrainingRate
[t] as the current state

Positions and other particle properties are not changing over time, so we can just return them for the current state and ignore them for the history and target, since they are the same for all timesteps (this is not true for general SPH simulations, but it is true for our wave equation simulations where particles are fixed on a grid).

Based on this we can define the following properties for our dataset:
- skipInitialSteps: number of initial steps to skip (since they may be unrepresentative due to initial transients)
- skipFinalSteps: number of final steps to skip (since they may be unrepresentative due to boundary effects or other end-of-simulation artifacts)
- temporalCoarseGrainingRate: the rate at which to sample timesteps (e.g. if c=2, we sample every 2 timesteps)
- unrollLength: the number of future timesteps to predict
- historyLength: the number of past timesteps to use as input

So given a simulation with N timesteps, the number of valid starting points for t is:
N - skipInitialSteps - skipFinalSteps - unrollLength * temporalCoarseGrainingRate - historyLength * temporalCoarseGrainingRate
'''

def computeValidStartingPoints(datasetProperties: DataSetProperties, numTimesteps: int):
    validStartPoints = numTimesteps - datasetProperties.skipInitialSteps - datasetProperties.skipFinalSteps - datasetProperties.unrollLength * datasetProperties.temporalCoarseGrainingRate - datasetProperties.historyLength * datasetProperties.temporalCoarseGrainingRate
    return validStartPoints

# print(f'Number of valid starting points per simulation: {computeValidStartingPoints(datasetProperties, T)}')

# validPoints = computeValidStartingPoints(datasetProperties, T)
# totalValidPoints = validPoints * S
# print(f'Total valid data points across all simulations: {totalValidPoints}')

'''
For the dataloader we can sample a random integer between 0 and totalValidPoints-1, and then determine which simulation and which starting point within that simulation it corresponds to.
'''

def getSimulationAndStartingPoint(globalIndex: int, datasetProperties: DataSetProperties, numTimesteps: int):
    validPointsPerSim = computeValidStartingPoints(datasetProperties, numTimesteps)
    simIndex = globalIndex // validPointsPerSim
    startingPoint = globalIndex % validPointsPerSim + datasetProperties.skipInitialSteps + datasetProperties.historyLength * datasetProperties.temporalCoarseGrainingRate
    return simIndex, startingPoint

# print(getSimulationAndStartingPoint(0, datasetProperties, T))  # Should be (0, skipInitialSteps + historyLength * temporalCoarseGrainingRate)

def loadSlice(fileNames: List[str], simIndex: int, startingPoint: int, datasetProperties: DataSetProperties, device = torch.device('cpu')):
    fileName = fileNames[simIndex]
    loadedFile = h5py.File(fileName, 'r')

    numParticles = loadedFile['particles']['positions'][:].shape[0]
    nx = int(math.sqrt(numParticles))
    dt = loadedFile['simulation'].attrs['dt']

    # print(f'Number of particles: {numParticles}, inferred nx: {nx}')


    config, domain, device, dtype, kernel = generateInitialVariables(
        nx, device = device
    )

    schemeToLoad = loadedFile['simulation'].attrs['integrationScheme']
    config['integrationScheme'] = None
    for scheme in IntegrationSchemeType:
        # print(f'Checking scheme: {scheme.value} against loaded scheme: {schemeToLoad}')
        if scheme.name == schemeToLoad:
            config['integrationScheme'] = scheme
            # print(f'Inferred integration scheme: {config["integrationScheme"]}')
            break

    
    positions = torch.tensor(loadedFile['particles']['positions'][:], dtype=torch.float32, device=device)
    densities = torch.tensor(loadedFile['particles']['densities'][:], dtype=torch.float32, device=device)
    supports = torch.tensor(loadedFile['particles']['supports'][:], dtype=torch.float32, device=device)
    volumes = torch.tensor(loadedFile['particles']['masses'][:], dtype=torch.float32, device=device)

    # particleState = BasicState(positions, supports, volumes, densities, torch.zeros_like(positions), torch.zeros(positions.shape[0], device = device, dtype = torch.int64), torch.zeros(positions.shape[0], device = device, dtype = torch.int64), torch.arange(positions.shape[0], device = device), positions.shape[0])
    # neighborhood, neighbors = evaluateNeighborhood(particleState, config['domain'], kernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
    # particleState.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particleState, neighbors.neighbors, which = 'noghost')).rowEntries

    historyIndices = [startingPoint - i * datasetProperties.temporalCoarseGrainingRate for i in range(datasetProperties.historyLength, 0, -1)]
    targetIndices = [startingPoint + i * datasetProperties.temporalCoarseGrainingRate for i in range(1, datasetProperties.unrollLength + 1)]
    
    historyU = torch.tensor(loadedFile['simulation']['u'][historyIndices], dtype=torch.float32, device=device)
    historyV = torch.tensor(loadedFile['simulation']['v'][historyIndices], dtype=torch.float32, device=device)
    historyC = torch.tensor(loadedFile['simulation']['c'][historyIndices], dtype=torch.float32, device=device)
    historyD = torch.tensor(loadedFile['simulation']['damping'][historyIndices], dtype=torch.float32, device=device)
    
    targetU = torch.tensor(loadedFile['simulation']['u'][targetIndices], dtype=torch.float32, device=device)
    targetV = torch.tensor(loadedFile['simulation']['v'][targetIndices], dtype=torch.float32, device=device)
    targetC = torch.tensor(loadedFile['simulation']['c'][targetIndices], dtype=torch.float32, device=device)
    targetD = torch.tensor(loadedFile['simulation']['damping'][targetIndices], dtype=torch.float32, device=device)
    
    currentU = torch.tensor(loadedFile['simulation']['u'][startingPoint], dtype=torch.float32, device=device)
    currentV = torch.tensor(loadedFile['simulation']['v'][startingPoint], dtype=torch.float32, device=device)
    currentC = torch.tensor(loadedFile['simulation']['c'][startingPoint], dtype=torch.float32, device=device)
    currentD = torch.tensor(loadedFile['simulation']['damping'][startingPoint], dtype=torch.float32, device=device)

    historyState = torch.stack([historyU, historyV, historyC, historyD], dim=-1).squeeze(-2)  # Shape: (historyLength, N, numFeatures)
    targetState = torch.stack([targetU, targetV, targetC, targetD], dim=-1).squeeze(-2)  # Shape: (unrollLength, N, numFeatures)
    currentState = torch.stack([currentU, currentV, currentC, currentD], dim=-1).squeeze(-2)  # Shape: (N, numFeatures)
    
    return historyState, targetState, currentState, positions, densities, supports, volumes, dt, scheme.value

def getDatasetItem(globalIndex: int, datasetProperties: DataSetProperties, fileNames: List[str], numTimesteps: int, device = torch.device('cpu')):
    simIndex, startingPoint = getSimulationAndStartingPoint(globalIndex, datasetProperties, numTimesteps)
    historyState, targetState, currentState, positions, densities, supports, volumes, dt, scheme = loadSlice(fileNames, simIndex, startingPoint, datasetProperties, device)
    return historyState, targetState, currentState, positions, densities, supports, volumes, dt, scheme, fileNames[simIndex], simIndex, startingPoint

def getDatasetProperties(simFolder):
    files = os.listdir(simFolder)
    files = [f for f in files if f.endswith('.h5')]
    files.sort()
    # print(files)
# 
    fileIndex = 0
    fileName = os.path.join(simFolder, files[fileIndex])
    fileNames = [os.path.join(simFolder, f) for f in files]
    # print(f"Loading from {fileName}")
    loadedFile = h5py.File(fileName, 'r')

    S = len(files)
    T = loadedFile['simulation']['u'].shape[0]
    N = loadedFile['simulation']['u'].shape[1]
    Nx = Ny = int(N**0.5)
    # print(f'Dataset has S={S} samples, T={T} timesteps, N={N} particles, Nx={Nx}, Ny={Ny}')

    loadedFile.close()

    return fileNames, S, T, N, Nx, Ny

class Dataset:
    def __init__(self, datasetProperties: DataSetProperties, fileNames: List[str], numTimesteps: int, device = torch.device('cpu')):
        self.datasetProperties = datasetProperties
        self.fileNames = fileNames
        self.numTimesteps = numTimesteps
        self.device = device
        self.totalValidPoints = computeValidStartingPoints(datasetProperties, numTimesteps) * len(fileNames)
    
    def __len__(self):
        return self.totalValidPoints
    
    def __getitem__(self, index):
        return getDatasetItem(index, self.datasetProperties, self.fileNames, self.numTimesteps, self.device)


def batchedToSimState(
        currentStateBatch, positionsBatch, densitiesBatch, supportsBatch, volumesBatch, dtBatch, schemeBatched, fileNameBatch, simIndexBatch, startingPointBatch
):
    B, N, F = currentStateBatch.shape
    B, N, D = positionsBatch.shape
    assert B == 1, f'Expected batch size of 1, but got {B}'

    dt = dtBatch.item()

    # print(f'Current state batch shape: {currentStateBatch.shape}')
    # print(f'Loaded from file: {fileNameBatch[0]}, sim index: {simIndexBatch[0]}, starting point: {startingPointBatch[0]}')
    scheme = IntegrationSchemeType(schemeBatched[0].cpu().item())
    # print(f'Integration scheme: {scheme}')
    nx = int(N**0.5)

    config, domain, device, dtype, kernel = generateInitialVariables(
        nx, device = currentStateBatch.device
        )
    config['integrationScheme'] = scheme

    integrator = getIntegrator(config['integrationScheme'])

    particleState = BasicState(positionsBatch[0], supportsBatch[0], volumesBatch[0], densitiesBatch[0], torch.zeros_like(positionsBatch[0]), torch.zeros(positionsBatch[0].shape[0], device = device, dtype = torch.int64), torch.zeros(positionsBatch[0].shape[0], device = device, dtype = torch.int64), torch.arange(positionsBatch[0].shape[0], device = device), positionsBatch[0].shape[0])
    neighborhood, neighbors = evaluateNeighborhood(particleState, config['domain'], kernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
    particleState.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particleState, neighbors.neighbors, which = 'noghost')).rowEntries
    particleState.densities = computeDensity(particleState, kernel, neighbors.get('noghost'), SupportScheme.Gather, config)

    timestep = 0

    waveState = WaveEquationState(
        u = currentStateBatch[0, :, 0],
        v = currentStateBatch[0, :, 1],
        c = currentStateBatch[0, :, 2],
        damping = currentStateBatch[0, :, 3],
    )
    # smoothedState = smoothState(waveState, particleState, 0, neighbors, config)

    waveSystem = WaveSystem(
        systemState = particleState,
        waveState = waveState,
        neighborhood = neighbors.get('noghost'),
        t = timestep * dt
    )

    return waveSystem, config, integrator, dt

def batchToSimulation(batch):
    historyStateBatch, targetStateBatch, currentStateBatch, positionsBatch, densitiesBatch, supportsBatch, volumesBatch, dtBatch, schemeBatched, fileNameBatch, simIndexBatch, startingPointBatch = batch

    waveSystem, config, integrator, dt = batchedToSimState(
        currentStateBatch, positionsBatch, densitiesBatch, supportsBatch, volumesBatch, dtBatch, schemeBatched, fileNameBatch, simIndexBatch, startingPointBatch
    )

    return waveSystem, config, integrator, dt, historyStateBatch[0], targetStateBatch[0], currentStateBatch[0]