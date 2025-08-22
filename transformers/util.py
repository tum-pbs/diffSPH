from diffSPH.schemes.states.wcsph import WeaklyCompressibleUpdate, WeaklyCompressibleState
from diffSPH.sampling import ParticleSet
from diffSPH.simple import *
import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm
from diffSPH.modules.particleShifting import solveShifting

def sampleOptimal(nx, domain, targetNeighbors, kernel, jitter = 0.1, shiftIters = 128, shiftScheme = 'IPS'):
    band = 0
    dim = domain.dim
    particles_full = sampleRegularParticles(nx, domain, targetNeighbors, jitter = 0.00, band = band)
    particleDx = particles_full.masses.pow(1/dim).mean().item()
    device = particles_full.positions.device

    particles = WeaklyCompressibleState(
        particles_full.positions,
        supports = particles_full.supports,
        masses = particles_full.masses,
        
        densities = torch.zeros_like(particles_full.masses),
        velocities = torch.zeros_like(particles_full.positions),
        
        pressures = torch.zeros_like(particles_full.masses),
        soundspeeds = torch.ones_like(particles_full.masses),
        
        kinds = torch.zeros_like(particles_full.masses),
        materials = torch.zeros_like(particles_full.masses),
        UIDs = torch.arange(particles_full.positions.shape[0], device = device, dtype = torch.int64),
    )

    config = {
        'domain': domain,
        'kernel': kernel,
        'verletScale': 1.4,
        'particle': {'dx': particleDx, 'support': particles.supports.max().item()},
        'shifting': {'scheme': shiftScheme,
                     'computeMach': False,
                     'summationDensity': True,
                    'solverThreshold': 0.5 * particleDx},
        'neighborhood': {'verletScale': 1.4, 'targetNeighbors': targetNeighbors,
        'computeHessian': True, 'computeDkDh': True},
    }

    neighborhood = None
    particleSystem = DeltaPlusSPHSystem(domain, neighborhood, 0., particles, 'momentum', None, [], [], config)
    particleSystem.systemState.positions += jitter * torch.randn_like(particleSystem.systemState.positions) * particleDx


    for i in tqdm(range(shiftIters), leave=False):
        dx, neighborhood, neighbors, overallStates, densities, velocities, fs, n, lMin = solveShifting(particleSystem, 0.1, config, verbose = False)
        particleSystem.neighborhoodInfo = neighborhood
        particleSystem.systemState.positions += dx

    return ParticleSet(particleSystem.systemState.positions, particleSystem.systemState.supports, particleSystem.systemState.masses, torch.ones_like(particleSystem.systemState.masses))

from diffSPH.dataLoaderUtils.state import WeaklyCompressibleSPHState

def wcstateToState(state):
    return WeaklyCompressibleSPHState(
        positions = state.positions,
        supports = state.supports,
        masses = state.masses,
        densities = state.densities,
        velocities = state.velocities if hasattr(state, 'velocities') else torch.zeros_like(state.positions),

        kinds = state.kinds if hasattr(state, 'kinds') else torch.zeros_like(state.masses),
        materials = state.materials if hasattr(state, 'materials') else torch.zeros_like(state.masses),
        UIDs = state.UIDs if hasattr(state, 'UIDs') else torch.arange(state.positions.shape[0], device=state.positions.device, dtype=torch.int64),

        numParticles = [state.positions.shape[0]],
        time = [0.],
        dt = [1e-3],
        timestep = [0],
        key = [''],

        boundaryNormals = None,
        boundaryDistances = None,
        boundaryIndices = None,

        rigidBodies = None,
        batches = None
    )

import torch
from diffSPH.sphOperations.shared import scatter_sum
from matplotlib.colors import LogNorm

def plotDistribution(fig, axis, particleState, neighborhood, logNorm = True, nnx = 63):
    ddx = 2 / (nnx)
    hij = (particleState.supports[neighborhood[0].row] + particleState.supports[neighborhood[0].col]) / 2
    # print(hij.shape)
    # print(neighborhood[0].row, neighborhood[0].col)

    positions = neighborhood[1].x_ij / hij.view(-1,1)
    # positions = positions[neighborhood['indices'][0] != neighborhood['indices'][1]]

    index = ((positions + 1) / ddx).to(torch.int64)
    linIdx = index[:,0] * nnx + index[:,1]
    # print(linIdx, linIdx.min(), linIdx.max())

    counter = scatter_sum(torch.ones_like(linIdx), dim = 0, index = linIdx, dim_size = nnx**2).reshape(nnx,nnx).cpu().numpy()
    if logNorm:
        sc = axis.imshow(counter, norm=LogNorm(), extent=(-1, 1, -1, 1))
    else:
        sc = axis.imshow(counter, extent=(-1, 1, -1, 1))

    # print(counter.min(), counter.max()) 

    # neighGrid = torch.zeros(64,64)
    cbar = fig.colorbar(sc, ax=axis)
    axis.set_aspect('equal')
    axis.set_title('Neighbor Distribution')
    return sc, cbar


from diffSPH.neighborhood import computeSparseDistanceTensor, SparseNeighborhood

def buildDenseNeighborhood(particleState, domain):
    batch_ids = torch.unique(particleState.batches) if particleState.batches is not None else None
    if batch_ids is not None:
        print(f'Building dense neighborhood for batches: {batch_ids}')
        cumulativeParticles = 0
        rowIndices = []
        colIndices = []
        for batch_id in batch_ids:
            batch_mask = particleState.batches == batch_id
            numPtcls = particleState.positions[batch_mask].shape[0]
            indices = torch.arange(numPtcls).reshape(-1, numPtcls).repeat(numPtcls, 1).to(device)
            rowIndices.append(indices.t().reshape(-1) + cumulativeParticles)
            colIndices.append(indices.reshape(-1) + cumulativeParticles)

            cumulativeParticles += numPtcls

        rowIndices = torch.cat(rowIndices)
        colIndices = torch.cat(colIndices)

        denseNeighborhood = SparseNeighborhood(
            row = rowIndices,
            col = colIndices,
            numRows = cumulativeParticles,
            numCols = cumulativeParticles,
            
            points_a = PointCloud(
                positions = particleState.positions[batch_mask],
                supports = particleState.supports[batch_mask],
            ),
            points_b = PointCloud(
                positions = particleState.positions[batch_mask],
                supports = particleState.supports[batch_mask],
            ),
            
            domain = domain,
        )

        return denseNeighborhood

    else:
        numPtcls = particleState.positions.shape[0]
        indices = torch.arange(numPtcls).reshape(-1, numPtcls).repeat(numPtcls, 1).to(device)
        rowIndices = indices.t().reshape(-1)
        colIndices = indices.reshape(-1)

        denseNeighborhood = SparseNeighborhood(
            row = rowIndices,
            col = colIndices,
            numRows = nx**2,
            numCols = nx**2,
            
            points_a = PointCloud(
                positions = particleState.positions,
                supports = particleState.supports,
            ),
            points_b = PointCloud(
                positions = particleState.positions,
                supports = particleState.supports,
            ),
            
            domain = domain,
        )

        return denseNeighborhood

def buildSparseNeighborhood(particleState, neighbors):
    filteredNeighborhood = filterNeighborhoodByKind(particleState, neighbors.neighbors, 'noghost')
    return filteredNeighborhood

from diffSPH.neighborhood import evaluateNeighborhood, filterNeighborhoodByKind, coo_to_csr, SparseNeighborhood
from diffSPH.neighborhood import buildNeighborhood, computeNeighborhoodStates, filterNeighborhoodByKind, coo_to_csr
from util import *

def buildSparseNeighborhood(neighborhood):
    i = neighborhood.row
    j = neighborhood.col

    return SparseNeighborhood(
        row = i,
        col = j,
        numRows = neighborhood.numRows,
        numCols = neighborhood.numCols,

        points_a = neighborhood.points_a,
        points_b = neighborhood.points_b,

        domain = neighborhood.domain,
    )


def prepareState(particles, domain, kernel, config):
    particleState = wcstateToState(particles)
    mode = SupportScheme.SuperSymmetric
    mode_str = mode.name
    mode_str = mode_str[0].lower() + mode_str[1:]

    neighborhood, sparseNeighborhood_ = buildNeighborhood(particles, particles, domain, verletScale = 1.0, mode = mode_str, priorNeighborhood=None)
    neighbors = computeNeighborhoodStates(particleState, sparseNeighborhood_, mode_str, kernel, kernel, True, False, False)

    numNeighbors = coo_to_csr(neighbors.neighbors).rowEntries
    particleState.densities = computeDensity(particleState, kernel, neighbors.get('noghost'), SupportScheme.Gather, config)

    device = particleState.positions.device
    inputFeatures = torch.cat([
        # particleState.velocities,
        torch.ones_like(particleState.velocities[:, 0])[:, None],  # bias term
    ], dim = 1).to(device)

    gt = particleState.densities.to(device)[:, None]  # ground truth densities
        
    filteredNeighborhood = neighbors.neighbors
    h_i = particleState.supports[filteredNeighborhood.row].to(device)
    rij, xij_ = computeSparseDistanceTensor(filteredNeighborhood)
    sparse_edge_attr = xij_ / h_i[:,None]

    # adjacency = buildDenseNeighborhood(particleState, domain)
    adjacency = buildSparseNeighborhood(neighbors.neighbors)

    h_i = particleState.supports[adjacency.row].to(device)
    rij, xij_ = computeSparseDistanceTensor(adjacency)

    edge_attr = xij_ / h_i[:,None]
    edge_index = torch.stack([adjacency.row, adjacency.col])

    return particleState, neighborhood, neighbors, numNeighbors, gt.view(1,-1,gt.shape[-1]), inputFeatures.view(1, -1, inputFeatures.shape[-1]), edge_attr, edge_index, adjacency
