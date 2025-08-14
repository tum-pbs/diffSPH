import torch
from diffSPH.dataLoaderUtils.neighborhood import DomainDescription
from diffSPH.operations import sph_op
from diffSPH.kernels import getSPHKernelv2
from diffSPH.dataLoaderUtils.util import buildRotationMatrix
from diffSPH.dataLoaderUtils.neighborhood import coo_to_csr
from diffSPH.dataLoaderUtils.neighborhood import DomainDescription
from diffSPH.operations import sph_op
from diffSPH.kernels import getSPHKernelv2
from diffSPH.dataLoaderUtils.util import buildRotationMatrix
from diffSPH.dataLoaderUtils.neighborhood import coo_to_csr
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py 
# from BasisConvolution.util.datautils import parseFile
import os

from diffSPH.dataLoaderUtils.state import DataConfiguration
from diffSPH.dataLoaderUtils.util import processFolder, getDataLoader
from diffSPH.dataLoaderUtils.newFormatLoader import loadFrame_newFormat_v2, convertNewFormatToWCSPH, loadNewFormatState
from diffSPH.dataLoaderUtils.loader import loadState
from diffSPH.dataLoaderUtils.loader import loadBatch
from diffSPH.dataLoaderUtils.neighborhood import neighborSearch, filterNeighborhoodByKind

import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm
from diffSPH.dataLoaderUtils.state import WeaklyCompressibleSPHState, CompressibleSPHState, RigidBodyState
from diffSPH.dataLoaderUtils.batch import mergeBatch, mergeTrajectoryStates
from diffSPH.dataLoaderUtils.augment import augmentDomain, rotateState, buildRotationMatrix
from diffSPH.dataLoaderUtils.util import kernelNameToKernel
from diffSPH.dataLoaderUtils.augment import loadAugmentedBatch
from diffSPH.plotting import visualizeParticles, updatePlot
import matplotlib.pyplot as plt



def getDomain(parsedMetaData, device = torch.device('cpu'), dtype = torch.float32):

    parsedMetaData['bounds'][0]
    dim = len(parsedMetaData['bounds'])
    minExtent = []
    maxExtent = []
    for i in range(dim):
        minExtent.append(parsedMetaData['bounds'][i][0])
        maxExtent.append(parsedMetaData['bounds'][i][1])
        

    # print(minExtent, maxExtent)

    periodic = parsedMetaData['periodic_boundary_conditions'][:dim]
    # print(periodic)


    domain = DomainDescription(
        min = torch.tensor(minExtent, dtype = dtype, device =device),
        max = torch.tensor(maxExtent, dtype = dtype, device =device),
        periodic = torch.tensor(periodic, dtype = torch.bool, device = device),
        dim = dim
    )

    # print(domain)
    return domain

def computeVelocity(positions_a, positions_b, dt, domain):
    dx = positions_b - positions_a
    for i in range(domain.dim):
        if domain.periodic[i]:
            box_size = domain.max[i] - domain.min[i]
            dx_component = dx[:, i]
            dx[:, i] = ((dx_component + box_size / 2) % box_size) - box_size / 2
    return dx / dt

def loadFrame(inFile, trajectory, frame, parsedMetaData, device = torch.device('cpu'), dtype = torch.float32, domain = None):
    dt = parsedMetaData['dt'] * parsedMetaData['write_every']
    positions = torch.from_numpy(inFile[trajectory]['position'][frame,:]).to(device = device, dtype=dtype)
    supports = torch.ones_like(positions[:,0], dtype=dtype, device=device) * parsedMetaData['default_connectivity_radius'] * 2

    masses = torch.ones_like(positions[:,0], dtype=dtype, device=device) * parsedMetaData['dx']**parsedMetaData['dim']

    densities = torch.ones_like(positions[:,0], dtype=dtype, device=device)

    if frame == 0:
        nextPositions = torch.from_numpy(inFile[trajectory]['position'][frame+1,:]).to(device = device, dtype=dtype)
        # speed = (nextPositions - positions) / dt
        speed = - computeVelocity(positions, nextPositions, dt, domain)
    else:
        nextPositions = torch.from_numpy(inFile[trajectory]['position'][frame-1,:]).to(device = device, dtype=dtype)
        speed = computeVelocity(positions, nextPositions, dt, domain)

    velocities = speed

    kinds = torch.from_numpy(inFile[trajectory]['particle_type'][:]).to(device = device, dtype=torch.int64).clamp(min=0, max=1)
    materials = torch.zeros_like(kinds, dtype=torch.int64, device=device)
    UIDs = torch.arange(positions.shape[0], dtype=torch.int64, device=device)

    numParticles = positions.shape[0]
    time = frame * dt
    dt = dt
    key = f'{trajectory}_{frame}'


    state = WeaklyCompressibleSPHState(
        positions=positions,
        supports=supports,
        masses=masses,
        velocities=velocities,
        densities=densities,
        kinds=kinds,
        materials=materials,
        UIDs=UIDs,
        numParticles=numParticles,
        time=time,
        dt=dt,
        key=key,
        timestep=frame,
    )

    neighborhood = neighborSearch(
        state=state,
        domain=domain,
        config=None
    )

    density = sph_op(
        state, state, domain, getSPHKernelv2('CubicSpline'), neighborhood, quantity=state.densities, supportScheme='gather', operation='density')

    state.densities = density
    filtered_csr = coo_to_csr(neighborhood)

    return state
