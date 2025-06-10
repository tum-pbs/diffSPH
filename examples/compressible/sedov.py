import torch
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'

import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from sphMath.operations import sph_operation, mod
from sphMath.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr
from sphMath.util import volumeToSupport
from sphMath.util import ParticleSet
from sphMath.sampling import buildDomainDescription, sampleRegularParticles

from sphMath.plotting import visualizeParticles
from sphMath.sampling import generateNoiseInterpolator
from sphMath.util import ParticleSetWithQuantity, mergeParticles
import random
from sphMath.sampling import getSpacing
from sphMath.operations import sph_op
from sphMath.sampling import generateTestData
from sphMath.modules.density import computeDensity, computeDensityGradient, computeRenormalizedDensityGradient, computeDensityDeltaTerm

from typing import NamedTuple, Tuple
from sphMath.kernels import KernelType
# from sphMath.neighborhood import buildSuperSymmetricNeighborhood
from sphMath.modules.adaptiveSmoothing import computeOmega, evaluateOptimalSupport
from sphMath.integration import semiImplicitEuler, TVDRK3, symplecticEuler, RungeKutta2
from sphMath.schemes.monaghanPrice import getPrice2007Config, MonaghanScheme, getMonaghan1997Config, getMonaghan1992Config, getMonaghanGingold1983Config
from sphMath.schemes.compSPH import compSPHScheme, getCompSPHConfig, CompSPHSystem
from sphMath.schemes.PressureEnergySPH import PressureEnergyScheme, getPressureEnergyConfig
from sphMath.schemes.CRKSPH import CRKScheme, getCRKConfig
from sphMath.schemes.gasDynamics import CompressibleSystem, CompressibleState
from sphMath.reference.sod import buildSod_reference, sodInitialState, generateSod1D
import copy

from sphMath.reference.hydrostatic import generateHydroStatic
from sphMath.modules.eos import idealGasEOS
from sphMath.kernels import Kernel
from sphMath.sampling import sampleShell
from sphMath.neighborhood import evaluateNeighborhood, filterNeighborhoodByKind, SupportScheme

def buildSedov(
        nx : int = 200,
        dim: int = 2,
        domainExtent: float = 2,
        periodicDomain: bool = True,

        rho0 : float = 1,
        E0 : float = 1,
        initialization: str = 'hat',
        gamma : float = 5/3,
        kernel: KernelType = KernelType.B7,
        targetNeighbors = 50,

        dtype = torch.float32,
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), shellSampling = False, SimulationSystem = CompressibleSystem):
    # dim = 2
    domain = buildDomainDescription(domainExtent, dim, periodic = periodicDomain, device = device, dtype = dtype)

    if initialization == 'singular':
        if nx % 2 == 0:
            warnings.warn('nx should be odd for singular initialization, setting to nx + 1')
            nx += 1
    elif initialization == 'quadrant':
        if nx % 2 == 1:
            warnings.warn('nx should be even for quadrant initialization, setting to nx - 1')
            nx -= 1

    if shellSampling:
        particles_ = sampleShell(nx, domain, targetNeighbors)
    else:
        particles_ = sampleRegularParticles(nx, domain, targetNeighbors)
    # domain = buildDomainDescription(domainExtent * 1.5, dim, periodic = periodicDomain, device = device, dtype = dtype)
    particles_ = particles_._replace(masses = particles_.masses * rho0)


    particles = CompressibleState(
        positions = particles_.positions,
        supports = particles_.supports,
        masses = particles_.masses,
        densities = particles_.densities,
        velocities = torch.zeros_like(particles_.positions),
        
        kinds = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        materials = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        UIDs = torch.arange(particles_.positions.shape[0], device = device, dtype = torch.int32),
        
        internalEnergies = None,
        totalEnergies = None,
        entropies = None,
        pressures = None,
        soundspeeds = None,

        divergence=torch.zeros_like(particles_.densities),
        alpha0s= torch.ones_like(particles_.densities),
        alphas= torch.ones_like(particles_.densities),
    )

    wrappedKernel = kernel
    neighborhood, neighbors = evaluateNeighborhood(particles, domain, wrappedKernel, verletScale = 1.0, mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
    numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries
    config = {'targetNeighbors': targetNeighbors, 'domain': domain, 'support': {'iterations': 16, 'scheme': 'Monaghan'}, 'neighborhood': {'algorithm': 'compact'}}
    rho, h, rhos, hs, neighborhood = evaluateOptimalSupport(particles, wrappedKernel, neighborhood, SupportScheme.Gather, config)


    particles .densities = rho
    particles.supports = h

    P_initial = torch.zeros_like(particles.densities)
    u = 1 / (gamma - 1) * (P_initial / rho)
    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = rho, gamma = gamma)
    A_, u_, P_, c_s = idealGasEOS(A = None, u = u, P = None, rho = rho, gamma = gamma)

    internalEnergy = u_ 
    kineticEnergy = torch.linalg.norm(torch.zeros_like(particles.positions), dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * particles.masses

    simulationState_ = CompressibleState(
        positions = particles.positions,
        supports = particles.supports,
        masses = particles.masses,
        densities = particles.densities,        
        velocities = torch.zeros_like(particles.positions),

        kinds = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        materials = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        UIDs = torch.arange(particles_.positions.shape[0], device = device, dtype = torch.int32),
        
        internalEnergies = u_,
        totalEnergies = totalEnergy,
        entropies = A_,
        pressures = P_,
        soundspeeds = c_s,

        alphas = torch.ones_like(particles.densities),
        alpha0s = torch.ones_like(particles.densities),
        divergence=torch.zeros_like(particles.densities),
    )
        

    if initialization == 'hat':
        W = Kernel(wrappedKernel, torch.tensor([[0,0]], device = device, dtype = dtype) - simulationState_.positions, simulationState_.supports)
        W = W/W.sum()
        simulationState_.internalEnergies = E0 * W / simulationState_.masses
        A_, u_, P_, c_s = idealGasEOS(A = None, u = simulationState_.internalEnergies, P = None, rho = rho, gamma = gamma)
        simulationState_.pressures = P_
    elif initialization == 'singular':
        dist = torch.linalg.norm(simulationState_.positions, dim = -1)
        sortedDist, idx = torch.sort(dist)
        u_ = torch.zeros_like(simulationState_.masses)
        u_[idx[0]] = E0 / simulationState_.masses[idx[0]]

        # if sampleShell:
        #     u_[0] = E0 / particles.masses[0]
        # else:
        #     u_grid = u_.reshape(nx, nx)
        #     middle = nx // 2
        #     u_grid[middle, middle] = E0 / particles.masses[middle * nx + middle]
        #     u_ = u_grid.reshape(-1)
        A_, u_, P_, c_s = idealGasEOS(A = None, u = u_, P = None, rho = rho, gamma = gamma)
        simulationState_.internalEnergies = u_
        simulationState_.pressures = P_
    elif initialization == 'quadrant':
        if shellSampling:
            raise ValueError('Shell sampling not supported for quadrant initialization')
        if dim == 2: 
            nptcl = 4
        elif dim == 1:
            nptcl = 2
            
        dist = torch.linalg.norm(simulationState_.positions, dim = -1)
        sortedDist, idx = torch.sort(dist)
        u_ = torch.zeros_like(simulationState_.masses)
        u_[idx[:nptcl]] = E0 / simulationState_.masses[idx[:nptcl]]
        


        # u_grid = u_.reshape(nx, nx)
        # middle = nx // 2
        # for ix in [middle - 1, middle]:
        #     for iy in [middle - 1, middle]:
        #         u_grid[ix, iy] = E0 / 4 / particles.masses[middle * nx + middle]
        # u_ = u_grid.reshape(-1)
        A_, u_, P_, c_s = idealGasEOS(A = None, u = u_, P = None, rho = rho, gamma = gamma)
        simulationState_.internalEnergies = u_
        simulationState_.pressures = P_
    else:
        raise ValueError(f'Unknown initialization {initialization}')

    particleSystem = SimulationSystem(
        systemState = simulationState_,
        domain = domain,
        neighborhoodInfo = neighborhood,
        t = 0
    )

    return particleSystem

def radius(beta, E0, t, rho0, nu):
    return beta *( E0 * t**2 / (rho0)) ** (1/(2+nu))
def velocity(beta, E0, t, rho0, nu):
    return 2/(nu+2) * radius(beta, E0, t, rho0, nu) / t

def beta(nu):
    if nu == 1:
        # return 1 / (answer.alpha ** nu1 )
        return 1.11
    elif nu == 2:
        return 1.12
    elif nu == 3:
        return 1.15