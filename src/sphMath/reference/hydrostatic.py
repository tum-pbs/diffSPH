import torch
from sphMath.schemes.gasDynamics import CompressibleSystem, CompressibleState
from sphMath.sampling import buildDomainDescription, sampleRegularParticles
from sphMath.modules.adaptiveSmoothingASPH import nH_to_n_h, n_h_to_nH
from sphMath.schemes.monaghanPrice import getPrice2007Config, MonaghanScheme, getMonaghan1997Config, getMonaghan1992Config, getMonaghanGingold1983Config

from dataclasses import dataclass, fields
from sphMath.finiteDifference import continuousGradient, centralDifferenceStencil
from sphMath.modules.adaptiveSmoothingASPH import evaluateOptimalSupportOwen, computeOwen
from sphMath.modules.eos import idealGasEOS
from sphMath.neighborhood import computeDistanceTensor, NeighborhoodInformation, SparseNeighborhood
from typing import Union
from sphMath.modules.density import computeDensity, computeDensityGradient, computeRenormalizedDensityGradient, computeDensityDeltaTerm
from sphMath.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr
from sphMath.util import volumeToSupport
from sphMath.util import ParticleSet
from sphMath.sampling import buildDomainDescription, sampleRegularParticles
from sphMath.modules.adaptiveSmoothing import computeOmega, evaluateOptimalSupport


def sdBox(p, b, c):
    d = torch.abs(p - c) - b
    return torch.linalg.norm(d.clamp(min = 0), dim = -1) + torch.max(d, dim = -1)[0].clamp(max=0)

def sampleSDF(x, sdf, h = 1e-4):
    dist = sdf(x)
    
    stencil, stencil_offsets = centralDifferenceStencil(1,2)#.to(device = particles.positions.device, dtype = particles.positions.dtype)
    stencil = stencil.to(device = x.device, dtype = x.dtype)
    stencil_offsets = stencil_offsets.to(device = x.device, dtype = x.dtype)

    normal = continuousGradient(sdf, x, stencil, stencil_offsets, h, 1)
    normal = torch.nn.functional.normalize(normal, p = 2, dim = -1)
    return dist, normal
    

def generateHydroStatic(
        rho_low = 1,
        rho_high = 4,
        nx = 100,
        l = 1,
        gamma = 1.5,
        samplingBefore = True,
        kernel = 'B7',
        device = torch.device('cuda'),
        dtype = torch.float64,
        supportScheme = 'Owen'
):
    dim = 2
    domain = buildDomainDescription(l, dim, periodic = True, device = device, dtype = dtype)
    # domain.min = torch.tensor([0, 0], device = device, dtype = dtype)
    # domain.max = torch.tensor([l, l], device = device, dtype = dtype)
    
    targetNeighbors = n_h_to_nH(4, dim)
    solverConfig = getPrice2007Config(gamma = gamma, kernel=kernel, targetNeighbors = targetNeighbors, domain = domain)
        
    particles = sampleRegularParticles(nx, domain, solverConfig['targetNeighbors'], jitter = 0)
    fluidParticles = CompressibleState(
        positions = particles.positions,
        supports = particles.supports,
        masses = particles.masses,
        densities = torch.ones_like(particles.masses),
        velocities= torch.zeros_like(particles.positions),
        
        internalEnergies= torch.zeros_like(particles.masses),
        totalEnergies= torch.zeros_like(particles.masses),
        entropies = torch.zeros_like(particles.masses),
        pressures = torch.ones_like(particles.masses),
        soundspeeds = torch.zeros_like(particles.masses),
        
        alphas = torch.ones_like(particles.masses),
        alpha0s = torch.ones_like(particles.masses),
        divergence = torch.zeros_like(particles.masses),
    )

    fluidParticles.supports = volumeToSupport(fluidParticles.masses, solverConfig['targetNeighbors'], fluidParticles.positions.shape[1])
    neighborhood, sparseNeighborhood = buildNeighborhood(fluidParticles, fluidParticles, domain, solverconfig['neighborhood']['verletScale'],  'superSymmetric', priorNeighborhood=None)
    actualNeighbors = filterNeighborhood(sparseNeighborhood)
    # csr = coo_to_csr(neighborhood.fullAdjacency)

    rij, xij = computeDistanceTensor(sparseNeighborhood, False)

    innerRegion = lambda x: sdBox(x, torch.tensor([l/4, l/4], device = device, dtype = dtype), torch.tensor([+0/4, +0/4], device = device, dtype = dtype))
    r   , n = sampleSDF(particles.positions, innerRegion)

    rho = computeDensity(fluidParticles, fluidParticles, domain, solverConfig['kernel'], actualNeighbors, 'gather')

    totalMass =( rho * particles.masses) .sum()

    minMass = fluidParticles.masses.min()

    fluidParticles.masses[r < 0] = fluidParticles.masses.min() * rho_high
    fluidParticles.densities[r < 0] = rho_high
    fluidParticles.masses[r >= 0] = fluidParticles.masses.min() * rho_low
    fluidParticles.densities[r >= 0] = rho_low
    P = torch.ones_like(fluidParticles.densities)
    A_, u_, p_, c_ = idealGasEOS(A=None, u = None, P = P, rho = fluidParticles.densities, gamma = gamma)


    fluidParticles.supports = volumeToSupport(fluidParticles.masses / fluidParticles.densities, solverConfig['targetNeighbors'], fluidParticles.positions.shape[1])

    neighborhood, sparseNeighborhood = buildNeighborhood(fluidParticles, fluidParticles, domain, solverconfig['neighborhood']['verletScale'],  'superSymmetric', priorNeighborhood=None)
    actualNeighbors = filterNeighborhood(sparseNeighborhood)

    rho = computeDensity(fluidParticles, fluidParticles, domain, solverConfig['kernel'], actualNeighbors, 'gather')

    solverConfig['owenSupport'] = computeOwen(solverConfig['kernel'], dim = domain.dim, nMin = 2.0, nMax = 6.0, nLUT = 1024)
    if supportScheme == 'Monaghan':
        rho, h_i_new, rhos, hs, neighborhood = evaluateOptimalSupport(fluidParticles, domain, solverConfig['kernel'], solverConfig['targetNeighbors'], nIter = 16, neighborhood=neighborhood)
    elif supportScheme == 'Owen':
        rho, h_i_new, rhos, hs, neighborhood = evaluateOptimalSupportOwen(particles, domain, solverConfig['kernel'], solverConfig['targetNeighbors'], solverConfig['owenSupport'], nIter = 16, neighborhood=neighborhood, verbose = False, eps = 1e-3)
    else:
        h_i_new = fluidParticles.supports
        rho = fluidParticles.densities

    fluidParticles.supports = h_i_new
    fluidParticles.densities = rho

    neighborhood, sparseNeighborhood = buildNeighborhood(fluidParticles, fluidParticles, domain, 1.0,  'gather', priorNeighborhood=None)
    # csr = coo_to_csr(filterNeighborhood(neighborhood))

    rho = computeDensity(fluidParticles, fluidParticles, domain, solverConfig['kernel'], neighborhood.fullAdjacency, 'gather')
    fluidParticles.densities = rho

    if not samplingBefore:
        A_, u_, p_, c_ = idealGasEOS(A=None, u = None, P = P, rho = fluidParticles.densities, gamma = gamma)

    fluidParticles.internalEnergies = u_
    fluidParticles.totalEnergies = u_ * fluidParticles.masses
    fluidParticles.entropies = A_
    fluidParticles.pressures = P
    fluidParticles.soundspeeds = c_

    return fluidParticles, domain, neighborhood, solverConfig