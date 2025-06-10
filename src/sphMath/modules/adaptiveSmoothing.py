from sphMath.util import ParticleSet, ParticleSetWithQuantity
from sphMath.neighborhood import DomainDescription, SparseCOO
from typing import Union, Tuple
from sphMath.kernels import SPHKernel
import torch
from sphMath.operations import sph_op
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product
from torch.profiler import profile, record_function, ProfilerActivity
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from typing import Dict


def computeOmega(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    if supportScheme != SupportScheme.Gather:
        raise NotImplementedError('Only gather is supported')
    with record_function("[SPH] - Compute Omega"):
        kernelValues = neighborhood[1]
        
        
        i = neighborhood[0].row
        j = neighborhood[0].col
        
        # periodicity = domain.periodic
        # minExtent = domain.min
        # maxExtent = domain.max
        
        dh_drho = - particles.supports / (particles.densities * particles.positions.shape[1])
        
        # print(particles_a.positions.shape)
        # print(particles_b.positions.shape)
        
        # positions = (particles_a.positions, particles_b.positions)
        
        
        # x_ij = mod_distance(get_i(positions, i), get_j(positions, j), periodicity, minExtent, maxExtent)
        
        x_ij = kernelValues.x_ij
        dWdh = kernelValues.ddh_W_i
        # dWdh = kernel.dkdh(x_ij, particles.supports[i])
        
        summation = scatter_sum(dWdh * particles.masses[j], i, dim = 0, dim_size = particles.positions.shape[0])
        return 1 - dh_drho * summation


from sphMath.neighborhood import buildNeighborhood, filterNeighborhood
from sphMath.modules.density import computeDensity

from sphMath.neighborhood import buildNeighborhood, filterNeighborhood
from sphMath.modules.density import computeDensity


# def computeH(rho, m, targetNeighbors):
#     V = m / rho
#     return targetNeighbors / 2 * V

# def F(h, rho, m, targetNeighbors):
#     return h - computeH(rho, m, targetNeighbors)

# def dhdrho(h, rho, d):
#     return -h / (rho * d)
# def drhodh(particles: ParticleSet, kernel: SPHKernel, domain: DomainDescription, neighborhood: SparseCOO):
#     i = neighborhood.row
#     j = neighborhood.col
#     positions = (particles.positions, particles.positions)
#     x_ij = mod_distance(get_i(positions, i), get_j(positions, j), domain.periodic, domain.min, domain.max)
#     dWdh = kernel.dkdh(x_ij, particles.supports[i])
    
#     summation = scatter_sum(dWdh * particles.masses[j], i, dim = 0, dim_size = particles.positions.shape[0])
#     return summation

# # this is just computeOmega
# def dFdh(particles: ParticleSet, kernel: SPHKernel, domain: DomainDescription, neighborhood: SparseCOO):
    
#     dh_drho = dhdrho(particles.supports, particles.densities, particles.positions.shape[1])
#     drho_dh = drhodh(particles, kernel, domain, neighborhood)
#     return 1 - dh_drho * drho_dh
    

# def evaluateOptimalSupport(particles, domain, kernel, targetNeighbors, nIter = 16, neighborhood = None):
#     rhos = [particles.densities]
#     hs = [particles.supports]
#     neighborhood_ = neighborhood
#     verletScale = 1.4 if neighborhood is None else neighborhood.verletScale
#     supportMode = 'superSymmetric' if neighborhood is None else neighborhood.mode
#     if supportMode not in ['gather', 'superSymmetric']:
#         supportMode = 'superSymmetric'
#     # verletScale = 1.0 if config is None else (config['neighborhood']['verletScale'] if 'verletScale' in config else 1.0)
#     for i in range(nIter):
#         with record_function(f"[SPH] - Optimal Support - Neighbors"):
#             # print(f'Iteration {i} | neighborhood: {neighborhood_ is not None}')
#             neighborhood_ = buildNeighborhood(particles, particles, domain, verletScale= verletScale, mode =supportMode, priorNeighborhood=neighborhood_, verbose = False)
#             # actualNeighbors = filterNeighborhood(neighborhood_)
#             actualNeighbors = neighborhood_.fullAdjacency
        
#         if isinstance(particles, Tuple):
#             particles = particles._replace(densities = computeDensity(particles, particles, domain, kernel, actualNeighbors, 'gather'))
#         else:
#             particles.densities = computeDensity(particles, particles, domain, kernel, actualNeighbors, 'gather')
            
#         h_prev = particles.supports
        
#         F_ = F(h_prev, particles.densities, particles.masses, targetNeighbors)
#         # dFdh_ = dFdh(particles, wrappedKernel, domain, actualNeighbors)
#         dFdh_ = computeOmega(particles, particles, domain, kernel, actualNeighbors, 'gather')
        
#         h_new = h_prev - F_ / dFdh_
#         h_new = h_new.clamp(min = h_prev * 0.25, max = h_prev * 4.0)

#         h_diff = h_new - h_prev
#         h_ratio = h_new / h_prev
        
#         # print(f'Support Update: {h_diff.min()} | {h_diff.max()} | {h_diff.mean()}')
        

#         # rhos.append(particles.densities)
#         # hs.append(h_new)
        
#         # actualNeighbors = buildSuperSymmetricNeighborhood(particles, domain, verletScale = 1.0)
#         # rho = computeDensity(particles, particles, domain, kernel, actualNeighbors, 'gather' )
#         # V = particles.masses / rho
#         # h = targetNeighbors / 2 * V
        
#         # print(f'Iteration {i} | Support: {h.min()} | {h.max()} | {h.mean()} | Ratio: {(h / hs[0]).min()} | {(h / hs[0]).max()} | {(h / hs[0]).mean()}')
        
#         if isinstance(particles, Tuple):
#             particles = particles._replace(supports = h_new)
#         else:
#             particles.supports = h_new
#             # particles.densities = rho

#         # particles = particles._replace(supports = h, densities = rho)
#         rhos.append(particles.densities)
#         hs.append(particles.supports)
        
#         # print(f'Iteration {i} | h_ratio: {h_ratio.min()} | {h_ratio.max()} | {h_ratio.mean()}')
        
#         if (h_ratio - 1).abs().max() < 1e-3:
#             # print('Stopping Early')
#             break

#     if neighborhood is not None:
#         return rhos[-1], hs[-1], rhos, hs, neighborhood_
#     return rhos[-1], hs[-1], rhos, hs




from sphMath.neighborhood import buildNeighborhood, filterNeighborhood
from sphMath.util import volumeToSupport
from sphMath.modules.density import computeDensity


def computeH(rho, m, targetNeighbors, dim):
    V = m / rho
    return volumeToSupport(V, targetNeighbors, dim)
    return targetNeighbors / 2 * V

def F(h, rho, m, targetNeighbors, dim):
    return h - computeH(rho, m, targetNeighbors, dim)

def dhdrho(h, rho, d):
    return -h / (rho * d)
def drhodh(particles: ParticleSet, kernel: SPHKernel, domain: DomainDescription, neighborhood: SparseCOO):
    i = neighborhood.row
    j = neighborhood.col
    positions = (particles.positions, particles.positions)
    x_ij = mod_distance(get_i(positions, i), get_j(positions, j), domain.periodic, domain.min, domain.max)
    dWdh = kernel.dkdh(x_ij, particles.supports[i])
    
    summation = scatter_sum(dWdh * particles.masses[j], i, dim = 0, dim_size = particles.positions.shape[0])
    return summation

# this is just computeOmega
def dFdh(particles: ParticleSet, kernel: SPHKernel, domain: DomainDescription, neighborhood: SparseCOO):
    
    dh_drho = dhdrho(particles.supports, particles.densities, particles.positions.shape[1])
    drho_dh = drhodh(particles, kernel, domain, neighborhood)
    return 1 - dh_drho * drho_dh
    
# from sphMath.ghost import filterNeighborsGhost, LiuLiuFirstOrder
from sphMath.schemes.gasDynamics import checkTensor
from sphMath.neighborhood import filterNeighborhoodByKind
from sphMath.kernels import getSPHKernelv2
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme, NeighborhoodInformation, evaluateNeighborhood
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict
from sphMath.util import getSetConfig
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState

def evaluateOptimalSupport(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel_: SPHKernel,
        neighborhood: NeighborhoodInformation = None,
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}
        ):
    kernel = getSPHKernelv2(kernel_)
    rhos = [particles.densities]
    hs = [particles.supports]
    neighborhood_ = neighborhood
    verletScale = 1.4 if neighborhood is None else neighborhood.verletScale
    nIter = getSetConfig(config, 'support', 'iterations', 16)
    adaptiveHThreshold = getSetConfig(config, 'support', 'adaptiveHThreshold', 1e-3)
    supportMode = 'superSymmetric' if neighborhood is None else neighborhood.mode
    if supportMode not in ['gather', 'superSymmetric']:
        supportMode = 'superSymmetric'
    # verletScale = 1.0 if config is None else (config['neighborhood']['verletScale'] if 'verletScale' in config else 1.0)

    hMin = particles.supports.min()
    hMax = particles.supports.max()
    for i in range(nIter):
        with record_function(f"[SPH] - Optimal Support - Neighbors"):
            # print(f'Iteration {i} | neighborhood: {neighborhood_ is not None}')
            neighborhood, neighbors = evaluateNeighborhood(particles, config['domain'], kernel_, verletScale = 1.0, mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
            # numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries
            # neighborhood_, sparseNeighborhood = buildNeighborhood(particles, particles, config['domain'], verletScale= verletScale, mode =supportMode, priorNeighborhood=neighborhood_, verbose = False, neighborhoodAlgorithm = config['neighborhood']['algorithm'])
            # actualNeighbors = filterNeighborhood(neighborhood_)
            # actualNeighbors = sparseNeighborhood
            # actualNeighbors = filterNeighborhoodByKind(particles, actualNeighbors)
            # actualNeighbors, _ = filterNeighborsGhost(neighborhood_, particles, ghostState)
        
        if isinstance(particles, Tuple):
            particles.densities = computeDensity(particles, kernel_, neighbors.get('noghost'), supportScheme, config)
            
            # particles = particles._replace(densities = computeDensity(particles, particles, domain, kernel, actualNeighbors, 'gather'))
        else:
            particles.densities = computeDensity(particles, kernel_, neighbors.get('noghost'), supportScheme, config)#computeDensity(particles, particles, domain, kernel, actualNeighbors, 'gather')
            # if ghostState is not None:
                # particles.densities = LiuLiuFirstOrder(particles.densities, particles, ghostState, domain, kernel)
            
        h_prev = particles.supports
        
        F_ = F(h_prev, particles.densities, particles.masses, config['targetNeighbors'], dim = particles.positions.shape[1])
        # dFdh_ = dFdh(particles, kernel, domain, actualNeighbors)
        dFdh_ = computeOmega(particles, kernel_, neighbors.get('noghost'), supportScheme, config)
        
        h_new = h_prev - F_ / (dFdh_ + 1e-6)
        # if ghostState is not None:
            # h_new[particles.species != 0] = h_prev[particles.species != 0]

        h_new = h_new.clamp(min = hMin * 0.25, max = hMax * 4.0)
        hMin = h_new.min()
        hMax = h_new.max()

        h_diff = h_new - h_prev
        h_ratio = h_new / (h_prev + 1e-6)
        
        verbose = False
        checkTensor(F_, particles.positions.dtype, particles.positions.device, 'F_', verbose = verbose)
        checkTensor(dFdh_, particles.positions.dtype, particles.positions.device, 'dFdh_', verbose = verbose)
        checkTensor(h_new, particles.positions.dtype, particles.positions.device, 'h_new', verbose = verbose)
        checkTensor(h_diff, particles.positions.dtype, particles.positions.device, 'h_diff', verbose = verbose)
        checkTensor(h_ratio, particles.positions.dtype, particles.positions.device, 'h_ratio', verbose = verbose)
        checkTensor(particles.densities, particles.positions.dtype, particles.positions.device, 'densities', verbose = verbose)
        
        # print(f'Support Update: {h_diff.min()} | {h_diff.max()} | {h_diff.mean()}')
        
        # print(f'Iteration: {i} | h_ratio: {h_ratio.min()} | {h_ratio.max()} | {h_ratio.mean()}')
        # print(f'Densities: {particles.densities.min()} | {particles.densities.max()} | {particles.densities.mean()}')
        # print(f'Supports: {h_new.min()} | {h_new.max()} | {h_new.mean()}')

        # rhos.append(particles.densities)
        # hs.append(h_new)
        
        # actualNeighbors = buildSuperSymmetricNeighborhood(particles, domain, verletScale = 1.0)
        # rho = computeDensity(particles, particles, domain, kernel, actualNeighbors, 'gather' )
        # V = particles.masses / rho
        # h = targetNeighbors / 2 * V
        
        # print(f'Iteration {i} | Support: {h.min()} | {h.max()} | {h.mean()} | Ratio: {(h / hs[0]).min()} | {(h / hs[0]).max()} | {(h / hs[0]).mean()}')
        
        
        if isinstance(particles, Tuple):
            particles = particles._replace(supports = h_new)
        else:
            particles.supports = h_new
            # particles.densities = rho

        # particles = particles._replace(supports = h, densities = rho)
        rhos.append(particles.densities)
        hs.append(particles.supports)
        
        # print(f'Iteration {i} | h_ratio: {h_ratio.min()} | {h_ratio.max()} | {h_ratio.mean()}')
        
        if (h_ratio - 1).abs().max() < adaptiveHThreshold:
            # print('Stopping Early')
            break

    # if neighborhood is not None:
    return rhos[-1], hs[-1], rhos, hs, neighborhood_
    # return rhos[-1], hs[-1], rhos, hs