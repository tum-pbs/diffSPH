
# from sphMath.neighborhood import SparseNeighborhood
# import torch
# from sphMath.schemes.gasDynamics import CompressibleState, CompressibleSystem
# from sphMath.neighborhood import DomainDescription
# from sphMath.neighborhood import SparseNeighborhood
# from sphMath.neighborhood import PointCloud
# import copy
# from typing import Callable

# def maskNeighborhoodSpecies(particles : CompressibleState, neighborhood: SparseNeighborhood, species_a: int, species_b: int):
#     spreadSpecies_i = particles.species[neighborhood.row] == species_a
#     spreadSpecies_j = particles.species[neighborhood.col] == species_b
    
#     num_i = torch.sum(particles.species == species_a)
#     num_j = torch.sum(particles.species == species_b)
    
#     return SparseNeighborhood(
#         row = neighborhood.row[torch.logical_and(spreadSpecies_i, spreadSpecies_j)],
#         col = neighborhood.col[torch.logical_and(spreadSpecies_i, spreadSpecies_j)],
        
#         numRows = neighborhood.numRows,
#         numCols= neighborhood.numCols,
        
#         # numRows = num_i,
#         # numCols = num_j,
        
#         points_a = neighborhood.points_a,
#         points_b = neighborhood.points_b,
        
#         # points_a = PointCloud(positions = neighborhood.points_a.positions[particles.species == species_a], supports = neighborhood.points_a.supports[particles.species == species_a]),
#         # points_b = PointCloud(positions = neighborhood.points_b.positions[particles.species == species_b], supports = neighborhood.points_b.supports[particles.species == species_b]),
        
#         domain = neighborhood.domain
#     )
# def maskNeighborhoodNonGhost(particles : CompressibleState, neighborhood: SparseNeighborhood):
#     spreadSpecies_i = particles.species[neighborhood.row] >= 0
#     spreadSpecies_j = particles.species[neighborhood.col] >= 0
    
#     num_i = torch.sum(particles.species >= 0)
#     num_j = torch.sum(particles.species >= 0)
    
#     return SparseNeighborhood(
#         row = neighborhood.row[torch.logical_and(spreadSpecies_i, spreadSpecies_j)],
#         col = neighborhood.col[torch.logical_and(spreadSpecies_i, spreadSpecies_j)],
        
#         numRows = neighborhood.numRows,
#         numCols= neighborhood.numCols,
        
#         # numRows = num_i,
#         # numCols = num_j,
        
#         points_a = neighborhood.points_a,
#         points_b = neighborhood.points_b,
        
#         # points_a = PointCloud(positions = neighborhood.points_a.positions[particles.species >= 0], supports = neighborhood.points_a.supports[particles.species >= 0]),
#         # points_b = PointCloud(positions = neighborhood.points_b.positions[particles.species >= 0], supports = neighborhood.points_b.supports[particles.species >= 0]),
        
#         domain = neighborhood.domain
#     )
    
    

# from sphMath.schemes.gasDynamics import CompressibleState
# from dataclasses import dataclass, fields
# from sphMath.finiteDifference import continuousGradient, centralDifferenceStencil

# def sdBox(p, b):
#     d = torch.abs(p) - b
#     return torch.linalg.norm(d.clamp(min = 0), dim = -1) + torch.max(d, dim = -1)[0].clamp(max=0)

# def sdBoxNormal(p, b):
#     d = torch.abs(p) - b
#     return torch.nn.functional.normalize(d, p = 2, dim = -1)

# def computeDomainDistances(particles, domain: DomainDescription, h = 1e-3):

#     # dist_left = particles.positions - domain.min
#     # dist_right = domain.max - particles.positions

#     # print(dist_left, dist_right)

#     # dist = torch.min(dist_left, dist_right)

#     # dist = torch.where(dist_left.abs() < dist_right.abs(), dist_left, dist_right)
#     # dist = torch.where(dist_left.abs() < dist_right.abs(), torch.ones_like(dist), torch.zeros_like(dist))
#     sdf = lambda x: sdBox(x - (domain.min + domain.max) / 2, (domain.max - domain.min) / 2)
#     dist = sdf(particles.positions)
    
#     stencil, stencil_offsets = centralDifferenceStencil(1,2)#.to(device = particles.positions.device, dtype = particles.positions.dtype)
#     stencil = stencil.to(device = particles.positions.device, dtype = particles.positions.dtype)
#     stencil_offsets = stencil_offsets.to(device = particles.positions.device, dtype = particles.positions.dtype)

#     normal = continuousGradient(sdf, particles.positions, stencil, stencil_offsets, h, 1)
#     normal = torch.nn.functional.normalize(normal, p = 2, dim = -1)
    
#     # print(dist.shape, normal.shape)

#     return dist, normal
#     return torch.linalg.norm(dist, dim = -1), dist #torch.nn.functional.normalize(dist, p = 2, dim = -1)


# def filterParticles(particles, mask):
#     filteredParticles = copy.deepcopy(particles)
#     for attr in fields(particles):
#         # print(attr.name)
#         if hasattr(particles, attr.name) and getattr(particles, attr.name) is not None:
#             setattr(filteredParticles, attr.name, getattr(particles, attr.name)[mask])
#     return filteredParticles
# def appendParticles(particles, newParticles):
#     combinedParticles = copy.deepcopy(particles)
    
#     for attr in fields(particles):
#         try:
#             if hasattr(particles, attr.name) and getattr(particles, attr.name) is not None:
#                 if getattr(newParticles, attr.name) is None:
#                     raise ValueError(f'Attribute {attr.name} is None in newParticles')
#                 setattr(combinedParticles, attr.name, torch.cat((getattr(particles, attr.name), getattr(newParticles, attr.name)), dim = 0))
#             elif hasattr(particles, attr.name) and (getattr(newParticles, attr.name) is not None and getattr(particles, attr.name) is None):
#                 raise ValueError(f'Attribute {attr.name} is None in particles')
#         except Exception as e:
#             print(f'Error in {attr.name}')
#             if hasattr(particles, attr.name):
#                 print(getattr(particles, attr.name))
#             if hasattr(newParticles, attr.name):
#                 print(getattr(newParticles, attr.name))
#             print(e)
#             raise e
#     return combinedParticles


# def getBufferParticlesLeft(system : CompressibleSystem, bufferWidth):
#     dxLeft = system.systemState.positions[1,0] - system.systemState.positions[0,0]
#     xLeft = system.systemState.positions[0,0] - dxLeft / 2
#     leftBuffer = torch.linspace(xLeft - dxLeft / 2, xLeft - dxLeft / 2 - (bufferWidth - 1) * dxLeft, bufferWidth, device = system.systemState.positions.device, dtype = system.systemState.positions.dtype).view(-1, 1)
    
#     # print((leftBuffer[1] - leftBuffer[0]) + dxLeft)
    
#     leftState = CompressibleState(
#         positions = leftBuffer,
#         supports = torch.ones_like(leftBuffer[:,0]) * system.systemState.supports[0],
#         masses = torch.ones_like(leftBuffer[:,0]) * system.systemState.masses[0],
#         densities= torch.ones_like(leftBuffer[:,0]) * system.systemState.densities[0],
#         velocities = torch.ones_like(leftBuffer) * system.systemState.velocities[0],
        
#         internalEnergies= torch.ones_like(leftBuffer[:,0]) * system.systemState.internalEnergies[0],
#         totalEnergies= torch.ones_like(leftBuffer[:,0]) * system.systemState.totalEnergies[0],
#         entropies= torch.ones_like(leftBuffer[:,0]) * system.systemState.entropies[0],
#         pressures= torch.ones_like(leftBuffer[:,0]) * system.systemState.pressures[0],
#         soundspeeds = torch.ones_like(leftBuffer[:,0]) * system.systemState.soundspeeds[0],
#         alphas = torch.ones_like(leftBuffer[:,0]) * system.systemState.alphas[0] if system.systemState.alphas is not None else None,
#         alpha0s = torch.ones_like(leftBuffer[:,0]) * system.systemState.alpha0s[0] if system.systemState.alpha0s is not None else None,
#         species = torch.ones_like(leftBuffer[:,0])
#     )
#     return leftState
# def getBufferParticlesRight(system: CompressibleSystem, bufferWidth):
#     dxRight = system.systemState.positions[-1,0] - system.systemState.positions[-2,0]
#     xRight = system.systemState.positions[-1,0] + dxRight / 2
#     rightBuffer = torch.linspace(xRight + dxRight / 2, xRight + dxRight / 2 + (bufferWidth-1) * dxRight, bufferWidth, device = system.systemState.positions.device, dtype = system.systemState.positions.dtype).view(-1, 1)
#     # print((rightBuffer[1] - rightBuffer[0]) - dxRight)
    
#     rightState = CompressibleState(
#         positions = rightBuffer,
#         supports = torch.ones_like(rightBuffer[:,0]) * system.systemState.supports[-1],
#         masses = torch.ones_like(rightBuffer[:,0]) * system.systemState.masses[-1],
#         densities= torch.ones_like(rightBuffer[:,0]) * system.systemState.densities[-1],
#         velocities = torch.ones_like(rightBuffer) * system.systemState.velocities[-1],
        
#         internalEnergies= torch.ones_like(rightBuffer[:,0]) * system.systemState.internalEnergies[-1],
#         totalEnergies= torch.ones_like(rightBuffer[:,0]) * system.systemState.totalEnergies[-1],
#         entropies= torch.ones_like(rightBuffer[:,0]) * system.systemState.entropies[-1],
#         pressures= torch.ones_like(rightBuffer[:,0]) * system.systemState.pressures[-1],
#         soundspeeds = torch.ones_like(rightBuffer[:,0]) * system.systemState.soundspeeds[-1],
#         alphas = torch.ones_like(rightBuffer[:,0]) * system.systemState.alphas[-1] if system.systemState.alphas is not None else None,
#         alpha0s = torch.ones_like(rightBuffer[:,0]) * system.systemState.alpha0s[-1] if system.systemState.alpha0s is not None else None,
#         species = torch.ones_like(rightBuffer[:,0])
#     )
#     return rightState

# from sphMath.schemes.gasDynamics import GhostParticleState

# def getBufferParticles(system: CompressibleSystem, bufferWidth, domain):
#     leftState = getBufferParticlesLeft(system, bufferWidth)
#     rightState = getBufferParticlesRight(system, bufferWidth)
#     mergedParticles = appendParticles(leftState, rightState)
    
#     r, n = computeDomainDistances(mergedParticles, domain)
    
#     ghostParticles = GhostParticleState(
#         positions = mergedParticles.positions,
#         masses = mergedParticles.masses,
#         supports = mergedParticles.supports,
        
#         velocities = mergedParticles.velocities,
#         densities = mergedParticles.densities,
        
#         boundaryDistances = r,
#         boundaryDirections= n,
#         ghostPositions = mergedParticles.positions - 2 * r[:, None] * n,
        
#         referenceParticleIndex=None,
#         kind = 'reflecting'
#     )
    
#     ghostState = CompressibleState(
#         positions = ghostParticles.ghostPositions,
#         supports = mergedParticles.supports,
#         masses = mergedParticles.masses,
#         densities= mergedParticles.densities,
#         velocities = mergedParticles.velocities,
        
#         internalEnergies= mergedParticles.internalEnergies,
#         totalEnergies= mergedParticles.totalEnergies,
#         entropies= mergedParticles.entropies,
#         pressures= mergedParticles.pressures,
#         soundspeeds = mergedParticles.soundspeeds,
#         alphas = mergedParticles.alphas,
#         alpha0s = mergedParticles.alpha0s,
#         species = mergedParticles.species * (-1)
#     )
    
    
#     return appendParticles(mergedParticles, ghostState), ghostParticles
    
    
# def getBufferParticles2(system: CompressibleSystem, bufferWidth, domain):    
#     r, n = computeDomainDistances(system.systemState, domain)
    
#     toFlip = r.abs() < bufferWidth
    
#     mergedParticles = CompressibleState(
#         positions = system.systemState.positions[toFlip] + 2 * r[toFlip][:, None] * n[toFlip],
#         masses = system.systemState.masses[toFlip],
#         supports = system.systemState.supports[toFlip],
        
#         velocities = system.systemState.velocities[toFlip],
#         densities = system.systemState.densities[toFlip],
        
#         internalEnergies = system.systemState.internalEnergies[toFlip],
#         totalEnergies = system.systemState.totalEnergies[toFlip],
#         entropies = system.systemState.entropies[toFlip],
#         pressures = system.systemState.pressures[toFlip],
#         soundspeeds = system.systemState.soundspeeds[toFlip],
#         alphas = system.systemState.alphas[toFlip] if system.systemState.alphas is not None else None,
#         alpha0s = system.systemState.alpha0s[toFlip] if system.systemState.alpha0s is not None else None,
#         species = system.systemState.species[toFlip]
#     )
    
#     ghostParticles = GhostParticleState(
#         positions = mergedParticles.positions,
#         masses = mergedParticles.masses,
#         supports = mergedParticles.supports,
        
#         velocities = mergedParticles.velocities,
#         densities = mergedParticles.densities,
        
#         boundaryDistances = r,
#         boundaryDirections= n,
#         ghostPositions = system.systemState.positions[toFlip],
        
#         referenceParticleIndex=None,
#         kind = 'reflecting'
#     )
    
#     ghostState = CompressibleState(
#         positions = ghostParticles.ghostPositions,
#         supports = mergedParticles.supports,
#         masses = mergedParticles.masses,
#         densities= mergedParticles.densities,
#         velocities = mergedParticles.velocities,
        
#         internalEnergies= mergedParticles.internalEnergies,
#         totalEnergies= mergedParticles.totalEnergies,
#         entropies= mergedParticles.entropies,
#         pressures= mergedParticles.pressures,
#         soundspeeds = mergedParticles.soundspeeds,
#         alphas = mergedParticles.alphas,
#         alpha0s = mergedParticles.alpha0s,
#         species = mergedParticles.species * (-1)
#     )
    
#     return mergedParticles, ghostParticles

# from sphMath.neighborhood import filterNeighborhood
# def filterNeighborsGhost(neighborhood, systemState, ghostState):
#     actualNeighborsAll = filterNeighborhood(neighborhood)

#     if ghostState is not None:
#         ghostState.particleToGhostNeighborhood = maskNeighborhoodSpecies(systemState, actualNeighborsAll, -1, 0)
#         ghostState.particleToBoundaryNeighborhood = maskNeighborhoodSpecies(systemState, actualNeighborsAll, 1, 0)
#         ghostState.boundaryToParticleNeighborhood = maskNeighborhoodSpecies(systemState, actualNeighborsAll, 0, 1)
#         ghostState.boundaryToBoundaryNeighborhood = maskNeighborhoodSpecies(systemState, actualNeighborsAll, 1, 1)
        
#         actualNeighbors = maskNeighborhoodNonGhost(systemState, actualNeighborsAll)
#         return actualNeighbors, ghostState
#     else:
#         return actualNeighborsAll, None


# from sphMath.operations import sph_op
# from sphMath.neighborhood import computeDistanceTensor

# def adjunctMatrix(M, c, i):
#     res = torch.empty_like(M)

#     for j in range(c.shape[1]):
#         res[:,:,j] = c if j == i else M[:,:,j]
#     return res

# def LiuLiuFirstOrder(quantity, particles: CompressibleState, ghostParticles: GhostParticleState, domain: DomainDescription, wrappedKernel, kind_ = None):
#     if ghostParticles is None:
#         return quantity
#     if len(quantity.shape) > 1:
#         flattenedQuantity = quantity.view(quantity.shape[0], -1)
#         for i in range(flattenedQuantity.shape[1]):
#             flattenedQuantity[:,i] = LiuLiuFirstOrder(flattenedQuantity[:,i], particles, ghostParticles, domain, wrappedKernel, kind_)
#         return flattenedQuantity.view(quantity.shape)
        
        
#     # particles=  combinedSystem.systemState
#     # ghostParticles = combinedSystem.ghostState
#     # quantity = torch.sin(particles.positions[:,0] * 2 * np.pi)


#     b_scalar = sph_op(particles, particles, domain, wrappedKernel, ghostParticles.particleToGhostNeighborhood, 'gather', operation = 'interpolate', quantity=(None, quantity))
#     b_grad = sph_op(particles, particles, domain, wrappedKernel, ghostParticles.particleToGhostNeighborhood, 'gather', operation = 'gradient', quantity=(None, quantity), gradientMode = 'naive')

#     b = torch.cat([b_scalar.view(-1,1), b_grad], dim = 1)

#     rij, xij = computeDistanceTensor(ghostParticles.particleToGhostNeighborhood, normalize=False)

#     q = (torch.ones_like(quantity), torch.ones_like(quantity))
#     M_0 = sph_op(particles, particles, domain, wrappedKernel, ghostParticles.particleToGhostNeighborhood, 'gather', operation = 'interpolate', quantity=q)
#     M_grad = sph_op(particles, particles, domain, wrappedKernel, ghostParticles.particleToGhostNeighborhood, 'gather', operation = 'gradient', quantity=q, gradientMode = 'naive')

#     M_x = sph_op(particles, particles, domain, wrappedKernel, ghostParticles.particleToGhostNeighborhood, 'gather', operation = 'interpolate', quantity=xij)
#     M_x_grad = sph_op(particles, particles, domain, wrappedKernel, ghostParticles.particleToGhostNeighborhood, 'gather', operation = 'gradient', quantity=xij, gradientMode = 'naive')


#     M = []
#     M.append(torch.cat([M_0.view(-1,1), M_x], dim = 1))
#     for i in range(M_grad.shape[1]):
#         M.append(torch.cat([M_grad[:,i].view(-1,1), M_x_grad[:,i,:]], dim = 1))
#     M = torch.stack([row.unsqueeze(2) for row in M], dim = 2)[:,:,:,0].mT

#     M = M[particles.species < 0]
#     b = b[particles.species < 0]

#     d0 = torch.linalg.det(M)
#     adjunctMatrices = [torch.linalg.det(adjunctMatrix(M, b, i)) for i in range(M.shape[1])]
#     solution = torch.stack([adj/(d0 + 1e-7) for adj in adjunctMatrices], dim = 1)

#     extrapolatedQuantity = quantity.clone()
#     extrapolatedQuantity[particles.species < 0] = solution[:,0]

#     kind = ghostParticles.kind if kind_ is None else kind_

#     if kind == 'reflecting':
#         extrapolatedQuantity[ghostParticles.referenceParticleIndex] = solution[:,0]# - 2 * torch.einsum('ij,ij->i', solution[:,1:], ghostParticles.boundaryDirections) * ghostParticles.boundaryDistances
#     elif kind == 'open':
#         extrapolatedQuantity[ghostParticles.referenceParticleIndex] = solution[:,0] - 2* torch.einsum('ij,ij->i', solution[:,1:], ghostParticles.boundaryDirections) * ghostParticles.boundaryDistances
#     else:
#         raise ValueError(f'Unknown ghost particle kind {ghostParticles.kind}')
#     return extrapolatedQuantity