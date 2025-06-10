from sphMath.kernels import SPHKernel
from sphMath.neighborhood import computeDistanceTensor

from dataclasses import dataclass
from typing import Callable
import torch
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood
from sphMath.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr
from sphMath.util import domainSDF, sampleDomainSDF
from sphMath.operations import sph_op
from sphMath.rigidBody import RigidBody, getTransformationMatrix

def addGhostParticle(particleState: CompressibleState, domainBody: RigidBody):
    device = particleState.positions.device
    dtype = particleState.positions.dtype
    mergedPositions = torch.cat([particleState.positions, domainBody.ghostParticlePositions], dim = 0)
    mergedKinds = torch.cat([particleState.kinds, torch.ones(domainBody.ghostParticlePositions.shape[0], device = device, dtype = torch.int32) * 2], dim = 0)
    mergedUIDs = torch.cat([particleState.UIDs, domainBody.ghostParticleUIDs], dim = 0)
    
    ghostIndices = particleState.ghostIndices if particleState.ghostIndices is not None else torch.ones_like(particleState.positions[:, 0], device = device, dtype = torch.int32) * -1
    
    boundaryIndices = domainBody.particleIndices
    ghostIndices = particleState.ghostIndices if particleState.ghostIndices is not None else torch.ones_like(particleState.positions[:, 0], device = device, dtype = torch.int32) * -1
    # ghostIndices = torch.cat([ghostIndices, boundaryIndices], dim = 0)
    ghostIndices[boundaryIndices] = torch.arange(boundaryIndices.shape[0], device = device, dtype = torch.int32) + particleState.positions.shape[0]
    # ghostIndices[particleState.positions.shape[0]:] = boundaryIndices
    ghostIndices = torch.cat([ghostIndices, boundaryIndices], dim = 0)
    
    ghostOffsets = particleState.ghostOffsets if particleState.ghostOffsets is not None else torch.zeros_like(particleState.positions)
    offsets = domainBody.particleBoundaryDistances.view(-1, 1) * domainBody.particleBoundaryNormals
    ghostOffsets = torch.cat([ghostOffsets, offsets], dim = 0)
    ghostOffsets[boundaryIndices] = -offsets
    
    return CompressibleState(
        positions = mergedPositions,
        supports = torch.cat([particleState.supports, particleState.supports[boundaryIndices]], dim = 0),
        masses = torch.cat([particleState.masses, particleState.masses[boundaryIndices]], dim = 0),
        densities = torch.cat([particleState.densities, particleState.densities[boundaryIndices]], dim = 0),
        velocities = torch.cat([particleState.velocities, particleState.velocities[boundaryIndices]], dim = 0),
        
        kinds = mergedKinds,
        materials = torch.cat([particleState.materials, particleState.materials[boundaryIndices]], dim = 0),
        UIDs = mergedUIDs,
        
        internalEnergies = torch.cat([particleState.internalEnergies, particleState.internalEnergies[boundaryIndices]], dim = 0),
        totalEnergies = torch.cat([particleState.totalEnergies, particleState.totalEnergies[boundaryIndices]], dim = 0),
        entropies = torch.cat([particleState.entropies, particleState.entropies[boundaryIndices]], dim = 0),
        pressures = torch.cat([particleState.pressures, particleState.pressures[boundaryIndices]], dim = 0),
        soundspeeds = torch.cat([particleState.soundspeeds, particleState.soundspeeds[boundaryIndices]], dim = 0),
        
        alphas = torch.cat([particleState.alphas, particleState.alphas[boundaryIndices]], dim = 0) if particleState.alphas is not None else None,
        alpha0s = torch.cat([particleState.alpha0s, particleState.alpha0s[boundaryIndices]], dim = 0) if particleState.alpha0s is not None else None,
        divergence= torch.cat([particleState.divergence, particleState.divergence[boundaryIndices]], dim = 0) if particleState.divergence is not None else None,

        ghostIndices= ghostIndices,
        ghostOffsets= ghostOffsets,
    )
    
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
def addGhostParticleWeaklyCompressible(particleState: WeaklyCompressibleState, domainBody: RigidBody):
    device = particleState.positions.device
    dtype = particleState.positions.dtype
    mergedPositions = torch.cat([particleState.positions, domainBody.ghostParticlePositions], dim = 0)
    mergedKinds = torch.cat([particleState.kinds, torch.ones(domainBody.ghostParticlePositions.shape[0], device = device, dtype = torch.int32) * 2], dim = 0)
    mergedUIDs = torch.cat([particleState.UIDs, domainBody.ghostParticleUIDs], dim = 0)
    
    ghostIndices = particleState.ghostIndices if particleState.ghostIndices is not None else torch.ones_like(particleState.positions[:, 0], device = device, dtype = torch.int32) * -1
    
    boundaryIndices = domainBody.particleIndices
    ghostIndices = particleState.ghostIndices if particleState.ghostIndices is not None else torch.ones_like(particleState.positions[:, 0], device = device, dtype = torch.int32) * -1
    # ghostIndices = torch.cat([ghostIndices, boundaryIndices], dim = 0)
    ghostIndices[boundaryIndices] = torch.arange(boundaryIndices.shape[0], device = device, dtype = torch.int32) + particleState.positions.shape[0]
    # ghostIndices[particleState.positions.shape[0]:] = boundaryIndices
    ghostIndices = torch.cat([ghostIndices, boundaryIndices], dim = 0)
    
    ghostOffsets = particleState.ghostOffsets if particleState.ghostOffsets is not None else torch.zeros_like(particleState.positions)
    offsets = domainBody.particleBoundaryDistances.view(-1, 1) * domainBody.particleBoundaryNormals
    ghostOffsets = torch.cat([ghostOffsets, offsets], dim = 0)
    ghostOffsets[boundaryIndices] = -offsets
    
    return WeaklyCompressibleState(
        positions = mergedPositions,
        supports = torch.cat([particleState.supports, particleState.supports[boundaryIndices]], dim = 0),
        masses = torch.cat([particleState.masses, particleState.masses[boundaryIndices]], dim = 0),
        densities = torch.cat([particleState.densities, particleState.densities[boundaryIndices]], dim = 0),
        velocities = torch.cat([particleState.velocities, particleState.velocities[boundaryIndices]], dim = 0),
        
        pressures = torch.cat([particleState.pressures, particleState.pressures[boundaryIndices]], dim = 0),
        soundspeeds = torch.cat([particleState.soundspeeds, particleState.soundspeeds[boundaryIndices]], dim = 0),
        
        kinds = mergedKinds,
        materials = torch.cat([particleState.materials, particleState.materials[boundaryIndices]], dim = 0),
        UIDs = mergedUIDs,
        
        covarianceMatrices = torch.cat([particleState.covarianceMatrices, particleState.covarianceMatrices[boundaryIndices]], dim = 0) if particleState.covarianceMatrices is not None else None,
        gradCorrectionMatrices = torch.cat([particleState.gradCorrectionMatrices, particleState.gradCorrectionMatrices[boundaryIndices]], dim = 0) if particleState.gradCorrectionMatrices is not None else None,
        eigenValues = torch.cat([particleState.eigenValues, particleState.eigenValues[boundaryIndices]], dim = 0) if particleState.eigenValues is not None else None,
        
        gradRho = torch.cat([particleState.gradRho, particleState.gradRho[boundaryIndices]], dim = 0) if particleState.gradRho is not None else None,
        gradRhoL = torch.cat([particleState.gradRhoL, particleState.gradRhoL[boundaryIndices]], dim = 0) if particleState.gradRhoL is not None else None,
        
        surfaceMask = torch.cat([particleState.surfaceMask, particleState.surfaceMask[boundaryIndices]], dim = 0) if particleState.surfaceMask is not None else None,
        surfaceNormals = torch.cat([particleState.surfaceNormals, particleState.surfaceNormals[boundaryIndices]], dim = 0) if particleState.surfaceNormals is not None else None,
        
        ghostIndices= ghostIndices,
        ghostOffsets= ghostOffsets,
    )

def adjunctMatrix(M: torch.Tensor, c: torch.Tensor, i: int):
    res = torch.empty_like(M)

    for j in range(c.shape[1]):
        res[:,:,j] = c if j == i else M[:,:,j]
    return res

from sphMath.neighborhood import SupportScheme, PrecomputedNeighborhood
from sphMath.operations import SPHOperationCompiled, Operation, GradientMode, access_optional
from typing import Tuple
# from sphMath.util import access_optional

@torch.jit.script
def LiuLiuGhostNodes(quantities: torch.Tensor, particles: WeaklyCompressibleState, neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood], num_nbrs: torch.Tensor, supportScheme: SupportScheme = SupportScheme.Scatter):
    
    b_scalar = SPHOperationCompiled(
        particles,
        quantity = quantities,
        neighborhood= neighborhood[0],
        kernelValues= neighborhood[1],
        operation = Operation.Interpolate,
        supportScheme= supportScheme,        
    )
    b_grad = SPHOperationCompiled(
        particles,
        quantity = quantities,
        neighborhood= neighborhood[0],
        kernelValues= neighborhood[1],
        operation = Operation.Gradient,
        gradientMode = GradientMode.Naive,
        supportScheme= supportScheme,        
    )
    
    b = torch.cat([b_scalar.view(-1,1), b_grad], dim = 1)

    # rij, xij = computeDistanceTensor(neighborhood, normalize=False)

    q = (torch.ones_like(quantities), torch.ones_like(quantities))
    
    M_0 = SPHOperationCompiled(
        particles,
        quantity=q,
        neighborhood= neighborhood[0],
        kernelValues= neighborhood[1],
        operation = Operation.Interpolate,
        supportScheme= supportScheme,
    )
    M_grad = SPHOperationCompiled(
        particles,
        quantity=q,
        neighborhood= neighborhood[0],
        kernelValues= neighborhood[1],
        operation = Operation.Gradient,
        gradientMode = GradientMode.Naive,
        supportScheme= supportScheme,
    )
    xij = neighborhood[1].x_ij

    M_x = SPHOperationCompiled(
        particles,
        quantity=xij,
        neighborhood= neighborhood[0],
        kernelValues= neighborhood[1],
        operation = Operation.Interpolate,
        supportScheme= supportScheme,
    )
    M_x_grad = SPHOperationCompiled(
        particles,
        quantity=xij,
        neighborhood= neighborhood[0],
        kernelValues= neighborhood[1],
        operation = Operation.Gradient,
        gradientMode = GradientMode.Naive,
        supportScheme= supportScheme,
    )
    
    MList = []
    MList.append(torch.cat([M_0.view(-1,1), M_x], dim = 1))
    for i in range(M_grad.shape[1]):
        MList.append(torch.cat([M_grad[:,i].view(-1,1), M_x_grad[:,i,:]], dim = 1))
        

    # print(M)

    M = torch.stack([row.unsqueeze(2) for row in MList], dim = 2)[:,:,:,0].mT
    
    # print(M.shape)
    # print(particleState.kinds.shape)
    ghostMask = particles.kinds == 2
    
    M = M[ghostMask,:]
    b = b[ghostMask,:]
    # solution = torch.linalg.solve(M, b)

    # d0 = torch.linalg.det(M)
    d0 = torch.linalg.det(M)
    adjunctMatrices = [torch.linalg.det(adjunctMatrix(M, b, i)) for i in range(M.shape[1])]
    solution = torch.stack([adj/(d0 + 1e-7) for adj in adjunctMatrices], dim = 1)
    
    # num_nbrs = coo_to_csr(neighborhood).rowEntries[ghostMask]
    solution[num_nbrs < 4, 0] = b[num_nbrs < 4, 0]
    solution[num_nbrs < 4, 1:] = 0
    
    solution[num_nbrs == 0,0] = quantities[access_optional(particles.ghostIndices, ghostMask)][num_nbrs == 0]
    solution[num_nbrs == 0,1:] = 0
    
    return solution


def evalGhostQuantity(quantities: torch.Tensor, particles: WeaklyCompressibleState, neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood], supportScheme: SupportScheme = SupportScheme.Scatter, kind: str = 'extend'):
    neighCounts = coo_to_csr(neighborhood[0]).rowEntries[particles.kinds == 2]
    qB = LiuLiuGhostNodes(quantities, particles, neighCounts, neighborhood, supportScheme)
    if kind == 'extend':
        qBoundary = qB[:,0] - torch.einsum('nd, nd -> n', qB[:,1:], particles.ghostOffsets[particles.kinds == 2])
    elif kind == 'project':
        qBoundary = qB[:,0]
    else:
        raise ValueError(f'kind = {kind} not recognized')
    bIndices = particles.ghostIndices[particles.kinds == 2]
    outQuantity = quantities.clone()
    outQuantity[bIndices] = qBoundary
    return outQuantity


from typing import Dict
from sphMath.neighborhood import SparseCOO
from typing import Union, Tuple, Dict
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme

def computeBoundaryVelocities(
    particleState: WeaklyCompressibleState, 
    kernel: SPHKernel, 
    neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
    supportScheme: SupportScheme = SupportScheme.Scatter,
    config: Dict = {}):
    
    if not torch.any(particleState.kinds == 1):
        return particleState.velocities
    device = particleState.positions.device
    dtype = particleState.positions.dtype
    
    materials = particleState.materials[particleState.kinds == 1]
    uniqueMaterials = torch.unique(materials)
    
    
    boundaryRegions = [region for region in config['regions'] if region['type'] == 'boundary']
    kinds = [region['kind'] for region in boundaryRegions]
    ghostMask = particleState.kinds == 2
    neighCounts = coo_to_csr(neighborhood[0]).rowEntries[ghostMask]

    if 'extend' in kinds or 'project' in kinds or 'free-slip' in kinds or 'no-slip' in kinds:        
        q_velsL = []
        for d in range(particleState.positions.shape[1]):
            q_velsL.append(LiuLiuGhostNodes(particleState.velocities[:,d], particleState, neighborhood, neighCounts, supportScheme))
        
        q_vels = torch.stack(q_velsL, dim = 1)
        ghostOffsets = particleState.ghostOffsets[particleState.kinds == 2]
        
    bIndices = particleState.ghostIndices[particleState.kinds == 2]
    
    boundaryVelocities = particleState.velocities.clone()
    
    for material in uniqueMaterials.detach().cpu().numpy():
        materialMask = torch.logical_and(particleState.kinds == 1, particleState.materials == material)        
        materialGhostMask = particleState.materials[bIndices] == material
        
        region = boundaryRegions[material]
        
        if region['kind'] == 'constant' or region['kind'] == 'none' or region['kind'] == 'driven':
            continue
        elif region['kind'] == 'extend':            
            projected_vels = q_vels[:,:,0] - torch.einsum('nid, nd -> ni', q_vels[:,:,1:], ghostOffsets)
            boundaryVelocities[materialMask,:] = projected_vels[materialGhostMask]
        elif region['kind'] == 'zero':
            boundaryVelocities[materialMask,:] = torch.zeros_like(boundaryVelocities[materialMask,:])
        elif region['kind'] == 'free-slip':
            projected_vels = q_vels[:,:,0]            
            normals = torch.nn.functional.normalize(ghostOffsets)
            projected_vels = projected_vels - torch.einsum('nd, nd -> n', projected_vels, normals).view(-1,1) * normals
            boundaryVelocities[materialMask,:] = projected_vels[materialGhostMask]
        elif region['kind'] == 'no-slip':
            projected_vels = q_vels[:,:,0]            
            normals = torch.nn.functional.normalize(ghostOffsets)
            projected_vels = projected_vels - torch.einsum('nd, nd -> n', projected_vels, normals).view(-1,1) * normals
            boundaryVelocities[materialMask,:] = -projected_vels[materialGhostMask]
        else:
            raise ValueError(f'Unknown boundary condition {region["kind"]}')
    
        
    return boundaryVelocities

from sphMath.neighborhood import ParticleSet
from sphMath.kernels import getSPHKernelv2


def LiuLiuGhostNodes2(quantities: torch.Tensor, ghostState : ParticleSet, particleState: WeaklyCompressibleState, domain: DomainDescription, kernel, neighborhood: SparseCOO, ghostIndices):
    wrappedKernel = getSPHKernelv2(kernel)
    b_scalar = sph_op(ghostState, particleState, domain, wrappedKernel, neighborhood, 'gather', operation = 'interpolate', quantity=(None, quantities))
    b_grad = sph_op(ghostState, particleState, domain, wrappedKernel, neighborhood, 'gather', operation = 'gradient', quantity=(None, quantities), gradientMode = 'naive')
    b = torch.cat([b_scalar.view(-1,1), b_grad], dim = 1)

    rij, xij = computeDistanceTensor(neighborhood, normalize=False)

    q = (torch.ones_like(quantities), torch.ones_like(quantities))
    M_0 = sph_op(ghostState, particleState, domain, wrappedKernel, neighborhood, 'gather', operation = 'interpolate', quantity=q)
    M_grad = sph_op(ghostState, particleState, domain, wrappedKernel, neighborhood, 'gather', operation = 'gradient', quantity=q, gradientMode = 'naive')

    M_x = sph_op(ghostState, particleState, domain, wrappedKernel, neighborhood, 'gather', operation = 'interpolate', quantity=xij)
    M_x_grad = sph_op(ghostState, particleState, domain, wrappedKernel, neighborhood, 'gather', operation = 'gradient', quantity=xij, gradientMode = 'naive')


    M = []
    M.append(torch.cat([M_0.view(-1,1), M_x], dim = 1))
    for i in range(M_grad.shape[1]):
        M.append(torch.cat([M_grad[:,i].view(-1,1), M_x_grad[:,i,:]], dim = 1))
        

    # print(M)

    M = torch.stack([row.unsqueeze(2) for row in M], dim = 2)[:,:,:,0].mT
    
    # print(M.shape)
    # print(particleState.kinds.shape)
    
    M = M[:ghostState.positions.shape[0]]
    b = b[:ghostState.positions.shape[0]]
    # solution = torch.linalg.solve(M, b)

    # d0 = torch.linalg.det(M)
    d0 = torch.linalg.det(M)
    adjunctMatrices = [torch.linalg.det(adjunctMatrix(M, b, i)) for i in range(M.shape[1])]
    solution = torch.stack([adj/(d0 + 1e-7) for adj in adjunctMatrices], dim = 1)
    
    # num_nbrs = coo_to_csr(neighborhood).rowEntries[:ghostState.positions.shape[0]]
    # solution[num_nbrs < 4, 0] = b[num_nbrs < 4, 0]
    # solution[num_nbrs < 4, 1:] = 0
    
    # solution[num_nbrs == 0,0] = quantities[ghostIndices][num_nbrs == 0]
    # solution[num_nbrs == 0,1:] = 0
    # solution[:,0] = num_nbrs
    return solution

def LiuLiuQ(quantity, ghostState, particleState, domain, wrappedKernel, sparseNeighborhood_, ghostIndices, offsets, mask):
    quantityFlatten = quantity.view(-1,1) if len(quantity.shape) == 1 else quantity.flatten(1)
    
    numNbrs = coo_to_csr(sparseNeighborhood_).rowEntries
    # liuMask = torch.logical_and(numNbrs > 4, mask)
    # ghostMask = (numNbrs > 4) 
    
    for d in range(quantityFlatten.shape[1]):
        q = quantityFlatten[:,d]
        qB = LiuLiuGhostNodes2(q, ghostState, particleState, domain, wrappedKernel, sparseNeighborhood_, ghostIndices)
        quantityFlatten[mask,d] = torch.where(numNbrs > 4, qB[:,0] - torch.einsum('nd, nd -> n', qB[:,1:], offsets), 
                                               quantityFlatten[mask,d])
    return quantityFlatten.reshape(quantity.shape)

