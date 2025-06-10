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

@dataclass(slots = True)
class RigidBody:
    centerOfMass: torch.Tensor
    orientation: torch.Tensor
    angularVelocity: torch.Tensor
    linearVelocity: torch.Tensor
    mass: torch.Tensor
    inertia: torch.Tensor
    
    # All in world coordinates
    particlePositions: torch.Tensor
    particleVelocities: torch.Tensor
    particleMasses: torch.Tensor
    particleUIDs: torch.Tensor
    particleIndices: torch.Tensor
    particleBoundaryDistances: torch.Tensor
    particleBoundaryNormals: torch.Tensor
    
    ghostParticlePositions: torch.Tensor
    ghostParticleIndices: torch.Tensor  
    ghostParticleUIDs: torch.Tensor
    ghostParticleBoundaryDistances: torch.Tensor
    ghostParticleBoundaryNormals: torch.Tensor
    
    
    sdf: Callable
    bodyID: int = 0  
    kind: str = 'constant'
    

def buildDomainBody(particleState : CompressibleState, domain: DomainDescription):
    d, n = sampleDomainSDF(particleState.positions, domain, invert = False)
    
    mask = d < 0
    device = domain.min.device
    dtype = domain.min.dtype
    
    positions = particleState.positions[mask]
    velocities = particleState.velocities[mask]
    masses = particleState.masses[mask]
    UIDs = particleState.UIDs[mask]
    indices = torch.arange(particleState.positions.shape[0], device = device, dtype = torch.int32)[mask]
    boundaryDistances = d[mask]
    boundaryNormals = torch.nn.functional.normalize(n[mask])
    
    sdf = lambda x: sampleDomainSDF(x, domain, invert = False)
    
    mass = torch.sum(masses)
    centerOfMass = torch.sum(positions * masses[:, None], dim = 0) / mass
    inertia = torch.sum(masses[:, None, None] * (positions - centerOfMass)[:, :, None] * (positions - centerOfMass)[:, None, :], dim = 0)
    angularVelocity = torch.zeros(positions.shape[1] - 1, device = device, dtype = dtype)
    linearVelocity = torch.zeros(positions.shape[1], device = device, dtype = dtype)
    orientation = torch.eye(positions.shape[1], device = device, dtype = dtype)
    
    ghostPositions = positions - 2 * boundaryDistances[:, None] * boundaryNormals
    ghostIndices = indices
    
    return RigidBody(
        centerOfMass = centerOfMass,
        orientation = orientation,
        angularVelocity = angularVelocity,
        linearVelocity = linearVelocity,
        mass = mass,
        inertia = inertia,
        
        particlePositions = positions,
        particleVelocities = velocities,
        particleMasses = masses,
        particleUIDs = UIDs,
        particleIndices = indices,
        particleBoundaryDistances = boundaryDistances,
        particleBoundaryNormals = boundaryNormals,
        
        ghostParticlePositions = ghostPositions,
        ghostParticleIndices = ghostIndices,       
        ghostParticleUIDs= -UIDs, 
        
        sdf = sdf,
    )

def buildRigidBody(particleState, config, bodyId):
    particleIndices = torch.logical_and(particleState.kinds == 1, particleState.materials == bodyId)
    ghostIndices = torch.logical_and(particleState.kinds == 2, particleState.materials == bodyId)

    boundaryRegions = [region for region in config['regions'] if region['type'] == 'boundary']
    currentRegion = boundaryRegions[bodyId]

    # print(torch.sum(particleIndices), torch.sum(ghostIndices))
    if(torch.sum(particleIndices) == 0):
        print('No particles in body', bodyId)
        return None

    masses = particleState.masses[particleIndices]
    positions = particleState.positions[particleIndices]
    device = particleState.positions.device
    dtype = particleState.positions.dtype

    ghostPositions = particleState.positions[ghostIndices]

    mass = torch.sum(masses)
    centerOfMass = torch.sum(masses.view(-1,1) * positions, dim = 0) / mass
    angularVelocity = torch.tensor(0.0, device = device, dtype = dtype)
    linearVelocity = torch.tensor([0.0, 0.0], device = device, dtype = dtype)
    inertia = torch.sum(masses * torch.linalg.norm(positions - centerOfMass, dim = 1)**2)
    orientation = torch.tensor(0.0, device = device, dtype = dtype)

    return RigidBody(
        centerOfMass=centerOfMass,
        orientation=orientation,
        angularVelocity=angularVelocity,
        linearVelocity=linearVelocity,
        mass=mass,
        inertia=inertia,

        particlePositions=positions - centerOfMass,
        ghostParticlePositions=ghostPositions - centerOfMass,
        particleVelocities=particleState.velocities[particleIndices],


        particleMasses = masses,
        particleUIDs = particleState.UIDs[particleIndices],
        ghostParticleUIDs = particleState.UIDs[ghostIndices],
        particleIndices = particleIndices,
        ghostParticleIndices=ghostIndices,

        particleBoundaryDistances=torch.linalg.norm(particleState.ghostOffsets[particleIndices], dim = -1),
        ghostParticleBoundaryDistances=torch.linalg.norm(particleState.ghostOffsets[ghostIndices], dim = -1),
        particleBoundaryNormals=particleState.ghostOffsets[particleIndices],
        ghostParticleBoundaryNormals=particleState.ghostOffsets[ghostIndices],

        bodyID=bodyId,
        sdf = currentRegion['sdf'],
        kind= currentRegion['kind'],
    )

def getTransformationMatrix(rigidBody : RigidBody):
    device = rigidBody.centerOfMass.device
    dtype = rigidBody.centerOfMass.dtype
    I = torch.eye(3, device = device, dtype = dtype)
    R = torch.tensor([[torch.cos(rigidBody.orientation), -torch.sin(rigidBody.orientation), 0],
                      [torch.sin(rigidBody.orientation), torch.cos(rigidBody.orientation), 0],
                      [0, 0, 1]], device = device, dtype = dtype)
    T = torch.tensor([[1, 0, rigidBody.centerOfMass[0]],
                      [0, 1, rigidBody.centerOfMass[1]],
                      [0, 0, 1]], device = device, dtype = dtype)
    return T @ R @ I


def integrateRigidBody(rigidBody, dudt, dwdt, dt):
    rigidBody.angularVelocity += dwdt * dt
    rigidBody.linearVelocity += dudt * dt
    
    rigidBody.orientation += rigidBody.angularVelocity * dt
    rigidBody.centerOfMass += rigidBody.linearVelocity * dt

    return rigidBody
