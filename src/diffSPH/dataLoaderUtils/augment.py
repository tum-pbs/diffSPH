import torch
from dataclasses import dataclass
from typing import Union, List  
from diffSPH.dataLoaderUtils.neighborhood import DomainDescription, AugmentedDomainDescription

from diffSPH.dataLoaderUtils.neighborhood import AugmentedDomainDescription, DomainDescription
from typing import Union
from typing import List
from diffSPH.dataLoaderUtils.state import PointCloudWithKinds
from diffSPH.dataLoaderUtils.neighborhood import batchNeighborsearch
from diffSPH.dataLoaderUtils.batch import mergeTrajectoryStates, mergeBatch
from diffSPH.dataLoaderUtils.loader import loadBatch

import numpy as np
from diffSPH.dataLoaderUtils.util import buildRotationMatrix
     
def getRotationMatrix(domain : AugmentedDomainDescription):
    if len(domain.angles) == 1:
        return buildRotationMatrix(0, domain.dim, device=domain.device, dtype=domain.dtype)
    elif len(domain.angles) == 2:
        return buildRotationMatrix(domain.angles, domain.dim, device=domain.device, dtype=domain.dtype)
    elif len(domain.angles) == 3:
        return buildRotationMatrix(domain.angles, domain.dim, device=domain.device, dtype=domain.dtype)
    else:
        raise ValueError(f"Unsupported number of angles: {len(domain.angles)}")
    
def getRandomRotationMatrix(domain : Union[DomainDescription, AugmentedDomainDescription], device: torch.device = None, dtype: torch.dtype = None):
    device_ = domain.device if isinstance(domain, AugmentedDomainDescription) else domain.min.device
    dtype_ = domain.dtype if isinstance(domain, AugmentedDomainDescription) else domain.min.dtype

    device_ = device if device is not None else device_
    dtype_ = dtype if dtype is not None else dtype_

    print(f"Using device: {device_}, dtype: {dtype_}")
    angles = (torch.rand(domain.dim - 1, device=device_, dtype=dtype_) - 0.5) * 2 * np.pi
    return buildRotationMatrix(angles, domain.dim, device=device_, dtype=dtype_), angles



def anglesFromRotationMatrix(rotMat: torch.Tensor, dim: int) -> List[float]:
    if dim == 1:
        return [0.0]
    elif dim == 2:
        angle = torch.atan2(rotMat[1, 0], rotMat[0, 0])
        return [angle.item()]
    elif dim == 3:
        angle_phi = torch.atan2(rotMat[1, 0], rotMat[0, 0])
        angle_theta = torch.acos(rotMat[2, 2])
        return [angle_phi.item(), angle_theta.item()]
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    
def augmentDomain(domain : Union[DomainDescription, AugmentedDomainDescription], rotationMatrix : torch.Tensor):

    if isinstance(domain, DomainDescription):
        domain = AugmentedDomainDescription(
            min=domain.min,
            max=domain.max,
            periodic=domain.periodic,
            dim=domain.dim,
            device=domain.min.device,
            dtype=domain.min.dtype,
            angles=[0.0] * (domain.dim - 1)  # Default angles for DomainDescription
        )

    angles = anglesFromRotationMatrix(rotationMatrix, domain.dim)
    newAngles = [cAngle + nAngle for cAngle, nAngle in zip(domain.angles, angles)]

    if domain.dim == 1:
        return domain
    elif domain.dim == 2:
        # newMin = rotationMatrix @ domain.min
        # newMax = rotationMatrix @ domain.max
        return AugmentedDomainDescription(
            min=domain.min,
            max=domain.max,
            periodic=domain.periodic,
            dim=domain.dim,
            angles=newAngles,
            device=domain.device,
            dtype=domain.dtype
        )
    elif domain.dim == 3:
        # newMin = rotationMatrix @ domain.min
        # newMax = rotationMatrix @ domain.max
        return AugmentedDomainDescription(
            min=domain.min,
            max=domain.max,
            periodic=domain.periodic,
            dim=domain.dim,
            angles=newAngles,
            device=domain.device,
            dtype=domain.dtype
        )
    else:
        raise ValueError(f"Unsupported dimension: {domain.dim}")
        

from diffSPH.dataLoaderUtils.state import RigidBodyState
def augmentRigidBody(body : RigidBodyState, rotationMatrix : torch.Tensor):
    newState = RigidBodyState(
        bodyID=body.bodyID,
        kind=body.kind,

        centerOfMass = body.centerOfMass @ rotationMatrix,

        angularVelocity= body.angularVelocity,
        linearVelocity= body.linearVelocity @ rotationMatrix,
        mass=body.mass,
        inertia=body.inertia @ rotationMatrix)
    angles = anglesFromRotationMatrix(rotationMatrix, body.centerOfMass.shape[0])
    if len(angles) == 1:
        newState.orientation = body.orientation + angles[0]
    elif len(angles) == 2:
        newState.orientation = [o + a for o, a in zip(body.orientation, angles)]
    

    return newState

from diffSPH.dataLoaderUtils.state import WeaklyCompressibleSPHState, CompressibleSPHState
def rotateState(state: Union[WeaklyCompressibleSPHState, CompressibleSPHState], rotationMatrix: torch.Tensor):
    if isinstance(state, WeaklyCompressibleSPHState):
        newState = WeaklyCompressibleSPHState(
            positions=torch.einsum('ij, ni->nj', rotationMatrix, state.positions),
            supports=state.supports,
            masses=state.masses,
            densities=state.densities,
            velocities=torch.einsum('ij, ni->nj', rotationMatrix, state.velocities),
            kinds=state.kinds,
            materials=state.materials,
            UIDs=state.UIDs,
            numParticles=state.numParticles,
            time=state.time,
            dt=state.dt,
            timestep=state.timestep,
            key=state.key,
            boundaryNormals=torch.einsum('ij, ni->nj', rotationMatrix, state.boundaryNormals) if state.boundaryNormals is not None else None,
            boundaryDistances=state.boundaryDistances if state.boundaryDistances is not None else None,
            rigidBodies=[augmentRigidBody(body, rotationMatrix) for body in state.rigidBodies] if state.rigidBodies is not None else None,
            batches=state.batches
        )
        return newState
    elif isinstance(state, CompressibleSPHState):
        newState = CompressibleSPHState(
            positions=torch.einsum('ij, ni->nj', rotationMatrix, state.positions),
            supports=state.supports,
            masses=state.masses,
            densities=state.densities,
            velocities=torch.einsum('ij, ni->nj', rotationMatrix, state.velocities),
            kinds=state.kinds,
            materials=state.materials,
            UIDs=state.UIDs,
            internalEnergies=state.internalEnergies,
            entropies=state.entropies,
            pressures=state.pressures,
            soundspeeds=state.soundspeeds,
            numParticles=state.numParticles,
            time=state.time,
            dt=state.dt,
            timestep=state.timestep,
            key=state.key,
            alphas=state.alphas if state.alphas is not None else None,
            alpha0s=state.alpha0s if state.alpha0s is not None else None,
            divergence=state.divergence if state.divergence is not None else None,
            batches=state.batches
        )
        return newState
    else:
        raise ValueError(f"Unsupported state type: {type(state)}")



def augmentStates(
        priorStates,
        currentStates,
        trajectoryStates,
        domains,
        configs,
        augmentAngle: bool = True,
        augmentFluidPosition: bool = False,
        augmentBoundaryPosition: bool = False,
        augmentPositionMagnitude: float = 0.01,
        augmentPositionTrajectory: bool = False,
        augmentPositionHistory: bool = False):

    rotMats = []

    augmentedDomains = []
    augmentedPriorStates = []
    augmentedCurrentStates = []
    augmentedTrajectoryStates = []

    if augmentAngle:
        for i in range(len(domains)):
            augmentAngle = torch.rand(1, device=domains[i].min.device, dtype=domains[i].min.dtype) * 2 * np.pi - np.pi
            rotMat = buildRotationMatrix(augmentAngle, domains[i].dim, device=domains[i].min.device, dtype=domains[i].min.dtype)
            rotMats.append(rotMat)

        # augmentedStates = []
        for i in range(len(currentStates)):
            cState = currentStates[i]
            rotMat = rotMats[i % len(rotMats)]
            augmentedState = rotateState(cState, rotMat)
            augmentedCurrentStates.append(augmentedState)
            augmentedDomains.append(augmentDomain(domains[i], rotMat))
            tStates = []
            for t in range(len(trajectoryStates[i])):
                tState = trajectoryStates[i][t]
                augmentedTState = rotateState(tState, rotMat)
                tStates.append(augmentedTState)
            augmentedTrajectoryStates.append(tStates)
            hStates = []
            for h in range(len(priorStates[i])):
                hState = priorStates[i][h]
                augmentedHState = rotateState(hState, rotMat)
                hStates.append(augmentedHState)
            augmentedPriorStates.append(hStates)
    else:
        augmentedCurrentStates = currentStates
        augmentedDomains = domains
        augmentedTrajectoryStates = trajectoryStates
        augmentedPriorStates = priorStates

    if augmentFluidPosition:
        for i in range(len(domains)):
            augmentedCurrentStates[i].positions[augmentedCurrentStates[i].kinds == 0] += torch.randn_like(augmentedCurrentStates[i].positions[augmentedCurrentStates[i].kinds == 0]) * augmentPositionMagnitude * augmentedCurrentStates[i].supports[augmentedCurrentStates[i].kinds == 0][:,None]
            for t in range(len(augmentedTrajectoryStates[i])):
                augmentedTrajectoryStates[i][t].positions[augmentedTrajectoryStates[i][t].kinds == 0] += torch.randn_like(augmentedTrajectoryStates[i][t].positions[augmentedTrajectoryStates[i][t].kinds == 0]) * augmentPositionMagnitude * augmentedTrajectoryStates[i][t].supports[augmentedTrajectoryStates[i][t].kinds == 0][:,None]
            for h in range(len(augmentedPriorStates[i])):
                augmentedPriorStates[i][h].positions[augmentedPriorStates[i][h].kinds == 0] += torch.randn_like(augmentedPriorStates[i][h].positions[augmentedPriorStates[i][h].kinds == 0]) * augmentPositionMagnitude * augmentedPriorStates[i][h].supports[augmentedPriorStates[i][h].kinds == 0][:,None]

    if augmentBoundaryPosition:
        for i in range(len(domains)):
            augmentedCurrentStates[i].positions[augmentedCurrentStates[i].kinds == 1] += torch.randn_like(augmentedCurrentStates[i].positions[augmentedCurrentStates[i].kinds == 1]) * augmentPositionMagnitude * augmentedCurrentStates[i].supports[augmentedCurrentStates[i].kinds == 1][:,None]
            for t in range(len(augmentedTrajectoryStates[i])):
                augmentedTrajectoryStates[i][t].positions[augmentedTrajectoryStates[i][t].kinds == 1] += torch.randn_like(augmentedTrajectoryStates[i][t].positions[augmentedTrajectoryStates[i][t].kinds == 1]) * augmentPositionMagnitude * augmentedTrajectoryStates[i][t].supports[augmentedTrajectoryStates[i][t].kinds == 1][:,None]
            for h in range(len(augmentedPriorStates[i])):
                augmentedPriorStates[i][h].positions[augmentedPriorStates[i][h].kinds == 1] += torch.randn_like(augmentedPriorStates[i][h].positions[augmentedPriorStates[i][h].kinds == 1]) * augmentPositionMagnitude * augmentedPriorStates[i][h].supports[augmentedPriorStates[i][h].kinds == 1][:,None]


    return augmentedPriorStates, augmentedCurrentStates, augmentedTrajectoryStates, augmentedDomains, rotMats



def loadAugmentedBatch(
        dataset, batch, configuration, device = 'cpu', dtype = torch.float32,
        augmentAngle: bool = True,
        augmentFluidPosition: bool = False,
        augmentBoundaryPosition: bool = False,
        augmentPositionMagnitude: float = 0.01,
        augmentPositionTrajectory: bool = False,
        augmentPositionHistory: bool = False
):
    priorStates, currentStates, trajectoryStates, domains, configs = loadBatch(dataset, batch, configuration, device=device, dtype=dtype)
    
    priorStates, currentStates, trajectoryStates, augmentedDomains, rotMats = augmentStates(
        priorStates,
        currentStates,
        trajectoryStates,
        domains,
        configs,
        augmentAngle=augmentAngle,
        augmentFluidPosition=augmentFluidPosition,
        augmentBoundaryPosition=augmentBoundaryPosition,
        augmentPositionMagnitude=augmentPositionMagnitude,
        augmentPositionTrajectory=augmentPositionTrajectory,
        augmentPositionHistory=augmentPositionHistory
    )
    currentState = mergeBatch(currentStates)
    trajectoryState = mergeTrajectoryStates(trajectoryStates)
    priorState = mergeTrajectoryStates(priorStates)


    mergedNeighborhood, neighborhoods = batchNeighborsearch(currentState, augmentedDomains, configs)

    return priorState, currentState, trajectoryState, augmentedDomains, rotMats, configs, mergedNeighborhood