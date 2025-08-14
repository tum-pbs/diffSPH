import torch
from dataclasses import dataclass
from typing import Union, List
from diffSPH.dataLoaderUtils.state import WeaklyCompressibleSPHState, CompressibleSPHState
# from neighborhood import SparseBatchedNeighborhood, evalDistanceTensor_

def mergeBatch(states):
    if len(states) == 0:
        raise ValueError("The batch is empty, cannot merge states.")
    if len(states) == 1:
        mergedState = states[0]
        mergedState.numParticles = [mergedState.numParticles]
        mergedState.time = [mergedState.time]
        mergedState.dt = [mergedState.dt]
        mergedState.timestep = [mergedState.timestep]
        mergedState.key = [mergedState.key]
        return mergedState
    
    if isinstance(states[0], WeaklyCompressibleSPHState):
        mergedState = WeaklyCompressibleSPHState(
            positions=torch.cat([s.positions for s in states], dim=0),
            supports= torch.cat([s.supports for s in states], dim=0),
            masses = torch.cat([s.masses for s in states], dim=0),
            densities = torch.cat([s.densities for s in states], dim=0),
            velocities= torch.cat([s.velocities for s in states], dim=0),
            kinds = torch.cat([s.kinds for s in states], dim=0),
            materials = torch.cat([s.materials for s in states], dim=0),
            UIDs = torch.cat([s.UIDs for s in states], dim=0),

            numParticles = [s.numParticles for s in states],
            time = [s.time for s in states],
            dt = [s.dt for s in states],
            timestep = [s.timestep for s in states],
            key =[s.key for s in states],

            boundaryNormals= None,
            boundaryDistances= None,
            rigidBodies = None,
            batches = torch.cat([s.batches for s in states], dim=0),
        )
        if any(s.boundaryNormals is not None for s in states):
            mergedState.boundaryNormals = torch.cat([s.boundaryNormals if s.boundaryNormals is not None else torch.zeros_like(s.positions) for s in states], dim=0)
        if any(s.boundaryDistances is not None for s in states):
            mergedState.boundaryDistances = torch.cat([s.boundaryDistances if s.boundaryDistances is not None else torch.zeros_like(s.positions) for s in states], dim=0)
        if any(s.rigidBodies is not None for s in states):
            rigidBodies = []
            for s in states:
                if s.rigidBodies is not None:
                    rigidBodies.append(s.rigidBodies)
            mergedState.rigidBodies = rigidBodies

    elif isinstance(states[0], CompressibleSPHState):
        mergedState = CompressibleSPHState(
            positions=torch.cat([s.positions for s in states], dim=0),
            supports= torch.cat([s.supports for s in states], dim=0),
            masses = torch.cat([s.masses for s in states], dim=0),
            densities = torch.cat([s.densities for s in states], dim=0),
            velocities= torch.cat([s.velocities for s in states], dim=0),

            kinds = torch.cat([s.kinds for s in states], dim=0),
            materials = torch.cat([s.materials for s in states], dim=0),
            UIDs = torch.cat([s.UIDs for s in states], dim=0),

            internalEnergies= torch.cat([s.internalEnergies for s in states], dim=0),
            entropies = None,
            pressures = None,
            soundspeeds= None,

            numParticles = [s.numParticles for s in states],
            time = [s.time for s in states],
            dt = [s.dt for s in states],
            timestep = [s.timestep for s in states],
            key =[s.key for s in states],

            alphas = None,
            alpha0s= None,
            divergence = None,

            batches = torch.cat([s.batches for s in states], dim=0),
        )

        alphas = [s.alphas for s in states]
        alpha0s = [s.alpha0s for s in states]
        divergences = [s.divergence for s in states]
        if any(a is not None for a in [s.entropies for s in states]):
            mergedState.entropies = torch.cat([s.entropies for s in states if s.entropies is not None], dim=0)
        if any(p is not None for p in [s.pressures for s in states]):
            mergedState.pressures = torch.cat([s.pressures for s in states if s.pressures is not None], dim=0)
        if any(s is not None for s in [s.soundspeeds for s in states]):
            mergedState.soundspeeds = torch.cat([s.soundspeeds for s in states if s.soundspeeds is not None], dim=0)

        if all(a is not None for a in alphas):
            mergedState.alphas = torch.cat(alphas, dim=0)
        if all(a0 is not None for a0 in alpha0s):
            mergedState.alpha0s = torch.cat(alpha0s, dim=0)
        if all(d is not None for d in divergences):
            mergedState.divergence = torch.cat(divergences, dim=0)

        # if any(a is None for a in alphas) and any(a is not None for a in alphas):
        #     raise ValueError("Some 'alphas' are None and some are not None in the batch.")
        # if any(a0 is None for a0 in alpha0s) and any(a0 is not None for a0 in alpha0s):
        #     raise ValueError("Some 'alpha0s' are None and some are not None in the batch.")
        # if any(d is None for d in divergences) and any(d is not None for d in divergences):
        #     raise ValueError("Some 'divergence' are None and some are not None in the batch.")
        

    return mergedState

def mergeTrajectoryStates(states):
    # states are a list of lists.
    # The first list is batchwise the second list is the trajectory.
    # The batches should be merged into a single batch, i.e., 
    merged = []
    for t in range(len(states[0])):
        batchStates = [states[b][t] for b in range(len(states))]
        mergedState = mergeBatch(batchStates)
        merged.append(mergedState)
    return merged