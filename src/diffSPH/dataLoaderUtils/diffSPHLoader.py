from diffSPH.dataLoaderUtils.state import WeaklyCompressibleSPHState, convertNewFormatToWCSPH
from diffSPH.dataLoaderUtils.state import CompressibleSPHState
from diffSPH.dataLoaderUtils.state import RigidBodyState
import torch
from dataclasses import dataclass
from typing import Optional, List


def loadFrameDiffSPH(inFile, rootGroup, key, device, dtype):
    inGroup = inFile['simulationData'][key]

    if 'internalEnergies' in inGroup:
        # print('Loading Compressible SPH State')

        state = CompressibleSPHState(
            positions=torch.tensor(inGroup['positions'][:], device=device, dtype=dtype) if 'positions' in inGroup else torch.tensor(rootGroup['positions'][:], device=device, dtype=dtype),
            supports=torch.tensor(inGroup['supports'][:], device=device, dtype=dtype) if 'supports' in inGroup else torch.tensor(rootGroup['supports'][:], device=device, dtype=dtype),
            masses=torch.tensor(inGroup['masses'][:], device=device, dtype=dtype) if 'masses' in inGroup else torch.tensor(rootGroup['masses'][:], device=device, dtype=dtype),
            densities=torch.tensor(inGroup['densities'][:], device=device, dtype=dtype) if 'densities' in inGroup else torch.tensor(rootGroup['densities'][:], device=device, dtype=dtype),
            velocities=torch.tensor(inGroup['velocities'][:], device=device, dtype=dtype) if 'velocities' in inGroup else torch.tensor(rootGroup['velocities'][:], device=device, dtype=dtype),
            kinds=torch.tensor(inGroup['kinds'][:], device=device, dtype=torch.int64) if 'kinds' in inGroup else torch.tensor(rootGroup['kinds'][:], device=device, dtype=torch.int64),
            materials=torch.tensor(inGroup['materials'][:], device=device, dtype=torch.int64) if 'materials' in inGroup else torch.tensor(rootGroup['materials'][:], device=device, dtype=torch.int64),
            UIDs=torch.tensor(inGroup['UIDs'][:], device=device, dtype=torch.int64) if 'UIDs' in inGroup else torch.tensor(rootGroup['UIDs'][:], device=device, dtype=torch.int64),
            internalEnergies=torch.tensor(inGroup['internalEnergies'][:], device=device, dtype=dtype) if 'internalEnergies' in inGroup else torch.tensor(rootGroup['internalEnergies'][:], device=device, dtype=dtype),
            entropies=torch.tensor(inGroup['entropies'][:], device=device, dtype=dtype) if 'entropies' in inGroup else None,
            pressures=torch.tensor(inGroup['pressures'][:], device=device, dtype=dtype) if 'pressures' in inGroup else None,
            soundspeeds=torch.tensor(inGroup['soundspeeds'][:], device=device, dtype=dtype) if 'soundspeeds' in inGroup else None,
            numParticles=inGroup.attrs['numParticles'],
            time=inGroup.attrs['time'],
            dt=inGroup.attrs['dt'],
            timestep=int(key),
            key=key,

            alphas=torch.tensor(inGroup['alphas'][:], device=device, dtype=dtype) if 'alphas' in inGroup else None,
            alpha0s=torch.tensor(inGroup['alpha0s'][:], device=device, dtype=dtype) if 'alpha0s' in inGroup else None,
            divergence=torch.tensor(inGroup['divergence'][:], device=device, dtype=dtype) if 'divergence' in inGroup else None
        )
        return state
    else:
        # print('Loading Weakly Compressible SPH State')

        rigidBodies = []
        if len(inGroup.keys()) > 0:
            for gkey in inGroup.keys():
                if 'rigidBody_' in gkey:
                    bodyGroup = inGroup[gkey]
                    rb = RigidBodyState(
                        bodyGroup.attrs['bodyID'],
                        kind=bodyGroup.attrs['kind'],

                        centerOfMass= torch.tensor(bodyGroup.attrs['centerOfMass'][:], device=device, dtype=dtype),
                        orientation= torch.tensor(bodyGroup.attrs['orientation'], device=device, dtype=dtype),
                        angularVelocity= torch.tensor(bodyGroup.attrs['angularVelocity'], device=device, dtype=dtype),
                        linearVelocity= torch.tensor(bodyGroup.attrs['linearVelocity'][:], device=device, dtype=dtype),
                        mass= torch.tensor(bodyGroup.attrs['mass'], device=device, dtype=dtype),
                        inertia= torch.tensor(bodyGroup.attrs['inertia'], device=device, dtype=dtype),
                    )

        boundaryNormals = None
        boundaryDistances = None

        if 'ghostOffsets' in inGroup:
            boundaryNormals = torch.tensor(inGroup['ghostOffsets'][:], device=device, dtype=dtype)
            boundaryNormals = torch.nn.functional.normalize(boundaryNormals, dim=1)
        elif 'ghostOffsets'in rootGroup:
            boundaryNormals = torch.tensor(rootGroup['ghostOffsets'][:], device=device, dtype=dtype)
            boundaryNormals = torch.nn.functional.normalize(boundaryNormals, dim=1)
        else:
            pass
        if 'ghostDistances' in inGroup:
            boundaryDistances = torch.tensor(inGroup['ghostDistances'][:], device=device, dtype=dtype) / 2
        elif 'ghostDistances' in rootGroup:
            boundaryDistances = torch.tensor(rootGroup['ghostDistances'][:], device=device, dtype=dtype) / 2
        else:
            pass
        

        state = WeaklyCompressibleSPHState(
            positions=torch.tensor(inGroup['positions'][:], device=device, dtype=dtype) if 'positions' in inGroup else torch.tensor(rootGroup['positions'][:], device=device, dtype=dtype),
            supports=torch.tensor(inGroup['supports'][:], device=device, dtype=dtype) if 'supports' in inGroup else torch.tensor(rootGroup['supports'][:], device=device, dtype=dtype),
            masses=torch.tensor(inGroup['masses'][:], device=device, dtype=dtype) if 'masses' in inGroup else torch.tensor(rootGroup['masses'][:], device=device, dtype=dtype),
            densities=torch.tensor(inGroup['densities'][:], device=device, dtype=dtype) if 'densities' in inGroup else torch.tensor(rootGroup['densities'][:], device=device, dtype=dtype),
            velocities=torch.tensor(inGroup['velocities'][:], device=device, dtype=dtype) if 'velocities' in inGroup else torch.tensor(rootGroup['velocities'][:], device=device, dtype=dtype),
            kinds=torch.tensor(inGroup['kinds'][:], device=device, dtype=torch.int64) if 'kinds' in inGroup else torch.tensor(rootGroup['kinds'][:], device=device, dtype=torch.int64),
            materials=torch.tensor(inGroup['materials'][:], device=device, dtype=torch.int64) if 'materials' in inGroup else torch.tensor(rootGroup['materials'][:], device=device, dtype=torch.int64),
            UIDs=torch.tensor(inGroup['UIDs'][:], device=device, dtype=torch.int64) if 'UIDs' in inGroup else torch.tensor(rootGroup['UIDs'][:], device=device, dtype=torch.int64),
            numParticles=inGroup.attrs['numParticles'],
            time=inGroup.attrs['time'],
            dt=inGroup.attrs['dt'],
            timestep=int(key),
            key=key,

            rigidBodies=rigidBodies,
            boundaryNormals=boundaryNormals,
            boundaryDistances=boundaryDistances
        )
        return state
    
    return []
try:
    from torchCompactRadius.util import DomainDescription
except ImportError as e:
    # raise e
    # print("torchCompactRadius not found, using fallback implementations.")

    from diffSPH.dataLoaderUtils.fallback import DomainDescription


import copy
from diffSPH.dataLoaderUtils.neighborhood import AugmentedDomainDescription

def loadDiffSPHState(inFile, key, configuration, device, dtype):
    rootGroup = inFile['simulationData']['%06d' % 0]

    currentState = loadFrameDiffSPH(inFile, rootGroup, key, device, dtype)

    if configuration.historyLength > 0:
        priorStates = []
        for h in range(configuration.historyLength):
            iPriorKey = int(key) - configuration.frameDistance * (h + 1)
            if iPriorKey < 0 or configuration.frameDistance == 0:
                priorState = copy.deepcopy(currentState)
            else:
                priorState = loadFrameDiffSPH(inFile, rootGroup, '%06d' % iPriorKey, device = device, dtype = dtype)
            priorStates.append(priorState)
        priorStates.reverse()

    else:
        priorStates  = []

    if configuration.maxRollout > 0:
        trajectoryStates = []
        for u in range(configuration.maxRollout):
            unrollKey = int(key) + configuration.frameDistance * (u + 1)
            nextState = loadFrameDiffSPH(inFile, rootGroup, '%06d' % unrollKey, device = device, dtype = dtype)
            trajectoryStates.append(nextState)
    else: 
        trajectoryStates = []


    domain = AugmentedDomainDescription(
        min = torch.tensor(inFile['domain'].attrs['min'], device = device, dtype = dtype),
        max = torch.tensor(inFile['domain'].attrs['max'], device = device, dtype = dtype),
        periodic = torch.tensor(inFile['domain'].attrs['periodic'], device = device, dtype = torch.bool),
        dim = len(inFile['domain'].attrs['min']),
        angles = [0.0] * (len(inFile['domain'].attrs['min']) - 1), 
        device = device,
        dtype = dtype
    )

    parsedConfig = {}

    for attr in inFile['config'].attrs:
        parsedConfig[attr] = inFile['config'].attrs[attr]

    for key in inFile['config'].keys():
        parsedConfig[key] = {}
        for attr in inFile['config'][key].attrs:
            parsedConfig[key][attr] = inFile['config'][key].attrs[attr]


    return priorStates, currentState, trajectoryStates, domain, parsedConfig