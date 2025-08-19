from diffSPH.dataLoaderUtils.state import DataConfiguration
try:
    from torchCompactRadius.util import DomainDescription
except ImportError:
    # raise e
    # print("torchCompactRadius not found, using fallback implementations.")

    from diffSPH.dataLoaderUtils.fallback import DomainDescription
import numpy as np
import torch
from diffSPH.dataLoaderUtils.state import WeaklyCompressibleSPHState

import warnings
import copy
def computeSupport(area, targetNumNeighbors, dim):
    if dim == 1:
        return targetNumNeighbors * area
    if dim == 2:
        if (isinstance(targetNumNeighbors, int) or isinstance(targetNumNeighbors, float)) and not isinstance(area, torch.Tensor):
            return np.sqrt(targetNumNeighbors * area / np.pi)
        return torch.sqrt(targetNumNeighbors * area / np.pi)
    if dim == 3:
        return (3 * targetNumNeighbors * area / (4 * np.pi))**(1/3)
    else:
        raise ValueError('Unsupported dimension %d' % dim)

def loadAdditional(inGrp, state, additionalData, device, dtype):
    for dataKey in additionalData:
        if dataKey in inGrp:
            state[dataKey] = torch.from_numpy(inGrp[dataKey][:]).to(device = device, dtype = dtype)
        else:
            warnings.warn('Additional data key %s not found in group' % dataKey)
    return state


def loadGroup_testcaseII(inFile, inGrp, staticBoundaryData, device = 'cpu', dtype = torch.float32, additionalData = []):
    if 'boundaryInformation' in inFile:
        dynamicBoundaryData = {}
        for k in staticBoundaryData.keys():
            if isinstance(staticBoundaryData[k], torch.Tensor):
                dynamicBoundaryData[k] = staticBoundaryData[k].clone()
            else:
                dynamicBoundaryData[k] = staticBoundaryData[k]
    else:
        dynamicBoundaryData = None

    areas = torch.from_numpy(inGrp['fluidArea'][:]).to(device = device, dtype = dtype)
    support = inGrp['fluidSupport'][0]
    state = {
        'fluid': {
            'positions': torch.from_numpy(inGrp['fluidPosition'][:]).to(device = device, dtype = dtype),
            'velocities': torch.from_numpy(inGrp['fluidVelocity'][:]).to(device = device, dtype = dtype),
            'gravityAcceleration': torch.from_numpy(inGrp['fluidGravity'][:]).to(device = device, dtype = dtype) if 'fluidGravity' not in inFile.attrs else torch.from_numpy(inFile.attrs['fluidGravity']).to(device = device, dtype = dtype) * torch.ones(inGrp['fluidDensity'][:].shape[0]).to(device = device, dtype = dtype)[:,None],
            'densities': torch.from_numpy(inGrp['fluidDensity'][:]).to(device = device, dtype = dtype) * inFile.attrs['restDensity'],
            'areas': areas,
            'masses': areas * inFile.attrs['restDensity'],
            'supports': torch.ones_like(areas) * support, #torch.from_numpy(inGrp['fluidSupport'][:]).to(device = device, dtype = dtype),
            'indices': torch.from_numpy(inGrp['UID'][:]).to(device = device, dtype = torch.int64),
            'numParticles': len(areas)
        },
        'boundary': dynamicBoundaryData if dynamicBoundaryData is not None else staticBoundaryData,
        'time': inGrp.attrs['time'],
        'dt': inGrp.attrs['dt'],# * hyperParameterDict['frameDistance'],
        'timestep': inGrp.attrs['timestep'],
    }
    loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
    # if hyperParameterDict['normalizeDensity']:
        # state['fluid']['densities'] = (state['fluid']['densities'] - 1) * inFile.attrs['restDensity']
    # for dataKey in additionalData:
        # state['fluid'][dataKey] = torch.from_numpy(np.array(inGrp[dataKey])).to(device = device, dtype = dtype)
    
    return state


def loadFrame_testcase_II(inFile, key, device = 'cpu', dtype = torch.float32):
    # print(key)

    inGrp = inFile['simulationExport'][key]
    support = np.max(inGrp['fluidSupport'][:]) if 'support' not in inFile.attrs else inFile.attrs['support']
    # if hyperParameterDict['numNeighbors'] > 0:
        # support = computeSupport(inGrp['fluidArea'][0], hyperParameterDict['numNeighbors'], 2)
    attributes = {
        'support': support,
        'targetNeighbors': inFile.attrs['targetNeighbors'],
        'restDensity': inFile.attrs['restDensity'],
        'dt': inGrp.attrs['dt'],#, * hyperParameterDict['frameDistance'],
        'time': inGrp.attrs['time'],
        'radius': inFile.attrs['radius'] if 'radius' in inFile.attrs else inGrp.attrs['radius'],
        'area': inFile.attrs['radius'] **2 * np.pi if 'area' not in inFile.attrs else inFile.attrs['area'],
    }
    config = {
        'domain':{
            'dim': 2,
            'minExtent': torch.tensor([-1.2, -1.2], device = device, dtype = dtype),
            'maxExtent': torch.tensor([1.2, 1.2], device = device, dtype = dtype),
            'periodicity': torch.tensor([False, False], device = device, dtype = torch.bool),
            'periodic': False
        },
        'neighborhood':{
            'scheme': 'compact',
            'verletScale': 1.4
        },
        'compute':{
            'device': device,
            'dtype': dtype,
            'precision': 'float32' if dtype == torch.float32 else 'float64',
        },
        'kernel':{
            'name': 'Wendland2',
            'targetNeighbors': 20,
            # 'function': getKernel('Wendland2')
        },
        'boundary':{
            'active': True
        },
        'fluid':{
            'rho0': 1000,
            'cs': 20,
        },
        'particle':{
            'support': attributes['support']
        },
        'shifting':{
            'CFL': 1.5
        }
    }

    if 'boundaryInformation' in inFile:
        staticBoundaryData = {
                'indices': torch.arange(0, inFile['boundaryInformation']['boundaryPosition'].shape[0], device = device, dtype = torch.int64),
                'positions': torch.from_numpy(inFile['boundaryInformation']['boundaryPosition'][:]).to(device = device, dtype = dtype),
                'normals': torch.from_numpy(inFile['boundaryInformation']['boundaryNormals'][:]).to(device = device, dtype = dtype),
                'areas': torch.from_numpy(inFile['boundaryInformation']['boundaryArea'][:]).to(device = device, dtype = dtype),
                'masses': torch.from_numpy(inFile['boundaryInformation']['boundaryArea'][:]).to(device = device, dtype = dtype) * config['fluid']['rho0'],
                'velocities': torch.from_numpy(inFile['boundaryInformation']['boundaryVelocity'][:]).to(device = device, dtype = dtype),
                'densities': torch.from_numpy(inFile['boundaryInformation']['boundaryRestDensity'][:]).to(device = device, dtype = dtype),
                'supports': torch.from_numpy(inFile['boundaryInformation']['boundarySupport'][:]).to(device = device, dtype = dtype),
                'bodyIDs': torch.from_numpy(inFile['boundaryInformation']['boundaryBodyAssociation'][:]).to(device = device, dtype = torch.int64),
                'numParticles': len(inFile['boundaryInformation']['boundaryPosition'][:]),
            } if 'boundaryInformation' in inFile else None
    else:
        staticBoundaryData = None

    if 'boundaryInformation' in inFile:
        dynamicBoundaryData = {}
        for k in staticBoundaryData.keys():
            if isinstance(staticBoundaryData[k], torch.Tensor):
                dynamicBoundaryData[k] = staticBoundaryData[k].clone()
            else:
                dynamicBoundaryData[k] = staticBoundaryData[k]

        dynamicBoundaryData['positions'] = torch.from_numpy(inGrp['boundaryPosition'][:]).to(device = device, dtype = dtype) if 'boundaryPosition' in inGrp else dynamicBoundaryData['positions']
        dynamicBoundaryData['normals'] = torch.from_numpy(inGrp['boundaryNormals'][:]).to(device = device, dtype = dtype) if 'boundaryNormals' in inGrp else dynamicBoundaryData['normals']
        dynamicBoundaryData['areas'] = torch.from_numpy(inGrp['boundaryArea'][:]).to(device = device, dtype = dtype) if 'boundaryArea' in inGrp else dynamicBoundaryData['areas']
        dynamicBoundaryData['velocities'] = torch.from_numpy(inGrp['boundaryVelocity'][:]).to(device = device, dtype = dtype) if 'boundaryVelocity' in inGrp else dynamicBoundaryData['velocities']
        dynamicBoundaryData['densities'] = torch.from_numpy(inGrp['boundaryDensity'][:]).to(device = device, dtype = dtype) if 'boundaryDensity' in inGrp else dynamicBoundaryData['densities']
        dynamicBoundaryData['supports'] = torch.from_numpy(inGrp['boundarySupport'][:]).to(device = device, dtype = dtype) if 'boundarySupport' in inGrp else dynamicBoundaryData['supports'],
        dynamicBoundaryData['bodyIDs'] = torch.from_numpy(inGrp['boundaryBodyAssociation'][:]).to(device = device, dtype = torch.int64) if 'boundaryBodyAssociation' in inGrp else dynamicBoundaryData['bodyIDs']
    else:
        dynamicBoundaryData = None

    state = loadGroup_testcaseII(inFile, inGrp, staticBoundaryData, dtype=dtype, device = device,)

    state['fluid']['numParticles'] = state['fluid']['positions'].shape[0]
    if state['boundary'] is not None:
        state['boundary']['numParticles'] = state['boundary']['positions'].shape[0]


    # iPriorKey = int(key) - hyperParameterDict['frameDistance']


    # priorStates = []
    # # print(f'Loading prior states [{max(hyperParameterDict["historyLength"], 1)}]')
    # for h in range(max(hyperParameterDict['historyLength'], 1)):
    #     priorState = None        
    #     iPriorKey = int(key) - hyperParameterDict['frameDistance'] * (h + 1)

    #     if buildPriorState or hyperParameterDict['adjustForFrameDistance']:
    #         if iPriorKey < 0 or hyperParameterDict['frameDistance'] == 0:
    #             priorState = copy.deepcopy(state)
    #         else:
    #             grp = inFile['simulationExport']['%05d' % iPriorKey] if '%05d' % iPriorKey in inFile['simulationExport'] else None
    #             # if grp is None:
    #                 # print('Key %s not found in file' % iPriorKey)
    #             priorState = loadGroup_testcaseII(inFile, inFile['simulationExport']['%05d' % iPriorKey], staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)
    #     # print('Loaded prior state %s' % iPriorKey)
    #     priorStates.append(priorState)

    # priorState = None
    # print(iPriorKey)
    # if buildPriorState:
    #     if iPriorKey < 0 or hyperParameterDict['frameDistance'] == 0:
    #         priorState = copy.deepcopy(state)
    #         print('copying state')
    #     else:
    #         priorState = loadGroup_testcaseII(inFile, inFile['simulationExport']['%05d' % iPriorKey], staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)
    #         print('loading prior state')
            
    # nextStates = []
    # if buildNextState:
    #     if unrollLength == 0 and hyperParameterDict['frameDistance'] == 0:
    #         nextStates = [copy.deepcopy(state)]
    #     if unrollLength == 0 and hyperParameterDict['frameDistance'] != 0:
    #         nextStates = [copy.deepcopy(state)]
    #         warnings.warn('Unroll length is zero, but frame distance is not zero')
    #     if unrollLength != 0 and hyperParameterDict['frameDistance'] == 0:
    #         nextStates = [copy.deepcopy(state)] * unrollLength
    #     if unrollLength != 0 and hyperParameterDict['frameDistance'] != 0:
    #         for u in range(unrollLength):
    #             unrollKey = int(key) + hyperParameterDict['frameDistance'] * (u + 1)
    #             nextState = loadGroup_testcaseII(inFile, inFile['simulationExport']['%05d' % unrollKey], staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)                
    #             nextStates.append(nextState)            




    return state, attributes, config#, priorStates, nextStates

from diffSPH.dataLoaderUtils.state import WeaklyCompressibleSPHState, convertNewFormatToWCSPH

from diffSPH.dataLoaderUtils.neighborhood import AugmentedDomainDescription

def loadTestcaseIIState(inFile, key, configuration : DataConfiguration, device = 'cpu', dtype = torch.float32):

    currentState, attributes, config = loadFrame_testcase_II(inFile, key, device = device, dtype = dtype)
    # print(currentState['boundary'])
    currentState = convertNewFormatToWCSPH(inFile, key, currentState)

    if configuration.historyLength > 0:
        priorStates = []
        for h in range(configuration.historyLength):
            iPriorKey = int(key) - configuration.frameDistance * (h + 1)
            if iPriorKey < 0 or configuration.frameDistance == 0:
                priorState = copy.deepcopy(currentState)
            else:
                priorState,*_ = loadFrame_testcase_II(inFile, '%05d' % iPriorKey, device = device, dtype = dtype)
            priorState = convertNewFormatToWCSPH(inFile, '%05d' % iPriorKey, priorState)
            priorStates.append(priorState)
        priorStates.reverse()

    else:
        priorStates  = []

    if configuration.maxRollout > 0:
        trajectoryStates = []
        for u in range(configuration.maxRollout):
            unrollKey = int(key) + configuration.frameDistance * (u + 1)
            nextState,*_ = loadFrame_testcase_II(inFile, '%05d' % unrollKey, device = device, dtype = dtype)
            nextState = convertNewFormatToWCSPH(inFile, '%05d' % unrollKey, nextState)
            trajectoryStates.append(nextState)
    else: 
        trajectoryStates = []

    domain = AugmentedDomainDescription(
        min = torch.tensor([-1.2, -1.2], device = device, dtype = dtype),
        max = torch.tensor([1.2, 1.2], device = device, dtype = dtype),
        periodic = torch.tensor([False, False], device = device, dtype = torch.bool),
        dim = 2,
        angles = [0.0],
        device = device,
        dtype = dtype
    )

    return priorStates, currentState, trajectoryStates, domain, config