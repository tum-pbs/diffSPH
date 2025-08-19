import torch
import numpy as np
import h5py
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



def loadGroup_cuMath(inFile, inGrp, staticFluidData, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    if staticFluidData is not None and int(key) == 0 or inGrp is None:
        state = {
            'fluid': staticFluidData,
            'boundary': staticBoundaryData,
            'time': 0.0,
            'dt': inFile.attrs['targetDt'] * hyperParameterDict['frameDistance'],
            'timestep': 0,
        }
        loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
        return state

    kinds = inGrp['kinds'][:] if 'kinds' in inGrp else inFile['initialState']['kinds'][:]
    hasBoundary = np.any(kinds == 1)
    if hasBoundary:
        boundaryMask = kinds == 1
        dynamicBoundaryData = {}
        for k in staticBoundaryData.keys():
            if isinstance(staticBoundaryData[k], torch.Tensor):
                dynamicBoundaryData[k] = staticBoundaryData[k].clone()
            else:
                dynamicBoundaryData[k] = staticBoundaryData[k]

        for k in staticFluidData.keys():
            if isinstance(dynamicBoundaryData[k], torch.Tensor):
                dynamicBoundaryData[k] = torch.from_numpy(inGrp[k][boundaryMask]).to(device = device, dtype = dtype) if k in inGrp else dynamicBoundaryData[k]
    else:
        dynamicBoundaryData = None

    fluidState = {}

    if staticFluidData is not None:
        for k in staticFluidData.keys():
            if isinstance(staticFluidData[k], torch.Tensor):
                fluidState[k] = staticFluidData[k].clone()
            else:
                fluidState[k] = staticFluidData[k]
    fluidMask = kinds == 0
    for k in staticFluidData.keys():
        if isinstance(fluidState[k], torch.Tensor):
            fluidState[k] = torch.from_numpy(inGrp[k][fluidMask]).to(device = device, dtype = dtype) if k in inGrp else fluidState[k]
    fluidState['numParticles'] = len(fluidState['densities'])

    # for k in inGrp.keys():
        # print(k, inGrp[k])

    # support = inFile.attrs['support'] if hyperParameterDict['numNeighbors'] < 0 else computeSupport(inFile.attrs['area'], hyperParameterDict['numNeighbors'], 2)
    # rho = torch.from_numpy(inGrp['fluidDensity'][:]).to(device = device, dtype = dtype)
    # areas = torch.ones_like(rho) * inFile.attrs['area']
    state = {
        'fluid': fluidState,
        'boundary': dynamicBoundaryData if dynamicBoundaryData is not None else staticBoundaryData,
        'time': inGrp.attrs['time'],
        'dt': inGrp.attrs['dt'] * hyperParameterDict['frameDistance'],
        'timestep': inGrp.attrs['timestep'],
    }
    loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
    # for dataKey in additionalData:
        # state['fluid'][dataKey] = torch.from_numpy(np.array(inGrp[dataKey])).to(device = device, dtype = dtype)
    
    return state

def loadFrame_cuMath(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    # print(f'Loading frame {key} from {fileName} ')
    # print(key)

    initialKinds = inFile['initialState']['fluid']['kinds'][:]
    fluidMask = initialKinds == 0
    boundaryMask = initialKinds == 1


    staticFluidData = {
        'positions': torch.from_numpy(inFile['initialState']['fluid']['positions'][fluidMask]).to(device = device, dtype = dtype),
        'velocities': torch.from_numpy(inFile['initialState']['fluid']['velocities'][fluidMask]).to(device = device, dtype = dtype),
        # 'gravityAcceleration': torch.zeros_like(torch.from_numpy(inFile['initialState']['fluid']['velocities'][fluidMask]).to(device = device, dtype = dtype)),
        'densities': torch.from_numpy(inFile['initialState']['fluid']['densities'][fluidMask]).to(device = device, dtype = dtype),
        'areas': torch.from_numpy(inFile['initialState']['fluid']['areas'][fluidMask]).to(device = device, dtype = dtype),
        'masses': torch.from_numpy(inFile['initialState']['fluid']['masses'][fluidMask]).to(device = device, dtype = dtype),
        'supports': torch.from_numpy(inFile['initialState']['fluid']['supports'][fluidMask]).to(device = device, dtype = dtype),
        'indices': torch.from_numpy(inFile['initialState']['fluid']['UIDs'][fluidMask]).to(device = device, dtype = torch.int32),
        'numParticles': len(inFile['initialState']['fluid']['positions'][fluidMask]),                                  
    }

    area = np.max(staticFluidData['areas'].detach().cpu().numpy())
    support = np.max(staticFluidData['supports'].detach().cpu().numpy())
    if hyperParameterDict['numNeighbors'] > 0:
        support = computeSupport(area, hyperParameterDict['numNeighbors'], 2)
    attributes = {
        'support': support,
        'targetNeighbors': inFile.attrs['targetNeighbors'],
        'restDensity': inFile.attrs['rho0'],
        'dt': inFile.attrs['targetDt'] * hyperParameterDict['frameDistance'],
        'time': 0.0,
        'radius': inFile.attrs['dx'],
        'area': area,
    }
    if key == '000000':
        inGrp = None
    else:
        inGrp = inFile['simulationData'][key] 


    initialKinds = inFile['initialState']['boundary']['kinds'][:]
    fluidMask = initialKinds == 0
    boundaryMask = initialKinds == 1

    staticBoundaryData = {
            'indices': torch.from_numpy(inFile['initialState']['boundary']['UIDs'][boundaryMask]).to(device = device, dtype = torch.int64),
            'positions': torch.from_numpy(inFile['initialState']['boundary']['positions'][boundaryMask]).to(device = device, dtype = dtype),
            'normals': torch.from_numpy(inFile['initialState']['boundary']['normals'][boundaryMask]).to(device = device, dtype = dtype),
            'areas': torch.from_numpy(inFile['initialState']['boundary']['areas'][boundaryMask]).to(device = device, dtype = dtype),
            'masses': torch.from_numpy(inFile['initialState']['boundary']['masses'][boundaryMask]).to(device = device, dtype = dtype),
            'velocities': torch.from_numpy(inFile['initialState']['boundary']['velocities'][boundaryMask]).to(device = device, dtype = dtype),
            'densities': torch.from_numpy(inFile['initialState']['boundary']['densities'][boundaryMask]).to(device = device, dtype = dtype),
            'supports': torch.from_numpy(inFile['initialState']['boundary']['supports'][boundaryMask]).to(device = device, dtype = dtype),
            'bodyIDs': torch.from_numpy(inFile['initialState']['boundary']['materials'][boundaryMask]).to(device = device, dtype = torch.int64),
            'numParticles': len(inFile['initialState']['boundary']['positions'][boundaryMask]),
        } if np.sum(boundaryMask)>0  else None

    # print(staticBoundaryData['numParticles'])
    # print(f'Fluid Particles: {staticFluidData["numParticles"]}, Boundary Particles: {staticBoundaryData["numParticles"] if staticBoundaryData is not None else 0}')

    # if 'boundaryInformation' in inFile:
    #     dynamicBoundaryData = {}
    #     for k in staticBoundaryData.keys():
    #         if isinstance(staticBoundaryData[k], torch.Tensor):
    #             dynamicBoundaryData[k] = staticBoundaryData[k].clone()
    #         else:
    #             dynamicBoundaryData[k] = staticBoundaryData[k]


    # else:
    #     dynamicBoundaryData = None

    state = loadGroup_cuMath(inFile, inGrp, staticFluidData, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = buildPriorState, buildNextState = buildNextState)


    priorStates = []
    # print(f'Loading prior states [{max(hyperParameterDict["historyLength"], 1)}]')
    historyLength = max(hyperParameterDict['historyLength'], 1)
    if 'dt' in hyperParameterDict['fluidFeatures'] or 'ddt' in hyperParameterDict['fluidFeatures'] or 'diff' in hyperParameterDict['fluidFeatures']:
        historyLength += 1
    for h in range(historyLength):
        priorState = None        
        iPriorKey = int(key) - hyperParameterDict['frameDistance'] * (h + 1)

        if buildPriorState or hyperParameterDict['adjustForFrameDistance']:
            if iPriorKey < 0 or hyperParameterDict['frameDistance'] == 0:
                priorState = copy.deepcopy(state)
            else:
                grp = inFile['simulationData']['%06d' % iPriorKey] if '%06d' % iPriorKey in inFile['simulationData'] else None
                # if grp is None:
                    # print('Key %s not found in file' % iPriorKey)
                priorState = loadGroup_cuMath(inFile, grp, staticFluidData, staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)
        # print('Loaded prior state %s' % iPriorKey)
        priorStates.append(priorState)

    nextStates = []
    if buildNextState:
        if unrollLength == 0 and hyperParameterDict['frameDistance'] == 0:
            nextStates = [copy.deepcopy(state)]
        if unrollLength == 0 and hyperParameterDict['frameDistance'] != 0:
            nextStates = [copy.deepcopy(state)]
            warnings.warn('Unroll length is zero, but frame distance is not zero')
        if unrollLength != 0 and hyperParameterDict['frameDistance'] == 0:
            nextStates = [copy.deepcopy(state)] * unrollLength
        if unrollLength != 0 and hyperParameterDict['frameDistance'] != 0:
            for u in range(unrollLength):
                unrollKey = int(key) + hyperParameterDict['frameDistance'] * (u + 1)
                nextState = loadGroup_cuMath(inFile, inFile['simulationData']['%05d' % unrollKey], staticFluidData, staticBoundaryData, fileName, unrollKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)                
                nextStates.append(nextState)            

    # if hyperParameterDict['adjustForFrameDistance']:

    # config['particle']['support'] = support

    # print('Loaded frame %s' % key)

    config = {
        'domain':{
            'dim': 2,
            'minExtent': torch.tensor(inFile.attrs['domainMin'], device = device, dtype = dtype),
            'maxExtent': torch.tensor(inFile.attrs['domainMax'], device = device, dtype = dtype),
            'periodic': torch.tensor(inFile.attrs['domainPeriodic'], device = device, dtype = torch.bool),
            'periodicity': torch.tensor(inFile.attrs['domainPeriodic'], device = device, dtype = torch.bool),
        },
        'neighborhood':{
            'scheme': 'compact',
            'verletScale': 1.0
        },
        'compute':{
            'device': device,
            'dtype': dtype,
            'precision': 'float32' if dtype == torch.float32 else 'float64',
        },
        'kernel':{
            'name': inFile.attrs['kernel'],
            'targetNeighbors': inFile.attrs['targetNeighbors'],
            # 'function': getKernel(inFile.attrs['kernel'])
        },
        'boundary':{
            'active': np.any(initialKinds == 1),
        },
        'fluid':{
            'rho0': inFile.attrs['rho0'],
            'cs': inFile.attrs['c_s'],
        },
        'particle':{
            'support': inFile.attrs['support']
        }
    }
    return config, attributes, state, priorStates, nextStates
