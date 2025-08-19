import torch
import numpy as np
import h5py
import warnings
import copy

try:
    from diffSPH.v2.parameters import parseDefaultParameters, parseModuleParameters
    # from torchCompactRadius import radiusSearch
    hasDiffSPH = True
except ModuleNotFoundError:
    # from BasisConvolution.neighborhoodFallback.neighborhood import radiusSearch
    hasDiffSPH = False
    # pass

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

def parseSPHConfig(inFile, device, dtype):
    # if not hasDiffSPH:
        # raise ModuleNotFoundError('diffSPH is not installed, cannot parse SPH config')
    config = {}
    for key in inFile['config'].keys():
        config[key] = {}
        for subkey in inFile['config'][key].attrs.keys():
            # print(key,subkey)
            config[key][subkey] = inFile['config'][key].attrs[subkey]
        # print(key, config[key])

    if 'domain' in config:
        if 'minExtent' in config['domain']:
            config['domain']['minExtent'] = config['domain']['minExtent'].tolist()
        if 'maxExtent' in config['domain']:
            # print(config['domain']['maxExtent'])
            config['domain']['maxExtent'] = config['domain']['maxExtent'].tolist()
        if 'periodicity' in config['domain']:
            config['domain']['periodicity'] = config['domain']['periodicity'].tolist()
        if 'periodic' in config['domain']:
            config['domain']['periodic'] = bool(config['domain']['periodic'])
    config['compute']['device'] = device
    config['compute']['dtype'] = dtype
    config['simulation']['correctArea'] = False

    # if hasDiffSPH:
    #     parseDefaultParameters(config)
    #     parseModuleParameters(config)
    # else:
    #     raise ModuleNotFoundError('diffSPH is not installed, cannot parse SPH config')
    
    return config

def loadGroup_newFormat(inFile, inGrp, staticFluidData, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    if staticFluidData is not None and int(key) == 0 or inGrp is None:
        state = {
            'fluid': staticFluidData,
            'boundary': staticBoundaryData,
            'time': 0.0,
            'dt': inFile['config']['timestep'].attrs['dt'] * hyperParameterDict['frameDistance'],
            'timestep': 0,
        }
        loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
        return state


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
        dynamicBoundaryData['supports'] = torch.from_numpy(inGrp['boundarySupport'][:]).to(device = device, dtype = dtype) if 'boundarySupport' in inGrp else dynamicBoundaryData['supports']
        dynamicBoundaryData['bodyIDs'] = torch.from_numpy(inGrp['boundaryBodyAssociation'][:]).to(device = device, dtype = torch.int64) if 'boundaryBodyAssociation' in inGrp else dynamicBoundaryData['bodyIDs']
    elif 'initial' in inFile:
        dynamicBoundaryData = {} if staticBoundaryData is not None else None
        if staticBoundaryData is not None:
            for k in staticBoundaryData.keys():
                if isinstance(staticBoundaryData[k], torch.Tensor):
                    dynamicBoundaryData[k] = staticBoundaryData[k].clone()
                else:
                    dynamicBoundaryData[k] = staticBoundaryData[k]

        if 'boundaryDensity' in inGrp:
            dynamicBoundaryData['densities'] = torch.from_numpy(inGrp['boundaryDensity'][:]).to(device = device, dtype = dtype)
        if 'boundaryVelocity' in inGrp:
            dynamicBoundaryData['velocities'] = torch.from_numpy(inGrp['boundaryVelocity'][:]).to(device = device, dtype = dtype)
        if 'boundaryPosition' in inGrp:
            dynamicBoundaryData['positions'] = torch.from_numpy(inGrp['boundaryPosition'][:]).to(device = device, dtype = dtype)
        if 'boundaryNormals' in inGrp:
            dynamicBoundaryData['normals'] = torch.from_numpy(inGrp['boundaryNormals'][:]).to(device = device, dtype = dtype)
    else:
        dynamicBoundaryData = None
    if 'boundaryDensity' in inGrp:
        dynamicBoundaryData['densities'] = torch.from_numpy(inGrp['boundaryDensity'][:]).to(device = device, dtype = dtype)
    if 'boundaryVelocity' in inGrp:
        dynamicBoundaryData['velocities'] = torch.from_numpy(inGrp['boundaryVelocity'][:]).to(device = device, dtype = dtype)
    if 'boundaryPosition' in inGrp:
        dynamicBoundaryData['positions'] = torch.from_numpy(inGrp['boundaryPosition'][:]).to(device = device, dtype = dtype)
    if 'boundaryNormals' in inGrp:
        dynamicBoundaryData['normals'] = torch.from_numpy(inGrp['boundaryNormals'][:]).to(device = device, dtype = dtype)


    fluidState = {}

    if staticFluidData is not None:
        for k in staticFluidData.keys():
            if isinstance(staticFluidData[k], torch.Tensor):
                fluidState[k] = staticFluidData[k].clone()
            else:
                fluidState[k] = staticFluidData[k]
    
    if 'fluidPosition' in inGrp:
        fluidState['positions'] = torch.from_numpy(inGrp['fluidPosition'][:]).to(device = device, dtype = dtype)
    if 'fluidVelocity' in inGrp:
        fluidState['velocities'] = torch.from_numpy(inGrp['fluidVelocity'][:]).to(device = device, dtype = dtype)
    if 'fluidDensity' in inGrp:
        fluidState['densities'] = torch.from_numpy(inGrp['fluidDensity'][:]).to(device = device, dtype = dtype)
    if 'fluidGravity' in inGrp:
        fluidState['gravityAcceleration'] = torch.from_numpy(inGrp['fluidGravity'][:]).to(device = device, dtype = dtype)
    
    support = inFile.attrs['support'] if hyperParameterDict['numNeighbors'] < 0 else computeSupport(inFile.attrs['area'], hyperParameterDict['numNeighbors'], 2)
    rho = fluidState['densities']
    areas = torch.ones_like(rho) * inFile.attrs['area']

    fluidState['densities'] = rho #- rho.mean()#* inFile.attrs['restDensity']
    if hyperParameterDict['normalizeDensity']:
        fluidState['densities'] = (fluidState['densities'] - 1.0) * inFile.attrs['restDensity']
    fluidState['areas'] = areas
    fluidState['masses'] = areas * inFile.attrs['restDensity']
    fluidState['supports'] = torch.ones_like(rho) * support
    fluidState['indices'] = torch.from_numpy(inGrp['UID'][:]).to(device = device, dtype = torch.int64)
    fluidState['numParticles'] = len(rho)

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

def loadFrame_newFormat(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    # print(f'Loading frame {key} from {fileName} ')
    # print(key)

    if 'initial' in inFile:
        targetNeighbors = inFile.attrs['targetNeighbors']

        staticFluidData = {
            'positions': torch.from_numpy(inFile['initial']['fluid']['positions'][:]).to(device = device, dtype = dtype),
            'velocities': torch.from_numpy(inFile['initial']['fluid']['velocities'][:]).to(device = device, dtype = dtype),
            'gravityAcceleration': torch.zeros_like(torch.from_numpy(inFile['initial']['fluid']['velocities'][:]).to(device = device, dtype = dtype)),
            'densities': torch.from_numpy(inFile['initial']['fluid']['densities'][:]).to(device = device, dtype = dtype),
            'areas': torch.from_numpy(inFile['initial']['fluid']['areas'][:]).to(device = device, dtype = dtype),
            'masses': torch.from_numpy(inFile['initial']['fluid']['masses'][:]).to(device = device, dtype = dtype),
            'supports': computeSupport(torch.from_numpy(inFile['initial']['fluid']['areas'][:]).to(device = device, dtype = dtype), targetNeighbors, 2),
            'indices': torch.from_numpy(inFile['initial']['fluid']['UID'][:]).to(device = device, dtype = torch.int32),
            'numParticles': len(inFile['initial']['fluid']['positions'][:]),                                  
        }

        config = parseSPHConfig(inFile, device, dtype)
        area = inFile.attrs['radius'] **2 if 'area' not in inFile.attrs else inFile.attrs['area']
        support = np.max(staticFluidData['supports'].detach().cpu().numpy()) if 'support' not in inFile.attrs else inFile.attrs['support']
        if hyperParameterDict['numNeighbors'] > 0:
            support = computeSupport(area, hyperParameterDict['numNeighbors'], 2)
        attributes = {
            'support': support,
            'targetNeighbors': inFile.attrs['targetNeighbors'],
            'restDensity': inFile.attrs['restDensity'],
            'dt': config['timestep']['dt'] * hyperParameterDict['frameDistance'],
            'time': 0.0,
            'radius': inFile.attrs['radius'],
            'area': area,
        }
        if key == '00000':
            inGrp = None
        else:
            inGrp = inFile['simulationExport'][key] 

    else:
        staticFluidData = None
        inGrp = inFile['simulationExport'][key]

        # print(inFile.attrs.keys())
        # for k in inFile.attrs.keys():
            # print(k, inFile.attrs[k])

        config = parseSPHConfig(inFile, device, dtype)
        area = inFile.attrs['radius'] **2 if 'area' not in inFile.attrs else inFile.attrs['area']
        support = np.max(inGrp['fluidSupport'][:]) if 'support' not in inFile.attrs else inFile.attrs['support']
        if hyperParameterDict['numNeighbors'] > 0:
            support = computeSupport(area, hyperParameterDict['numNeighbors'], 2)
        attributes = {
            'support': support,
            'targetNeighbors': inFile.attrs['targetNeighbors'],
            'restDensity': inFile.attrs['restDensity'],
            'dt': inGrp.attrs['dt'] * hyperParameterDict['frameDistance'],
            'time': inGrp.attrs['time'],
            'radius': inFile.attrs['radius'] if 'radius' in inFile.attrs else inGrp.attrs['radius'],
            'area': area,
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
    elif 'initial' in inFile:
        staticBoundaryData = {
            'indices': torch.from_numpy(inFile['initial']['boundary']['UID'][:]).to(device = device, dtype = torch.int64),
            'positions': torch.from_numpy(inFile['initial']['boundary']['positions'][:]).to(device = device, dtype = dtype),
            'normals': torch.from_numpy(inFile['initial']['boundary']['normals'][:]).to(device = device, dtype = dtype),
            'distances': torch.from_numpy(inFile['initial']['boundary']['distances'][:]).to(device = device, dtype = dtype),
            'areas': torch.from_numpy(inFile['initial']['boundary']['areas'][:]).to(device = device, dtype = dtype),
            'masses': torch.from_numpy(inFile['initial']['boundary']['masses'][:]).to(device = device, dtype = dtype),
            'velocities': torch.from_numpy(inFile['initial']['boundary']['velocities'][:]).to(device = device, dtype = dtype),
            'densities': torch.from_numpy(inFile['initial']['boundary']['densities'][:]).to(device = device, dtype = dtype),
            'supports': computeSupport(torch.from_numpy(inFile['initial']['boundary']['areas'][:]).to(device = device, dtype = dtype), inFile.attrs['targetNeighbors'], 2),
            'bodyIDs': torch.from_numpy(inFile['initial']['boundary']['bodyIDs'][:]).to(device = device, dtype = torch.int64),
            'numParticles': len(inFile['initial']['boundary']['UID'][:]),

        } if 'boundary' in inFile['initial'] else None
    else:
        staticBoundaryData = None

    # if 'boundaryInformation' in inFile:
    #     dynamicBoundaryData = {}
    #     for k in staticBoundaryData.keys():
    #         if isinstance(staticBoundaryData[k], torch.Tensor):
    #             dynamicBoundaryData[k] = staticBoundaryData[k].clone()
    #         else:
    #             dynamicBoundaryData[k] = staticBoundaryData[k]


    # else:
    #     dynamicBoundaryData = None

    state = loadGroup_newFormat(inFile, inGrp, staticFluidData, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = buildPriorState, buildNextState = buildNextState)


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
                grp = inFile['simulationExport']['%05d' % iPriorKey] if '%05d' % iPriorKey in inFile['simulationExport'] else None
                # if grp is None:
                    # print('Key %s not found in file' % iPriorKey)
                priorState = loadGroup_newFormat(inFile, grp, staticFluidData, staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)
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
                nextState = loadGroup_newFormat(inFile, inFile['simulationExport']['%05d' % unrollKey], staticFluidData, staticBoundaryData, fileName, unrollKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)                
                nextStates.append(nextState)            

    # if hyperParameterDict['adjustForFrameDistance']:

    config['particle']['support'] = support

    # print('Loaded frame %s' % key)

    return config, attributes, state, priorStates, nextStates



def loadGroup_newFormat_v2(inFile, inGrp, staticFluidData, staticBoundaryData, key, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    if staticFluidData is not None and int(key) == 0 or inGrp is None:
        state = {
            'fluid': staticFluidData,
            'boundary': staticBoundaryData,
            'time': 0.0,
            'dt': inFile['config']['timestep'].attrs['dt'],# * hyperParameterDict['frameDistance'],
            'timestep': 0,
        }
        loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
        return state


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
        dynamicBoundaryData['densities'] = torch.from_numpy(inGrp['boundaryDensity'][:]).to(device = device, dtype = dtype) * inFile.attrs['restDensity']    if 'boundaryDensity' in inGrp else dynamicBoundaryData['densities']
        dynamicBoundaryData['supports'] = torch.from_numpy(inGrp['boundarySupport'][:]).to(device = device, dtype = dtype) if 'boundarySupport' in inGrp else dynamicBoundaryData['supports']
        dynamicBoundaryData['bodyIDs'] = torch.from_numpy(inGrp['boundaryBodyAssociation'][:]).to(device = device, dtype = torch.int64) if 'boundaryBodyAssociation' in inGrp else dynamicBoundaryData['bodyIDs']
    elif 'initial' in inFile:
        dynamicBoundaryData = {} if staticBoundaryData is not None else None
        if staticBoundaryData is not None:
            for k in staticBoundaryData.keys():
                if isinstance(staticBoundaryData[k], torch.Tensor):
                    dynamicBoundaryData[k] = staticBoundaryData[k].clone()
                else:
                    dynamicBoundaryData[k] = staticBoundaryData[k]

        if 'boundaryDensity' in inGrp:
            dynamicBoundaryData['densities'] = torch.from_numpy(inGrp['boundaryDensity'][:]).to(device = device, dtype = dtype) * inFile.attrs['restDensity']   
        if 'boundaryVelocity' in inGrp:
            dynamicBoundaryData['velocities'] = torch.from_numpy(inGrp['boundaryVelocity'][:]).to(device = device, dtype = dtype)
        if 'boundaryPosition' in inGrp:
            dynamicBoundaryData['positions'] = torch.from_numpy(inGrp['boundaryPosition'][:]).to(device = device, dtype = dtype)
        if 'boundaryNormals' in inGrp:
            dynamicBoundaryData['normals'] = torch.from_numpy(inGrp['boundaryNormals'][:]).to(device = device, dtype = dtype)
    else:
        dynamicBoundaryData = None
    if 'boundaryDensity' in inGrp:
        dynamicBoundaryData['densities'] = torch.from_numpy(inGrp['boundaryDensity'][:]).to(device = device, dtype = dtype) * inFile.attrs['restDensity']   
    if 'boundaryVelocity' in inGrp:
        dynamicBoundaryData['velocities'] = torch.from_numpy(inGrp['boundaryVelocity'][:]).to(device = device, dtype = dtype)
    if 'boundaryPosition' in inGrp:
        dynamicBoundaryData['positions'] = torch.from_numpy(inGrp['boundaryPosition'][:]).to(device = device, dtype = dtype)
    if 'boundaryNormals' in inGrp:
        dynamicBoundaryData['normals'] = torch.from_numpy(inGrp['boundaryNormals'][:]).to(device = device, dtype = dtype)


    fluidState = {}

    if staticFluidData is not None:
        for k in staticFluidData.keys():
            if isinstance(staticFluidData[k], torch.Tensor):
                fluidState[k] = staticFluidData[k].clone()
            else:
                fluidState[k] = staticFluidData[k]
    
    if 'fluidPosition' in inGrp:
        fluidState['positions'] = torch.from_numpy(inGrp['fluidPosition'][:]).to(device = device, dtype = dtype)
    if 'fluidVelocity' in inGrp:
        fluidState['velocities'] = torch.from_numpy(inGrp['fluidVelocity'][:]).to(device = device, dtype = dtype)
    if 'fluidDensity' in inGrp:
        fluidState['densities'] = torch.from_numpy(inGrp['fluidDensity'][:]).to(device = device, dtype = dtype) * inFile.attrs['restDensity']   
    if 'fluidGravity' in inGrp:
        fluidState['gravityAcceleration'] = torch.from_numpy(inGrp['fluidGravity'][:]).to(device = device, dtype = dtype)
    
    support = inFile.attrs['support'] #if hyperParameterDict['numNeighbors'] < 0 else computeSupport(inFile.attrs['area'], hyperParameterDict['numNeighbors'], 2)
    rho = fluidState['densities']
    areas = torch.ones_like(rho) * inFile.attrs['area']

    fluidState['densities'] = rho #- rho.mean()#* inFile.attrs['restDensity']
    # if hyperParameterDict['normalizeDensity']:
        # fluidState['densities'] = (fluidState['densities'] - 1.0) * inFile.attrs['restDensity']
    fluidState['areas'] = areas
    fluidState['masses'] = areas * inFile.attrs['restDensity']
    fluidState['supports'] = torch.ones_like(rho) * support
    fluidState['indices'] = torch.from_numpy(inGrp['UID'][:]).to(device = device, dtype = torch.int64)
    fluidState['numParticles'] = len(rho)

    # for k in inGrp.keys():
        # print(k, inGrp[k])

    # support = inFile.attrs['support'] if hyperParameterDict['numNeighbors'] < 0 else computeSupport(inFile.attrs['area'], hyperParameterDict['numNeighbors'], 2)
    # rho = torch.from_numpy(inGrp['fluidDensity'][:]).to(device = device, dtype = dtype)
    # areas = torch.ones_like(rho) * inFile.attrs['area']
    state = {
        'fluid': fluidState,
        'boundary': dynamicBoundaryData if dynamicBoundaryData is not None else staticBoundaryData,
        'time': inGrp.attrs['time'],
        'dt': inGrp.attrs['dt'],
        'timestep': inGrp.attrs['timestep'],
    }
    loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
    # for dataKey in additionalData:
        # state['fluid'][dataKey] = torch.from_numpy(np.array(inGrp[dataKey])).to(device = device, dtype = dtype)
    
    return state


def loadFrame_newFormat_v2(inFile, key: str, device: str = 'cpu', dtype: torch.dtype = torch.float32):
    # print(f'Loading frame {key} from {inFile.filename}')

    if 'initial' in inFile:
        targetNeighbors = inFile.attrs['targetNeighbors']

        staticFluidData = {
            'positions': torch.from_numpy(inFile['initial']['fluid']['positions'][:]).to(device = device, dtype = dtype),
            'velocities': torch.from_numpy(inFile['initial']['fluid']['velocities'][:]).to(device = device, dtype = dtype),
            'gravityAcceleration': torch.zeros_like(torch.from_numpy(inFile['initial']['fluid']['velocities'][:]).to(device = device, dtype = dtype)),
            'densities': torch.from_numpy(inFile['initial']['fluid']['densities'][:]).to(device = device, dtype = dtype) * inFile.attrs['restDensity']   ,
            'areas': torch.from_numpy(inFile['initial']['fluid']['areas'][:]).to(device = device, dtype = dtype),
            'masses': torch.from_numpy(inFile['initial']['fluid']['masses'][:]).to(device = device, dtype = dtype),
            'supports': computeSupport(torch.from_numpy(inFile['initial']['fluid']['areas'][:]).to(device = device, dtype = dtype), targetNeighbors, 2),
            'indices': torch.from_numpy(inFile['initial']['fluid']['UID'][:]).to(device = device, dtype = torch.int32),
            'numParticles': len(inFile['initial']['fluid']['positions'][:]),                                  
        }

        config = parseSPHConfig(inFile, device, dtype)
        area = inFile.attrs['radius'] **2 if 'area' not in inFile.attrs else inFile.attrs['area']
        support = np.max(staticFluidData['supports'].detach().cpu().numpy()) if 'support' not in inFile.attrs else inFile.attrs['support']
        # if hyperParameterDict['numNeighbors'] > 0:
            # support = computeSupport(area, hyperParameterDict['numNeighbors'], 2)
        # attributes = {
        #     'support': support,
        #     'targetNeighbors': inFile.attrs['targetNeighbors'],
        #     'restDensity': inFile.attrs['restDensity'],
        #     'dt': config['timestep']['dt'] * hyperParameterDict['frameDistance'],
        #     'time': 0.0,
        #     'radius': inFile.attrs['radius'],
        #     'area': area,
        # }
        if key == '00000':
            inGrp = None
        else:
            inGrp = inFile['simulationExport'][key] 

    else:
        staticFluidData = None
        inGrp = inFile['simulationExport'][key]

        # print(inFile.attrs.keys())
        # for k in inFile.attrs.keys():
            # print(k, inFile.attrs[k])

        config = parseSPHConfig(inFile, device, dtype)
        area = inFile.attrs['radius'] **2 if 'area' not in inFile.attrs else inFile.attrs['area']
        # support = np.max(inGrp['fluidSupport'][:]) if 'support' not in inFile.attrs else inFile.attrs['support']
        # if hyperParameterDict['numNeighbors'] > 0:
        #     support = computeSupport(area, hyperParameterDict['numNeighbors'], 2)
        # attributes = {
        #     'support': support,
        #     'targetNeighbors': inFile.attrs['targetNeighbors'],
        #     'restDensity': inFile.attrs['restDensity'],
        #     'dt': inGrp.attrs['dt'] * hyperParameterDict['frameDistance'],
        #     'time': inGrp.attrs['time'],
        #     'radius': inFile.attrs['radius'] if 'radius' in inFile.attrs else inGrp.attrs['radius'],
        #     'area': area,
        # }



    if 'boundaryInformation' in inFile:
        staticBoundaryData = {
                'indices': torch.arange(0, inFile['boundaryInformation']['boundaryPosition'].shape[0], device = device, dtype = torch.int64),
                'positions': torch.from_numpy(inFile['boundaryInformation']['boundaryPosition'][:]).to(device = device, dtype = dtype),
                'normals': torch.from_numpy(inFile['boundaryInformation']['boundaryNormals'][:]).to(device = device, dtype = dtype),
                'areas': torch.from_numpy(inFile['boundaryInformation']['boundaryArea'][:]).to(device = device, dtype = dtype),
                'masses': torch.from_numpy(inFile['boundaryInformation']['boundaryArea'][:]).to(device = device, dtype = dtype) * config['fluid']['rho0'],
                'velocities': torch.from_numpy(inFile['boundaryInformation']['boundaryVelocity'][:]).to(device = device, dtype = dtype),
                'densities': torch.from_numpy(inFile['boundaryInformation']['boundaryRestDensity'][:]).to(device = device, dtype = dtype) * inFile.attrs['restDensity']   ,
                'supports': torch.from_numpy(inFile['boundaryInformation']['boundarySupport'][:]).to(device = device, dtype = dtype),
                'bodyIDs': torch.from_numpy(inFile['boundaryInformation']['boundaryBodyAssociation'][:]).to(device = device, dtype = torch.int64),
                'numParticles': len(inFile['boundaryInformation']['boundaryPosition'][:]),
            } if 'boundaryInformation' in inFile else None
    elif 'initial' in inFile:
        staticBoundaryData = {
            'indices': torch.from_numpy(inFile['initial']['boundary']['UID'][:]).to(device = device, dtype = torch.int64),
            'positions': torch.from_numpy(inFile['initial']['boundary']['positions'][:]).to(device = device, dtype = dtype),
            'normals': torch.from_numpy(inFile['initial']['boundary']['normals'][:]).to(device = device, dtype = dtype),
            'distances': torch.from_numpy(inFile['initial']['boundary']['distances'][:]).to(device = device, dtype = dtype),
            'areas': torch.from_numpy(inFile['initial']['boundary']['areas'][:]).to(device = device, dtype = dtype),
            'masses': torch.from_numpy(inFile['initial']['boundary']['masses'][:]).to(device = device, dtype = dtype),
            'velocities': torch.from_numpy(inFile['initial']['boundary']['velocities'][:]).to(device = device, dtype = dtype),
            'densities': torch.from_numpy(inFile['initial']['boundary']['densities'][:]).to(device = device, dtype = dtype) * inFile.attrs['restDensity']   ,
            'supports': computeSupport(torch.from_numpy(inFile['initial']['boundary']['areas'][:]).to(device = device, dtype = dtype), inFile.attrs['targetNeighbors'], 2),
            'bodyIDs': torch.from_numpy(inFile['initial']['boundary']['bodyIDs'][:]).to(device = device, dtype = torch.int64),
            'numParticles': len(inFile['initial']['boundary']['UID'][:]),

        } if 'boundary' in inFile['initial'] else None
    else:
        staticBoundaryData = None

    # if 'boundaryInformation' in inFile:
    #     dynamicBoundaryData = {}
    #     for k in staticBoundaryData.keys():
    #         if isinstance(staticBoundaryData[k], torch.Tensor):
    #             dynamicBoundaryData[k] = staticBoundaryData[k].clone()
    #         else:
    #             dynamicBoundaryData[k] = staticBoundaryData[k]


    # else:
    #     dynamicBoundaryData = None
    additionalData = []
    state = loadGroup_newFormat_v2(inFile, inGrp, staticFluidData, staticBoundaryData, key, device = device, dtype = dtype, additionalData = additionalData)
    return state, config


from diffSPH.dataLoaderUtils.state import WeaklyCompressibleSPHState, convertNewFormatToWCSPH


from diffSPH.dataLoaderUtils.state import DataConfiguration
try:
    from torchCompactRadius.util import DomainDescription
except ImportError:
    from diffSPH.dataLoaderUtils.fallback import DomainDescription
from diffSPH.dataLoaderUtils.neighborhood import AugmentedDomainDescription

def loadNewFormatState(inFile, key, configuration : DataConfiguration, device = 'cpu', dtype = torch.float32):

    currentState, config = loadFrame_newFormat_v2(inFile, key, device = device, dtype = dtype)
    currentState = convertNewFormatToWCSPH(inFile, key, currentState)

    if configuration.historyLength > 0:
        priorStates = []
        for h in range(configuration.historyLength):
            iPriorKey = int(key) - configuration.frameDistance * (h + 1)
            if iPriorKey < 0 or configuration.frameDistance == 0:
                priorState = copy.deepcopy(currentState)
            else:
                priorState,_ = loadFrame_newFormat_v2(inFile, '%05d' % iPriorKey, device = device, dtype = dtype)
            priorState = convertNewFormatToWCSPH(inFile, '%05d' % iPriorKey, priorState)
            priorStates.append(priorState)
        priorStates.reverse()

    else:
        priorStates  = []

    if configuration.maxRollout > 0:
        trajectoryStates = []
        for u in range(configuration.maxRollout):
            unrollKey = int(key) + configuration.frameDistance * (u + 1)
            nextState,_ = loadFrame_newFormat_v2(inFile, '%05d' % unrollKey, device = device, dtype = dtype)
            nextState = convertNewFormatToWCSPH(inFile, '%05d' % unrollKey, nextState)
            trajectoryStates.append(nextState)
    else: 
        trajectoryStates = []

    # print(config['domain'])

    domain = AugmentedDomainDescription(
        min = torch.tensor(config['domain']['minExtent'][:], device = device, dtype = dtype),
        max = torch.tensor(config['domain']['maxExtent'][:], device = device, dtype = dtype),
        periodic = torch.tensor(config['domain']['periodicity'][:], device = device, dtype = torch.bool),
        dim = 2,
        angles = [0.0],
        device = device,
        dtype = dtype
    )

    return priorStates, currentState, trajectoryStates, domain, config
