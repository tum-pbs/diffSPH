import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py 
# from BasisConvolution.util.datautils import parseFile
import os
from diffSPH.dataLoaderUtils.state import DataConfiguration

class datasetLoader(Dataset):
    def __init__(self, data):
        self.frameCounts = [len(s['samples']) for s in data]
        self.fileNames = [s['fileName'] for s in data]
        
        self.indices = [s['samples'] for s in data]
        self.fileFormat = [s['style'] for s in data][0]
        self.data = data
        #self.counters = [indices[1] for s, indices in data]
        
#         print(frameCounts)
        
        
    def __len__(self):
#         print('len', np.sum(self.frameCounts))
        return np.sum(self.frameCounts)
    
    def __getitem__(self, idx):
#         print(idx , ' / ', np.sum(self.frameCounts))
        cs = np.cumsum(self.frameCounts)
        p = 0
        for i in range(cs.shape[0]):
#             print(p, idx, cs[i])
            if idx < cs[i] and idx >= p:
#                 print('Found index ', idx, 'in dataset ', i)
#                 print('Loading frame ', self.indices[i][idx - p], ' from dataset ', i, ' for ', idx, p)
                return self.fileNames[i], self.indices[i][idx - p], self.data[i], i, idx - p
        

                return (i, self.indices[i][idx - p]), (i, self.indices[i][idx-p])
#                 return torch.rand(10,1), 2
            p = cs[i]
        return None, None
    

def isTemporalData(inFile):
    if 'dataType' in inFile.attrs:
        return True

    if 'simulation' in inFile:
        return True
    if 'simulationExport' in inFile:
        return True
    if 'simulationData' in inFile:
        if 'simulator' in inFile.attrs:
            return True
        if 'fluidPosition' in inFile['simulationData']:
            return True    
    return False

import warnings
def getFrameCount(inFile):
    if 'dataType' in inFile.attrs and inFile.attrs['dataType'] == 'diffSPH':
        # print(f'DiffSPH format detected, returning frame count {len(inFile["simulationData"].keys())}')
        # print(f'Keys: {list(inFile["simulationData"].keys())}')
        return len(inFile['simulationData'].keys())


    if 'simulationExport' in inFile:
        if 'initial' in inFile['simulationExport']:
            return int(len(inFile['simulationExport'].keys()) -1) + 1
        return int(len(inFile['simulationExport'].keys()) -1)
    if 'simulation' in inFile:
        return int(len(inFile['simulation'].keys()) -1)
    if 'simulationData' in inFile:
        if 'simulator' in inFile.attrs:
            # print('Simulator found')
            # print(len(inFile['simulationData'].keys()))
            return int(len(inFile['simulationData'].keys()) -1)
        if 'fluidPosition' in inFile['simulationData']:
            return inFile['simulationData']['fluidPosition'].shape[0] - 1
        else:
            return int(len(inFile['simulationData'].keys()))
    raise ValueError('Could not parse file')

def getFrames(inFile):
    if 'dataType' in inFile.attrs and inFile.attrs['dataType'] == 'diffSPH':
        return np.arange(len(inFile['simulationData'].keys())).tolist(), list(inFile['simulationData'].keys())


    if 'simulationExport' in inFile:
        if 'initial' in inFile['simulationExport']:
            return [0] + [int(i) for i in inFile['simulationExport'].keys()], ['00000'] + list(inFile['simulationExport'].keys())
        return [int(i) for i in inFile['simulationExport'].keys()], list(inFile['simulationExport'].keys())
    if 'simulation' in inFile:
        return np.arange(len(inFile['simulation'].keys())).tolist(), list(inFile['simulation'].keys())
    if 'simulationData' in inFile:
        if 'simulator' in inFile.attrs:
            return np.arange(len(inFile['simulationData'].keys())).tolist(), list(inFile['simulationData'].keys())
        if 'fluidPosition' in inFile['simulationData']:
            return np.arange(inFile['simulationData']['fluidPosition'].shape[0]).tolist(), np.arange(inFile['simulationData']['fluidPosition'].shape[0]).tolist()
        else:
            return [int(i) for i in inFile['simulationData'].keys()], list(inFile['simulationData'].keys())
    raise ValueError('Could not parse file')


def getSamples(inFile, frameSpacing = 1, frameDistance = 1, maxRollout = 0, skip = 0, limit = 0):
    temporalData = isTemporalData(inFile)
    frameCount = getFrameCount(inFile)
    frames, samples = getFrames(inFile)

    if maxRollout > 0 and not temporalData:
        raise ValueError('Max rollout only supported for temporal data')
    if frameCount < maxRollout:
        raise ValueError('Max rollout larger than frame count')
    
    if skip > 0:
        if skip > len(frames):
            warnings.warn(f'Skip larger than frame count for {inFile.filename}')
        frames = frames[min(len(frames), skip):]
        samples = samples[min(len(samples), skip):]
    if limit > 0:
        if limit > len(frames):
            warnings.warn(f'Limit larger than frame count for {inFile.filename}')
        frames = frames[:min(len(frames), limit)]
        samples = samples[:min(len(samples), limit)]

    if not temporalData or maxRollout == 0:
        return frames[::frameSpacing], samples[::frameSpacing]
    else:
        if 'dataType' in inFile.attrs and inFile.attrs['dataType'] == 'diffSPH':
            lastPossible = len(frames) - maxRollout * frameDistance
        else:
            lastPossible = len(frames) - 1 - maxRollout * frameDistance
        if lastPossible < 0:
            raise ValueError('Frame count too low for max rollout')
        return frames[:lastPossible:frameSpacing], samples[:lastPossible:frameSpacing]


def getStyle(inFile):
    try:
        if 'dataType' in inFile.attrs and inFile.attrs['dataType'] == 'diffSPH':
            return 'diffSPH'

        if 'simulationExport' in inFile:
            if 'config' in inFile: # New format
                return 'newFormat'
            if 'config' not in inFile:
                if isTemporalData(inFile): # temporal old format data, test case II/III
                    return 'testcase_II'
                else:
                    raise ValueError('Unsupported Format for file')
        else:
            if 'simulationData' in inFile and 'simulator' in inFile.attrs:
                return 'cuMath'
            
            if 'simulation' in inFile: 
                return 'waveEquation'
            # This should be test case I with flat 1D data
            if isTemporalData(inFile):
                return 'testcase_I'
            else:
                return 'testcase_IV'
    except Exception as e:
        print('Unable to load frame (unknown format)')
        raise e
    
class DatasetLoader(Dataset):
    def __init__(self, data):
        self.frameCounts = [len(s['samples']) for s in data]
        self.fileNames = [s['fileName'] for s in data]
        
        self.indices = [s['samples'] for s in data]
        self.fileFormat = [s['style'] for s in data][0]
        self.data = data
        #self.counters = [indices[1] for s, indices in data]
        
#         print(frameCounts)
        
        
    def __len__(self):
#         print('len', np.sum(self.frameCounts))
        return np.sum(self.frameCounts)
    
    def __getitem__(self, idx):
#         print(idx , ' / ', np.sum(self.frameCounts))
        cs = np.cumsum(self.frameCounts)
        p = 0
        for i in range(cs.shape[0]):
#             print(p, idx, cs[i])
            if idx < cs[i] and idx >= p:
#                 print('Found index ', idx, 'in dataset ', i)
#                 print('Loading frame ', self.indices[i][idx - p], ' from dataset ', i, ' for ', idx, p)
                return self.fileNames[i], self.indices[i][idx - p], self.data[i], i, idx - p
        

                return (i, self.indices[i][idx - p]), (i, self.indices[i][idx-p])
#                 return torch.rand(10,1), 2
            p = cs[i]
        return None, None
    

def getDataLoader(data, batch_size, shuffle = True, verbose = False):
    if verbose:
        print('Setting up data loaders')
    train_ds = DatasetLoader(data)
    train_dataloader = DataLoader(train_ds, shuffle=shuffle, batch_size = batch_size).batch_sampler
    return train_ds, train_dataloader

def parseFile(inFile, config: DataConfiguration):
    # print(config.historyLength)
    frameDistance = config.frameDistance
    frameSpacing = config.frameSpacing
    maxRollout = config.maxRollout
    historyLength = config.historyLength

    temporalData = isTemporalData(inFile)
    skip = historyLength * frameDistance + config.skipInitialFrames
    limit = config.cutoff 

    # print(f'Parsing file {inFile.filename} with frameDistance={frameDistance}, frameSpacing={frameSpacing}, maxRollout={maxRollout}, skip={skip}, limit={limit}')
    # print(f'skipInitialFrames={config.skipInitialFrames}, historyLength={historyLength}, temporalData={temporalData}')
    # print(f'FrameCount = {getFrameCount(inFile)}')

    frames, samples = getSamples(inFile, frameSpacing=frameSpacing, frameDistance=frameDistance, maxRollout=maxRollout, skip=skip, limit=limit)

    data = {
        'fileName': inFile.filename,
        'frames': frames,
        'samples': samples,
        'frameDistance': frameDistance,
        'frameSpacing': frameSpacing,
        'maxRollout': maxRollout,
        'skip': skip,
        'limit': limit,
        'isTemporalData': temporalData,
        'style': getStyle(inFile)
    }
    # print(f'Found {len(frames)} frames and {len(samples)} samples in file {inFile.filename} with style {data["style"]}')
    
    return data


def processFolder(folder, config: DataConfiguration):
    folder = os.path.expanduser(folder)
    simulationFiles = sorted([folder + '/' + f for f in os.listdir(folder) if f.endswith('.hdf5') or f.endswith('.h5')])
    # print(f'Found {len(simulationFiles)} simulation files in {folder}, [{[s.split("/")[-1] for s in simulationFiles]}]')
    data = []
    for s in simulationFiles:
        try:
            # print(f'Processing file {s}')
            inFile = h5py.File(s, 'r')
            data.append(parseFile(inFile, config))
            inFile.close()
        except Exception as e:
            print(f'Error processing file {s}: {e}')
            continue
    if len(data) == 0:
        raise ValueError(f'No valid files found in folder {folder}')
    return data


from diffSPH.enums import KernelType
def kernelNameToKernel(kernel: str):
    if isinstance(kernel, dict):
        kernelName = kernel['name']
    if isinstance(kernel, np.int64) or isinstance(kernel, int):
        kernelName = KernelType(kernel).name
    
    if kernelName == 'Wendland2':
        return KernelType.Wendland2
    
    for k in KernelType:
        if k.name == kernelName:
            return k
    raise ValueError(f'Unknown kernel name: {kernelName}')

from typing import List
def buildRotationMatrix(angles : List[float], dim: int, device: torch.device = None, dtype: torch.dtype = None):
    if dim == 1:
        return torch.tensor([[1.0]], device=device, dtype=dtype)
    elif dim == 2:
        return torch.tensor([[torch.cos(angles), -torch.sin(angles)],
                             [torch.sin(angles), torch.cos(angles)]], device=device, dtype=dtype)
    elif dim == 3:
        angle_phi = angles[0]
        angle_theta = angles[1]
        return torch.tensor([
            [torch.cos(angle_phi) * torch.sin(angle_theta), -torch.sin(angle_phi), torch.cos(angle_phi) * torch.cos(angle_theta)],
            [torch.sin(angle_phi) * torch.sin(angle_theta), torch.cos(angle_phi), torch.sin(angle_phi) * torch.cos(angle_theta)],
            [torch.cos(angle_theta), 0, -torch.sin(angle_theta)]
        ], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")