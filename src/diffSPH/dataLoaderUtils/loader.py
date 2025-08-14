from diffSPH.dataLoaderUtils.testcaseIILoader import loadTestcaseIIState
from diffSPH.dataLoaderUtils.newFormatLoader import loadNewFormatState
from diffSPH.dataLoaderUtils.diffSPHLoader import loadDiffSPHState
from diffSPH.dataLoaderUtils.state import WeaklyCompressibleSPHState, CompressibleSPHState, RigidBodyState

from diffSPH.dataLoaderUtils.util import getStyle

def loadState(inFile, key, configuration, device, dtype):
    style = getStyle(inFile)

    if style == 'newFormat':
        # print(f'Loading new format state for key: {key}')
        return loadNewFormatState(inFile, key, configuration, device, dtype)
    elif style == 'testcase_II':
        # print(f'Loading Testcase II state for key: {key}')
        return loadTestcaseIIState(inFile, key, configuration, device, dtype)
    elif style == 'diffSPH':
        return loadDiffSPHState(inFile, key, configuration, device, dtype)
    else:
        raise ValueError(f'Unknown style: {style}. Supported styles are "newFormat" and "testcaseII".')
    
import h5py
import torch
def loadEntry(entry, configuration, device = 'cpu', dtype = torch.float32):
    inFile = h5py.File(entry[0], 'r')
    key = entry[1]
    
    result = loadState(inFile, key, configuration, device, dtype)
    inFile.close()
    return result

def loadBatch(dataset, batch, configuration, device = 'cpu', dtype = torch.float32):
    priorStates, currentState, nextStates, domains, configs, = [], [], [], [], []
    batched = isinstance(batch, list)
    for ib, b in enumerate(batch):
        if isinstance(b, int):
            entry = dataset[b]
        else:
            entry = b

        prior, current, nexts, dom, cfg = loadEntry(entry, configuration, device, dtype)

        if batched:
            for p in range(len(prior)):
                prior[p].batches = torch.ones_like(prior[p].UIDs) * ib
            current.batches = torch.ones_like(current.UIDs) * ib
            for n in range(len(nexts)):
                nexts[n].batches = torch.ones_like(nexts[n].UIDs) * ib

            if isinstance(current, WeaklyCompressibleSPHState):
                if current.rigidBodies is not None:
                    for rb in current.rigidBodies:
                        rb.batchID = ib
                    for rb in prior:
                        if rb.rigidBodies is not None:
                            for rbb in rb.rigidBodies:
                                rbb.batchID = ib
                    for n in nexts:
                        if n.rigidBodies is not None:
                            for rb in n.rigidBodies:
                                rb.batchID = ib

        priorStates.append(prior)
        currentState.append(current)
        nextStates.append(nexts)
        domains.append(dom)
        configs.append(cfg)
    
    return priorStates, currentState, nextStates, domains, configs