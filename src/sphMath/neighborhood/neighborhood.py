import torch
from typing import Union, Tuple, Optional, List


from .naive import radiusNaive, radiusNaiveFixed
import numpy as np

# import functorch.experimental.control_flow.cond

def radiusSearch( 
        queryPositions : torch.Tensor,
        referencePositions : Optional[torch.Tensor],
        support : Optional[Union[torch.Tensor,Tuple[torch.Tensor, torch.Tensor]]] = None,
        fixedSupport : Optional[torch.Tensor] = None,
        mode : str = 'gather',
        domainMin : Optional[torch.Tensor] = None,
        domainMax : Optional[torch.Tensor] = None,
        periodicity : Optional[Union[bool, torch.Tensor]] = None,
        hashMapLength = 4096,
        algorithm: str = 'naive',
        verbose: bool = False,
        returnStructure : bool = False
        ):
    
    assert algorithm in ['naive'], f'algorithm = {algorithm} not supported'
    assert mode in ['symmetric', 'scatter', 'gather'], f'mode = {mode} not supported'
    assert queryPositions.shape[1] == referencePositions.shape[1] if referencePositions is not None else True, f'queryPositions.shape[1] = {queryPositions.shape[1]} != referencePositions.shape[1] = {referencePositions.shape[1]}'
    assert hashMapLength > 0, f'hashMapLength = {hashMapLength} <= 0'
    assert periodicity.shape[0] == queryPositions.shape[1] if isinstance(periodicity, torch.Tensor) else True, f'len(periodicity) = {len(periodicity)} != queryPositions.shape[1] = {queryPositions.shape[1]}'
    assert domainMin.shape[0] == queryPositions.shape[1] if domainMin is not None else True, f'domainMin.shape[0] = {domainMin.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
    assert domainMax.shape[0] == queryPositions.shape[1] if domainMax is not None else True, f'domainMax.shape[0] = {domainMax.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
    # assert isinstance(support, float) or support.shape[0] == queryPositions.shape[0] if isinstance(support, torch.Tensor) else True, f'support.shape[0] = {support.shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
    # assert support[0].shape[0] == queryPositions.shape[0] if isinstance(support, tuple) else True, f'support[0].shape[0] = {support[0].shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
    # assert support[1].shape[0] == referencePositions.shape[0] if isinstance(support, tuple) else True, f'support[1].shape[0] = {support[1].shape[0]} != referencePositions.shape[0] = {referencePositions.shape[0]}'



    if referencePositions is None:
        referencePositions = queryPositions

    if fixedSupport is not None:
        supportRadius = fixedSupport
        querySupport = None
        referenceSupport = None
    elif support is not None and isinstance(support, torch.Tensor):
        supportRadius = None
        querySupport = support
        if mode == 'gather':
            referenceSupport = torch.zeros(referencePositions.shape[0], device = referencePositions.device)
        assert mode == 'gather', f'mode = {mode} != gather'
        assert querySupport.shape[0] == queryPositions.shape[0], f'querySupport.shape[0] = {querySupport.shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
    elif support is not None and isinstance(support, tuple):
        supportRadius = None
        querySupport = support[0]
        referenceSupport = support[1]
        assert querySupport.shape[0] == queryPositions.shape[0], f'querySupport.shape[0] = {querySupport.shape[0]} != queryPositions.shape[0] = {queryPositions.shape[0]}'
        assert referenceSupport.shape[0] == referencePositions.shape[0], f'referenceSupport.shape[0] = {referenceSupport.shape[0]} != referencePositions.shape[0] = {referencePositions.shape[0]}'
    if periodicity is not None:
        if isinstance(periodicity, bool):
            periodicTensor = torch.tensor([periodicity] * queryPositions.shape[1], device = queryPositions.device, dtype = torch.bool)
            if periodicity:
                assert domainMin is not None, f'domainMin = {domainMin} is None'
                assert domainMax is not None, f'domainMax = {domainMax} is None'
                assert domainMin.shape[0] == queryPositions.shape[1], f'domainMin.shape[0] = {domainMin.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
                assert domainMax.shape[0] == queryPositions.shape[1], f'domainMax.shape[0] = {domainMax.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
        else:
            periodicTensor = periodicity
            # assert len(periodicTensor) == queryPositions.shape[1], f'len(periodicTensor) = {len(periodicTensor)} != queryPositions.shape[1] = {queryPositions.shape[1]}'
            # if np.any(periodicTensor):
            #     assert domainMin is not None, f'domainMin = {domainMin} is None'
            #     assert domainMax is not None, f'domainMax = {domainMax} is None'
            #     assert domainMin.shape[0] == queryPositions.shape[1], f'domainMin.shape[0] = {domainMin.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'
            #     assert domainMax.shape[0] == queryPositions.shape[1], f'domainMax.shape[0] = {domainMax.shape[0]} != queryPositions.shape[1] = {queryPositions.shape[1]}'    
    else:
        periodicTensor = torch.tensor([False] * queryPositions.shape[1], dtype = torch.bool, device = queryPositions.device)

    # if torch.any(periodicTensor):
        # if algorithm == 'cluster':
            # raise ValueError(f'algorithm = {algorithm} not supported for periodic search')
            
    x = torch.stack([queryPositions[:,i] if not periodic_i else torch.remainder(queryPositions[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i, periodic_i in enumerate(periodicTensor)], dim = 1)
    y = torch.stack([referencePositions[:,i] if not periodic_i else torch.remainder(referencePositions[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i, periodic_i in enumerate(periodicTensor)], dim = 1)
    # else:
        # x = queryPositions
        # y = referencePositions

    if domainMin is None:
        domainMin = torch.zeros(queryPositions.shape[1], device = queryPositions.device)
    if domainMax is None:
        domainMax = torch.ones(queryPositions.shape[1], device = queryPositions.device)
    if supportRadius is not None:
        if algorithm == 'naive':
            if verbose:
                print('Calling radiusNaiveFixed, arguments:')
                print(f'queryPositions = {queryPositions.shape} on {queryPositions.device}')
                print(f'referencePositions = {referencePositions.shape} on {referencePositions.device}')
                print(f'supportRadius = {supportRadius}')
                print(f'periodicTensor = {periodicTensor}')
                print(f'domainMin = {domainMin.shape} on {domainMin.device}')
                print(f'domainMax = {domainMax.shape} on {domainMax.device}')
            return radiusNaiveFixed(x, y, supportRadius, periodicTensor, domainMin, domainMax)
        else:
            raise ValueError(f'algorithm = {algorithm} not supported')
    else:
        if algorithm == 'naive':
            if verbose:
                print('Calling radiusNaive, arguments:')
                print(f'queryPositions = {queryPositions.shape} on {queryPositions.device}')
                print(f'querySupport = {querySupport.shape} on {querySupport.device}')
                print(f'referencePositions = {referencePositions.shape} on {referencePositions.device}')
                print(f'referenceSupport = {referenceSupport.shape} on {referenceSupport.device}')
                print(f'periodicTensor = {periodicTensor}')
                print(f'domainMin = {domainMin.shape} on {domainMin.device}')
                print(f'domainMax = {domainMax.shape} on {domainMax.device}')
            return radiusNaive(x, y, querySupport, referenceSupport, periodicTensor, domainMin, domainMax, mode)
        else:
            raise ValueError(f'algorithm = {algorithm} not supported')
    pass


def radius(queryPositions : torch.Tensor,
        referencePositions : Optional[torch.Tensor],
        support : Union[float, torch.Tensor,Tuple[torch.Tensor, torch.Tensor]],
        batch_x : Optional[torch.Tensor] = None, batch_y : Optional[torch.Tensor] = None,
        mode : str = 'gather',
        domainMin : Optional[torch.Tensor] = None,
        domainMax : Optional[torch.Tensor] = None,
        periodicity : Optional[Union[bool, torch.Tensor]] = None,
        hashMapLength = 4096,
        algorithm: str = 'naive',
        verbose: bool = False,
        returnStructure : bool = False):
    if batch_x is None and batch_y is None:
        return radiusSearch(queryPositions, referencePositions, support, mode, domainMin, domainMax, periodicity, hashMapLength, algorithm, verbose, returnStructure)
    else:
        batchIDs = torch.unique(batch_x) if batch_x is not None else torch.unique(batch_y)
        if returnStructure:
            i = torch.empty(0, dtype = torch.long, device = queryPositions.device)
            j = torch.empty(0, dtype = torch.long, device = queryPositions.device)
            ds = {}
            offsets = []
            for batchID in batchIDs:
                if batch_x is not None:
                    mask_x = batch_x == batchID
                else:
                    mask_x = torch.ones_like(queryPositions, dtype = torch.bool)
                if batch_y is not None:
                    mask_y = batch_y == batchID
                else:
                    mask_y = torch.ones_like(referencePositions if referencePositions is not None else queryPositions, dtype = torch.bool)
                x = queryPositions[mask_x]
                y = referencePositions[mask_y] if referencePositions is not None else queryPositions[mask_y]
                i_batch, j_batch, ds_batch = radiusSearch(x, y, support, mode, domainMin, domainMax, periodicity, hashMapLength, algorithm, verbose, returnStructure)
                i = torch.cat([i, i_batch + offsets[0]])
                j = torch.cat([j, j_batch + offsets[1]])
                ds[batchID] = ds_batch
                if batch_x is not None:
                    offsets[0] += x.shape[0]
                if batch_y is not None:
                    offsets[1] += y.shape[0]
            return i, j, ds
        else:
            i = torch.empty(0, dtype = torch.long, device = queryPositions.device)
            j = torch.empty(0, dtype = torch.long, device = queryPositions.device)
            offsets = []
            for batchID in batchIDs:
                if batch_x is not None:
                    mask_x = batch_x == batchID
                else:
                    mask_x = torch.ones_like(queryPositions, dtype = torch.bool)
                if batch_y is not None:
                    mask_y = batch_y == batchID
                else:
                    mask_y = torch.ones_like(referencePositions, dtype = torch.bool)
                x = queryPositions[mask_x]
                y = referencePositions[mask_y] if referencePositions is not None else queryPositions[mask_y]
                i_batch, j_batch = radiusSearch(x, y, support, mode, domainMin, domainMax, periodicity, hashMapLength, algorithm, verbose, returnStructure)
                i = torch.cat([i, i_batch + offsets[0]])
                j = torch.cat([j, j_batch + offsets[1]])
                if batch_x is not None:
                    offsets[0] += x.shape[0]
                if batch_y is not None:
                    offsets[1] += y.shape[0]
            return i, j