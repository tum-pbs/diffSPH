import copy
import torch
from typing import NamedTuple, Optional, List, Union
from sphMath.schemes.baseScheme import SPHSystem

def updateStateEuler(systemState_: SPHSystem, systemUpdate, dt, copyState = True, **kwargs):
    if copyState:
        systemState = systemState_.initializeNewState(**kwargs)
    else:
        systemState = systemState_
    return systemState.integrate(systemUpdate, dt, **kwargs)

def updateStateSemiImplicitEuler(systemState_: SPHSystem, systemUpdate, dt, copyState = True, **kwargs):
    if copyState:
        systemState = systemState_.initializeNewState(**kwargs)
    else:
        systemState = systemState_
    systemState.integrateVelocity(systemUpdate, dt, semiImplicit = True, **kwargs)
    systemState.integratePosition(systemUpdate, 0., semiImplicitScale=dt, **kwargs)
    systemState.integrateQuantities(systemUpdate, dt, **kwargs)
    systemState.t = systemState.t + dt
    return systemState


from enum import Enum
import torch

from sphMath.enums import IntegrationSchemeType
from typing import Callable
class IntegrationScheme(NamedTuple):
    function: Callable
    name: str    
    identifier: IntegrationSchemeType
    order: int
    
    dissipation: bool = False
    nonLagrangian: bool = False
    
    def __call__(self, state, dt, f):
        return self.function(state, dt, f)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    
IntegrationSchemes = []

def integrateQ(q : torch.Tensor, dqdt : Union[torch.Tensor, List[torch.Tensor]], dt : Union[float, List[float]],
                selfScale: Optional[float] = None,
                verletScale: Optional[float] = None,
                verletValue: Optional[torch.Tensor] = None,
                referenceValue: Optional[torch.Tensor] = None,
                referenceWeight: Optional[float] = None,
                integrateSpecies: Optional[List[int]] = None,
                species: Optional[torch.Tensor] = None):
    if integrateSpecies is not None:
        mask = torch.zeros(q.shape[0], device=q.device, dtype=torch.bool)
        for i in integrateSpecies:
            mask = torch.logical_or(mask, species == i)
        newValue = q.clone()
        if selfScale is not None:
            newValue[mask] = newValue[mask] * selfScale
        if referenceValue is not None:
            newValue[mask] = newValue[mask] + referenceValue[mask] * referenceWeight
        if verletScale is not None:
            newValue[mask] = newValue[mask] + verletScale * verletValue[mask]
        if isinstance(dt, list):
            for i in range(len(dt)):
                newValue[mask] = newValue[mask] + dt[i] * dqdt[mask]
        else:
            newValue[mask] = newValue[mask] + dt * dqdt[mask]
        return newValue
    else:        
        newValue = q.clone()
        if selfScale is not None:
            newValue = newValue * selfScale
        if referenceValue is not None:
            newValue = newValue + referenceValue * referenceWeight
        if verletScale is not None:
            newValue = newValue + verletScale * verletValue
            
        if isinstance(dt, list):
            for i in range(len(dt)):
                newValue = newValue + dt[i] * dqdt[i]
        else:
            newValue = newValue + dt * dqdt
        return newValue

def is_multiple_return_values(func, *args, **kwargs):
    result = func(*args, **kwargs)
    return isinstance(result, tuple) and len(result) > 1
def split_return(rv):
    if isinstance(rv, tuple):
        return rv[0], rv[1:]
    return rv, None

from torch.profiler import record_function

def initializeSystem(currentSystem : SPHSystem, dt, *args, **kwargs):
    # print('Arguments: ', args)
    # print('Kwargs: ', kwargs)
    with record_function("[Integration] Preprocess"):
        return currentSystem.initialize(dt, *args, **kwargs)
    return currentSystem

def preprocessSystem(currentSystem : SPHSystem, initialSystem : SPHSystem, dt, *args, **kwargs):
    with record_function("[Integration] Preprocess"):
        return currentSystem.preprocess(initialSystem, dt, *args, **kwargs)
    return currentSystem

def postprocessSystem(initialSystem : SPHSystem, currentSystem : SPHSystem, dt, r, *args, **kwargs):
    with record_function("[Integration] Postprocess"):
        return initialSystem.postprocess(currentSystem, dt, r, *args, **kwargs)
    return initialSystem

def finalizeSystem(currentSystem : SPHSystem, initialSystem : SPHSystem, dt, *args, **kwargs):
    with record_function("[Integration] Finalize"):
        return currentSystem.finalize(initialSystem, dt, *args, **kwargs)
    return currentSystem


def updateStep(initialState, currentState, dt, f, *args, **kwargs):
    with record_function("[Integration] Update Step"):
        preprocessSystem(currentState, initialState, dt, *args, **kwargs)
        k, r = split_return(f(currentState, dt, *args, **kwargs))
        postprocessSystem(initialState, currentState, dt, r, *args, **kwargs)
        return k, r