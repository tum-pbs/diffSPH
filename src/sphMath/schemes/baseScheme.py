import torch
from dataclasses import dataclass
import dataclasses
from typing import Union, Tuple, Optional, List
from sphMath.neighborhood import DomainDescription, SparseCOO
from sphMath.neighborhood import NeighborhoodInformation


@dataclass(slots=True)
class SPHUpdate:
    positions: torch.Tensor
    velocities: torch.Tensor
    densities: Optional[torch.Tensor] = None

@dataclass
class SPHSystem:
    domain: DomainDescription
    
    neighborhoodInfo: NeighborhoodInformation
    t: float

    def integratePosition(self, 
                          systemUpdate : Union[SPHUpdate, List[SPHUpdate]], dt : Union[float, List[float]], 
                          verletScale: Optional[float] = None,
                          semiImplicitScale: Optional[float] = None,
                          selfScale: Optional[float] = None,
                          referenceState: Optional['SPHSystem'] = None,
                          referenceWeight: Optional[float] = None, **kwargs):
        raise NotImplementedError
    def integrateVelocity(self,
                          systemUpdate : Union[SPHUpdate, List[SPHUpdate]], dt : Union[float, List[float]],
                          selfScale: Optional[float] = None,
                          referenceState: Optional['SPHSystem'] = None,
                          referenceWeight: Optional[float] = None, **kwargs):
        raise NotImplementedError
    def integrateQuantities(self,
                            systemUpdate : Union[SPHUpdate, List[SPHUpdate]], dt : Union[float, List[float]],
                            selfScale: Optional[float] = None,
                            referenceState: Optional['SPHSystem'] = None,
                            referenceWeight: Optional[float] = None, 
                            densitySwitch = False, **kwargs):
        raise NotImplementedError
    def integrate(self,
                    systemUpdate : Union[SPHUpdate, List[SPHUpdate]], dt : Union[float, List[float]], **kwargs):
        raise NotImplementedError
        
    def initializeNewState(self, *args, **kwargs):
        raise NotImplementedError
    def initialize(self, dt, *args, **kwargs):
        raise NotImplementedError
    def preprocess(self, initialState, dt, *args, **kwargs):
        return self
    def postprocess(self, initialState, dt, returnValues, *args, **kwargs):
        return self
    def finalize(self, initialState, dt, returnValues, *args, **kwargs):
        return self
    
    
def verbosePrint(verbose, *args):
    if verbose:
        print(*args)

def printPositionIntegration(systemUpdate, dt : Union[float, List[float]], 
     selfScale: Optional[float] = None,
    verletScale: Optional[float] = None,
    verletValue: Optional[torch.Tensor] = None,
    semiImplicitScale: Optional[float] = None,
    referenceValue: Optional[torch.Tensor] = None,
    referenceWeight: Optional[float] = None,
    verbose = False):
    if not verbose:
        return
    string = f'x^[n+1] = '
    if selfScale is not None and selfScale != 1.:
        string += f'{selfScale} * x^[n] '
    elif selfScale is not None and selfScale == 1.:
        string += f'x^[n] '
    elif selfScale is None:
        string += f'x^[n] '
    if referenceValue is not None:
        string += f'+ {referenceWeight} * x^[r]'
    if verletScale is not None:
        string += f'+ {verletScale} * v^[n]'
    if semiImplicitScale is not None:
        string += f'+ {semiImplicitScale} * v^[n+1]'
    if dt > 0.:
        string += f'+ {dt} * dx/dt'
    print(string)
def printVelocityIntegration(systemUpdate, dt : Union[float, List[float]], 
     selfScale: Optional[float] = None,
    referenceValue: Optional[torch.Tensor] = None,
    referenceWeight: Optional[float] = None,
    verbose = False):
    if not verbose:
        return
    string = f'v^[n+1] = '
    if selfScale is not None and selfScale != 1.:
        string += f'{selfScale} * v^[n] '
    elif selfScale is not None and selfScale == 1.:
        string += f'v^[n] '
    elif selfScale is None:
        string += f'v^[n] '
    if referenceValue is not None:
        string += f'+ {referenceWeight} * v^[r]'
    if dt > 0.:
        string += f'+ {dt} * dv/dt'
    print(string)
def printQuantityIntegration(systemUpdate, dt : Union[float, List[float]],
     selfScale: Optional[float] = None,
    referenceValue: Optional[torch.Tensor] = None,
    referenceWeight: Optional[float] = None,
    verbose = False):
    if not verbose:
        return
    string = f'Q^[n+1] = '
    if selfScale is not None and selfScale != 1.:
        string += f'{selfScale} * Q^[n] '
    elif selfScale is not None and selfScale == 1.:
        string += f'Q^[n] '
    elif selfScale is None:
        string += f'Q^[n] '
    if referenceValue is not None:
        string += f'+ {referenceWeight} * Q^[r]'
    if dt > 0.:
        string += f'+ {dt} * dQ/dt'
    print(string)
