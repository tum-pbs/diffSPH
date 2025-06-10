from typing import List
import numpy as np
from typing import Union, Tuple, Optional, List
import torch
from dataclasses import dataclass
import dataclasses
from sphMath.neighborhood import DomainDescription, SparseCOO
from sphMath.schemes.baseScheme import SPHSystem, verbosePrint, printPositionIntegration, printVelocityIntegration, printQuantityIntegration
from sphMath.integrationSchemes.util import integrateQ
from sphMath.neighborhood import NeighborhoodInformation
from sphMath.schemes.states.compressiblesph import CompressibleState, CompressibleUpdate

def cloneOptionalTensor(T : Optional[torch.Tensor]):
    if T is None:
        return None
    else:
        return T.clone()
def detachOptionalTensor(T : Optional[torch.Tensor]):
    if T is None:
        return None
    else:
        return T.detach()

def checkTensor(T : torch.Tensor, dtype, device, label, verbose = False):
    if T is None:
        return
    if verbose:
        print(f'Tensor {label:16s}: {T.min().item():10.3e} - {T.max().item():10.3e} - {T.median().item():10.3e} - {T.std().item():10.3e} - \t {T.dtype} - {T.device} - {T.shape}')
    
    if T.dtype != dtype:
        raise ValueError(f'{label} has incorrect dtype {T.dtype} != {dtype}')
    if T.device != device:
        raise ValueError(f'{label} has incorrect device {T.device} != {device}')
    # if not T.is_contiguous():
        # raise ValueError(f'{label} is not contiguous')
    if torch.isnan(T).any():
        raise ValueError(f'{label} has NaN values')
    if torch.isinf(T).any():
        raise ValueError(f'{label} has Inf values')
    
from sphMath.neighborhood import SparseCOO, DomainDescription, SparseNeighborhood



# @dataclass(slots = True)
# class GhostParticleState:
#     positions: torch.Tensor
#     masses: torch.Tensor
#     supports: torch.Tensor

#     velocities: torch.Tensor
#     densities: torch.Tensor

#     boundaryDistances: torch.Tensor
#     boundaryDirections: torch.Tensor
#     ghostPositions: torch.Tensor
    
#     referenceParticleIndex: torch.Tensor
#     # Information flows from fluid particles to ghost particles
#     # Consequently rows are GHOST particles and columns are FLUID particles
#     particleToGhostNeighborhood: Optional[SparseNeighborhood] = None
#     # Information flows from fluid particles to boundary particles
#     particleToBoundaryNeighborhood: Optional[SparseNeighborhood] = None
#     # Inforamtion flows between boundary particles
#     boundaryToBoundaryNeighborhood: Optional[SparseNeighborhood] = None
#     # Information flows from boundary particles to fluid particles
#     boundaryToParticleNeighborhood: Optional[SparseNeighborhood] = None
    
#     kind: str = 'reflecting'
#     species: int = 1

import warnings
import copy
from dataclasses import dataclass, fields
@dataclass
class CompressibleSystem(SPHSystem):
    systemState: CompressibleState
    domain: DomainDescription
    
    neighborhoodInfo: NeighborhoodInformation
    t: float
    scheme: str = 'internalEnergy'
    
    # ghostState: Optional[GhostParticleState] = None

    priorStep: Optional[Tuple[CompressibleUpdate, CompressibleState]] = None
    
    def integratePosition(self, 
                          systemUpdate : Union[CompressibleUpdate, List[CompressibleUpdate]], dt : Union[float, List[float]], 
                          verletScale: Optional[float] = None,
                          semiImplicitScale: Optional[float] = None,
                          selfScale: Optional[float] = None,
                          referenceState: Optional['CompressibleSystem'] = None,
                          referenceWeight: Optional[float] = None, verbose = False, **kwargs):
        assert not (verletScale is not None and semiImplicitScale is not None), "Cannot use both verlet and semi-implicit integration"
        
        if verbose:
            print('[Integrate x]', end=' ')
            printPositionIntegration(systemUpdate, dt, selfScale=selfScale, verletScale=verletScale, semiImplicitScale=semiImplicitScale, referenceValue=referenceState.systemState.positions if referenceState is not None else None, referenceWeight=referenceWeight, verbose = verbose)


        updates = systemUpdate.positions if not isinstance(systemUpdate, list) else [su.positions for su in systemUpdate]     
        
        if updates is None or (isinstance(updates, list) and updates[0] is None):
            warnings.warn('No updates provided for positions!')
            return self
         
        velocities = systemUpdate.velocities if not isinstance(systemUpdate, list) else [su.velocities for su in systemUpdate]  
        
        self.systemState.positions = integrateQ(self.systemState.positions, updates, dt,
                                    selfScale=selfScale,
                                    verletScale=verletScale if verletScale is not None else semiImplicitScale,
                                    verletValue=velocities if verletScale is not None else self.systemState.velocities,
                                    referenceWeight=referenceWeight,
                                    referenceValue=referenceState.systemState.positions if referenceState is not None else None,
                                    integrateSpecies=[0] if self.systemState.kinds is not None else None,
                                    species=self.systemState.kinds)
        return self
    def integrateVelocity(self, 
                          systemUpdate : Union[CompressibleUpdate, List[CompressibleUpdate]], dt : Union[float, List[float]],
                          selfScale: Optional[float] = None,
                          referenceState: Optional['CompressibleSystem'] = None,
                          referenceWeight: Optional[float] = None, verbose = False, **kwargs):
        updates = systemUpdate.velocities if not isinstance(systemUpdate, list) else [su.velocities for su in systemUpdate]   
        if updates is None or (isinstance(updates, list) and updates[0] is None):
            warnings.warn('No updates provided for velocities!')
            return self
        if verbose:
            print('[Integrate v]', end=' ')
            printVelocityIntegration(systemUpdate, dt, selfScale=selfScale, referenceValue=referenceState.systemState.velocities if referenceState is not None else None, referenceWeight=referenceWeight, verbose = verbose)

        self.systemState.velocities = integrateQ(self.systemState.velocities, updates, dt,
                                    selfScale=selfScale,
                                    referenceWeight=referenceWeight,
                                    referenceValue=referenceState.systemState.velocities if referenceState is not None else None,
                                    integrateSpecies=[0] if self.systemState.kinds is not None else None,
                                    species=self.systemState.kinds)
        return self
    def integrateQuantities(self, 
                            systemUpdate : Union[CompressibleUpdate, List[CompressibleUpdate]], dt : Union[float, List[float]],
                            selfScale: Optional[float] = None,
                            referenceState: Optional['CompressibleSystem'] = None,
                            referenceWeight: Optional[float] = None, 
                            densitySwitch = False, verbose = False, **kwargs):
        if (isinstance(systemUpdate, list) and hasattr(systemUpdate[0], 'totalEnergies') and systemUpdate[0].totalEnergies is not None) or (hasattr(systemUpdate, 'totalEnergies') and systemUpdate.totalEnergies is not None):
            if self.systemState.totalEnergies is not None:
                if verbose:
                    print('[Integrate E]', end=' ')
                    printQuantityIntegration(systemUpdate, dt, selfScale=selfScale, referenceValue=referenceState.systemState.totalEnergies if referenceState is not None else None, referenceWeight=referenceWeight, verbose = verbose)

                updates = systemUpdate.totalEnergies if not isinstance(systemUpdate, list) else [su.totalEnergies for su in systemUpdate]   
                
                self.systemState.totalEnergies = integrateQ(self.systemState.totalEnergies, updates, dt,
                                            selfScale=selfScale,
                                            referenceWeight=referenceWeight,
                                            referenceValue=referenceState.systemState.totalEnergies if referenceState is not None else None,
                                    integrateSpecies=[0] if self.systemState.kinds is not None else None,
                                    species=self.systemState.kinds)
        if (isinstance(systemUpdate, list) and hasattr(systemUpdate[0], 'internalEnergies') and systemUpdate[0].internalEnergies is not None) or (hasattr(systemUpdate, 'internalEnergies') and systemUpdate.internalEnergies is not None):
            if self.systemState.internalEnergies is not None:
                if verbose:
                    print('[Integrate u]', end=' ')
                    printQuantityIntegration(systemUpdate, dt, selfScale=selfScale, referenceValue=referenceState.systemState.internalEnergies if referenceState is not None else None, referenceWeight=referenceWeight, verbose = verbose)

                updates = systemUpdate.internalEnergies if not isinstance(systemUpdate, list) else [su.internalEnergies for su in systemUpdate]   
                
                self.systemState.internalEnergies = integrateQ(self.systemState.internalEnergies, updates, dt,
                                            selfScale=selfScale,
                                            referenceWeight=referenceWeight,
                                            referenceValue=referenceState.systemState.internalEnergies if referenceState is not None else None,
                                    integrateSpecies=[0] if self.systemState.kinds is not None else None,
                                    species=self.systemState.kinds)
        if (isinstance(systemUpdate, list) and hasattr(systemUpdate[0], 'densities') and systemUpdate[0].densities is not None) or (hasattr(systemUpdate, 'densities') and systemUpdate.densities is not None):
            if self.systemState.densities is not None:
                if verbose:
                    print('[Integrate rho]', end=' ')
                    printQuantityIntegration(systemUpdate, dt, selfScale=selfScale, referenceValue=referenceState.systemState.densities if referenceState is not None else None, referenceWeight=referenceWeight, verbose = verbose)

                updates = systemUpdate.densities if not isinstance(systemUpdate, list) else [su.densities for su in systemUpdate]
                
                self.systemState.densities = integrateQ(self.systemState.densities, updates, dt,
                                            selfScale=selfScale,
                                            referenceWeight=referenceWeight,
                                            referenceValue=referenceState.systemState.densities if referenceState is not None else None,
                                    integrateSpecies=[0] if self.systemState.kinds is not None else None,
                                    species=self.systemState.kinds)

        return self
    def integrate(self, 
                    systemUpdate : Union[CompressibleUpdate, List[CompressibleUpdate]], dt : Union[float, List[float]], verbose = False, **kwargs):
        verbosePrint(verbose, f'Integration  - {dt}')

        self.integratePosition(systemUpdate, dt, verbose = verbose, **kwargs)
        self.integrateVelocity(systemUpdate, dt, verbose = verbose, **kwargs)
        self.integrateQuantities(systemUpdate, dt, verbose = verbose, **kwargs)
        self.t = self.t + dt if not isinstance(dt, list) else self.t + np.sum(dt)
        return self 
    def initializeNewState(self, *args, verbose = False, **kwargs):
        verbosePrint(verbose, 'Initializing new state')
        newSystem = CompressibleSystem(
            systemState = self.systemState.initializeNewState(self.scheme),
            domain = self.domain,
            neighborhoodInfo = self.neighborhoodInfo,
            t = self.t,
            scheme = self.scheme,
            # ghostState = self.ghostState
        )
        # print(self.scheme, 'density' not in self.scheme.lower())
        # newSystem.systemState.densities = None if 'density' not in self.scheme.lower() else newSystem.systemState.densities
        return newSystem
    def preprocess(self, initialState, dt, *args, verbose = False, **kwargs):
        return self
    def postprocess(self, currentState, dt, r, *args, verbose = False, **kwargs):
        verbosePrint(verbose, f'[Post] Integration [{currentState.systemState.positions.shape} - {self.t}/{currentState.t} - {dt}]')
        self.neighborhoodInfo = r[-1]

        return self
    def initialize(self, dt, *args, verbose = False, **kwargs):
        verbosePrint(verbose, '[Init] Integration')
        return self

    def finalize(self, initialState, dt, returnValues, updateValues, butcherValues, *args, verbose = False, **kwargs):
        verbosePrint(verbose, f'Finalizing system [{initialState.systemState.positions.shape} - {self.t}/{initialState.t} - {dt}]')

        verbosePrint(verbose, '[Fin]\tUpdating state')
        lastState = returnValues[-1][0]
        firsState = returnValues[0][0]
        lastUpdate = updateValues[-1]
        
        # if lastUpdate.passive is not None:
        #     # print('passive particles:', torch.sum(lastUpdate.passive))
            # self.systemState.velocities[lastUpdate.passive,:] = lastUpdate.positions[lastUpdate.passive,:]
            # self.systemState.velocities[lastUpdate.passive,:] = (self.systemState.positions - initialState.systemState.positions)[lastUpdate.passive,:]/dt
        #     if 'momentum' in self.scheme.lower():
        #         self.systemState.densities[lastUpdate.passive] = lastState.densities[lastUpdate.passive]
            
        self.systemState.pressures = lastState.pressures
        self.systemState.soundspeeds = lastState.soundspeeds
        if 'entropy' not in self.scheme.lower():
            self.systemState.entropies = lastState.entropies
        # if 'internalenergy' not in self.scheme.lower():
            # self.systemState.internalEnergies = lastState.internalEnergies
        if 'totalenergy' not in self.scheme.lower():
            self.systemState.totalEnergies = lastState.totalEnergies
        if 'density' not in self.scheme.lower():
            self.systemState.densities = lastState.densities
        self.systemState.densities = lastState.densities
        self.systemState.supports = lastState.supports
        
        self.systemState.n = lastState.n
        self.systemState.dndh = lastState.dndh
        self.systemState.P = lastState.P
        self.systemState.dPdh = lastState.dPdh
        
        self.systemState.alphas = lastState.alphas
        self.systemState.alpha0s = lastState.alpha0s
        self.systemState.divergence = firsState.divergence
        
        self.systemState.A = lastState.A
        self.systemState.B = lastState.B
        self.systemState.gradA = lastState.gradA
        self.systemState.gradB = lastState.gradB
        
        self.systemState.f_ij = lastState.f_ij
        self.systemState.av_ij = lastState.av_ij
        self.systemState.ap_ij = lastState.ap_ij
        
        self.systemState.kinds = lastState.kinds
        self.systemState.UIDs = lastState.UIDs
        self.systemState.materials = lastState.materials
        self.systemState.ghostIndices = lastState.ghostIndices
        self.systemState.ghostOffsets = lastState.ghostOffsets
        self.systemState.numNeighbors = lastState.numNeighbors

        referenceDtype = lastState.positions.dtype
        referenceDevice = lastState.positions.device
        
        if verbose:
            print('[Fin]\tChecking tensors')
        checkTensor(self.systemState.positions, referenceDtype, referenceDevice, 'positions', verbose = verbose)
        checkTensor(self.systemState.supports, referenceDtype, referenceDevice, 'supports', verbose = verbose)
        checkTensor(self.systemState.masses, referenceDtype, referenceDevice, 'masses', verbose = verbose)
        checkTensor(self.systemState.densities, referenceDtype, referenceDevice, 'densities', verbose = verbose)
        checkTensor(self.systemState.velocities, referenceDtype, referenceDevice, 'velocities', verbose = verbose)
        checkTensor(self.systemState.internalEnergies, referenceDtype, referenceDevice, 'internalEnergies', verbose = verbose)
        checkTensor(self.systemState.totalEnergies, referenceDtype, referenceDevice, 'totalEnergies', verbose = verbose)
        checkTensor(self.systemState.entropies, referenceDtype, referenceDevice, 'entropies', verbose = verbose)
        checkTensor(self.systemState.pressures, referenceDtype, referenceDevice, 'pressures', verbose = verbose)
        checkTensor(self.systemState.soundspeeds, referenceDtype, referenceDevice, 'soundspeeds', verbose = verbose)
        checkTensor(self.systemState.alphas, referenceDtype, referenceDevice, 'alphas', verbose = verbose)
        checkTensor(self.systemState.alpha0s, referenceDtype, referenceDevice, 'alpha0s', verbose = verbose)
        checkTensor(self.systemState.divergence, referenceDtype, referenceDevice, 'divergence', verbose = verbose)
        checkTensor(self.systemState.n, referenceDtype, referenceDevice, 'n', verbose = verbose)
        checkTensor(self.systemState.dndh, referenceDtype, referenceDevice, 'dndh', verbose = verbose)
        checkTensor(self.systemState.P, referenceDtype, referenceDevice, 'P', verbose = verbose)
        checkTensor(self.systemState.dPdh, referenceDtype, referenceDevice, 'dPdh', verbose = verbose)
        checkTensor(self.systemState.A, referenceDtype, referenceDevice, 'A', verbose = verbose)
        checkTensor(self.systemState.B, referenceDtype, referenceDevice, 'B', verbose = verbose)
        checkTensor(self.systemState.gradA, referenceDtype, referenceDevice, 'gradA', verbose = verbose)
        checkTensor(self.systemState.gradB, referenceDtype, referenceDevice, 'gradB', verbose = verbose)
        checkTensor(self.systemState.f_ij, referenceDtype, referenceDevice, 'f_ij', verbose = verbose)
        checkTensor(self.systemState.av_ij, referenceDtype, referenceDevice, 'av_ij', verbose = verbose)
        checkTensor(self.systemState.ap_ij, referenceDtype, referenceDevice, 'ap_ij', verbose = verbose) 
        # checkTensor(self.systemState.kinds, referenceDtype, referenceDevice, 'species', verbose = verbose)

        return self
    

    def nograd(self):
        return type(self)(
            systemState = self.systemState.nograd(),
            domain = self.domain,
            neighborhoodInfo = None,
            t = self.t,
            scheme = self.scheme
        )
    
    def to(self, dtype = torch.float32, device = torch.device('cpu')):
        convertedState = copy.deepcopy(self.systemState)
        convertedDomain = DomainDescription(
            min = self.domain.min.to(dtype=dtype, device=device),
            max = self.domain.max.to(dtype=dtype, device=device),
            periodic = self.domain.periodic,
            dim = self.domain.dim
        ) 

        for field in fields(convertedState):
            if getattr(convertedState, field.name) is not None and isinstance(getattr(convertedState, field.name), torch.Tensor):
                setattr(convertedState, field.name, getattr(convertedState, field.name).to(dtype=dtype, device=device))
        
        return type(self)(
            systemState = convertedState,
            domain = convertedDomain,
            neighborhoodInfo = None,
            t = self.t,
            scheme = self.scheme,
            # ghostState = None
        )
    

