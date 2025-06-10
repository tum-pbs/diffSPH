from typing import Union, Tuple
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood
from sphMath.operations import GradientMode, LaplacianMode
from sphMath.kernels import SPHKernel
from sphMath.enums import *
from sphMath.operations import SPHOperation, Operation
from sphMath.schemes.states.common import BasicState
from sphMath.neighborhood import SupportScheme, evaluateNeighborhood
from sphMath.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr, filterNeighborhoodByKind

from dataclasses import dataclass
def verbosePrint(verbose, *args):
    if verbose:
        print(*args)


@torch.jit.script
@dataclass(slots  = True)
class WaveEquationState:
    u: torch.Tensor
    v: torch.Tensor
    c: torch.Tensor
    damping: torch.Tensor

    def initializeNewState(self, scheme : str = 'internalEnergy'):
        return WaveEquationState(
            u = self.u.clone(),
            v = self.v.clone(),
            c = self.c.clone(),
            damping = self.damping.clone()
        )
    def detach(self):
        return WaveEquationState(
            u = self.u.detach(),
            v = self.v.detach(),
            c = self.c.detach(),
            damping = self.damping.detach()
        )
    


@dataclass
class WaveSystemUpdate:
    dudt: torch.Tensor
    dvdt: torch.Tensor
    


from dataclasses import dataclass
from typing import List, Optional, Union
import warnings
from sphMath.integrationSchemes.util import integrateQ
from sphMath.integration import getIntegrator


import numpy as np
@dataclass
class WaveSystem:
    systemState: BasicState
    waveState: WaveEquationState
    neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood]
    t: float

    def initialize(self, dt, *args, verbose = False, **kwargs):
        verbosePrint(verbose, '[Init] Integration')
        return self
    def initializeNewState(self, *args, verbose = False, **kwargs):
        verbosePrint(verbose, 'Initializing new state')
        newSystem = WaveSystem(
            systemState = self.systemState.initializeNewState('unused'),
            waveState = self.waveState.initializeNewState('unused'),
            neighborhood = self.neighborhood,
            t = self.t
        )
        # print(self.scheme, 'density' not in self.scheme.lower())
        # newSystem.systemState.densities = None if 'density' not in self.scheme.lower() else newSystem.systemState.densities
        return newSystem
    def preprocess(self, initialState, dt, *args, verbose = False, **kwargs):
        verbosePrint(verbose, f'[Pre] Integration [{self.t}/{initialState.t} - {dt}]')
        return self    
    def postprocess(self, currentState, dt, r, *args, verbose = False, **kwargs):
        verbosePrint(verbose, f'[Post] Integration [{currentState.systemState.positions.shape} - {self.t}/{currentState.t} - {dt}]')
        return self
    def finalize(self, initialState, dt, returnValues, updateValues, butcherValues, *args, verbose = False, **kwargs):
        verbosePrint(verbose, f'Finalizing system [{initialState.systemState.positions.shape} - {self.t}/{initialState.t} - {dt}]')
        return self
    def integratePosition(self, 
                          systemUpdate : Union[WaveSystemUpdate, List[WaveSystemUpdate]], dt : Union[float, List[float]], 
                          verletScale: Optional[float] = None,
                          semiImplicitScale: Optional[float] = None,
                          selfScale: Optional[float] = None,
                          referenceState: Optional['WaveSystem'] = None,
                          referenceWeight: Optional[float] = None, verbose = False, **kwargs):
        assert not (verletScale is not None and semiImplicitScale is not None), "Cannot use both verlet and semi-implicit integration"
        
        if verbose:
            print('[Integrate u]')

        updates = systemUpdate.dudt if not isinstance(systemUpdate, list) else [su.dudt for su in systemUpdate]     
        
        if updates is None or (isinstance(updates, list) and updates[0] is None):
            warnings.warn('No updates provided for positions!')
            return self
         
        velocities = systemUpdate.dvdt if not isinstance(systemUpdate, list) else [su.dvdt for su in systemUpdate]  
        
        self.waveState.u = integrateQ(self.waveState.u, updates, dt,
                                    selfScale=selfScale,
                                    verletScale=verletScale if verletScale is not None else semiImplicitScale,
                                    verletValue=velocities if verletScale is not None else self.waveState.v,
                                    referenceWeight=referenceWeight,
                                    referenceValue=referenceState.systemState.positions if referenceState is not None else None,
                                    integrateSpecies=[0] if self.systemState.kinds is not None else None,
                                    species=self.systemState.kinds)
    def integrateVelocity(self, 
                          systemUpdate : Union[WaveSystemUpdate, List[WaveSystemUpdate]], dt : Union[float, List[float]],
                          selfScale: Optional[float] = None,
                          referenceState: Optional['WaveSystem'] = None,
                          referenceWeight: Optional[float] = None, verbose = False, **kwargs):
        updates = systemUpdate.dvdt if not isinstance(systemUpdate, list) else [su.dvdt for su in systemUpdate]   
        if updates is None or (isinstance(updates, list) and updates[0] is None):
            warnings.warn('No updates provided for velocities!')
            return self
        if verbose:
            print('[Integrate v]')

        self.waveState.v = integrateQ(self.waveState.v, updates, dt,
                                    selfScale=selfScale,
                                    referenceWeight=referenceWeight,
                                    referenceValue=referenceState.waveState.v if referenceState is not None else None,
                                    integrateSpecies=[0] if self.systemState.kinds is not None else None,
                                    species=self.systemState.kinds)
        return self
    def integrateQuantities(self, 
                            systemUpdate : Union[WaveSystemUpdate, List[WaveSystemUpdate]], dt : Union[float, List[float]],
                            selfScale: Optional[float] = None,
                            referenceState: Optional['WaveSystem'] = None,
                            referenceWeight: Optional[float] = None, 
                            densitySwitch = False, verbose = False, **kwargs):
        return
    def integrate(self, 
                    systemUpdate : Union[WaveSystemUpdate, List[WaveSystemUpdate]], dt : Union[float, List[float]], verbose = False, **kwargs):
        verbosePrint(verbose, f'Integration  - {dt}')

        self.integratePosition(systemUpdate, dt, verbose = verbose, **kwargs)
        self.integrateVelocity(systemUpdate, dt, verbose = verbose, **kwargs)
        self.integrateQuantities(systemUpdate, dt, verbose = verbose, **kwargs)
        self.t = self.t + dt if not isinstance(dt, list) else self.t + np.sum(dt)
        return self 
    


@torch.jit.script
def waveEquation2(
    waveState: WaveEquationState,

    particleState : BasicState,
    kernel: KernelType,
    neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],

    gradientMode: GradientMode = GradientMode.Difference,
    laplacianMode: LaplacianMode = LaplacianMode.Brookshaw,

    supportScheme: SupportScheme = SupportScheme.Scatter):
    laplacian_u = SPHOperation(
        particleState,
        waveState.u,
        kernel,
        neighborhood[0],
        neighborhood[1],
        operation = Operation.Laplacian,
        supportScheme = supportScheme,
        gradientMode = gradientMode,
        laplacianMode = laplacianMode
    )
    if isinstance(waveState.c, torch.Tensor):
        return waveState.v, waveState.c**2 * laplacian_u
    else:
        return waveState.v, waveState.c**2 * laplacian_u
    

# def integrateState(time : float, dt : float, state: WaveEquationState, 
#     particleState : BasicState,
#     kernel: KernelType,
#     neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
#     scheme :str = 'RK4',

#     gradientMode: GradientMode = GradientMode.Difference,
#     laplacianMode: LaplacianMode = LaplacianMode.Brookshaw,

#     supportScheme: SupportScheme = SupportScheme.Gather):

#     if scheme == 'RK4':
#         du_k1, dv_k1 = waveEquation2(state, particleState, kernel, neighborhood, gradientMode, laplacianMode, supportScheme)
#         state_k1 = WaveEquationState(state.u + 0.5 * dt * du_k1, state.v + 0.5 * dt * dv_k1, state.c, state.damping)

#         du_k2, dv_k2 = waveEquation2(state_k1, particleState, kernel, neighborhood, gradientMode, laplacianMode, supportScheme)
#         state_k2 = WaveEquationState(state.u + 0.5 * dt * du_k2, state.v + 0.5 * dt * dv_k2, state.c, state.damping)


#         du_k3, dv_k3 = waveEquation2(state_k2, particleState, kernel, neighborhood, gradientMode, laplacianMode, supportScheme)
#         state_k3 = WaveEquationState(state.u + dt * du_k3, state.v + dt * dv_k3, state.c, state.damping)

#         du_k4, dv_k4 = waveEquation2(state_k3, particleState, kernel, neighborhood, gradientMode, laplacianMode, supportScheme)

#         dudt = (du_k1 + 2 * du_k2 + 2 * du_k3 + du_k4) / 6
#         dvdt = (dv_k1 + 2 * dv_k2 + 2 * dv_k3 + dv_k4) / 6
#         # dudt = dt * dvdt

#         return (state.u + dt * dudt) * state.damping, (state.v + dt * dvdt) * state.damping
#     elif scheme == 'explicitEuler':
#         du, dv = waveEquation2(state, particleState, kernel, neighborhood, gradientMode, laplacianMode, supportScheme)
#         # return (u + dt**2 * dv) * damping, dv# (v + dt * dv) * damping
#         return (state.u + dt * du) * state.damping, (state.v + dt * dv) * state.damping
#     elif scheme == 'implicitEuler':
#         du, dv = waveEquation2(state, particleState, kernel, neighborhood, gradientMode, laplacianMode, supportScheme)
#         return (state.u + dt * du) * state.damping, (state.v + dt * dv) * state.damping
#     else:
#         raise ValueError(f'Unknown scheme {scheme}')
    


def waveSystemFunction(waveSystem : WaveSystem, dt : float, config : dict, verbose : bool = False):
    du, dv = waveEquation2(
        waveSystem.waveState,
        waveSystem.systemState,
        kernel = config['kernel'],
        neighborhood = waveSystem.neighborhood,

        gradientMode = config['gradientMode'],
        laplacianMode = config['laplacianMode'],
        supportScheme = config['supportScheme'])
    return WaveSystemUpdate(
        dudt = du,
        dvdt = dv
    )
