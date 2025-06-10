from typing import NamedTuple, Optional
import torch
from dataclasses import dataclass
import dataclasses


# @dataclass(slots=True)
# class CompressibleParticleSet:
#     positions: torch.Tensor
#     supports: torch.Tensor
#     masses: torch.Tensor
#     densities: torch.Tensor
    
#     velocities: torch.Tensor
#     pressures: torch.Tensor
#     soundspeeds: torch.Tensor
#     thermalEnergy: torch.Tensor
#     omega: torch.Tensor

#     quantities: Optional[torch.Tensor] = None

#     def _replace(self, **kwargs):
#         return dataclasses.replace(self, **kwargs)
    
    
import torch
from sphMath.kernels import SPHKernel
from typing import Union, Tuple, Optional, List
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product
from sphMath.neighborhood import DomainDescription, SparseCOO
from sphMath.integrationSchemes.util import integrateQ
from sphMath.neighborhood import NeighborhoodInformation
import copy

from dataclasses import dataclass
import dataclasses

# @dataclass(slots=True)
# class systemUpdate:
#     positions: torch.Tensor
#     velocities: torch.Tensor
#     thermalEnergies: torch.Tensor

# @dataclass(slots=True)
# class SPHSystem:
#     positions: torch.Tensor
#     supports: torch.Tensor
#     masses: torch.Tensor
#     densities: torch.Tensor
    
#     velocities: torch.Tensor
#     pressures: torch.Tensor
#     soundspeeds: torch.Tensor
#     thermalEnergies: torch.Tensor
#     omega: torch.Tensor

#     neighborhoodInfo: NeighborhoodInformation
#     t: float

#     def integratePosition(self, 
#                           systemUpdate : Union[systemUpdate, List[systemUpdate]], dt : Union[float, List[float]], 
#                           verletScale: Optional[float] = None,
#                           semiImplicitScale: Optional[float] = None,
#                           selfScale: Optional[float] = None,
#                           referenceState: Optional['SPHSystem'] = None,
#                           referenceWeight: Optional[float] = None, **kwargs):
#         assert not (verletScale is not None and semiImplicitScale is not None), "Cannot use both verlet and semi-implicit integration"
        
#         # print(f'Integrating positions [{self.positions.shape} x {systemUpdate.positions.shape} - {self.positions.dtype} x {systemUpdate.positions.dtype} - {self.positions.device} x {systemUpdate.positions.device} - {dt}]')

#         updates = systemUpdate.positions if not isinstance(systemUpdate, list) else [su.positions for su in systemUpdate]        
#         self.positions = integrateQ(self.positions, updates, dt,
#                                     selfScale=selfScale,
#                                     verletScale=verletScale if verletScale is not None else semiImplicitScale,
#                                     verletValue=systemUpdate.velocities if verletScale is not None else self.velocities,
#                                     referenceWeight=referenceWeight,
#                                     referenceValue=referenceState.positions if referenceState is not None else None)
#         return self
#     def integrateVelocity(self, 
#                           systemUpdate : Union[systemUpdate, List[systemUpdate]], dt : Union[float, List[float]],
#                           selfScale: Optional[float] = None,
#                           referenceState: Optional['SPHSystem'] = None,
#                           referenceWeight: Optional[float] = None, **kwargs):
#         updates = systemUpdate.velocities if not isinstance(systemUpdate, list) else [su.velocities for su in systemUpdate]   
#         # print(f'Integrating velocities [{self.velocities.shape} x {systemUpdate.velocities.shape} - {self.velocities.dtype} x {systemUpdate.velocities.dtype} - {self.velocities.device} x {systemUpdate.velocities.device} - {dt}]')
#         self.velocities = integrateQ(self.velocities, updates, dt,
#                                     selfScale=selfScale,
#                                     referenceWeight=referenceWeight,
#                                     referenceValue=referenceState.velocities if referenceState is not None else None)
#         return self
    
#     def integrateQuantities(self, 
#                             systemUpdate : Union[systemUpdate, List[systemUpdate]], dt : Union[float, List[float]],
#                             selfScale: Optional[float] = None,
#                             referenceState: Optional['SPHSystem'] = None,
#                             referenceWeight: Optional[float] = None, 
#                             densitySwitch = False, **kwargs):
#         if (isinstance(systemUpdate, list) and hasattr(systemUpdate[0], 'thermalEnergies')) or hasattr(systemUpdate, 'thermalEnergies'):
#             updates = systemUpdate.thermalEnergies if not isinstance(systemUpdate, list) else [su.thermalEnergies for su in systemUpdate]   
#             self.thermalEnergies = integrateQ(self.thermalEnergies, updates, dt,
#                                         selfScale=selfScale,
#                                         referenceWeight=referenceWeight,
#                                         referenceValue=referenceState.thermalEnergies if referenceState is not None else None)

#         return self
#     def integrate(self, 
#                     systemUpdate : Union[systemUpdate, List[systemUpdate]], dt : Union[float, List[float]], **kwargs):
#         self.integratePosition(systemUpdate, dt, **kwargs)
#         self.integrateVelocity(systemUpdate, dt, **kwargs)
#         self.integrateQuantities(systemUpdate, dt, **kwargs)
#         self.t = self.t + dt
#         return self
    
#     def preprocess(self, dt, *args, **kwargs):
#         return self


#     def postprocess(self, dt, r, *args, **kwargs):
#         # print('PostProcess')
#         # print(r[1])
#         self.neighborhoodInfo = r[-1]
#         # print(self.neighborhoodInfo)
#         return self

#     def finalize(self, dt, returnValues, *args, **kwargs):
#         self.pressures = returnValues[-1][0].pressures
#         self.soundspeeds = returnValues[-1][0].soundspeeds
#         self.omega = returnValues[-1][0].omega
#         self.densities = returnValues[-1][0].densities
#         self.supports = returnValues[-1][0].supports

#         return self
    

# @dataclass(slots=True)
# class BaseSPHSystem:
#     positions: torch.Tensor
#     supports: torch.Tensor
#     masses: torch.Tensor
#     densities: torch.Tensor    
#     velocities: torch.Tensor
    
#     neighborhoodInfo: NeighborhoodInformation
#     t: float

#     def integratePosition(self, 
#                           systemUpdate : Union[systemUpdate, List[systemUpdate]], dt : Union[float, List[float]], 
#                           verletScale: Optional[float] = None,
#                           semiImplicitScale: Optional[float] = None,
#                           selfScale: Optional[float] = None,
#                           referenceState: Optional['SPHSystem'] = None,
#                           referenceWeight: Optional[float] = None, **kwargs):
#         assert not (verletScale is not None and semiImplicitScale is not None), "Cannot use both verlet and semi-implicit integration"
        
#         updates = systemUpdate.positions if not isinstance(systemUpdate, list) else [su.positions for su in systemUpdate]        
#         self.positions = integrateQ(self.positions, updates, dt,
#                                     selfScale=selfScale,
#                                     verletScale=verletScale if verletScale is not None else semiImplicitScale,
#                                     verletValue=systemUpdate.velocities if verletScale is not None else self.velocities,
#                                     referenceWeight=referenceWeight,
#                                     referenceValue=referenceState.positions if referenceState is not None else None)
#         return self
#     def integrateVelocity(self, 
#                           systemUpdate : Union[systemUpdate, List[systemUpdate]], dt : Union[float, List[float]],
#                           selfScale: Optional[float] = None,
#                           referenceState: Optional['SPHSystem'] = None,
#                           referenceWeight: Optional[float] = None, **kwargs):
#         updates = systemUpdate.velocities if not isinstance(systemUpdate, list) else [su.velocities for su in systemUpdate]   
#         self.velocities = integrateQ(self.velocities, updates, dt,
#                                     selfScale=selfScale,
#                                     referenceWeight=referenceWeight,
#                                     referenceValue=referenceState.velocities if referenceState is not None else None)
#         return self
    
#     def integrateQuantities(self, 
#                             systemUpdate : Union[systemUpdate, List[systemUpdate]], dt : Union[float, List[float]],
#                             selfScale: Optional[float] = None,
#                             referenceState: Optional['SPHSystem'] = None,
#                             referenceWeight: Optional[float] = None, 
#                             densitySwitch = False, **kwargs):

#         return self
#     def integrate(self, 
#                     systemUpdate : Union[systemUpdate, List[systemUpdate]], dt : Union[float, List[float]], **kwargs):
#         self.integratePosition(systemUpdate, dt, **kwargs)
#         self.integrateVelocity(systemUpdate, dt, **kwargs)
#         self.integrateQuantities(systemUpdate, dt, **kwargs)
#         self.t = self.t + dt
#         return self
    
#     def preprocess(self, dt, *args, **kwargs):
#         return self


#     def postprocess(self, dt, r, *args, **kwargs):
#         # print('PostProcess')
#         # print(r[1])
#         self.neighborhoodInfo = r[-1]
#         # print(self.neighborhoodInfo)
#         return self

#     def finalize(self, dt, returnValues, *args, **kwargs):

#         return self
    
    

# def systemToParticles(SPHSystem):
#     return CompressibleParticleSet(
#         positions = SPHSystem.positions.clone(),
#         supports = SPHSystem.supports.clone(),
#         masses = SPHSystem.masses.clone(),
#         densities = SPHSystem.densities.clone(),

#         velocities = SPHSystem.velocities.clone(),
#         pressures = SPHSystem.pressures,
#         soundspeeds = SPHSystem.soundspeeds,
#         thermalEnergy = SPHSystem.thermalEnergies,
#         omega = SPHSystem.omega
#     )
    
    
    

from sphMath.schemes.baseScheme import SPHUpdate, SPHSystem

    
from typing import List
import numpy as np


from sphMath.neighborhood import buildNeighborhood, filterNeighborhood
from sphMath.modules.compSPH import computeDeltaU, compSPH_deltaU_multistep

from sphMath.schemes.gasDynamics import CompressibleState, CompressibleUpdate
from sphMath.schemes.baseScheme import SPHSystem, printPositionIntegration, printVelocityIntegration, printQuantityIntegration, verbosePrint
