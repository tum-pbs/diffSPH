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
# from sphMath.boundary import RigidBody

    
from sphMath.neighborhood import SparseCOO, DomainDescription, SparseNeighborhood

from sphMath.rigidBody import RigidBody
from sphMath.regions import processInlets, processOutlets
from sphMath.util import checkTensor, cloneOptionalTensor
from sphMath.schemes.states.wcsph import WeaklyCompressibleState, WeaklyCompressibleUpdate


import warnings
import copy
from dataclasses import dataclass, fields
from typing import Dict
from sphMath.rigidBody import integrateRigidBody, RigidBody, getTransformationMatrix
# from sphMath.boundary import updateBodyParticles

from dataclasses import dataclass, field
from sphMath.schemes.initializers import updateBodyParticles

@dataclass
class WeaklyCompressibleSystem(SPHSystem):
    systemState: WeaklyCompressibleState
    domain: DomainDescription
    
    neighborhoodInfo: NeighborhoodInformation
    t: float
    scheme: str = 'momentum'
    
    # ghostState: Optional[GhostParticleState] = None

    priorStep: Optional[Tuple[WeaklyCompressibleUpdate, WeaklyCompressibleState]] = None
    rigidBodies: List[RigidBody] = field(default_factory=[])
    regions: List[Dict] = field(default_factory=[])
    config: Dict = field(default_factory={})
    
    def integratePosition(self, 
                          systemUpdate : Union[WeaklyCompressibleUpdate, List[WeaklyCompressibleUpdate]], dt : Union[float, List[float]], 
                          verletScale: Optional[float] = None,
                          semiImplicitScale: Optional[float] = None,
                          selfScale: Optional[float] = None,
                          referenceState: Optional['WeaklyCompressibleSystem'] = None,
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
                                    integrateSpecies=[0] if self.systemState.kinds  is not None else None,
                                    species=self.systemState.kinds)
        return self
    def integrateVelocity(self, 
                          systemUpdate : Union[WeaklyCompressibleUpdate, List[WeaklyCompressibleUpdate]], dt : Union[float, List[float]],
                          selfScale: Optional[float] = None,
                          referenceState: Optional['WeaklyCompressibleSystem'] = None,
                          referenceWeight: Optional[float] = None, verbose = False,
                          semiImplicit: bool = False, **kwargs):
        updates = systemUpdate.velocities if not isinstance(systemUpdate, list) else [su.velocities for su in systemUpdate]   
        if updates is None or (isinstance(updates, list) and updates[0] is None):
            warnings.warn('No updates provided for velocities!')
            return self
        if verbose:
            print('[Integrate v]', end=' ')
            printVelocityIntegration(systemUpdate, dt, selfScale=selfScale, referenceValue=referenceState.systemState.velocities if referenceState is not None else None, referenceWeight=referenceWeight, verbose = verbose)

        referenceVelocity = self.systemState.velocities
        if semiImplicit:
            referenceVelocity = systemUpdate.positions
        self.systemState.velocities = integrateQ(referenceVelocity, updates, dt,
                                    selfScale=selfScale,
                                    referenceWeight=referenceWeight,
                                    referenceValue=referenceState.systemState.velocities if referenceState is not None else None,
                                    integrateSpecies=[0] if self.systemState.kinds is not None else None,
                                    species=self.systemState.kinds)
        return self
    def integrateQuantities(self, 
                            systemUpdate : Union[WeaklyCompressibleUpdate, List[WeaklyCompressibleUpdate]], dt : Union[float, List[float]],
                            selfScale: Optional[float] = None,
                            referenceState: Optional['WeaklyCompressibleSystem'] = None,
                            referenceWeight: Optional[float] = None, 
                            densitySwitch = False, verbose = False, **kwargs):
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
                    systemUpdate : Union[WeaklyCompressibleUpdate, List[WeaklyCompressibleUpdate]], dt : Union[float, List[float]], verbose = False, **kwargs):
        verbosePrint(verbose, f'Integration  - {dt}')

        self.integratePosition(systemUpdate, dt, verbose = verbose, **kwargs)
        self.integrateVelocity(systemUpdate, dt, verbose = verbose, **kwargs)
        self.integrateQuantities(systemUpdate, dt, verbose = verbose, **kwargs)
        self.t = self.t + dt if not isinstance(dt, list) else self.t + np.sum(dt)
        return self 
    def initializeNewState(self, *args, verbose = False, **kwargs):
        verbosePrint(verbose, 'Initializing new state')
        newSystem = type(self)(
            systemState = self.systemState.initializeNewState(self.scheme),
            domain = self.domain,
            neighborhoodInfo = self.neighborhoodInfo,
            t = self.t,
            scheme = self.scheme,
            config= self.config,
            rigidBodies= self.rigidBodies,
            regions= self.regions,
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

    def finalizeState(self, initialState, dt, returnValues, updateValues, butcherValues, *args, verbose = False, **kwargs):
        verbosePrint(verbose, '[Fin]\t Finalizing state')
        lastState = returnValues[-1][0]
        # firsState = returnValues[0][0]
        lastUpdate = updateValues[-1]
        
        if lastUpdate.passive is not None:
            # print('passive particles:', torch.sum(lastUpdate.passive))
            self.systemState.velocities[lastUpdate.passive,:] = lastState.velocities[lastUpdate.passive,:]
            if 'momentum' in self.scheme.lower():
                self.systemState.densities[lastUpdate.passive] = lastState.densities[lastUpdate.passive]
        # self.systemState.positions[lastUpdate.passive,:] = lastState.positions[lastUpdate.passive,:]
        
        
        self.systemState.pressures = lastState.pressures
        self.systemState.soundspeeds = lastState.soundspeeds
        
        if 'momentum' not in self.scheme.lower():
            self.systemState.densities = lastState.densities
        else:
            self.systemState.densities[self.systemState.kinds != 0] = lastState.densities[self.systemState.kinds != 0]
            
        self.systemState.covarianceMatrices = lastState.covarianceMatrices
        self.systemState.gradCorrectionMatrices = lastState.gradCorrectionMatrices
        self.systemState.eigenValues = lastState.eigenValues
        
        self.systemState.gradRho = lastState.gradRho
        self.systemState.gradRhoL = lastState.gradRhoL

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
        checkTensor(self.systemState.pressures, referenceDtype, referenceDevice, 'pressures', verbose = verbose)
        checkTensor(self.systemState.soundspeeds, referenceDtype, referenceDevice, 'soundspeeds', verbose = verbose)
        # checkTensor(self.systemState.kinds, referenceDtype, referenceDevice, 'kinds', verbose = verbose)
        # checkTensor(self.systemState.materials, referenceDtype, referenceDevice, 'materials', verbose = verbose)
        # checkTensor(self.systemState.UIDs, referenceDtype, referenceDevice, 'UIDs', verbose = verbose)  
        
        checkTensor(self.systemState.covarianceMatrices, referenceDtype, referenceDevice, 'covarianceMatrices', verbose = verbose)
        checkTensor(self.systemState.gradCorrectionMatrices, referenceDtype, referenceDevice, 'gradCorrectionMatrices', verbose = verbose)
        checkTensor(self.systemState.eigenValues, referenceDtype, referenceDevice, 'eigenValues', verbose = verbose)
        
        checkTensor(self.systemState.gradRho, referenceDtype, referenceDevice, 'gradRho', verbose = verbose)
        checkTensor(self.systemState.gradRhoL, referenceDtype, referenceDevice, 'gradRhoL', verbose = verbose)

    def finalizePhysics(self, initialState, dt, returnValues, updateValues, butcherValues, *args, verbose = False, **kwargs):
        verbosePrint(verbose, '[Fin]\t Finalizing physics')
        lastState = returnValues[-1][0]
        # firsState = returnValues[0][0]
        lastUpdate = updateValues[-1]
        self.priorStep = (lastUpdate, lastState)
        self.t = initialState.t + dt if not isinstance(dt, list) else initialState.t + np.sum(dt)
        verbosePrint(verbose, f'[Fin]\t Processing Inlets')
        self.systemState, emitted = processInlets(self.systemState, self.config, self.config['domain'])
        verbosePrint(verbose, f'[Fin]\t Processing Outlets')
        self.systemState, removed = processOutlets(self.systemState, self.config, self.config['domain'])
        if verbose:
            print(f'[Fin]\tRemoved {removed} particles')
            print(f'[Fin]\tEmitted {emitted} particles')
        if emitted > 0 or removed > 0:
            self.neighborhoodInfo = None
            self.priorStep = None

            idx = torch.arange(self.systemState.positions.shape[0], device = self.systemState.positions.device, dtype = torch.int32)

            for rigidBody in self.rigidBodies:                
                rigidBody.particleIndices = idx[torch.logical_and(self.systemState.kinds == 1, self.systemState.materials == rigidBody.bodyID)]
                rigidBody.ghostParticleIndices = idx[torch.logical_and(self.systemState.kinds == 2, self.systemState.materials == rigidBody.bodyID)]


        verbosePrint(verbose, f'[Fin]\t Updating rigid bodies')
        for rigidBody in self.rigidBodies:
            rigidBody = integrateRigidBody(rigidBody, 0, 0, dt)
            self.systemState = updateBodyParticles(self.config['scheme'], self.systemState, rigidBody)

    def finalize(self, initialState, dt, returnValues, updateValues, butcherValues, *args, verbose = False, **kwargs):
        verbosePrint(verbose, f'(Base) Finalizing system [{initialState.systemState.positions.shape} - {self.t}/{initialState.t} - {dt}]')

        verbosePrint(verbose, '[Fin]\tUpdating state')
        # lastState = returnValues[-1][0]
        # firsState = returnValues[0][0]
        # lastUpdate = updateValues[-1]
        
        self.finalizeState(initialState, dt, returnValues, updateValues, butcherValues, *args, verbose = verbose, **kwargs)

        self.finalizePhysics(initialState, dt, returnValues, updateValues, butcherValues, *args, verbose = verbose, **kwargs)

        return self
    def to(self, dtype = torch.float32, device = torch.device('cpu')):
        convertedState = copy.deepcopy(self.systemState)
        convertedDomain = DomainDescription(
            min = self.domain.min.to(dtype=dtype, device=device),
            max = self.domain.max.to(dtype=dtype, device=device),
            periodicity = self.domain.periodic,
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
    

from sphMath.schemes.states.wcsph import WeaklyCompressibleState

# def initializeWeaklyCompressibleState(particles, config, verbose = True):
#     print(particles)
#     if 'fluid' not in config:
#         if verbose:
#             warnings.warn('No fluid configuration found. Using default values.')
#         config['fluid'] = {}
#     if 'c_s' not in config['fluid']:
#         if verbose:
#             warnings.warn('No speed of sound found. Using default value.')
#         config['fluid']['c_s'] = 10       
#     if 'rho0' not in config['fluid']:
#         if verbose:
#             warnings.warn('No reference density found. Using default value.')
#         config['fluid']['rho0'] = 1

#     rho0 = config.get('fluid',{}).get('rho0', 1)
#     c_s = config.get('fluid',{}).get('c_s', 1)
#     dim = config['domain'].dim

#     if 'particle' not in config:
#         if verbose:
#             warnings.warn('No particle configuration found.')
#         config['particle'] = {}

#     if 'support' not in config['particle']:
#         if verbose:
#             warnings.warn('Using default support configuration.')
#         config['particle']['support'] = particles.supports.mean().item()
#     if 'dx' not in config['particle']:
#         if verbose:
#             warnings.warn('Using default dx configuration.')
#         config['particle']['dx'] = particles.masses.pow(1/dim).mean().item()

#     device = particles.positions.device
#     dtype = particles.positions.dtype

#     particleState = WeaklyCompressibleState(
#         particles.positions,
#         supports = particles.supports,
#         masses = particles.masses * rho0,
        
#         densities = torch.zeros_like(particles.masses),
#         velocities = torch.zeros_like(particles.positions),
        
#         pressures = torch.zeros_like(particles.masses),
#         soundspeeds = torch.ones_like(particles.masses) * c_s,
        
#         kinds = torch.zeros_like(particles.masses),
#         materials = torch.zeros_like(particles.masses),
#         UIDs = torch.arange(particles.positions.shape[0], device = device, dtype = torch.int64),
#         UIDcounter = particles.positions.shape[0],
#     )

#     return particleState, config


