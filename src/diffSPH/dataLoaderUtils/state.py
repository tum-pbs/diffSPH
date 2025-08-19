from dataclasses import dataclass
import torch
from typing import Optional, List, Union

@dataclass
class DataConfiguration:
    frameDistance: int = 1
    frameSpacing: int = 1
    maxRollout: int = 16
    historyLength: int = 1

    skipInitialFrames: int = 0
    cutoff: int = 0

@dataclass
class RigidBodyState:
    bodyID: int
    kind: str

    centerOfMass: torch.Tensor
    orientation: torch.Tensor
    angularVelocity: torch.Tensor
    linearVelocity: torch.Tensor
    mass: torch.Tensor
    inertia: torch.Tensor
    batchID: int = 0
from typing import List, Tuple, Dict

# @torch.jit.script
@dataclass(slots=True)
class WeaklyCompressibleSPHState:
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    densities: torch.Tensor    
    velocities: torch.Tensor
    
    # meta properties for particles
    kinds: torch.Tensor # 0 for fluid, 1 for boundary, 2 for ghost
    materials: torch.Tensor # specific subsets of particles, also indicates body id
    UIDs : torch.Tensor # unique identifiers for particles, ghost particles have negative UIDs

    
    numParticles: Union[int, List[int]]
    time: Union[float, List[float]]
    dt: Union[float, List[float]]
    timestep: Union[int, List[int]]
    key: Union[str, List[str]]

    boundaryNormals: Optional[torch.Tensor] = None
    boundaryDistances: Optional[torch.Tensor] = None

    rigidBodies: Optional[List[RigidBodyState]] = None
    batches: Optional[torch.Tensor] = None # batch IDs for particles, used for parallel processing

@dataclass(slots=True)
class CompressibleSPHState:
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    densities: torch.Tensor    
    velocities: torch.Tensor
    
    # meta properties for particles
    kinds: torch.Tensor
    materials: torch.Tensor # specific subsets of particles, also indicates body id
    UIDs : torch.Tensor # unique identifiers for particles, ghost particles have negative UIDs

    internalEnergies: torch.Tensor
    entropies: torch.Tensor
    pressures: torch.Tensor
    soundspeeds: torch.Tensor

    numParticles: Union[int, List[int]]
    time: Union[float, List[float]]
    dt: Union[float, List[float]]
    timestep: Union[int, List[int]]
    key: Union[str, List[str]]

    alphas: Optional[torch.Tensor] = None # alpha values for compressible SPH
    alpha0s: Optional[torch.Tensor] = None # initial alpha values for compressible SPH
    divergence: Optional[torch.Tensor] = None # divergence of the velocity field
    batches: Optional[torch.Tensor] = None # batch IDs for particles, used for parallel processing

@dataclass(slots = True)
class PointCloudWithKinds:
    positions: torch.Tensor
    supports: torch.Tensor
    kinds: torch.Tensor

def convertNewFormatToWCSPH(inFile, key, state):
    numFluidParticles = state['fluid']['numParticles']
    numBoundaryParticles = state['boundary']['numParticles'] if state['boundary'] is not None else 0

    # print(f'Converting frame {key} with {numFluidParticles} fluid particles and {numBoundaryParticles} boundary particles to Weakly Compressible SPH format')

    combinedPositions = torch.cat([state['fluid']['positions'], state['boundary']['positions']]) if state['boundary'] is not None else state['fluid']['positions']
    combinedSupports = torch.cat([state['fluid']['supports'], state['boundary']['supports']]) if state['boundary'] is not None else state['fluid']['supports']
    combinedMasses = torch.cat([state['fluid']['masses'], state['boundary']['masses']]) if state['boundary'] is not None else state['fluid']['masses']
    combinedDensities = torch.cat([state['fluid']['densities'], state['boundary']['densities']]) if state['boundary'] is not None else state['fluid']['densities']
    combinedVelocities = torch.cat([state['fluid']['velocities'], state['boundary']['velocities']]) if state['boundary'] is not None else state['fluid']['velocities']

    combinedKinds = torch.cat([torch.zeros_like(state['fluid']['indices'], dtype=torch.int64), torch.ones_like(state['boundary']['indices'], dtype=torch.int64)]) if state['boundary'] is not None else torch.zeros_like(state['fluid']['indices'], dtype=torch.int64)
    combinedMaterials = torch.cat([torch.zeros_like(state['fluid']['indices'], dtype=torch.int64), state['boundary']['bodyIDs']]) if state['boundary'] is not None else torch.zeros_like(state['fluid']['indices'], dtype=torch.int64)
    combinedUIDs = torch.cat([state['fluid']['indices'], state['boundary']['indices'] + numFluidParticles]) if state['boundary'] is not None else state['fluid']['indices']

    combinedNormals = torch.cat([torch.zeros_like(state['fluid']['positions'], dtype=torch.float32), state['boundary']['normals']]) if state['boundary'] is not None else None
    if 'distance' in state['boundary']:
        combinedDistances = torch.cat([torch.zeros_like(state['fluid']['densities'], dtype=torch.float32), state['boundary']['distances']]) if state['boundary'] is not None else None
    else:
        combinedDistances = None

    

    numParticles = numFluidParticles + numBoundaryParticles if state['boundary'] is not None else numFluidParticles
    time = state['time']
    dt = state['dt']
    timestep = state['timestep']
    key = key

    return WeaklyCompressibleSPHState(
        positions=combinedPositions,
        supports=combinedSupports,
        masses=combinedMasses,
        densities=combinedDensities,
        velocities=combinedVelocities,
        kinds=combinedKinds,
        materials=combinedMaterials,
        UIDs=combinedUIDs,
        numParticles=numParticles,
        time=time,
        dt=dt,
        timestep=timestep,
        key = key,

        boundaryNormals=combinedNormals,
        boundaryDistances=combinedDistances,
    )