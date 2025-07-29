import h5py
import torch

def writeAttributesWCSPH(outFile, scheme, kernel, integrationScheme, targetNeighbors, targetDt, L, nx, band, rho0, c_s, freeSurface, dim, timeLimit, config, particleState):
    outFile.attrs['simulator'] = 'diffSPH v0.1'
    outFile.attrs['simulationScheme'] = scheme.name
    outFile.attrs['kernel'] = kernel.name
    outFile.attrs['integrationScheme'] = integrationScheme.name
    outFile.attrs['targetNeighbors'] = targetNeighbors
    outFile.attrs['targetDt'] = targetDt
    outFile.attrs['L'] = L
    outFile.attrs['nx'] = nx
    outFile.attrs['band'] = band
    outFile.attrs['rho0'] = rho0
    outFile.attrs['c_s'] = c_s
    outFile.attrs['freeSurface'] = freeSurface
    outFile.attrs['dim'] = dim
    outFile.attrs['timeLimit'] = timeLimit
    outFile.attrs['diffusion'] = config['diffusion']['alpha']

    outFile.attrs['shiftingScheme'] = 'deltaSPH'
    outFile.attrs['shiftingEnabled'] = True
    # outFile.attrs['radius'] = 
    outFile.attrs['dx'] = config['particle']['dx']
    outFile.attrs['support'] = config['particle']['support']
    outFile.attrs['boundary'] = torch.any(particleState.kinds > 0).cpu().item()

    outFile.attrs['domainMin'] = config['domain'].min.cpu().numpy()
    outFile.attrs['domainMax'] = config['domain'].max.cpu().numpy()
    outFile.attrs['domainPeriodic'] = config['domain'].periodic.cpu().numpy()

    if 'gravity' in config and config['gravity']['active']:
        # print(config['gravity']['direction'].cpu().numpy())
        outFile.attrs['gravity'] = True 
        outFile.attrs['gravityMode'] = config['gravity']['mode']
        direction = config['gravity']['direction'].cpu().numpy() if 'direction' in config['gravity'] else []
        # print(direction)
        outFile.attrs['gravityDirection'] = direction
        magnitude = (config['gravity']['magnitude'].cpu().numpy() if isinstance(config['gravity']['magnitude'], torch.Tensor)else config['gravity']['magnitude']) if 'magnitude' in config['gravity'] else []
        outFile.attrs['gravityMagnitude'] = magnitude
    else:
        outFile.attrs['gravity'] = False 



from diffSPH.schemes.states.compressiblesph import CompressibleState
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState

def getState(state, mask, config, static = True):
    if isinstance(state, WeaklyCompressibleState):
        hasBoundary = torch.any(state.kinds > 0).cpu().item()
        stateDict = {
            'positions': state.positions[mask],
            'supports': state.supports[mask],
            'areas': state.masses[mask] / config['fluid']['rho0'],
            'masses': state.masses[mask],
            'densities': state.densities[mask],
            'velocities': state.velocities[mask],
            'kinds': state.kinds[mask],
            'materials': state.materials[mask],
            'UIDs': state.UIDs[mask],
        } if static else {
            'positions': state.positions[mask],
            'densities': state.densities[mask],
            'velocities': state.velocities[mask],
        }
        if hasBoundary:
            stateDict['normals'] = state.ghostOffsets[mask]
        return stateDict
    else:
        return {
            'positions': state.positions[mask],
            'supports': state.supports[mask],
            'areas': state.masses[mask] / config['fluid']['rho0'],
            'masses': state.masses[mask],
            'densities': state.densities[mask],
            'internalEnergies': state.internalEnergies[mask],
            'velocities': state.velocities[mask],
            'kinds': state.kinds[mask],
            'materials': state.materials[mask],
            'UIDs': state.UIDs[mask],
            'alpha0s': state.alpha0s[mask],
        } if static else {
            'positions': state.positions[mask],
            'supports': state.supports[mask],
            'densities': state.densities[mask],
            'internalEnergies': state.internalEnergies[mask],
            'velocities': state.velocities[mask],
            'alpha0s': state.alpha0s[mask],

        }
def saveState(state, mask, group, config, static = True):
    state = getState(state, mask, config, static)
    for key in state:
        state[key] = state[key].cpu().numpy()
        group.create_dataset(key, data=state[key])


    
import h5py
from enum import Enum
from torchCompactRadius.util import DomainDescription
import diffSPH

def initializeOutputFile(outFileName, particleSystem, config, simulationName):
    outFile = h5py.File(outFileName, 'w')

    outFile.attrs['dataType'] = 'diffSPH'
    outFile.attrs['version'] = diffSPH.__version__
    
    outFile.attrs['simulationName'] = simulationName
    outFile.attrs['simulationKind'] = 'weaklyCompressible' if 'c_s' in config['fluid'] else 'compressible'
    outFile.attrs['scheme'] = config['scheme'].value
    outFile.attrs['kernel'] = config['kernel'].value
    outFile.attrs['integrationScheme'] = config['integrationScheme'].value
    # outFile.attrs['targetDt'] = targetDt
    outFile.attrs['targetNeighbors'] = config['support']['targetNeighbors']
    if 'c_s' in config['fluid']:
        outFile.attrs['c_s'] = config['fluid']['c_s'] 
        outFile.attrs['rho0'] = config['fluid']['rho0']
    outFile.attrs['dim'] = particleSystem.domain.dim
    # outFile.attrs['CFL'] = CFL
    # outFile.attrs['timeLimit'] = timeLimit


    outFile.create_group('domain')
    outFile['domain'].attrs['min'] = particleSystem.domain.min.cpu().numpy()
    outFile['domain'].attrs['max'] = particleSystem.domain.max.cpu().numpy()
    outFile['domain'].attrs['periodic'] = particleSystem.domain.periodic.cpu().numpy()
    outFile['domain'].attrs['dim'] = particleSystem.domain.dim

    rigidBodiesGroup = outFile.create_group('rigidBodies')

    if hasattr(particleSystem, 'rigidBodies'):
        for r, rigidBody in enumerate(particleSystem.rigidBodies):
            rigidBodyGroup = rigidBodiesGroup.create_group(f'rigidBody_{r:03d}')
            rigidBodyGroup.attrs['bodyID'] = rigidBody.bodyID
            rigidBodyGroup.attrs['kind'] = rigidBody.kind

            rigidBodyGroup.attrs['centerOfMass'] = rigidBody.centerOfMass.cpu().numpy()
            rigidBodyGroup.attrs['orientation'] = rigidBody.orientation.cpu().numpy()
            rigidBodyGroup.attrs['angularVelocity'] = rigidBody.angularVelocity.cpu().numpy()
            rigidBodyGroup.attrs['linearVelocity'] = rigidBody.linearVelocity.cpu().numpy()
            rigidBodyGroup.attrs['mass'] = rigidBody.mass.cpu().numpy()
            rigidBodyGroup.attrs['inertia'] = rigidBody.inertia.cpu().numpy()

            rigidBodyGroup.create_dataset('particlePositions', data = rigidBody.particlePositions.cpu().numpy())
            rigidBodyGroup.create_dataset('particleVelocities', data = rigidBody.particleVelocities.cpu().numpy())
            rigidBodyGroup.create_dataset('particleMasses', data = rigidBody.particleMasses.cpu().numpy())
            rigidBodyGroup.create_dataset('particleUIDs', data = rigidBody.particleUIDs.cpu().numpy())
            rigidBodyGroup.create_dataset('particleIndices', data = rigidBody.particleIndices.cpu().numpy())
            rigidBodyGroup.create_dataset('particleBoundaryDistances', data = rigidBody.particleBoundaryDistances.cpu().numpy())
            rigidBodyGroup.create_dataset('particleBoundaryNormals', data = rigidBody.particleBoundaryNormals.cpu().numpy())

            rigidBodyGroup.create_dataset('ghostParticlePositions', data = rigidBody.ghostParticlePositions.cpu().numpy())
            rigidBodyGroup.create_dataset('ghostParticleIndices', data = rigidBody.ghostParticleIndices.cpu().numpy())
            rigidBodyGroup.create_dataset('ghostParticleUIDs', data = rigidBody.ghostParticleUIDs.cpu().numpy())
            rigidBodyGroup.create_dataset('ghostParticleBoundaryDistances', data = rigidBody.ghostParticleBoundaryDistances.cpu().numpy())
            rigidBodyGroup.create_dataset('ghostParticleBoundaryNormals', data = rigidBody.ghostParticleBoundaryNormals.cpu().numpy())

    configGroup = outFile.create_group('config')
    for key, value in config.items():
        # print(f'Writing config key: {key}')
        if key == 'regions' or key == 'rigidBodies':
            continue
        
        if isinstance(value, torch.Tensor):
            configGroup.create_dataset(key, data = value.cpu().numpy())
        elif isinstance(value, dict):
            subGroup = configGroup.create_group(key)
            for subKey, subValue in value.items():
                # print(f'Writing config subkey: {subKey}')
                if subValue is None:
                    continue
                if isinstance(subValue, torch.Tensor):
                    subGroup.create_dataset(subKey, data = subValue.cpu().numpy())
                elif isinstance(subValue, Enum):
                    subGroup.attrs[subKey] = subValue.value
                else:
                    subGroup.attrs[subKey] = subValue
        elif isinstance(value, DomainDescription):
            domainGroup = configGroup.create_group(key)
            domainGroup.attrs['min'] = value.min.cpu().numpy()
            domainGroup.attrs['max'] = value.max.cpu().numpy()
            domainGroup.attrs['periodic'] = value.periodic.cpu().numpy()
            domainGroup.attrs['dim'] = value.dim
        elif isinstance(value, Enum):
            configGroup.attrs[key] = value.value
        else:
            configGroup.attrs[key] = value

    simulationData = outFile.create_group('simulationData')
    outFile.create_group('caseSpecificData')
    return outFile

def writeParticleDataWCSPH(outGroup, particleSystem, step, dt):
    frameGroup = outGroup.create_group(f'{step:06d}')

    frameGroup.attrs['time'] = particleSystem.t.cpu().item() if isinstance(particleSystem.t, torch.Tensor) else particleSystem.t
    frameGroup.attrs['dt'] = dt.cpu().item() if isinstance(dt, torch.Tensor) else dt
    frameGroup.attrs['numParticles'] = particleSystem.systemState.positions.shape[0]
    frameGroup.attrs['numRigidBodies'] = len(particleSystem.rigidBodies)
    frameGroup.attrs['UIDcounter'] = particleSystem.systemState.UIDcounter.cpu().item() if isinstance(particleSystem.systemState.UIDcounter, torch.Tensor) else particleSystem.systemState.UIDcounter

    for r, rigidBody in enumerate(particleSystem.rigidBodies):
        rigidBodyGroup = frameGroup.create_group(f'rigidBody_{r:03d}')
        rigidBodyGroup.attrs['bodyID'] = rigidBody.bodyID
        rigidBodyGroup.attrs['kind'] = rigidBody.kind

        rigidBodyGroup.attrs['centerOfMass'] = rigidBody.centerOfMass.cpu().numpy()
        rigidBodyGroup.attrs['orientation'] = rigidBody.orientation.cpu().numpy()
        rigidBodyGroup.attrs['angularVelocity'] = rigidBody.angularVelocity.cpu().numpy()
        rigidBodyGroup.attrs['linearVelocity'] = rigidBody.linearVelocity.cpu().numpy()
        rigidBodyGroup.attrs['mass'] = rigidBody.mass.cpu().numpy()
        rigidBodyGroup.attrs['inertia'] = rigidBody.inertia.cpu().numpy()

    frameGroup.create_dataset('positions', data = particleSystem.systemState.positions.cpu().numpy())
    frameGroup.create_dataset('supports', data = particleSystem.systemState.supports.cpu().numpy())
    frameGroup.create_dataset('masses', data = particleSystem.systemState.masses.cpu().numpy())
    frameGroup.create_dataset('densities', data = particleSystem.systemState.densities.cpu().numpy())
    frameGroup.create_dataset('velocities', data = particleSystem.systemState.velocities.cpu().numpy())
    frameGroup.create_dataset('pressures', data = particleSystem.systemState.pressures.cpu().numpy())
    frameGroup.create_dataset('soundspeeds', data = particleSystem.systemState.soundspeeds.cpu().numpy())
    frameGroup.create_dataset('kinds', data = particleSystem.systemState.kinds.cpu().numpy())
    frameGroup.create_dataset('materials', data = particleSystem.systemState.materials.cpu().numpy())
    frameGroup.create_dataset('UIDs', data = particleSystem.systemState.UIDs.cpu().numpy())
    
    if particleSystem.systemState.numNeighbors is not None:
        frameGroup.create_dataset('numNeighbors', data = particleSystem.systemState.numNeighbors.cpu().numpy())
    if particleSystem.systemState.ghostIndices is not None:
        frameGroup.create_dataset('ghostIndices', data = particleSystem.systemState.ghostIndices.cpu().numpy())
        frameGroup.create_dataset('ghostOffsets', data = particleSystem.systemState.ghostOffsets.cpu().numpy())

    return frameGroup

def writeParticleDataCompressible(outGroup, particleSystem, step, dt):
    frameGroup = outGroup.create_group(f'{step:06d}')

    frameGroup.attrs['time'] = particleSystem.t.cpu().item() if isinstance(particleSystem.t, torch.Tensor) else particleSystem.t
    frameGroup.attrs['dt'] = dt.cpu().item() if isinstance(dt, torch.Tensor) else dt
    frameGroup.attrs['numParticles'] = particleSystem.systemState.positions.shape[0]
    frameGroup.attrs['numRigidBodies'] = len(particleSystem.rigidBodies) if hasattr(particleSystem, 'rigidBodies') else 0
    frameGroup.attrs['UIDcounter'] = particleSystem.systemState.UIDcounter.cpu().item() if isinstance(particleSystem.systemState.UIDcounter, torch.Tensor) else particleSystem.systemState.UIDcounter

    frameGroup.create_dataset('positions', data = particleSystem.systemState.positions.cpu().numpy())
    frameGroup.create_dataset('supports', data = particleSystem.systemState.supports.cpu().numpy())
    frameGroup.create_dataset('masses', data = particleSystem.systemState.masses.cpu().numpy())
    frameGroup.create_dataset('densities', data = particleSystem.systemState.densities.cpu().numpy())
    frameGroup.create_dataset('velocities', data = particleSystem.systemState.velocities.cpu().numpy())


    frameGroup.create_dataset('kinds', data = particleSystem.systemState.kinds.cpu().numpy())
    frameGroup.create_dataset('materials', data = particleSystem.systemState.materials.cpu().numpy())
    frameGroup.create_dataset('UIDs', data = particleSystem.systemState.UIDs.cpu().numpy())
    
    if particleSystem.systemState.numNeighbors is not None:
        frameGroup.create_dataset('numNeighbors', data = particleSystem.systemState.numNeighbors.cpu().numpy())
    if particleSystem.systemState.ghostIndices is not None:
        frameGroup.create_dataset('ghostIndices', data = particleSystem.systemState.ghostIndices.cpu().numpy())
        frameGroup.create_dataset('ghostOffsets', data = particleSystem.systemState.ghostOffsets.cpu().numpy())

    if particleSystem.systemState.alphas is not None:
        frameGroup.create_dataset('alphas', data = particleSystem.systemState.alphas.cpu().numpy())
    if particleSystem.systemState.alpha0s is not None:
        frameGroup.create_dataset('alpha0s', data = particleSystem.systemState.alpha0s.cpu().numpy())
    if particleSystem.systemState.divergence is not None:
        frameGroup.create_dataset('divergence', data = particleSystem.systemState.divergence.cpu().numpy())

    frameGroup.create_dataset('internalEnergies', data = particleSystem.systemState.internalEnergies.cpu().numpy())
    # frameGroup.create_dataset('totalEnergies', data = particleSystem.systemState.totalEnergies.cpu().numpy())
    frameGroup.create_dataset('entropies', data = particleSystem.systemState.entropies.cpu().numpy())
    frameGroup.create_dataset('pressures', data = particleSystem.systemState.pressures.cpu().numpy())
    frameGroup.create_dataset('soundspeeds', data = particleSystem.systemState.soundspeeds.cpu().numpy())

    return frameGroup


def writeParticleData(outFile, particleSystem, step, dt):
    if isinstance(particleSystem.systemState, WeaklyCompressibleState):
        return writeParticleDataWCSPH(outFile, particleSystem, step, dt)
    elif isinstance(particleSystem.systemState, CompressibleState):
        return writeParticleDataCompressible(outFile, particleSystem, step, dt)
    else:
        raise ValueError(f'Unknown state type: {type(particleSystem.systemState)}')