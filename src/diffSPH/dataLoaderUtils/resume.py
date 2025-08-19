from diffSPH.util import ParticleSet
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState
from diffSPH.enums import *
from diffSPH.simple import *
from diffSPH.rigidBody import buildRigidBody
from diffSPH.schemes.initializers import updateBodyParticlesWCSPH
from diffSPH.dataLoaderUtils.state import CompressibleSPHState, WeaklyCompressibleSPHState
import copy

def convertConfigKey(currentValue, newValue):
    if currentValue is None:
        return newValue
    
    elif isinstance(currentValue, torch.Tensor):
        return torch.tensor(newValue, device=currentValue.device, dtype=currentValue.dtype)
    elif isinstance(currentValue, dict):
        return {k: convertConfigKey(currentValue[k] if k in currentValue else None, newValue[k]) for k in newValue}
    elif isinstance(currentValue, DomainDescription):
        # print(f'Converting DomainDescription from {currentValue} to {newValue}')
        return DomainDescription(
            min=torch.tensor(newValue['min'], device=currentValue.min.device, dtype=currentValue.min.dtype),
            max=torch.tensor(newValue['max'], device=currentValue.max.device, dtype=currentValue.max.dtype),
            dim=newValue['dim'],
            periodic=torch.tensor(newValue['periodic'], device=currentValue.periodic.device, dtype=torch.bool)
        )
    elif isinstance(currentValue, Enum):
        return [e for e in type(currentValue) if e.value == newValue][0] if newValue is not None else None
    else:
        return type(currentValue)(newValue) if newValue is not None else None


def stateToCState(currentState, config, domain_, batch=0):
    state = CompressibleState(
        positions = currentState.positions[currentState.batches == batch],
        supports = currentState.supports[currentState.batches == batch],
        masses = currentState.masses[currentState.batches == batch],
        densities = currentState.densities[currentState.batches == batch],
        velocities = currentState.velocities[currentState.batches == batch],
        
        kinds = currentState.kinds[currentState.batches == batch] if hasattr(currentState, 'kinds') else None,
        materials = currentState.materials[currentState.batches == batch] if hasattr(currentState, 'materials') else None,
        UIDs = currentState.UIDs[currentState.batches == batch] if hasattr(currentState, 'UIDs') else None,

        UIDcounter= currentState.UIDs[currentState.batches == batch].max(),

        alphas = currentState.alphas[currentState.batches == batch] if currentState.alphas is not None else None,
        alpha0s= currentState.alpha0s[currentState.batches == batch] if currentState.alpha0s is not None else None,

        internalEnergies= currentState.internalEnergies[currentState.batches == batch],

        totalEnergies=None,
        pressures = None,
        entropies = None,
        soundspeeds= None,
    )

    state.totalEnergies = state.internalEnergies + 0.5 * state.masses * torch.sum(state.velocities**2, dim=1)

    A_, u_, P_, c_s = idealGasEOS(A = None, u = state.internalEnergies, P = None, rho = state.densities, gamma = config['fluid']['gamma'])
    # idealGasEOS

    state.pressures = P_
    state.soundspeeds = c_s
    state.entropies = A_

    caseName = config['attributes']['caseName'] if 'caseName' in config['attributes'] else 'default'


    kernel = [k for k in KernelType if k.value == config['kernel']][0] if config['kernel'] is not None else None
    scheme = [s for s in SimulationScheme if s.value == config['scheme']][0] if config['scheme'] is not None else None
    integrationScheme = [i for i in IntegrationSchemeType if i.value == config['integrationScheme']][0] if config['integrationScheme'] is not None else None
    targetNeighbors = config['targetNeighbors'] if config['targetNeighbors'] is not None else n_h_to_nH(4, domain_.dim)


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32

    domain = DomainDescription(
        min = torch.tensor(domain_.min, device=device, dtype=dtype),
        max = torch.tensor(domain_.max, device=device, dtype=dtype),
        dim = domain_.dim,
        periodic = torch.tensor(domain_.periodic, device=device, dtype=torch.bool)
    )

    # kernel = KernelType.Wendland4
    # scheme = SimulationScheme.DeltaSPH
    # integrationScheme = IntegrationSchemeType.symplecticEuler

    device = currentState.positions.device if hasattr(currentState, 'positions') else torch.device('cpu')
    dtype = currentState.positions.dtype if hasattr(currentState, 'positions') else torch.float32
    
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # dtype = torch.float32

    wrappedKernel = kernel
    simulator, SimulationSystem, config, integrator = getSimulationScheme(
        scheme, kernel, integrationScheme, 
        1.0, targetNeighbors, domain)


    config['caseName'] = caseName

    particleSystem = SimulationSystem(config['domain'], None, 0., copy.deepcopy(state))

    dt = float(currentState.dt[batch])
    
    return state, particleSystem, config, domain, wrappedKernel, simulator, SimulationSystem, integrator, dt


def stateToWCState(currentState, currentConfig, domain_, batch = 0):
    """
    Converts the current state to a WeaklyCompressibleState.
    This is useful for resuming simulations or for compatibility with different state types.
    """

    state = WeaklyCompressibleState(
        positions = currentState.positions[currentState.batches == batch],
        supports = currentState.supports[currentState.batches == batch],
        masses = currentState.masses[currentState.batches == batch],
        densities = currentState.densities[currentState.batches == batch],
        velocities = currentState.velocities[currentState.batches == batch],

        pressures = torch.zeros_like(currentState.densities[currentState.batches == batch], device=currentState.densities.device, dtype=currentState.densities.dtype),
        soundspeeds= torch.ones_like(currentState.densities[currentState.batches == batch], device=currentState.densities.device, dtype=currentState.densities.dtype) * currentConfig['fluid']['c_s'],

        kinds = currentState.kinds[currentState.batches == batch],
        materials = currentState.materials[currentState.batches == batch],
        UIDs = currentState.UIDs[currentState.batches == batch],

        UIDcounter = currentState.UIDs[currentState.batches == batch].max(),

        ghostIndices = currentState.boundaryIndices[currentState.batches == batch],
        ghostOffsets = currentState.boundaryNormals[currentState.batches == batch] * currentState.boundaryDistances[currentState.batches == batch][:,None] * 2
    )

    caseName = config['attributes']['caseName'] if 'caseName' in config['attributes'] else 'default'


    kernel = [k for k in KernelType if k.value == config['kernel']][0] if config['kernel'] is not None else None
    scheme = [s for s in SimulationScheme if s.value == config['scheme']][0] if config['scheme'] is not None else None
    integrationScheme = [i for i in IntegrationSchemeType if i.value == config['integrationScheme']][0] if config['integrationScheme'] is not None else None
    targetNeighbors = config['targetNeighbors'] if config['targetNeighbors'] is not None else n_h_to_nH(4, domain_.dim)


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32

    domain = DomainDescription(
        min = torch.tensor(domain_.min, device=device, dtype=dtype),
        max = torch.tensor(domain_.max, device=device, dtype=dtype),
        dim = domain_.dim,
        periodic = torch.tensor(domain_.periodic, device=device, dtype=torch.bool)
    )

    # kernel = KernelType.Wendland4
    # scheme = SimulationScheme.DeltaSPH
    # integrationScheme = IntegrationSchemeType.symplecticEuler

    device = currentState.positions.device if hasattr(currentState, 'positions') else torch.device('cpu')
    dtype = currentState.positions.dtype if hasattr(currentState, 'positions') else torch.float32
    
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # dtype = torch.float32

    wrappedKernel = kernel
    simulator, SimulationSystem, config, integrator = getSimulationScheme(
        scheme, kernel, integrationScheme, 
        1.0, targetNeighbors, domain)

    for key in config.keys():
        config[key] = convertConfigKey(config[key] if key in config else None, config[key])


    rigidBodyIDs = torch.unique(state.materials[state.kinds == 1]).cpu().numpy()
    # print(rigidBodyIDs)
    rigidBodies = []
    for id in rigidBodyIDs:
        # print('Processing rigid body', id)
        rigidBody = buildRigidBody(state, config, id)
        if(rigidBody is not None):
            rigidBodies.append(rigidBody)


    for rigidBody in rigidBodies:
        state = updateBodyParticlesWCSPH(state, rigidBody)

    config['rigidBodies'] = rigidBodies
    config['caseName'] = caseName

    particleSystem = SimulationSystem(config['domain'], None, 0., copy.deepcopy(state), 'momentum', None, rigidBodies = config['rigidBodies'], regions = config['regions'], config = config)

    dt = float(currentState.dt[batch])
    return state, particleSystem, config, domain, wrappedKernel, simulator, SimulationSystem, integrator, dt


def stateToState(currentState, currentConfig, domain, batch = 0):
    """
    Converts the current state to a State.
    This is useful for resuming simulations or for compatibility with different state types.
    """
    if isinstance(currentState, WeaklyCompressibleSPHState):
        return stateToWCState(currentState, currentConfig[batch], domain_ = domain[batch], batch=batch)
    elif isinstance(currentState, CompressibleSPHState):
        return stateToCState(currentState, currentConfig[batch], domain_ = domain[batch], batch=batch)
    else:
        raise ValueError(f'Unknown state type: {type(currentState)}. Expected WeaklyCompressibleSPHState or CompressibleSPHState.')