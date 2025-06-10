import h5py
import torch

def writeAttributesWCSPH(outFile, scheme, kernel, integrationScheme, targetNeighbors, targetDt, L, nx, band, rho0, c_s, freeSurface, dim, timeLimit, config, particleState):
    outFile.attrs['simulator'] = 'sphMath v0.1'
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



from sphMath.schemes.states.compressiblesph import CompressibleState
from sphMath.schemes.states.wcsph import WeaklyCompressibleState

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