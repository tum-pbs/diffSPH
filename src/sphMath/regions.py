import torch
from sphMath.neighborhood import filterNeighborhood, filterNeighborhoodByKind, coo_to_csr, buildNeighborhood, computeDistanceTensor
from sphMath.sampling import sampleRegularParticles, ParticleSet
import warnings
from skimage import measure
import numpy as np

def find_contour(f, minExtent, maxExtent, nGrid, level = 0):
    contours = measure.find_contours(f.numpy(), level)
    for ic in range(len(contours)):
        contours[ic][:,0] = (contours[ic][:,0]) / (f.shape[0] - 1) * (maxExtent[0] - minExtent[0]).numpy() + minExtent[0].numpy()
        contours[ic][:,1] = (contours[ic][:,1]) / (f.shape[1] - 1) * (maxExtent[1] - minExtent[1]).numpy() + minExtent[1].numpy()
    return contours



def sampleParticles(config, sdfs = [], minExtent = None, maxExtent = None, filter = True):
    # print(config['particle'])
    particlesA = sampleRegularParticles(config['particle']['nx'], config['domain'], config['particle']['targetNeighbors'], 0.0, 0, shortEdge = config.get('particle', {}).get('shortEdge', True))

    mask = torch.ones_like(particlesA.masses, dtype = torch.bool)
    if len(sdfs) > 0:
        distances = particlesA.masses.new_ones(particlesA.masses.shape) * np.inf
        mask = torch.zeros_like(particlesA.masses, dtype = torch.bool)

        for sdf_func in sdfs:            
            sdfDist, sdfNormal = sdf_func(particlesA.positions)
            maskA = sdfDist < 0
            mask = mask | maskA
            distances = torch.min(distances, sdfDist)
        if filter:
            particlesA = ParticleSet(
                positions = particlesA.positions[mask],
                supports = particlesA.supports[mask],
                masses = particlesA.masses[mask],
                densities = particlesA.densities[mask]            
            )
    else:
        mask = torch.ones_like(particlesA.masses, dtype = torch.bool)
        distances = particlesA.masses.new_ones(particlesA.masses.shape) * np.inf
    return particlesA, mask, distances

def buildRegion(sdf, config, ngrid = 255, kind : str = 'zero', type : str = 'fluid', dirichletValues = {}, updateValues = {}, bufferValues = [], initialConditions = {}):
    if config['domain'].dim == 2:
        Lx = config['domain'].max[0] - config['domain'].min[0]
        Ly = config['domain'].max[1] - config['domain'].min[1]
        minL = min(Lx, Ly)
        dL = minL / ngrid
        nx = (Lx//dL).to(torch.int32).cpu().item()
        ny = (Ly//dL).to(torch.int32).cpu().item()

        # print(nx, ny, ngrid)

        x = torch.linspace(config['domain'].min[0]-Lx * 0.01, config['domain'].max[0] + Lx*0.01, nx, dtype = torch.float32)
        y = torch.linspace(config['domain'].min[1]-Ly * 0.01, config['domain'].max[1] + Ly*0.01, ny, dtype = torch.float32)
        X, Y = torch.meshgrid(x, y, indexing = 'ij')
        P = torch.stack([X,Y], dim=-1)
        domain = config['domain']
        points = P.reshape(-1,2).to(domain.min.device).to(domain.min.dtype)
    
    return {
        'sdf': sdf,
        'type': type,
        'dirichlet': dirichletValues,
        'dirichletUpdate': updateValues, 
        'bufferValues': bufferValues,
        'kind': kind,
        'initialConditions': initialConditions,
        'particles': sampleParticles(config, [sdf])[0],
        'contour': find_contour(sdf(points)[0].reshape(nx, ny).cpu(), domain.min.cpu(), domain.max.cpu(), ngrid, 0) if config['domain'].dim == 2 else None,        
    }
    
def filterRegion(region, regions):
    particles = region['particles']
    if region['type'] == 'boundary' or region['type'] == 'dirichlet' or region['type'] == 'inlet' or region['type'] == 'outlet':
        return region
    for region_ in regions:
        if region_['type'] == 'boundary':
            sdfValues, sdfNormals = region_['sdf'](particles.positions)
            mask = sdfValues > 0
            particles = ParticleSet(
                positions = particles.positions[mask],
                supports = particles.supports[mask],
                masses = particles.masses[mask],
                densities = particles.densities[mask]            
            )
            region['particles'] = particles
    return region



def plotRegions(regions, axis, plotFluid = True, plotParticles = True):
    for region in regions:
        # visualizeParticles(region['particles'], axis[0,0], config)
        for ic, contour in enumerate(region['contour']):
            color = 'black'
            style = '-'
            if region['type'] == 'inlet':
                color = 'green'
                style = '--'
            if region['type'] == 'dirichlet':
                color = 'blue'
                style = ':'
            if region['type'] == 'outlet':
                color = 'red'
                style = ':'
            if region['type'] == 'mirror':
                color = 'black'
                style = ':'
            if region['type'] == 'boundary':
                color = 'grey'
                style = '--'
            if region['type'] == 'fluid':
                color = 'purple'
                style = '--'
            if region['type'] == 'buffer':
                color = 'orange'
                style = '--'
            if region['type'] == 'forcing':
                color = 'black'
                style = '-'
            if region['type'] == 'fluid' and not plotFluid:
                continue
            # axis[0,0].plot(contour[:,0], contour[:,1], color=color)
            axis.plot(contour[:,0], contour[:,1], color = color, ls = style, label = region['type'] if ic == 0 else None)
        if plotParticles:
            if region['type'] == 'inlet' and plotFluid: 
                axis.scatter(region['particles'].positions[:,0].detach().cpu().numpy(), region['particles'].positions[:,1].detach().cpu().numpy(), color = 'green', s = 1)
            if region['type'] == 'fluid' and plotFluid: 
                axis.scatter(region['particles'].positions[:,0].detach().cpu().numpy(), region['particles'].positions[:,1].detach().cpu().numpy(), color = 'purple', s = 1)
            if region['type'] == 'boundary':
                axis.scatter(region['particles'].positions[:,0].detach().cpu().numpy(), region['particles'].positions[:,1].detach().cpu().numpy(), color = 'grey', s = 1)
    # axis[0,0].legend()
    
    
def getInitialValue(region, pos, quantity, default):
    if quantity not in region['initialConditions']:
        return default
    if callable(region['initialConditions'][quantity]):
        return region['initialConditions'][quantity](pos)
    else:
        temp = torch.zeros_like(default)
        temp[:] = region['initialConditions'][quantity]
        return temp

from sphMath.schemes.states.wcsph import WeaklyCompressibleState
def initializeWeaklyCompressibleState(regions, config, verbose = True):
    if 'fluid' not in config:
        if verbose:
            warnings.warn('No fluid configuration found. Using default values.')
        config['fluid'] = {}
    if 'c_s' not in config['fluid']:
        if verbose:
            warnings.warn('No speed of sound found. Using default value.')
        config['fluid']['c_s'] = 10       
    if 'rho0' not in config['fluid']:
        if verbose:
            warnings.warn('No reference density found. Using default value.')
        config['fluid']['rho0'] = 1

    rho0 = config.get('fluid',{}).get('rho0', 1)
    c_s = config.get('fluid',{}).get('c_s', 1)
    dim = config['domain'].dim
    
    fluidParticles = []
    for region in regions:
        if region['type'] == 'fluid' or region['type'] == 'inlet':
            fluidParticles.append(region)
    if len(fluidParticles) == 0:
        raise ValueError('No fluid particles found. Please check the regions.')

    if 'particle' not in config:
        if verbose:
            warnings.warn('No particle configuration found.')
        config['particle'] = {}

    if 'support' not in config['particle']:
        if verbose:
            warnings.warn('Using default support configuration.')
        config['particle']['support'] = fluidParticles[0]['particles'].supports.mean().item()
    if 'dx' not in config['particle']:
        if verbose:
            warnings.warn('Using default dx configuration.')
        config['particle']['dx'] = fluidParticles[0]['particles'].masses.pow(1/dim).mean().item()

    device = fluidParticles[0]['particles'].positions.device
    dtype = fluidParticles[0]['particles'].positions.dtype

    positions = []
    supports = []
    masses = []
    velocities = []
    densities = []
    pressures = []
    soundspeeds = []
    kinds = []
    materials = []

    particleRegions = []
    for region in regions:
        if region['type'] == 'fluid' or region['type'] == 'boundary':
            particleRegions.append(region)
            positions.append(region['particles'].positions)
            supports.append(getInitialValue(region, region['particles'].positions, 'supports', region['particles'].supports))
            masses.append(getInitialValue(region, region['particles'].positions, 'masses', region['particles'].masses))
            velocities.append(getInitialValue(region, region['particles'].positions, 'velocities', torch.zeros_like(region['particles'].positions)))
            densities.append(getInitialValue(region, region['particles'].positions, 'densities', torch.ones_like(region['particles'].masses) * config['fluid']['rho0']))
            pressures.append(getInitialValue(region, region['particles'].positions, 'pressures', torch.zeros_like(region['particles'].masses)))
            soundspeeds.append(getInitialValue(region, region['particles'].positions, 'soundspeeds', torch.ones_like(region['particles'].masses) * config['fluid']['c_s']))
    fluidMaterials = 0
    boundaryMaterials = 0
    
    for region in particleRegions:
        if region['type'] == 'fluid':
            kinds.append(torch.zeros_like(region['particles'].masses, dtype = torch.int64))
            materials.append(torch.ones_like(region['particles'].masses, dtype = torch.int64) * fluidMaterials)
            fluidMaterials += 1
        elif region['type'] == 'boundary':
            kinds.append(torch.ones_like(region['particles'].masses, dtype = torch.int64))
            materials.append(torch.ones_like(region['particles'].masses, dtype = torch.int64)* boundaryMaterials)
            boundaryMaterials += 1
            
    kinds = torch.cat(kinds, dim = 0)
    materials = torch.cat(materials, dim = 0)
    positions = torch.cat(positions, dim = 0)
    UIDs = torch.arange(positions.shape[0], device = device, dtype = torch.int64)

    supports = torch.cat(supports, dim = 0)
    masses = torch.cat(masses, dim = 0)
    velocities = torch.cat(velocities, dim = 0)
    densities = torch.cat(densities, dim = 0)
    pressures = torch.cat(pressures, dim = 0)
    soundspeeds = torch.cat(soundspeeds, dim = 0)

    particleState = WeaklyCompressibleState(
        positions,
        supports = supports,
        masses = masses,
        
        densities = densities,
        velocities = velocities,
        
        pressures = pressures,
        soundspeeds = soundspeeds,
        
        kinds = kinds,
        materials = materials,
        UIDs = UIDs,
        
        UIDcounter = positions.shape[0],
    )

    return particleState, config


from sphMath.schemes.states.compressiblesph import CompressibleState
from sphMath.modules.eos import idealGasEOS
def initializeCompressibleState(regions, config, verbose = True):
    if 'fluid' not in config:
        if verbose:
            warnings.warn('No fluid configuration found. Using default values.')
        config['fluid'] = {}
    if 'c_s' not in config['fluid']:
        if verbose:
            warnings.warn('No speed of sound found. Using default value.')
        config['fluid']['c_s'] = 10       
    if 'rho0' not in config['fluid']:
        if verbose:
            warnings.warn('No reference density found. Using default value.')
        config['fluid']['rho0'] = 1

    rho0 = config.get('fluid',{}).get('rho0', 1)
    c_s = config.get('fluid',{}).get('c_s', 1)
    dim = config['domain'].dim
    
    fluidParticles = []
    for region in regions:
        if region['type'] == 'fluid' or region['type'] == 'inlet':
            fluidParticles.append(region)
    if len(fluidParticles) == 0:
        raise ValueError('No fluid particles found. Please check the regions.')

    if 'particle' not in config:
        if verbose:
            warnings.warn('No particle configuration found.')
        config['particle'] = {}

    if 'support' not in config['particle']:
        if verbose:
            warnings.warn('Using default support configuration.')
        config['particle']['support'] = fluidParticles[0]['particles'].supports.mean().item()
    if 'dx' not in config['particle']:
        if verbose:
            warnings.warn('Using default dx configuration.')
        config['particle']['dx'] = fluidParticles[0]['particles'].masses.pow(1/dim).mean().item()

    device = fluidParticles[0]['particles'].positions.device
    dtype = fluidParticles[0]['particles'].positions.dtype

    positions = []
    supports = []
    masses = []
    velocities = []
    densities = []
    pressures = []
    soundspeeds = []
    kinds = []
    materials = []
    internalEnergies = []
    totalEnergies = []
    entropies = []

    particleRegions = []
    for region in regions:
        if region['type'] == 'fluid' or region['type'] == 'boundary':
            particleRegions.append(region)
            positions.append(region['particles'].positions)
            supports.append(getInitialValue(region, region['particles'].positions, 'supports', region['particles'].supports))
            masses.append(getInitialValue(region, region['particles'].positions, 'masses', region['particles'].masses))
            velocities.append(getInitialValue(region, region['particles'].positions, 'velocities', torch.zeros_like(region['particles'].positions)))
            densities.append(getInitialValue(region, region['particles'].positions, 'densities', torch.ones_like(region['particles'].masses) * config['fluid']['rho0']))
            # P_default = torch.ones_like(region['particles'].masses)
            if \
                'pressures' not in region['initialConditions'] and \
                'internalEnergies' not in region['initialConditions'] and \
                'energies' not in region['initialConditions'] and \
                'entropies' not in region['initialConditions']:
                warnings.warn('No pressure or internal energy found. Using default value.')
                P = torch.ones_like(region['particles'].masses)
            else:
                P = None

            A_, u_, P_, c_s = idealGasEOS(
                A = region['initialConditions']['entropies'] if 'entropies' in region['initialConditions'] else None,
                u = region['initialConditions']['internalEnergies'] if 'internalEnergies' in region['initialConditions'] else None,
                P = region['initialConditions']['pressures'] if 'pressures' in region['initialConditions'] else P,
                rho = densities[-1],
                gamma = config['fluid']['gamma'])
            # print(A_, u_, P_, c_s)

            pressures.append(getInitialValue(region, region['particles'].positions, 'pressures', P_))
            soundspeeds.append(getInitialValue(region, region['particles'].positions,             'soundspeeds', c_s))
            internalEnergies.append(getInitialValue(region, region['particles'].positions, 'internalEnergies', u_))
            totalEnergies.append(getInitialValue(region, region['particles'].positions, 'totalEnergies', u_ + 1/2 * masses[-1] * torch.linalg.norm(velocities[-1], dim = -1)**2))
            entropies.append(getInitialValue(region, region['particles'].positions, 'entropies', A_))

    fluidMaterials = 0
    boundaryMaterials = 0
    
    for region in particleRegions:
        if region['type'] == 'fluid':
            kinds.append(torch.zeros_like(region['particles'].masses, dtype = torch.int64))
            materials.append(torch.ones_like(region['particles'].masses, dtype = torch.int64) * fluidMaterials)
            fluidMaterials += 1
        elif region['type'] == 'boundary':
            kinds.append(torch.ones_like(region['particles'].masses, dtype = torch.int64))
            materials.append(torch.ones_like(region['particles'].masses, dtype = torch.int64)* boundaryMaterials)
            boundaryMaterials += 1
            
    # print(kinds)
    kinds = torch.cat(kinds, dim = 0)
    materials = torch.cat(materials, dim = 0)
    positions = torch.cat(positions, dim = 0)
    UIDs = torch.arange(positions.shape[0], device = device, dtype = torch.int64)

    supports = torch.cat(supports, dim = 0)
    masses = torch.cat(masses, dim = 0)
    velocities = torch.cat(velocities, dim = 0)
    densities = torch.cat(densities, dim = 0)
    pressures = torch.cat(pressures, dim = 0)
    soundspeeds = torch.cat(soundspeeds, dim = 0)
    internalEnergies = torch.cat(internalEnergies, dim = 0)
    totalEnergies = torch.cat(totalEnergies, dim = 0)
    entropies = torch.cat(entropies, dim = 0)

    particleState = CompressibleState(
        positions,
        supports = supports,
        masses = masses,
        densities = densities,
        velocities = velocities,
        
        kinds = kinds,
        materials = materials,
        UIDs = UIDs,

        internalEnergies=internalEnergies,
        totalEnergies=totalEnergies,
        entropies=entropies,
        pressures=pressures,
        soundspeeds=soundspeeds,
        
        UIDcounter = positions.shape[0],

        divergence=torch.zeros_like(densities),
        alpha0s= torch.ones_like(densities),
        alphas= torch.ones_like(densities),
    )
    return particleState, config


def addBoundaryGhostParticles(regions, particleState : WeaklyCompressibleState):
    device = particleState.positions.device
    dtype = particleState.positions.dtype
    
    if not torch.any(particleState.kinds == 1):
        return particleState
    
    ghostIndices = particleState.positions.new_ones(particleState.positions.shape[0], dtype = torch.int64) * -1
    ghostOffsets = torch.zeros_like(particleState.positions)
    
    boundaryMaterial = 0
    numParticles = particleState.positions.shape[0]
    ghostPositions = []
    boundaryIndices = []
    
    for region in regions:
        if region['type'] == 'boundary':
            particleIndices = torch.arange(particleState.positions.shape[0], device = device, dtype = torch.int64)
            relevantParticles = torch.logical_and(particleState.kinds == 1, particleState.materials == boundaryMaterial)
            relevantParticles = particleIndices[relevantParticles]
            
            sdfValues, sdfNormals = region['sdf'](particleState.positions[relevantParticles])
            clampedDist = sdfValues - torch.min(-sdfValues, particleState.supports.mean())
            offsets = clampedDist.view(-1,1) * sdfNormals
            
            bIndices = relevantParticles
            boundaryIndices.append(bIndices)
            gUIDs = particleState.UIDs[relevantParticles]
            gIndices = torch.arange(numParticles, numParticles + offsets.shape[0], device = device, dtype = torch.int64)
            ghostIndices[bIndices] = gIndices
            ghostOffsets[bIndices] = offsets
            numParticles += offsets.shape[0]
            # print(sdfValues)
            
            
            ghostPositions.append(particleState.positions[relevantParticles] - offsets)
            boundaryMaterial += 1
            
    boundaryIndices = torch.cat(boundaryIndices, dim = 0)
    return WeaklyCompressibleState(
        positions = torch.cat([particleState.positions, torch.cat(ghostPositions, dim = 0)], dim = 0),
        supports = torch.cat([particleState.supports, particleState.supports[boundaryIndices]], dim = 0),
        masses = torch.cat([particleState.masses, particleState.masses[boundaryIndices]], dim = 0),
        densities = torch.cat([particleState.densities, particleState.densities[boundaryIndices]], dim = 0),
        velocities = torch.cat([particleState.velocities, particleState.velocities[boundaryIndices]], dim = 0),
        
        pressures= torch.cat([particleState.pressures, particleState.pressures[boundaryIndices]], dim = 0),
        soundspeeds= torch.cat([particleState.soundspeeds, particleState.soundspeeds[boundaryIndices]], dim = 0),
        
        kinds = torch.cat([particleState.kinds, torch.ones_like(boundaryIndices) * 2], dim = 0),
        materials = torch.cat([particleState.materials, particleState.materials[boundaryIndices]], dim = 0),
        UIDs = torch.cat([particleState.UIDs, -particleState.UIDs[boundaryIndices]], dim = 0),
        
        covarianceMatrices= torch.cat([particleState.covarianceMatrices, particleState.covarianceMatrices[boundaryIndices]], dim = 0) if particleState.covarianceMatrices is not None else None,
        gradCorrectionMatrices= torch.cat([particleState.gradCorrectionMatrices, particleState.gradCorrectionMatrices[boundaryIndices]], dim = 0) if particleState.gradCorrectionMatrices is not None else None,
        eigenValues= torch.cat([particleState.eigenValues, particleState.eigenValues[boundaryIndices]], dim = 0) if particleState.eigenValues is not None else None,
        
        gradRho= torch.cat([particleState.gradRho, particleState.gradRho[boundaryIndices]], dim = 0) if particleState.gradRho is not None else None,
        gradRhoL= torch.cat([particleState.gradRhoL, particleState.gradRhoL[boundaryIndices]], dim = 0) if particleState.gradRhoL is not None else None,
        
        surfaceMask= torch.cat([particleState.surfaceMask, particleState.surfaceMask[boundaryIndices]], dim = 0) if particleState.surfaceMask is not None else None,
        surfaceNormals = torch.cat([particleState.surfaceNormals, particleState.surfaceNormals[boundaryIndices]], dim = 0) if particleState.surfaceNormals is not None else None,
        
        ghostIndices = torch.cat([ghostIndices, boundaryIndices], dim = 0),
        ghostOffsets= torch.cat([ghostOffsets, ghostOffsets[boundaryIndices]], dim = 0),
        
        UIDcounter = particleState.UIDcounter
    )
            
    
    
from sphMath.neighborhood import SparseNeighborhood
from sphMath.util import getPeriodicPositions
from sphMath.boundary import LiuLiuQ
from torch.profiler import record_function

def enforceDirichlet(particleState, config, t, dt):
    with record_function("[SPH] - [Dirichlet BC]"):
        dirichletRegions = [region for region in config['regions'] if region['type'] == 'dirichlet' or region['type'] == 'inlet' or region['type'] == 'buffer']
        if len(dirichletRegions) == 0:
            return particleState
        device = particleState.positions.device
        dtype = particleState.positions.dtype
        modPos = getPeriodicPositions(particleState.positions, config['domain'])
        
        for ir, region in enumerate(dirichletRegions):
            with record_function(f"[SPH] - [Dirichlet BC] - region [{ir} - {region['type']}]"):
                mask = torch.logical_and(region['sdf'](modPos)[0] < 0, particleState.kinds != 2)
            
                if torch.sum(mask) == 0:
                    continue                
                with record_function(f"[SPH] - [Dirichlet BC] - attributes [{ir}]"):
                    for key in region['dirichlet']:
                        # print(key)  
                        if callable(region['dirichlet'][key]):
                            getattr(particleState, key)[mask] = region['dirichlet'][key](modPos[mask], mask, particleState, t, dt)
                        else:
                            getattr(particleState, key)[mask] = region['dirichlet'][key]
                    

                if len(region['bufferValues']) == 0:
                    continue
                # print('Buffering', region['bufferValues'], 'for', torch,sum(mask), 'particles')
                with record_function(f"[SPH] - [Dirichlet BC] - buffer [{ir}]"):
                    positions = modPos[mask]
                    sdfValues, sdfNormals = region['sdf'](positions)
                    clampedDist = sdfValues - torch.min(-sdfValues, particleState.supports.mean() * 2)
                    offsets = clampedDist.view(-1,1) * sdfNormals

                    ghostState = ParticleSet(
                        positions = positions - offsets, supports = particleState.supports[mask], masses = particleState.masses[mask], densities = particleState.densities[mask])
                    ghostIndices = torch.arange(particleState.positions.shape[0], device = device, dtype = torch.int64)[mask]

                    neighborhood, sparseNeighborhood_ = buildNeighborhood(ghostState, particleState, config['domain'], verletScale = 1.0, mode = 'superSymmetric', priorNeighborhood=None)
                    
                    fluidMask = particleState.kinds[sparseNeighborhood_.col] == 0
                    sparseNeighborhood = SparseNeighborhood(
                        row = sparseNeighborhood_.row[fluidMask],
                        col = sparseNeighborhood_.col[fluidMask],
                        
                        numRows = sparseNeighborhood_.numRows,
                        numCols = sparseNeighborhood_.numCols,
                        
                        points_a = sparseNeighborhood_.points_a,
                        points_b = sparseNeighborhood_.points_b,
                        
                        domain = sparseNeighborhood_.domain,
                    )
                
                    with record_function(f"[SPH] - [Dirichlet BC] - buffer [{ir}] - LiuLiuQ"):
                        for key in region['bufferValues']:
                            # print('Buffering', key)
                            setattr(particleState, key, LiuLiuQ(getattr(particleState, key), ghostState, particleState, config['domain'],config['kernel'], sparseNeighborhood, ghostIndices, offsets, mask))
                
        return particleState
    

def applyForcing(particleState, config, t, dt):
    with record_function("[SPH] - [Dirichlet BC]"):
        dirichletRegions = [region for region in config['regions'] if region['type'] == 'forcing']
        result = torch.zeros_like(particleState.velocities)
        if len(dirichletRegions) == 0:
            return result
        device = particleState.positions.device
        dtype = particleState.positions.dtype
        
        modPos = getPeriodicPositions(particleState.positions, config['domain'])
        for ir, region in enumerate(dirichletRegions):
            with record_function(f"[SPH] - [Dirichlet BC] - region [{ir} - {region['type']}]"):

                mask = torch.logical_and(region['sdf'](modPos)[0] < 0, particleState.kinds != 2)
            
                if torch.sum(mask) == 0:
                    continue                
                with record_function(f"[SPH] - [Dirichlet BC] - attributes [{ir}]"):
                    for key in region['dirichlet']:
                        if key != 'velocities':
                            raise ValueError('Forcing can only be applied to velocities')
                        # print(key)  
                        if callable(region['dirichlet'][key]):
                            result[mask] += region['dirichlet'][key](modPos[mask], mask, particleState, t, dt)
                            # getattr(particleState, key)[mask] = region['dirichlet'][key](modPos[mask])
                        else:
                            result[mask] += region['dirichlet'][key]
                    
        return result

def enforceDirichletUpdate(systemUpdate, particleState, config, t, dt):
    dirichletRegions = [region for region in config['regions'] if region['type'] == 'dirichlet' or region['type'] == 'inlet' or region['type'] == 'buffer']
    if len(dirichletRegions) == 0:
        return systemUpdate
    device = particleState.positions.device
    dtype = particleState.positions.dtype
    modPos = getPeriodicPositions(particleState.positions, config['domain'])
    
    for region in dirichletRegions:
        mask = region['sdf'](modPos)[0] < 0
        
        for key in region['dirichletUpdate']:
            if callable(region['dirichletUpdate'][key]):
                getattr(systemUpdate, key)[mask] = region['dirichletUpdate'][key](modPos[mask], mask, particleState, t, dt)
            else:
                getattr(systemUpdate, key)[mask] = region['dirichletUpdate'][key]
        systemUpdate.passive[mask] = True
    return systemUpdate


def processInlets(particleState, config, domain):
    regions = config['regions']
    inletRegions = [ region for region in regions if region['type'] == 'inlet']
    emitted = 0
    if len(inletRegions) == 0:
        return particleState, emitted
    
    for inletRegion in inletRegions:
        inletParticles = inletRegion['particles']
        _, neighborhood = buildNeighborhood(inletParticles, particleState, domain, 1.0, 'superSymmetric', None, False)
        
        rij, xij = computeDistanceTensor(neighborhood, False, 'superSymmetric')
        dx = config['particle']['dx']
        
        newDistance = torch.ones_like(inletParticles.masses) * np.inf
        minDistance = newDistance.index_reduce_(dim = 0, index = neighborhood.row, source = rij, include_self = False, reduce = 'amin')
        emitterMask = minDistance > dx
        particlesToEmit = torch.sum(emitterMask)
        # print('Particles to emit', particlesToEmit.item())
        if torch.sum(emitterMask) == 0:
            continue
        
        emitted += particlesToEmit.item()
        newUIDs = torch.arange(particleState.UIDcounter, particleState.UIDcounter + particlesToEmit, device = particleState.positions.device, dtype = torch.int64)
        # particleState.UIDcounter += particlesToEmit
        newPositions = inletParticles.positions[emitterMask]
        newSupports = inletParticles.supports[emitterMask]
        newMasses = inletParticles.masses[emitterMask]
        newDensities = inletParticles.densities[emitterMask]
        
        
        particleState = WeaklyCompressibleState(
            positions= torch.cat([particleState.positions, newPositions], dim = 0),
            supports= torch.cat([particleState.supports, newSupports], dim = 0),
            masses= torch.cat([particleState.masses, newMasses], dim = 0),
            densities= torch.cat([particleState.densities, newDensities], dim = 0),
            velocities= torch.cat([particleState.velocities, torch.zeros_like(newPositions)], dim = 0),
            
            pressures= torch.cat([particleState.pressures, torch.zeros_like(newMasses)], dim = 0),
            soundspeeds= torch.cat([particleState.soundspeeds, torch.ones_like(newMasses) * config['fluid']['c_s']], dim = 0),
            
            kinds= torch.cat([particleState.kinds, torch.zeros_like(newMasses, dtype = torch.int64)], dim = 0),
            materials= torch.cat([particleState.materials, torch.zeros_like(newMasses, dtype = torch.int64)], dim = 0),
            UIDs = torch.cat([particleState.UIDs, newUIDs], dim = 0),
            
            UIDcounter= particleState.UIDcounter + particlesToEmit,
            covarianceMatrices= None,
            gradCorrectionMatrices= None,
            eigenValues= None,
            gradRho= None,
            gradRhoL= None,
            
            surfaceMask= None,
            surfaceNormals= None,
            
            ghostIndices= torch.cat([particleState.ghostIndices, torch.ones_like(newMasses, dtype = torch.int64) * -1], dim = 0) if particleState.ghostIndices is not None else None,
            ghostOffsets= torch.cat([particleState.ghostOffsets, torch.zeros_like(newPositions)], dim = 0) if particleState.ghostOffsets is not None else None,
        )
        
        for key in inletRegion['dirichlet']:
            # print(key)  
            mask = torch.ones_like(particleState.masses, dtype = torch.bool)
            mask[:-particlesToEmit] = False
            if callable(inletRegion['dirichlet'][key]):
                getattr(particleState, key)[mask] = inletRegion['dirichlet'][key](particleState.positions[mask])
            else:
                getattr(particleState, key)[mask] = inletRegion['dirichlet'][key]
            

        
    return particleState, emitted
        
        
from sphMath.schemes.states.wcsph import WeaklyCompressibleState

def processOutlets(particleState, config, domain):
    regions = config['regions']
    outletRegions = [region for region in regions if region['type'] == 'outlet']
    removedParticles = 0
    if len(outletRegions) == 0:
        return particleState, removedParticles
    
    modPos = getPeriodicPositions(particleState.positions, config['domain'])
    for region in outletRegions:
        mask = torch.logical_and(
            particleState.kinds == 0,
            region['sdf'](modPos)[0] < 0
        )
        # print(mask.sum())
        if torch.sum(mask) > 0:
            removedParticles += mask.sum().item()
            
            invertedMask = ~mask
            
            particleState = WeaklyCompressibleState(
                positions = particleState.positions[invertedMask],
                supports = particleState.supports[invertedMask],
                masses = particleState.masses[invertedMask],
                densities= particleState.densities[invertedMask],
                velocities = particleState.velocities[invertedMask],
                
                pressures = particleState.pressures[invertedMask],
                soundspeeds= particleState.soundspeeds[invertedMask],
                
                kinds = particleState.kinds[invertedMask],
                materials = particleState.materials[invertedMask],
                UIDs = particleState.UIDs[invertedMask],
                
                UIDcounter= particleState.UIDcounter,
                
                covarianceMatrices= particleState.covarianceMatrices[invertedMask] if particleState.covarianceMatrices is not None else None,
                gradCorrectionMatrices= particleState.gradCorrectionMatrices[invertedMask] if particleState.gradCorrectionMatrices is not None else None,
                eigenValues= particleState.eigenValues[invertedMask] if particleState.eigenValues is not None else None,
                
                gradRho= particleState.gradRho[invertedMask] if particleState.gradRho is not None else None,
                gradRhoL= particleState.gradRhoL[invertedMask] if particleState.gradRhoL is not None else None,
                
                surfaceMask= particleState.surfaceMask[invertedMask] if particleState.surfaceMask is not None else None,
                surfaceNormals= particleState.surfaceNormals[invertedMask] if particleState.surfaceNormals is not None else None,
                
                ghostIndices= particleState.ghostIndices[invertedMask] if particleState.ghostIndices is not None else None,
                ghostOffsets= particleState.ghostOffsets[invertedMask] if particleState.ghostOffsets is not None else None,
            )
    return particleState, removedParticles