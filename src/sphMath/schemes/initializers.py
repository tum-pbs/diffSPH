
from sphMath.regions import addBoundaryGhostParticles, processInlets
from sphMath.rigidBody import buildRigidBody, RigidBody, getTransformationMatrix
# from sphMath.neighborhood import buildNeighborhood, filterNeighborhoodByKind
from sphMath.regions import initializeCompressibleState
# from sphMath.regions import initializeWeaklyCompressibleState
import torch
from sphMath.schemes.states.compressiblesph import CompressibleState
from sphMath.schemes.states.wcsph import WeaklyCompressibleState

def updateBodyParticlesCSPH(particleState, rigidBody: RigidBody):
    T = getTransformationMatrix(rigidBody)
    # print(T)

    particlePositions = torch.einsum('uv, nv -> nu', T, torch.hstack([
        rigidBody.particlePositions, 
        torch.ones_like(rigidBody.particlePositions[:,0]).view(-1,1)]))[:,:2]
    ghostParticlePositions  = torch.einsum('uv, nv -> nu', T, torch.hstack([
        rigidBody.ghostParticlePositions, 
        torch.ones_like(rigidBody.ghostParticlePositions[:,0]).view(-1,1)]))[:,:2]
    
    offsets = particlePositions - ghostParticlePositions
    relativePositions = particlePositions - rigidBody.centerOfMass
    
    # print(rigidBody.angularVelocity)
    # print(torch.stack([relativePositions[:,1], -relativePositions[:,0]], dim = 1))

    particleVelocities = torch.stack([-relativePositions[:,1], relativePositions[:,0]], dim = 1) * rigidBody.angularVelocity + rigidBody.linearVelocity


    # particleVelocities += rigidBody.linearVelocity

    updatedPositions = particleState.positions.clone()
    updatedPositions[rigidBody.particleIndices] = particlePositions
    updatedPositions[rigidBody.ghostParticleIndices] = ghostParticlePositions

    updatedVelocities = particleState.velocities.clone()
    if rigidBody.kind != 'constant':
        # print('updating velocities')
        # print(particleVelocities)
        updatedVelocities[rigidBody.particleIndices] = particleVelocities
        updatedVelocities[rigidBody.ghostParticleIndices] = particleVelocities

    updatedOffsets = particleState.ghostOffsets.clone()
    updatedOffsets[rigidBody.ghostParticleIndices] = offsets
    updatedOffsets[rigidBody.ghostParticleIndices] = offsets

    return CompressibleState(
        positions = updatedPositions,
        supports = particleState.supports,
        masses = particleState.masses,
        densities = particleState.densities,
        velocities=updatedVelocities,

        internalEnergies = particleState.internalEnergies,
        totalEnergies = particleState.totalEnergies,
        entropies = particleState.entropies,
        pressures = particleState.pressures,
        soundspeeds = particleState.soundspeeds,

        kinds = particleState.kinds,
        materials = particleState.materials,
        UIDs = particleState.UIDs,

        UIDcounter=particleState.UIDcounter,

        ghostIndices = particleState.ghostIndices,
        ghostOffsets = updatedOffsets,

        n = particleState.n,
        dndh = particleState.dndh,
        P = particleState.P,
        dPdh = particleState.dPdh,
        A = particleState.A,
        B = particleState.B,
        gradA = particleState.gradA,
        gradB = particleState.gradB,
        V = particleState.V,
        omega = particleState.omega,
        alphas = particleState.alphas,
        alpha0s = particleState.alpha0s,
        divergence = particleState.divergence,
        ap_ij = particleState.ap_ij,
        av_ij = particleState.av_ij,
        f_ij = particleState.f_ij,
        numNeighbors = particleState.numNeighbors,
        # adjacency = particleState.adjacency,
    )

def initializeCompressibleSimulation(config, regions):
    config['regions'] = regions
    particleState_ = initializeCompressibleState(regions, config)[0]
    particleState = addBoundaryGhostParticles(regions, particleState_)
    particleState, emitted = processInlets(particleState, config, config['domain'])
    # particleState.densities[:] = config['fluid']['rho0']
    # particleState.velocities = forcing(particleState.positions)

    # neighborhood, sparseNeighborhood_ = buildNeighborhood(particleState, particleState, config['domain'], verletScale = config['neighborhood']['verletScale'], mode = 'superSymmetric', priorNeighborhood=None)         
    # ghostNeighbors = filterNeighborhoodByKind(particleState, sparseNeighborhood_, which = 'fluidToGhost')

    rigidBodyIDs = torch.unique(particleState.materials[particleState.kinds == 1]).cpu().numpy()
    # print(rigidBodyIDs)
    rigidBodies = []
    for id in rigidBodyIDs:
        rigidBody = buildRigidBody(particleState, config, id)
        if(rigidBody is not None):
            rigidBodies.append(rigidBody)

    # rigidBodies[0].angularVelocity = np.pi / 2
    for rigidBody in rigidBodies:
        particleState = updateBodyParticlesCSPH(particleState, rigidBody)
    config['rigidBodies'] = rigidBodies

    return particleState, config, rigidBodies




def updateBodyParticlesWCSPH(particleState, rigidBody: RigidBody):
    T = getTransformationMatrix(rigidBody)
    # print(T)

    particlePositions = torch.einsum('uv, nv -> nu', T, torch.hstack([
        rigidBody.particlePositions, 
        torch.ones_like(rigidBody.particlePositions[:,0]).view(-1,1)]))[:,:2]
    ghostParticlePositions  = torch.einsum('uv, nv -> nu', T, torch.hstack([
        rigidBody.ghostParticlePositions, 
        torch.ones_like(rigidBody.ghostParticlePositions[:,0]).view(-1,1)]))[:,:2]
    
    offsets = particlePositions - ghostParticlePositions
    relativePositions = particlePositions - rigidBody.centerOfMass
    
    # print(rigidBody.angularVelocity)
    # print(torch.stack([relativePositions[:,1], -relativePositions[:,0]], dim = 1))

    particleVelocities = torch.stack([-relativePositions[:,1], relativePositions[:,0]], dim = 1) * rigidBody.angularVelocity + rigidBody.linearVelocity


    # particleVelocities += rigidBody.linearVelocity

    updatedPositions = particleState.positions.clone()
    updatedPositions[rigidBody.particleIndices] = particlePositions
    updatedPositions[rigidBody.ghostParticleIndices] = ghostParticlePositions

    updatedVelocities = particleState.velocities.clone()
    if rigidBody.kind != 'constant':
        # print('updating velocities')
        # print(particleVelocities)
        updatedVelocities[rigidBody.particleIndices] = particleVelocities
        updatedVelocities[rigidBody.ghostParticleIndices] = particleVelocities

    updatedOffsets = particleState.ghostOffsets.clone()
    updatedOffsets[rigidBody.ghostParticleIndices] = offsets
    updatedOffsets[rigidBody.ghostParticleIndices] = offsets

    return WeaklyCompressibleState(
        positions = updatedPositions,
        supports = particleState.supports,
        masses = particleState.masses,
        densities = particleState.densities,
        velocities=updatedVelocities,

        pressures = particleState.pressures,
        soundspeeds=particleState.soundspeeds,

        kinds = particleState.kinds,
        materials = particleState.materials,
        UIDs = particleState.UIDs,

        UIDcounter=particleState.UIDcounter,

        covarianceMatrices=particleState.covarianceMatrices,
        gradCorrectionMatrices=particleState.gradCorrectionMatrices,
        eigenValues = particleState.eigenValues,

        gradRho=particleState.gradRho,
        gradRhoL=particleState.gradRhoL,

        ghostIndices = particleState.ghostIndices,
        ghostOffsets = updatedOffsets,

        surfaceMask=particleState.surfaceMask,
        surfaceNormals=particleState.surfaceNormals,
    )



from sphMath.regions import addBoundaryGhostParticles
from sphMath.rigidBody import buildRigidBody
# from sphMath.neighborhood import buildNeighborhood, filterNeighborhoodByKind
from sphMath.regions import initializeWeaklyCompressibleState


def initializeWeaklyCompressibleSimulation(config, regions):
    config['regions'] = regions
    particleState_ = initializeWeaklyCompressibleState(regions, config)[0]
    particleState = addBoundaryGhostParticles(regions, particleState_)
    particleState, emitted = processInlets(particleState, config, config['domain'])
    # particleState.densities[:] = config['fluid']['rho0']
    # particleState.velocities = forcing(particleState.positions)

    # neighborhood, sparseNeighborhood_ = buildNeighborhood(particleState, particleState, config['domain'], verletScale = config['neighborhood']['verletScale'], mode = 'superSymmetric', priorNeighborhood=None)         
    # ghostNeighbors = filterNeighborhoodByKind(particleState, sparseNeighborhood_, which = 'fluidToGhost')

    rigidBodyIDs = torch.unique(particleState.materials[particleState.kinds == 1]).cpu().numpy()
    # print(rigidBodyIDs)
    rigidBodies = []
    for id in rigidBodyIDs:
        # print('Processing rigid body', id)
        rigidBody = buildRigidBody(particleState, config, id)
        if(rigidBody is not None):
            rigidBodies.append(rigidBody)

    # rigidBodies[0].angularVelocity = np.pi / 2
    for rigidBody in rigidBodies:
        particleState = updateBodyParticlesWCSPH(particleState, rigidBody)
    config['rigidBodies'] = rigidBodies
    return particleState, config, rigidBodies

    
# @torch.jit.script
# class SimulationScheme(Enum):
#     CompSPH = 0
#     PESPH = 1
#     CRKSPH = 2
#     Price2007 = 3
#     Monaghan1997 = 4
#     Monaghan1992 = 5
#     MonaghanGingold1983 = 6
#     DeltaSPH = 7
    
from sphMath.enums import SimulationScheme
def initializeSimulation(simulationScheme: SimulationScheme, config, regions):
    config['regions'] = regions
    if simulationScheme in [SimulationScheme.CompSPH, SimulationScheme.PESPH, SimulationScheme.CRKSPH, SimulationScheme.Price2007, SimulationScheme.Monaghan1997, SimulationScheme.Monaghan1992, SimulationScheme.MonaghanGingold1983]:
        return initializeCompressibleSimulation(config, regions)
    elif simulationScheme in [SimulationScheme.DeltaSPH]:
        return initializeWeaklyCompressibleSimulation(config, regions)
    else:
        raise ValueError(f"Unknown simulation scheme: {SimulationScheme}")
    
def updateBodyParticles(simulationScheme: SimulationScheme, particleState, rigidBody: RigidBody):
    if simulationScheme in [SimulationScheme.CompSPH, SimulationScheme.PESPH, SimulationScheme.CRKSPH, SimulationScheme.Price2007, SimulationScheme.Monaghan1997, SimulationScheme.Monaghan1992, SimulationScheme.MonaghanGingold1983]:
        return updateBodyParticlesCSPH(particleState, rigidBody)
    elif simulationScheme in [SimulationScheme.DeltaSPH]:
        return updateBodyParticlesWCSPH(particleState, rigidBody)
    else:
        raise ValueError(f"Unknown simulation scheme: {simulationScheme}")