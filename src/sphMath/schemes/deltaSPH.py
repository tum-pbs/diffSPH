from sphMath.schemes.weaklyCompressible import verbosePrint
from sphMath.neighborhood import filterNeighborhood, filterNeighborhoodByKind, coo_to_csr, evaluateNeighborhood, SupportScheme
from sphMath.modules.density import computeDensity
from torch.profiler import record_function

from sphMath.kernels import SPHKernel
from sphMath.modules.viscosity import compute_Pi
from sphMath.neighborhood import DomainDescription, SparseCOO
from sphMath.sphOperations.shared import getTerms, compute_xij
from sphMath.kernels import evalW, evalGradW, evalDerivativeW
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product
import torch
from sphMath.operations import sph_op
from sphMath.neighborhood import computeDistanceTensor
from sphMath.schemes.weaklyCompressible import WeaklyCompressibleUpdate, WeaklyCompressibleSystem, WeaklyCompressibleState, checkTensor
from sphMath.math import pinv2x2

from sphMath.modules.velocityDiffusion import computeViscosity_deltaSPH_inviscid
from sphMath.modules.densityDiffusion import computeDensityDeltaTerm
from sphMath.modules.renorm import computeCovarianceMatrices
from sphMath.neighborhood import buildNeighborhood, SparseNeighborhood, filterNeighborhoodByKind
from sphMath.modules.density import computeDensity, computeGradRhoL, computeGradRho
from sphMath.modules.gravity import computeGravity
from sphMath.modules.surfaceDetection import surfaceDetection
from sphMath.modules.pressureForce import computePressureForce
from sphMath.modules.eos import computeEOS_WC
from sphMath.modules.mDBC import mDBCDensity
from sphMath.modules.sps import computeSPSTurbulence
from sphMath.boundary import computeBoundaryVelocities

from sphMath.regions import enforceDirichlet, enforceDirichletUpdate, applyForcing
from torch.profiler import record_function
from sphMath.modules.velocityDiffusion import computeViscosity_deltaSPH_inviscid
from sphMath.modules.momentum import computeMomentum

from typing import Optional, Union
verbosePacking = True

# def pack_hook(tensor, targetDevice = 'cpu', name = None):
#     if verbosePacking:
#         print(f'Packing {name} to {targetDevice}')    
#     return tensor.to(device = targetDevice)

# def unpack_hook(tensor, targetDevice = 'cuda', name = None):
#     if verbosePacking:
#         print(f'Unpacking {name} to {targetDevice}')    
#     return tensor.to(device = targetDevice)

# def hook_tensor(tensor : Optional[torch.Tensor], device, targetDevice = 'cpu', name : str = None):
#     if tensor is not None:
#         # print(f'{name}: {tensor}')
#         if tensor.grad_fn is not None:
#             # if verbosePacking:
#                 # print(f'Hooking {name} to {targetDevice}')
#             tensor.grad_fn._raw_saved_self.register_hooks(lambda t: pack_hook(t, targetDevice, name), lambda t: unpack_hook(t, device, name))
#             # tensor.register_hook(lambda t: pack_hook(t, targetDevice, name))
#             # tensor.storage().register_hook(lambda t: unpack_hook(t, device, name))
#     return tensor


def deltaPlusSPHScheme(SPHSystem, dt, config, verbose = False):    
    with record_function("[deltaSPH]"):
        domain          = config['domain']
        wrappedKernel   = config['kernel']
        particles       = SPHSystem.systemState
        neighborhood    = SPHSystem.neighborhoodInfo
        hadDensity      = particles.densities is not None
        priorDensity    = particles.densities.clone() if hadDensity else None
        
        
    
    verbosePrint(verbose, '[PESPH]\tNeighborsearch')
    with record_function("[deltaSPH] - 02 - Neighborsearch"):
        neighborhood, neighbors = evaluateNeighborhood(particles, domain, wrappedKernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=neighborhood, computeHessian=config['neighborhood']['computeHessian'], computeDkDh=config['neighborhood']['computeDkDh'], only_j = config['neighborhood']['only_j'])
        particles.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries
    
    if not hadDensity:
        verbosePrint(verbose, '[PESPH]\tDensity')
        with record_function("[deltaSPH] - 03 - Density"):
            particles.densities = computeDensity(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather)
    else:
        particles.densities = priorDensity
        
    checkTensor(particles.densities, domain.min.dtype, domain.min.device, 'density')

    if torch.any(particles.kinds > 1):
        with record_function("[deltaSPH] - 03 - mDBC"):
            particles.densities,_ = mDBCDensity(particles, wrappedKernel, neighbors.get('fluidToGhost'), SupportScheme.Scatter, config)
    checkTensor(particles.densities, domain.min.dtype, domain.min.device, 'density (aftert mdbc)')
        # print('Density already computed', particles.densities)
        
    with record_function("[deltaSPH] - 05 - Dirichlet BC"):
        particles = enforceDirichlet(particles, config, SPHSystem.t, dt)    
        
    with record_function("[deltaSPH] - 04 - EOS"):
        particles.pressures = computeEOS_WC(particles, config)
        checkTensor(particles.pressures, domain.min.dtype, domain.min.device, 'pressure')
    
    checkTensor(particles.velocities, domain.min.dtype, domain.min.device, 'velocity (after Dirichlet)')
    checkTensor(particles.densities, domain.min.dtype, domain.min.device, 'density (after Dirichlet)')
    checkTensor(particles.pressures, domain.min.dtype, domain.min.device, 'pressure (after Dirichlet)')
        
    if torch.any(particles.kinds > 0):
        with record_function("[deltaSPH] - 06 - Boundary Velocities"):
            particles.velocities = computeBoundaryVelocities(particles, wrappedKernel, neighbors.get('fluidToGhost'), SupportScheme.Scatter, config)
    checkTensor(particles.velocities, domain.min.dtype, domain.min.device, 'velocity (after boundary)')   
    
    with record_function("[deltaSPH] - 07 - Covariance Matrices"):
        particles.covarianceMatrices, particles.gradCorrectionMatrices, particles.eigenValues = computeCovarianceMatrices(particles, wrappedKernel, neighbors.get('normal'), SupportScheme.Scatter, config)

        checkTensor(particles.covarianceMatrices, domain.min.dtype, domain.min.device, 'covariance matrices')
        checkTensor(particles.gradCorrectionMatrices, domain.min.dtype, domain.min.device, 'correction matrices')
        checkTensor(particles.eigenValues, domain.min.dtype, domain.min.device, 'eigen values')
    with record_function("[deltaSPH] - 08 - Surface Detection"):
        if config.get('surfaceDetection', {}).get('active', False):
            fs, particles.surfaceMask, particles.surfaceNormals, lMin = surfaceDetection(particles, wrappedKernel, neighbors.get('normal'), SupportScheme.Gather, config, False)

    with record_function("[deltaSPH] - 09 - Density Diffusion"):
        particles.gradRhoL = computeGradRhoL(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
        # if torch.any(particles.kinds > 1):
        #     with record_function("[deltaSPH] - 09 - Boundary Density Diffusion"):
        #         particles.gradRhoL += computeGradRhoL(particles, wrappedKernel, neighbors.get('boundaryToFluid'), SupportScheme.Gather, config)

        particles.gradRho = computeGradRho(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
            
        checkTensor(particles.gradRhoL, domain.min.dtype, domain.min.device, 'density diffusion gradient')
        checkTensor(particles.gradRho, domain.min.dtype, domain.min.device, 'density gradient')

        # print(dvdt_diss.shape)
        drhodt_diss = computeDensityDeltaTerm(particles, wrappedKernel, neighbors.get('fluid'), SupportScheme.Gather, config)
        
        checkTensor(drhodt_diss, domain.min.dtype, domain.min.device, 'density diffusion')
    # if config.get('freeSurface', {}).get('active', False):

    with record_function("[deltaSPH] - 10 - Velocity Diffusion"):
        dvdt_diss = computeViscosity_deltaSPH_inviscid(particles, wrappedKernel, neighbors.get('fluid'), SupportScheme.Gather, config)
        if torch.any(particles.kinds > 1):
            with record_function("[deltaSPH] - 10 - Boundary Viscosity"):
                dvdt_diss += computeViscosity_deltaSPH_inviscid(particles, wrappedKernel, neighbors.get('boundaryToFluid'), SupportScheme.Gather, config)
        checkTensor(dvdt_diss, domain.min.dtype, domain.min.device, 'viscosity diffusion')
    
    with record_function("[deltaSPH] - 11 - Divergence"):
        drhodt = computeMomentum(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
        checkTensor(drhodt, domain.min.dtype, domain.min.device, 'density divergence')

    pressureAccel = computePressureForce(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
    checkTensor(pressureAccel, domain.min.dtype, domain.min.device, 'pressure acceleration')

    with record_function("[deltaSPH] - 12 - Gravity"):
        gravityAccel = computeGravity(particles, config)
        checkTensor(gravityAccel, domain.min.dtype, domain.min.device, 'gravity acceleration')
    
    forcing = applyForcing(particles, config, SPHSystem.t, dt)
    # spsTerm = computeSPSTurbulence(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)

    update = WeaklyCompressibleUpdate(
        positions=particles.velocities,
        velocities=pressureAccel + gravityAccel + forcing + dvdt_diss,
        densities=drhodt + drhodt_diss,
        passive = torch.zeros(particles.velocities.shape[0], dtype = torch.bool, device = particles.velocities.device),
    )

    with record_function("[deltaSPH] - 13 - Dirchlet Update"):
        update = enforceDirichletUpdate(update, particles, config, SPHSystem.t, dt)
    checkTensor(update.positions, domain.min.dtype, domain.min.device, 'position update (after Dirichlet update)')
    checkTensor(update.velocities, domain.min.dtype, domain.min.device, 'velocity update (after Dirichlet update)')
    checkTensor(update.densities, domain.min.dtype, domain.min.device, 'density update (after Dirichlet update)')

# @torch.jit.script
# @dataclass(slots = True)
# class PrecomputedNeighborhood:
#     r_ij: torch.Tensor # [n]
#     x_ij: torch.Tensor # [n, d]
    
#     W_i: torch.Tensor # [n], uses h_i
#     W_j: torch.Tensor # [n], uses h_j
#     gradW_i: torch.Tensor # [n, d], uses h_i
#     gradW_j: torch.Tensor # [n, d], uses h_j
    
#     H_i: Optional[torch.Tensor] # [n, d, d], uses h_i
#     H_j: Optional[torch.Tensor] # [n, d, d], uses h_j
    
#     ddh_W_i: Optional[torch.Tensor] # [n], uses h_i
#     ddh_W_j: Optional[torch.Tensor] # [n], uses h_j

    

    # hook_tensor(neighbors.kernelValues.x_ij, particles.positions.device, targetDevice = 'cpu', name = 'x_ij')
    # hook_tensor(neighbors.kernelValues.r_ij, particles.positions.device, targetDevice = 'cpu', name = 'r_ij')
    # hook_tensor(neighbors.kernelValues.W_i, particles.positions.device, targetDevice = 'cpu', name = 'W_i')
    # hook_tensor(neighbors.kernelValues.W_j, particles.positions.device, targetDevice = 'cpu', name = 'W_j')
    # hook_tensor(neighbors.kernelValues.gradW_i, particles.positions.device, targetDevice = 'cpu', name = 'gradW_i')
    # hook_tensor(neighbors.kernelValues.gradW_j, particles.positions.device, targetDevice = 'cpu', name = 'gradW_j')
    # hook_tensor(neighbors.kernelValues.H_i, particles.positions.device, targetDevice = 'cpu', name = 'H_i')
    # hook_tensor(neighbors.kernelValues.H_j, particles.positions.device, targetDevice = 'cpu', name = 'H_j')
    # hook_tensor(neighbors.kernelValues.ddh_W_i, particles.positions.device, targetDevice = 'cpu', name = 'ddh_W_i')
    # hook_tensor(neighbors.kernelValues.ddh_W_j, particles.positions.device, targetDevice = 'cpu', name = 'ddh_W_j')


    return update, particles, neighborhood


from sphMath.schemes.weaklyCompressible import WeaklyCompressibleSystem

    # particleSystem.priorStep = [updates[-1], currentState[-1]]
    # t += dt
    # dx, neighborhood, neighbors, overallStates, particleSystem.systemState.densities, particleSystem.systemState.velocities = solveShifting(particleSystem, 0.1, config, verbose = False)

    # oldVelocity = particleSystem.systemState.velocities.clone()

    # # neighborhood, sparseNeighborhood_ = buildNeighborhood(particleSystem.systemState, particleSystem.systemState, domain, verletScale = config['neighborhood']['verletScale'], mode = 'superSymmetric', priorNeighborhood=particleSystem.neighborhoodInfo)
    # # sparseNeighborhood = filterNeighborhoodByKind(particleSystem.systemState, sparseNeighborhood_, which = 'normal')

    
    # passiveMask = updates[-1].passive == 0 if updates[-1].passive is not None else torch.ones_like(updates[-1].UIDs, dtype = torch.bool)
    # particleSystem.systemState.positions[passiveMask,:] += dx[passiveMask,:]
    # particleSystem.systemState, emitted = processInlets(particleSystem.systemState, config, domain)
    # particleSystem.systemState, removedParticles = processOutlets(particleSystem.systemState, config, domain)
    # if emitted > 0 or removedParticles > 0:
    #     # print(f'Emitted: {emitted}, Removed: {removedParticles}')
    #     particleSystem.neighborhoodInfo = None
    #     particleSystem.priorStep = None
    # # dt = computeTimestep(dt, particleSystem, updates[-1], config, verbose = False)

    # for rigidBody in rigidBodies:
    #     rigidBody = integrateRigidBody(rigidBody, 0, 0, dt)
    #     particleSystem.systemState = updateBodyParticles(particleSystem.systemState, rigidBody)
    
from sphMath.modules.particleShifting import solveShifting

from dataclasses import dataclass, field
from sphMath.neighborhood import NeighborhoodInformation
from sphMath.rigidBody import RigidBody
from typing import Optional, Tuple, Dict, List

@dataclass
class DeltaPlusSPHSystem(WeaklyCompressibleSystem):

    def finalize(self, initialState, dt, returnValues, updateValues, butcherValues, *args, verbose = False, **kwargs):
        verbosePrint(verbose, f'(Deriv) Finalizing system [{initialState.systemState.positions.shape} - {self.t}/{initialState.t} - {dt}]')

        verbosePrint(verbose, '[Fin]\tUpdating state')
        # lastState = returnValues[-1][0]
        # firsState = returnValues[0][0]
        lastUpdate = updateValues[-1]
        
        initialRho = initialState.systemState.densities
        midRho = returnValues[-1][0].densities
        
        drhodtMid = updateValues[-1].densities
        epsilon = -dt * drhodtMid / midRho
        self.systemState.densities = initialRho * (2 - epsilon) / (2+epsilon)


        if self.config['shifting']['active']:
            verbosePrint(verbose, '[Fin]\tUpdating state - 01 - Shifting')
            dx, neighborhood, neighbors, overallStates, self.systemState.densities, self.systemState.velocities, self.systemState.surfaceMask, self.systemState.surfaceNormals, self.systemState.eigenValues = solveShifting(self, 0.1, self.config, verbose = False)
            self.systemState.shiftVectors = dx

            verbosePrint(verbose, '[Fin]\tUpdating state - 02 - Update positions')
            passiveMask = lastUpdate.passive == 0 if lastUpdate.passive is not None else torch.ones_like(lastUpdate.UIDs, dtype = torch.bool)
            self.systemState.positions[passiveMask,:] += dx[passiveMask,:]
            self.systemState.shiftVectors[~passiveMask,:] = 0.0

        self.finalizeState(initialState, dt, returnValues, updateValues, butcherValues, *args, verbose = verbose, **kwargs)

        self.finalizePhysics(initialState, dt, returnValues, updateValues, butcherValues, *args, verbose = verbose, **kwargs)

        return self

from sphMath.kernels import KernelType
# from sphMath.neighborhood import computeOwen
from sphMath.sampling import buildDomainDescription
from sphMath.enums import *

def getDeltaSPHConfig(
    gamma: float = 5./3.,
    kernel: KernelType = KernelType.B7,
    targetNeighbors: int = 50,  
    domain: DomainDescription = buildDomainDescription(1, 2, True, torch.device('cpu'), torch.float32),
    verletScale: float = 1.0,
):
    return {
        # 'gamma': gamma,
        'targetNeighbors': targetNeighbors,
        'domain': domain,
        'kernel': kernel,
        # 'supportIter': 4,
        # 'verletScale': 1.4,
        # 'supportScheme': 'Owen', # Could also use Owen (following CRKSPH)

        # 'adaptiveHThreshold': 1e-9,
        'correctiveOmega': True, # Use Omega terms to correct for adaptive support, seems to not be used in the comp SPH paper but is used in the CRKSPH variant of it
        'neighborhood':{
            'targetNeighbors': targetNeighbors,
            'verletScale': verletScale,
            'scheme': 'compact',
            'computeHessian': False,
            'computeDkDh': False,
            'only_j': False,
        },
        'support':{
          'iterations': 1,
          'adaptiveHThreshold' :1e-3,
          'scheme': AdaptiveSupportScheme.NoScheme,
          'targetNeighbors': targetNeighbors,
          'LUT': None  
        },
        'fluid':{
            'gamma': gamma,
            'backgroundPressure': 0.0,
        },
        'diffusion':{
            'alpha': 0.01
        },
        'diffusionSwitch':{
            'scheme': ViscositySwitch.NoneSwitch,
            'limitXi': False,
        },
        'shifting':{	
            'active': True,
            'scheme': 'delta',
            'freeSurface': False,
        },
        'surfaceDetection':{
            'active': False,
        },
        'pressure':{
            'term': 'Antuono',
        },
        'gravity':{
            'active': False,
        },
        'regions': [],

        # 'owenSupport': computeOwen(kernel, dim = domain.dim, nMin = 2.0, nMax = 10.0, nLUT = 1024),

        # 'C_l': 1, # Linear and quadratic viscosity terms
        # 'C_q': 2,
        # 'Cu_l: 1, # Linear and quadratic viscosity terms for the internal energy
        # 'Cu_q: 2, # However, compsph does not use internal energy dissipation


        # 'monaghanSwitch': True, # Use the viscosity switch (required)
        # 'viscosityTerm': 'Monaghan', # Use the standard viscosity term
        # 'correctXi': True, # Correct the xi term in the viscosity
        # 'signalTerm': 'Monaghan1997', # Not required for this scheme
        # 'thermalConductivity' : 0., # No explicit thermal conductivity
        # 'K': 1.0, # Scaling factor of viscosity

        # Possible energySchemes = ['equalWork', 'PdV', 'diminishing', 'monotonic', 'hybrid', 'CRK']
        'energyScheme': 'CRK',

        # 'limitXi': False, # Limiter for the cullen dehnen viscosity switch for 0 divergence fields,

        # 'viscosityFormulation': 'Monaghan1992', # closest match, is computed in a different symmetrized form here so we have to manually overwrite the use_cbar and use_rho_bar flags
        # 'use_cbar': False, # Use the average speed of sound
        # 'use_rho_bar': False, # Use the average density
        # 'use_h_bar': False, # Use the average support
        # 'scaleBeta': False, # Scale the beta term by the linear viscosity term
        
        'schemeName' : 'CompSPH',
    }
