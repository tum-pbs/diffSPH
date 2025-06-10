from sphMath.modules.adaptiveSmoothingASPH import n_h_to_nH
from sphMath.modules.compSPH import compSPH_acceleration, computeDeltaU
# from sphMath.modules.compressible import systemToParticles, systemUpdate
from sphMath.modules.eos import idealGasEOS
from sphMath.operations import sph_operation, mod
from sphMath.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr, filterNeighborhoodByKind
from sphMath.util import volumeToSupport
from sphMath.util import ParticleSet
from sphMath.sampling import buildDomainDescription, sampleRegularParticles

from sphMath.plotting import visualizeParticles
from sphMath.sampling import generateNoiseInterpolator
from sphMath.util import ParticleSetWithQuantity, mergeParticles
import random
from sphMath.sampling import getSpacing
from sphMath.operations import sph_op
from sphMath.sampling import generateTestData
from sphMath.modules.density import computeDensity, computeDensityGradient, computeRenormalizedDensityGradient, computeDensityDeltaTerm
from sphMath.reference.sod import solve, sodInitialState, plotSod, generateSod1D

from typing import NamedTuple, Tuple
from sphMath.kernels import KernelType
# from sphMath.neighborhood import buildSuperSymmetricNeighborhood
from sphMath.modules.adaptiveSmoothing import computeOmega, evaluateOptimalSupport
from sphMath.modules.adaptiveSmoothingASPH import n_h_to_nH, evaluateOptimalSupportOwen

from sphMath.modules.CRKSPH import computeCRKVolume, computeGeometricMoments, computeCRKTerms, computeCRKDensity, computeVelocityTensor, computeCRKdvdt, computeCRKdudt

from sphMath.modules.adaptiveSmoothingASPH import evaluateOptimalSupportOwen
from sphMath.schemes.monaghanPrice import computeViscosity_Monaghan1997
from sphMath.schemes.gasDynamics import CompressibleSystem, CompressibleUpdate, verbosePrint
# from sphMath.modules.pressuresESPH import PESPH_deltaU_multistep
from dataclasses import dataclass
from sphMath.modules.viscositySwitch import computeViscositySwitch, updateViscositySwitch
from torch.profiler import record_function
import torch
from sphMath.modules.compressible import CompressibleUpdate #, CompressibleState, CompressibleSystem
from sphMath.modules.switches.common import computeM
from sphMath.modules.compSPH import compute_fij, compSPH_dudt, compSPH_acceleration
from sphMath.neighborhood import SupportScheme, evaluateNeighborhood
from sphMath.regions import enforceDirichlet, enforceDirichletUpdate, applyForcing
from sphMath.modules.momentum import computeMomentum, computeMomentumConsistent

from sphMath.modules.gravity import computeGravity
from sphMath.enums import *
def CRKScheme(SPHSystem, dt, config, verbose = False):
    domain          = config['domain']
    wrappedKernel   = config['kernel']
    particles       = SPHSystem.systemState
    neighborhood    = SPHSystem.neighborhoodInfo
    hadDensity      = 'density' in SPHSystem.scheme.lower()
    priorDensity    = particles.densities.clone() if hadDensity else None
    
    verbosePrint(verbose, '[CompSPH]\tOptimizing Support')
    with record_function("[CompSPH] - 01 - Optimize Support"):
        if config['support']['scheme'] == AdaptiveSupportScheme.MonaghanScheme:
            verbosePrint(verbose, '[CompSPH]\t\tMonaghan Scheme')
            rho, h_i_new, rhos, hs, neighborhood = evaluateOptimalSupport(particles, wrappedKernel, neighborhood, SupportScheme.Gather, config)
        elif config['support']['scheme'] == AdaptiveSupportScheme.OwenScheme:
            verbosePrint(verbose, '[CompSPH]\t\tOwen Scheme')
            rho, h_i_new, rhos, hs, neighborhood = evaluateOptimalSupportOwen(particles, wrappedKernel, neighborhood, SupportScheme.Gather, config)
        else:
            verbosePrint(verbose, '[CompSPH]\t\tNo Support Scheme')
            h_i_new = particles.supports
            rho = particles.densities
    particles.supports = h_i_new
    
    if not hadDensity:
        verbosePrint(verbose, '[CompSPH]\tUpdating Density')
        particles.densities = rho
    else:
        verbosePrint(verbose, '[CompSPH]\tDensity computed in previous step')
        particles.densities = priorDensity

    verbosePrint(verbose, '[CompSPH]\tNeighborsearch')
    with record_function("[CompSPH] - 02 - Neighborsearch"):
        neighborhood, neighbors = evaluateNeighborhood(particles, config['domain'], wrappedKernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
        particles.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries
    
    with record_function("[PESPH] - 03 - Compute Volumes"):
        particles.omega = torch.ones_like(particles.masses)
        particles.apparentArea = computeCRKVolume(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)

    with record_function("[PESPH] - 04 - Compute Moments"):
        kernelMoments = computeGeometricMoments(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config)
    
    with record_function("[PESPH] - 05 - Compute Kernal Renormalization Terms"):
        particles.A, particles.B, particles.gradA, particles.gradB = computeCRKTerms(kernelMoments, particles.numNeighbors)

    with record_function("[PESPH] - 06 - Compute Density"):
        if not hadDensity:
            particles.densities = computeCRKDensity(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config)

    # if 'diffusionSwitch' in config and config['diffusionSwitch']['scheme'] is not None:
    #     with record_function("[CompSPH] - 05.5 - Cullen Dehnen Viscosity Terms"):
    #         verbosePrint(verbose, '[CompSPH]\tComputing Cullen Terms')
    #         particles.alphas, switchState = computeViscositySwitch(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config, dt)   

    with record_function("[deltaSPH] - 05 - Dirichlet BC"):
        particles = enforceDirichlet(particles, config, SPHSystem.t, dt)    
    with record_function("[CompSPH] - 05 - Compute EOS"):
        verbosePrint(verbose, '[CompSPH]\tComputing EOS')
        particles.entropies, _, particles.pressures, particles.soundspeeds  = idealGasEOS(A = None, u = particles.internalEnergies, P = None, rho = particles.densities, gamma = config['fluid']['gamma'])
        particles.pressures                                                 = particles.pressures + config['backgroundPressure'] if 'backgroundPressure' in config else particles.pressures

    with record_function("[PESPH] - 08 - Compute Velocity Tensor"):
        # Correct with M, based on SPHeral artificial viscosity handle.cc 294+
        velocityTensor = computeVelocityTensor(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
    
        # But this does not seem to be working in 1D?
        # verbosePrint(verbose, '[Cullen]\t\tComputing M')
        # if config['domain'].dim > 1:
        #     M = computeM(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
        #     M_inv = torch.linalg.pinv(M)     

        #     velocityTensor = torch.einsum('ijk, ikl -> ijl', velocityTensor, M_inv)

    with record_function("[PESPH] - 09 - Compute Acceleration"):
        # dvdt, ap_ij, av_ij = compSPH_acceleration(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config)
        dvdt, _, ap_ij, av_ij = computeCRKdvdt(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config, velocityTensor)
        
        particles.ap_ij = ap_ij
        particles.av_ij = av_ij
        v_halfstep = particles.velocities + 0.5 * dt * dvdt

        verbosePrint(verbose, '[CompSPH]\tComputing Work Distribution')
        particles.f_ij = compute_fij(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config, dt, v_halfstep, ap_ij, av_ij)

    with record_function("[PESPH] - 10 - Compute Energy Equation"):
        dudt = computeCRKdudt(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config, velocityTensor, dvdt, dt)

        # dudt = compSPH_dudt(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config)

    # if 'diffusionSwitch' in config and config['diffusionSwitch']['scheme'] is not None:
    #     with record_function("[CompSPH] - 09 - Compute Cullen Update"):
    #         verbosePrint(verbose, '[CompSPH]\tComputing Cullen Update')
    #         particles.alpha0s, switchState = updateViscositySwitch(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config, dt, dvdt, switchState)

    with record_function("[PESPH] - 08 - Compute Density Update"):
        verbosePrint(verbose, '[Price07]\tComputing Density Update')
        drhodt = computeMomentumConsistent(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
        

    with record_function("[PESPH] - 08 - Compute Energy Update"):
        verbosePrint(verbose, '[Price07]\tComputing Energy Update')
        dEdt = particles.masses * torch.einsum('ij,ij->i', particles.velocities, (dvdt)) + particles.masses * (dudt)

    particles.divergence = drhodt
    forcing = applyForcing(particles, config, SPHSystem.t, dt)
    forcing += computeGravity(particles, config)
    
    update = CompressibleUpdate(
            positions           = particles.velocities.clone(),
            velocities          = dvdt + forcing,
            totalEnergies       = dEdt,
            internalEnergies    = dudt,
            densities           = drhodt,
        passive = torch.zeros(particles.velocities.shape[0], dtype = torch.bool, device = particles.velocities.device),
    )
    with record_function("[deltaSPH] - 13 - Dirchlet Update"):
        update = enforceDirichletUpdate(update, particles, config, SPHSystem.t, dt)
       
    
    return update, particles, neighborhood


from sphMath.modules.adaptiveSmoothingASPH import computeOwen
from sphMath.enums import *

def getCRKConfig(gamma, kernel, targetNeighbors, domain, verletScale):
    
    return {
        # 'gamma': gamma,
        'targetNeighbors': targetNeighbors,
        'domain': domain,
        'kernel': kernel,
        # 'supportIter': 4,
        # 'verletScale': 1.4,
        # 'supportScheme': 'Monaghan', # Could also use Owen
        'correctiveOmega': False, # Use Omega terms to correct for adaptive support, seems to not be used         
        'neighborhood':{
            'targetNeighbors': targetNeighbors,
            'verletScale': verletScale,
            'scheme': 'compact'
        },
        'support':{
          'iterations': 1,
          'adaptiveHThreshold' :1e-3,
          'scheme': AdaptiveSupportScheme.MonaghanScheme,
          'targetNeighbors': targetNeighbors,
          'LUT': None, #computeOwen(kernel, dim = domain.dim, nMin = 2.0, nMax = 10.0, nLUT = 1024),  
        },
        'fluid':{
            'gamma': gamma,
            'backgroundPressure': 0.0,
        },
        # 'owenSupport': computeOwen(kernel, dim = domain.dim, nMin = 2.0, nMax = 6.0, nLUT = 1024),

        'diffusion':{
            'C_l': 2,
            'C_q': 1,
            # 'Cu_l': 1,
            # 'Cu_q': 2,
            'monaghanSwitch': True,
            'viscosityTerm': 'Monaghan',
            'correctXi': True,
            
            'viscosityFormulation': 'Monaghan1992',
            'use_cbar': True,
            'use_rho_bar': False,
            'K': 1.0,
            
            'thermalConductivity' : 0.0,
        },
        'shifting':{	
            'active': False,
            'scheme': 'delta',
            'freeSurface': False,
        },
        'surfaceDetection':{
            'active': False,
        },
        'pressure':{
            'term': 'symmetric',
        },
        'gravity':{
            'active': False,
        },
        'diffusionSwitch':{
            'scheme': ViscositySwitch.NoneSwitch,
            'limitXi': False,
        },
        'regions': [],
        # 'C_l': 2, # Linear and quadratic viscosity terms
        # 'C_q': 1,
        # 'Cu_l: 1, # Linear and quadratic viscosity terms for the internal energy
        # 'Cu_q: 2, # However, PESPH does not use internal energy dissipation

        # 'use_cbar': True, # Use the average speed of sound
        # 'use_rho_bar': False, # Use the average density

        # 'viscositySwitch': None,
        # 'monaghanSwitch': True, # Use the viscosity switch (required)
        # 'viscosityTerm': 'Monaghan', # Use the standard viscosity term
        # 'correctXi': True, # Correct the xi term in the viscosity
        # 'signalTerm': 'Monaghan1997', # Not required for this scheme
        # 'thermalConductivity' : 0., # No explicit thermal conductivity
        # 'K': 1.0, # Scaling factor of viscosity

        # Possible energySchemes = ['equalWork', 'PdV', 'diminishing', 'monotonic', 'hybrid', 'CRK']
        'energyScheme': EnergyScheme.CRK,
        'schemeName': 'CRKSPH',
    }