from sphMath.modules.adaptiveSmoothingASPH import n_h_to_nH
from sphMath.modules.compSPH import compSPH_acceleration, computeDeltaU, compSPH_dudt, compute_fij
# from sphMath.modules.compressible import systemToParticles#, systemUpdate
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
# from sphMath.reference.sod import solve, sodInitialState, plotSod, generateSod1D

from typing import NamedTuple, Tuple
from sphMath.kernels import KernelType
# from sphMath.neighborhood import buildSuperSymmetricNeighborhood
from sphMath.modules.adaptiveSmoothing import computeOmega, evaluateOptimalSupport
from sphMath.modules.adaptiveSmoothingASPH import n_h_to_nH, evaluateOptimalSupportOwen

# from sphMath.modules.compressible import CompressibleSystem, CompressibleUpdate, verbosePrint
import torch
import copy
from torch.profiler import record_function
from sphMath.modules.switches.CullenDehnen2010 import computeCullenTerms, computeCullenUpdate

from sphMath.schemes.gasDynamics import CompressibleSystem, CompressibleUpdate, verbosePrint
from sphMath.modules.compSPH import compSPH_deltaU_multistep
from dataclasses import dataclass
from sphMath.modules.viscositySwitch import computeViscositySwitch, updateViscositySwitch
from sphMath.modules.switches.common import computeDivergence
from dataclasses import dataclass, fields
@dataclass
class CompSPHSystem(CompressibleSystem):
    
    def finalize(self, initialState, dt, returnValues, updateValues, butcherValues, solverConfig, *args, verbose = False, **kwargs):
        verbosePrint(verbose, '[Fin]\tComputing delta u')
        delta_u = compSPH_deltaU_multistep(
            dt,
            initialState.systemState,
            [r[0] for r in returnValues],
            updateValues,
            butcherValues,
            solverConfig,
            verbose = verbose
        )
        if self.systemState.species is None:
            self.systemState.internalEnergies = initialState.systemState.internalEnergies + delta_u * dt
        else:
            mask = self.systemState.species == 0
            self.systemState.internalEnergies[mask] = initialState.systemState.internalEnergies[mask] + delta_u[mask] * dt
        return super().finalize(initialState, dt, returnValues, updateValues, butcherValues, solverConfig, *args, verbose=verbose,  **kwargs)
    
    def preprocess(self, initialState, dt, *args, verbose = False, **kwargs):
        verbosePrint(verbose, f'[Pre ] integration [{initialState.systemState.positions.shape} - {initialState.t}/{self.t} - {dt}]')
        if self.t == initialState.t and 'density' not in self.scheme.lower():
            self.systemState.densities = None
        return super().preprocess(initialState, dt, *args, **kwargs)
    

    def to(self, dtype = torch.float32, device = torch.device('cpu')):
        convertedState = copy.deepcopy(self.systemState)
        convertedDomain = DomainDescription(
            min = self.domain.min.to(dtype=dtype, device=device),
            max = self.domain.max.to(dtype=dtype, device=device),
            periodic = self.domain.periodic,
            dim = self.domain.dim
        ) 

        for field in fields(convertedState):
            if getattr(convertedState, field.name) is not None and isinstance(getattr(convertedState, field.name), torch.Tensor):
                setattr(convertedState, field.name, getattr(convertedState, field.name).to(dtype=dtype, device=device))
        
        return CompSPHSystem(
            systemState = convertedState,
            domain = convertedDomain,
            neighborhoodInfo = None,
            t = self.t,
            scheme = self.scheme,
            ghostState = None
        )
from sphMath.neighborhood import SupportScheme, evaluateNeighborhood
from sphMath.regions import enforceDirichlet, enforceDirichletUpdate, applyForcing
from sphMath.modules.momentum import computeMomentum, computeMomentumConsistent
from sphMath.enums import *
from sphMath.modules.gravity import computeGravity

def compSPHScheme(SPHSystem : CompSPHSystem, dt : float, config : dict, verbose : bool = False):
    verbosePrint(verbose, '[CompSPH] Scheme Step')

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
        neighborhood, neighbors = evaluateNeighborhood(particles, config['domain'], wrappedKernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=neighborhood, computeDkDh = True, computeHessian = False, useCheckpoint=False)
        particles.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries

    # $$\rho_i = \sum_j m_j W_{ij}(h_i)$$
    with record_function("[CompSPH] - 03 - Compute Density"):
        if not hadDensity:
            verbosePrint(verbose, '[CompSPH]\tComputing Density')
            particles.densities = computeDensity(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
        else:
            verbosePrint(verbose, '[CompSPH]\tDensity computed in previous step')

    # $$\frac{Dv_i^\alpha}{Dt} = -\sum_j m_j \left[\left(\frac{1}{\Omega_i}\frac{P_i}{\rho_i^2} + \frac{1}{2}\Pi_i\right)\partial_\alpha W_{ij}(h_i) + \left(\frac{1}{\Omega_j}\frac{P_j}{\rho_j^2} + \frac{1}{2}\Pi_j\right)\partial_\alpha W_{ij}(h_j)\right]$$
    with record_function("[deltaSPH] - 05 - Dirichlet BC"):
        particles = enforceDirichlet(particles, config, SPHSystem.t, dt)    

    with record_function("[CompSPH] - 05 - Compute EOS"):
        verbosePrint(verbose, '[CompSPH]\tComputing EOS')
        particles.entropies, _, particles.pressures, particles.soundspeeds  = idealGasEOS(A = None, u = particles.internalEnergies, P = None, rho = particles.densities, gamma = config['fluid']['gamma'])
        particles.pressures                                                 = particles.pressures + config['backgroundPressure'] if 'backgroundPressure' in config else particles.pressures


    # $$\Omega_i = 1 - \frac{\partial h_i}{\partial\rho_i}\sum_j m_j \frac{\partial W_{ij}(h_i)}{\partial h_i} = - \frac{1}{\nu\rho_i}\sum_j m_j \eta_i \frac{\partial W_{ij}(h_i)}{\partial \eta_i}$$
    with record_function("[CompSPH] - 04 - Compute Omega"):
        if config['correctiveOmega']:
            verbosePrint(verbose, '[CompSPH]\tComputing gradH Terms')
            particles.omega = computeOmega(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
        else:
            verbosePrint(verbose, '[CompSPH]\tNo gradH Correction')
            particles.omega = torch.ones_like(particles.densities)

    if 'diffusionSwitch' in config and config['diffusionSwitch']['scheme'] is not None:
        with record_function("[CompSPH] - 05.5 - Cullen Dehnen Viscosity Terms"):
            verbosePrint(verbose, '[CompSPH]\tComputing Cullen Terms')
            particles.alphas, switchState = computeViscositySwitch(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config, dt)   

    with record_function("[CompSPH] - 06 - Compute Acceleration"):
        verbosePrint(verbose, '[CompSPH]\tComputing Acceleration')
        dvdt, ap_ij, av_ij = compSPH_acceleration(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config)
        
    with record_function("[CompSPH] - 07 - Compute Energy Update"):
        verbosePrint(verbose, '[CompSPH]\tComputing Energy Update')
        dudt = compSPH_dudt(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config)

    with record_function("[CompSPH] - 08 - Compute Work Terms"):
        verbosePrint(verbose, '[CompSPH]\tComputing Halfstep Velocity')
        particles.ap_ij = ap_ij
        particles.av_ij = av_ij
        v_halfstep = particles.velocities + 0.5 * dt * dvdt

        verbosePrint(verbose, '[CompSPH]\tComputing Work Distribution')
        particles.f_ij = compute_fij(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config, dt, v_halfstep, ap_ij, av_ij)

    with record_function("[CompSPH] - 08 - Compute Density Update"):
        verbosePrint(verbose, '[CompSPH]\tComputing Density Update')
        drhodt = computeMomentumConsistent(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
        
        # -particles.densities / particles.omega * sph_op(particles, particles, domain, wrappedKernel, actualNeighbors, 'gather', 'divergence', gradientMode = 'difference' , consistentDivergence = True, quantity=(particles.velocities, particles.velocities))

    with record_function("[CompSPH] - 08 - Compute Energy Update"):
        verbosePrint(verbose, '[CompSPH]\tComputing Energy Update')
        dEdt = particles.masses * torch.einsum('ij,ij->i', particles.velocities, dvdt) + particles.masses * dudt

    if 'diffusionSwitch' in config and config['diffusionSwitch']['scheme'] is not None:
        with record_function("[CompSPH] - 09 - Compute Cullen Update"):
            verbosePrint(verbose, '[CompSPH]\tComputing Cullen Update')
            particles.alpha0s, switchState = updateViscositySwitch(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config, dt, dvdt, switchState)

    with record_function("[CompSPH] - 10 - Compute Divergence"):
        verbosePrint(verbose, '[CompSPH]\tComputing Divergence')
        particles.divergence = drhodt
    

    dxdt = particles.velocities.clone()

    # if 'imposeSolution' in config and config['imposeSolution'] is not None:
    #     verbosePrint(verbose, '[CompSPH]\tImposing Solution')
    #     dxdt, dvdt, dEdt, dudt, drhodt = config['imposeSolution'](particles, domain, wrappedKernel, actualNeighbors, config, SPHSystem.t, dxdt, dvdt, dEdt, dudt, drhodt)
    #     # particles.velocities = dxdt

    with record_function("[deltaSPH] - 12 - Gravity"):
        gravityAccel = computeGravity(particles, config)
        # checkTensor(gravityAccel, domain.min.dtype, domain.min.device, 'gravity acceleration')
    forcing = applyForcing(particles, config, SPHSystem.t, dt)
    # print(torch.linalg.norm(gravityAccel, dim = -1))
        
    update = CompressibleUpdate(
        positions           = dxdt,
        velocities          = dvdt + forcing + gravityAccel,
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


def getCompSPHConfig(gamma, kernel, targetNeighbors, domain, verletScale):
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
            'scheme': 'compact'
        },
        'support':{
          'iterations': 1,
          'adaptiveHThreshold' :1e-3,
          'scheme': AdaptiveSupportScheme.OwenScheme,
          'targetNeighbors': targetNeighbors,
          'LUT': None, #computeOwen(kernel, dim = domain.dim, nMin = 2.0, nMax = 10.0, nLUT = 1024),  
        },
        'fluid':{
            'gamma': gamma,
            'backgroundPressure': 0.0,
        },
        'diffusion':{
            'C_l': 1,
            'C_q': 2,
            'Cu_l': 1,
            'Cu_q': 2,
            'monaghanSwitch': True,
            'viscosityTerm': 'Monaghan',
            'correctXi': True,
            
            'viscosityFormulation': 'Monaghan1992',
            'use_cbar': False,
            'use_rho_bar': False,
            'use_h_bar': False,
            'scaleBeta': False,
            'K': 1.0,
            
            'thermalConductivity' : 0.0,
        },
        'diffusionSwitch':{
            'scheme': ViscositySwitch.CullenHopkins,
            'limitXi': False,
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
        'energyScheme': EnergyScheme.CRK,

        # 'limitXi': False, # Limiter for the cullen dehnen viscosity switch for 0 divergence fields,

        # 'viscosityFormulation': 'Monaghan1992', # closest match, is computed in a different symmetrized form here so we have to manually overwrite the use_cbar and use_rho_bar flags
        # 'use_cbar': False, # Use the average speed of sound
        # 'use_rho_bar': False, # Use the average density
        # 'use_h_bar': False, # Use the average support
        # 'scaleBeta': False, # Scale the beta term by the linear viscosity term
        
        'schemeName' : 'CompSPH',
    }

def getCompSPHConfigQ0(gamma, kernel, targetNeighbors, domain, verletScale):
    config = getCompSPHConfig(gamma, kernel, targetNeighbors, domain, verletScale)
    config['C_l'] = 0
    config['C_q'] = 0
    return config

def getCompSPHConfigCullen(gamma, kernel, targetNeighbors, domain, verletScale):
    config = getCompSPHConfig(gamma, kernel, targetNeighbors, domain, verletScale)
    config['diffusionSwitch']['scheme'] = ViscositySwitch.CullenDehnen2010
    return config