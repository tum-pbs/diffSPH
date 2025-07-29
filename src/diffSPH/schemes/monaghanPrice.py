from diffSPH.kernels import SPHKernel
from diffSPH.neighborhood import SparseCOO
from diffSPH.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product

from diffSPH.modules.eos import idealGasEOS
# from diffSPH.modules.compressible import CompressibleState
# from diffSPH.modules.velocityDiffusion import computePrice2012_velocityDissipation
# from diffSPH.modules.energyDiffusion import computePrice2012_energyDissipation
from torch.profiler import record_function
# from diffSPH.modules.compressible import CompressibleState, SPHSystem, systemToParticles, systemUpdate
# from diffSPH.modules.compressible import CompressibleState, systemToParticles


from diffSPH.sampling import buildDomainDescription, sampleRegularParticles
from diffSPH.modules.density import computeDensity, computeDensityGradient, computeRenormalizedDensityGradient, computeDensityDeltaTerm
from typing import NamedTuple, Tuple
from diffSPH.kernels import KernelType
# from diffSPH.neighborhood import buildSuperSymmetricNeighborhood
from diffSPH.modules.adaptiveSmoothing import computeOmega, evaluateOptimalSupport
from diffSPH.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr, filterNeighborhoodByKind
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH
import torch

from diffSPH.operations import sph_op
from diffSPH.sphOperations.shared import compute_xij, getTerms
from diffSPH.modules.viscosity import compute_Pi
from diffSPH.modules.adaptiveSmoothingASPH import evaluateOptimalSupportOwen
from diffSPH.schemes.gasDynamics import CompressibleSystem, CompressibleUpdate, CompressibleState
from diffSPH.kernels import evalW, evalGradW, evalDerivativeW
import copy
from diffSPH.modules.velocityDiffusion import computeViscosity_Monaghan1997
from diffSPH.modules.energyDiffusion import computeMonaghan1997Dissipation
from diffSPH.modules.switches.CullenDehnen2010 import computeCullenTerms, computeCullenUpdate
from diffSPH.schemes.gasDynamics import CompressibleSystem, CompressibleUpdate, verbosePrint, checkTensor
from diffSPH.modules.viscositySwitch import computeViscositySwitch, updateViscositySwitch
from diffSPH.neighborhood import SupportScheme, evaluateNeighborhood
from diffSPH.regions import enforceDirichlet, enforceDirichletUpdate, applyForcing
from diffSPH.modules.momentum import computeMomentum, computeMomentumConsistent


from diffSPH.sphOperations.shared import getTerms, compute_xij
from diffSPH.modules.viscosity import compute_Pi
from diffSPH.schemes.baseScheme import verbosePrint
from diffSPH.operations import sph_op
from torch.profiler import record_function
from diffSPH.operations import sph_op, SPHOperation
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState
from diffSPH.schemes.gasDynamics import CompressibleState
from diffSPH.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme, NeighborhoodInformation
from diffSPH.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple, Union
from diffSPH.util import getSetConfig
from diffSPH.modules.PSPH import compute_dvdt, compute_dndh, compute_dPdh, computeKernelMoment, computePressure_PSH, compute_ArtificialConductivity, compute_dEdt

from diffSPH.enums import *
def pressureForce(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather,
        config: Dict = {}):
    
    rho = particles.densities
    return -SPHOperation(
        particles,
        particles.pressures,
        kernel,
        neighborhood[0],
        neighborhood[1],
        operation = Operation.Gradient,
        supportScheme= supportScheme,
        gradientMode = GradientMode.Symmetric,
        correctionTerms=[KernelCorrectionScheme.gradH]
    ) / rho.view(-1, 1)

def compute_dudt(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather,
        config: Dict = {}):
    
    pressure = particles.pressures
    omega = particles.omega if hasattr(particles, 'omega') and particles.omega is not None else torch.ones_like(particles.densities)
    rho = particles.densities

    term = -pressure / (omega * rho)
    return term * SPHOperation(
        particles,
        particles.velocities,
        kernel,
        neighborhood[0],
        neighborhood[1],
        operation = Operation.Divergence,
        supportScheme= supportScheme,
        gradientMode = GradientMode.Difference,
        consistentDivergence=True)


from diffSPH.enums import *
from diffSPH.modules.gravity import computeGravity

def MonaghanScheme(SPHSystem, dt, config, verbose = False):        
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

    # $$\rho_i = \sum_j m_j W_{ij}(h_i)$$
    with record_function("[CompSPH] - 03 - Compute Density"):
        if not hadDensity:
            verbosePrint(verbose, '[CompSPH]\tComputing Density')
            particles.densities = computeDensity(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
        else:
            verbosePrint(verbose, '[CompSPH]\tDensity computed in previous step')

    with record_function("[deltaSPH] - 05 - Dirichlet BC"):
        particles = enforceDirichlet(particles, config, SPHSystem.t, dt)    
    # $$\frac{Dv_i^\alpha}{Dt} = -\sum_j m_j \left[\left(\frac{1}{\Omega_i}\frac{P_i}{\rho_i^2} + \frac{1}{2}\Pi_i\right)\partial_\alpha W_{ij}(h_i) + \left(\frac{1}{\Omega_j}\frac{P_j}{\rho_j^2} + \frac{1}{2}\Pi_j\right)\partial_\alpha W_{ij}(h_j)\right]$$
    with record_function("[CompSPH] - 05 - Compute EOS"):
        verbosePrint(verbose, '[CompSPH]\tComputing EOS')
        particles.entropies, _, particles.pressures, particles.soundspeeds  = idealGasEOS(A = None, u = particles.internalEnergies, P = None, rho = particles.densities, gamma = config['fluid']['gamma'])
        particles.pressures                                                 = particles.pressures + config['backgroundPressure'] if 'backgroundPressure' in config else particles.pressures

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

    with record_function("[SPH] - 06 - dvdt [pressure]"):
        dvdt = pressureForce(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config)

    with record_function("[SPH] - 07 - dvdt [dissipation]"):
        dvdt_diss = computeViscosity_Monaghan1997(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Symmetric, config)
        # dvdt += dvdt_diss
    with record_function("[SPH] - 08 - dudt [velocity]"):
        dudt = compute_dudt(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
    with record_function("[SPH] - 09 - dudt [dissipation]"):
        dudt_diss = computeMonaghan1997Dissipation(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Symmetric, config)

    with record_function("[CompSPH] - 08 - Compute Density Update"):
        verbosePrint(verbose, '[CompSPH]\tComputing Density Update')
        drhodt = computeMomentumConsistent(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)

    with record_function("[Price07] - 08 - Compute Energy Update"):
        verbosePrint(verbose, '[Price07]\tComputing Energy Update')
        dEdt = particles.masses * torch.einsum('ij,ij->i', particles.velocities, (dvdt + dvdt_diss)) + particles.masses * (dudt + dudt_diss)

    if 'diffusionSwitch' in config and config['diffusionSwitch']['scheme'] is not None:
        with record_function("[CompSPH] - 09 - Compute Cullen Update"):
            verbosePrint(verbose, '[CompSPH]\tComputing Cullen Update')
            particles.alpha0s, switchState = updateViscositySwitch(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config, dt, dvdt, switchState)

    with record_function("[CompSPH] - 10 - Compute Divergence"):
        verbosePrint(verbose, '[CompSPH]\tComputing Divergence')
        particles.divergence = drhodt

    # dudt[:] = 0

    if verbose:
        dtype = particles.positions.dtype
        device = particles.positions.device
        checkTensor(particles.supports, dtype, device, 'Support', verbose)
        checkTensor(particles.densities, dtype, device, 'Density', verbose)
        checkTensor(particles.pressures, dtype, device, 'Pressure', verbose)
        checkTensor(particles.soundspeeds, dtype, device, 'Sound Speed', verbose)
        checkTensor(particles.velocities, dtype, device, 'Velocity', verbose)
        checkTensor(particles.internalEnergies, dtype, device, 'Internal Energy', verbose)
        # checkTensor(particles.quantities, dtype, device, 'Quantity', verbose)
        checkTensor(particles.omega, dtype, device, 'Omega', verbose)
        checkTensor(particles.alphas, dtype, device, 'Alpha', verbose)
        checkTensor(particles.alpha0s, dtype, device, 'Alpha0', verbose)
        checkTensor(dvdt, dtype, device, 'dvdt', verbose)
        # checkTensor(dvdt_diss, dtype, device, 'dvdt_diss', verbose)
        checkTensor(dudt, dtype, device, 'dudt', verbose)
        checkTensor(dudt_diss, dtype, device, 'dudt_diss', verbose)
        checkTensor(drhodt, dtype, device, 'drhodt', verbose)
        checkTensor(dEdt, dtype, device, 'dEdt', verbose)
            

    forcing = applyForcing(particles, config, SPHSystem.t, dt)
    forcing += computeGravity(particles, config)

    update = CompressibleUpdate(
        positions           = particles.velocities.clone(),
        velocities          = dvdt + dvdt_diss + forcing,
        totalEnergies       = dEdt,
        internalEnergies    = dudt + dudt_diss,
        densities           = drhodt,
        passive = torch.zeros(particles.velocities.shape[0], dtype = torch.bool, device = particles.velocities.device),
    )
    with record_function("[deltaSPH] - 13 - Dirchlet Update"):
        update = enforceDirichletUpdate(update, particles, config, SPHSystem.t, dt)
    
    return update, particles, neighborhood

    # return systemUpdate(
    #     positions = particles.velocities,
    #     velocities = dvdt + dvdt_diss,
    #     thermalEnergies = dudt + dudt_diss
    # ), particles, neighborhood



from diffSPH.modules.adaptiveSmoothingASPH import computeOwen
from diffSPH.enums import *

def getPrice2007Config(gamma, kernel, targetNeighbors, domain, verletScale):
    return {
        # 'gamma': gamma,
        'targetNeighbors': targetNeighbors,
        'domain': domain,
        'kernel': kernel,
        # 'supportIter': 4,
        # 'verletScale': 1.4,
        # 'supportScheme': 'Monaghan', # Could also use Owen
        'correctiveOmega': True, # Use Omega terms to correct for adaptive support, seems to not be used in the comp SPH paper
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
            
            'viscosityFormulation': 'Price2012_98',
            'thermalConductivityFormulation': 'Price2008',
            'signalTerm': 'Price2019',
            'K': 1.0,
            
            'thermalConductivity' : 0.5,
        },
        'diffusionSwitch':{
            'scheme': ViscositySwitch.NoneSwitch,
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
        # 'owenSupport': computeOwen(kernel, dim = domain.dim, nMin = 2.0, nMax = 6.0, nLUT = 1024),

        # 'correctXi': True, # Correct the xi term in the viscosity

        # 'viscosityFormulation': 'Price2012_98', 
        # 'signalTerm': 'Price2019', # Not required for this scheme
        # 'C_l': 1, # Linear and quadratic viscosity terms
        # 'C_q': 2,
        # 'thermalConductivityFormulation': 'Price2008',
        # 'Cu_l': 1, # Linear and quadratic viscosity terms for the internal energy
        # 'Cu_q': 2, # However, compsph does not use internal energy dissipation
        # 'viscosityTerm': 'Monaghan1997', # Use the standard viscosity term

        # 'monaghanSwitch': True, # Use the viscosity switch (required)
        # 'thermalConductivity' : 0.5, # Scaling factor of thermal conductivity
        # 'K': 1.0, # Scaling factor of viscosity
        
        # # 'use_cbar': False, # Use the average speed of sound
        # # 'use_rho_bar': True, # Use the average density
        # 'signalTerm': 'Price2019', # Not required for this scheme
        
        # 'viscositySwitch': None,
        # 'viscositySwitchScaling': 'ij',
        
        'schemeName': 'Price2007'

    }
    
# Based on Monaghan 1997: SPH and Riemann Solvers
# The respective viscosity and thermal conductivity formulations are used from (4.7)
# This scheme does not differentiate between thermal and kinetic dissipation terms
def getMonaghan1997Config(gamma, kernel, targetNeighbors, domain, verletScale):
    baseConfig = getPrice2007Config(gamma, kernel, targetNeighbors, domain, verletScale)
    baseConfig['diffusion']['viscosityFormulation'] = 'Monaghan1997b'
    baseConfig['diffusion']['thermalConductivityFormulation'] = 'Monaghan1997b'
    baseConfig['diffusion']['viscosityTerm'] = 'Monaghan1997'
    baseConfig['diffusion']['thermalConductivity'] = 1.0
    baseConfig['schemeName'] = 'Monaghan1997'
    
    return baseConfig
    

def getMonaghan1992Config(gamma, kernel, targetNeighbors, domain, verletScale):
    baseConfig = getPrice2007Config(gamma, kernel, targetNeighbors, domain, verletScale)
    baseConfig['diffusion']['viscosityFormulation'] = 'Monaghan1992'
    baseConfig['diffusion']['thermalConductivityFormulation'] = 'Monaghan1992'
    baseConfig['diffusion']['viscosityTerm'] = 'Monaghan1992'
    baseConfig['diffusion']['thermalConductivity'] = 1.0
    baseConfig['schemeName'] = 'Monaghan1992'
    
    return baseConfig

def getMonaghanGingold1983Config(gamma, kernel, targetNeighbors, domain, verletScale):
    baseConfig = getPrice2007Config(gamma, kernel, targetNeighbors, domain, verletScale)
    baseConfig['diffusion']['viscosityFormulation'] = 'MonaghanGingold1983'
    baseConfig['diffusion']['thermalConductivityFormulation'] = 'MonaghanGingold1983'
    baseConfig['diffusion']['viscosityTerm'] = 'Monaghan1992'
    baseConfig['diffusion']['thermalConductivity'] = 1.0
    baseConfig['schemeName'] = 'MonaghanGingold1983'
    
    return baseConfig