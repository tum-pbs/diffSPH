from sphMath.modules.switches.CullenDehnen2010 import computeCullenTerms, computeCullenUpdate
from sphMath.modules.PSPH import compute_dndh, compute_dPdh, computePressure_PSH, computeKernelMoment, compute_dvdt, compute_dEdt, compute_ArtificialConductivity
from sphMath.modules.compressible import CompressibleUpdate
from sphMath.modules.eos import idealGasEOS
# from sphMath.modules.compressible import CompressibleParticleSet
# from sphMath.modules.velocityDiffusion import computePrice2012_velocityDissipation
# from sphMath.modules.energyDiffusion import computePrice2012_energyDissipation
from torch.profiler import record_function
# from sphMath.modules.compressible import CompressibleParticleSet, SPHSystem, systemToParticles, systemUpdate
# from sphMath.modules.compressible import CompressibleParticleSet, systemToParticles


from sphMath.sampling import buildDomainDescription, sampleRegularParticles
from sphMath.modules.density import computeDensity, computeDensityGradient, computeRenormalizedDensityGradient, computeDensityDeltaTerm
from typing import NamedTuple, Tuple
from sphMath.kernels import KernelType
# from sphMath.neighborhood import buildSuperSymmetricNeighborhood
from sphMath.modules.adaptiveSmoothing import computeOmega, evaluateOptimalSupport
from sphMath.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr, filterNeighborhoodByKind
from sphMath.modules.adaptiveSmoothingASPH import n_h_to_nH
import torch

from sphMath.operations import sph_op
# from sphMath.modules.PESPH import PESPH_viscosity, PESPH_dissipation


from sphMath.modules.adaptiveSmoothingASPH import evaluateOptimalSupportOwen
from sphMath.schemes.monaghanPrice import computeViscosity_Monaghan1997
from sphMath.schemes.gasDynamics import CompressibleSystem, CompressibleUpdate, verbosePrint
# from sphMath.modules.PESPH import PESPH_deltaU_multistep
from dataclasses import dataclass
from sphMath.modules.viscositySwitch import computeViscositySwitch, updateViscositySwitch
from sphMath.neighborhood import SupportScheme, evaluateNeighborhood
from sphMath.regions import enforceDirichlet, enforceDirichletUpdate, applyForcing
from sphMath.modules.momentum import computeMomentum, computeMomentumConsistent


# PBSPH based on CRK SPH

# $$P_i = \sum_j (\gamma -1) m_j u_j W_{ij}(h_i)$$
# $$\rho_i = \sum_j m_j W_{ij}(h_i)$$
# $$n_i = \sum_j W_{ij}(h_i)$$

# $$\frac{D v_i^\alpha}{Dt} = -\sum_j m_j \left[ (\gamma - 1)^2 u_i u_j \left(\frac{f_{ij}}{P_i}\partial_\alpha W_{ij}(h_i) + \frac{f_{ij}}{P_j}\partial_\alpha W_{ij}(h_i)\right) + q_{\text{acc}ij}^\alpha\right]$$

# $$\frac{DE_i}{Dt} = m_i v_i^\alpha + \sum_j m_i m_j \left[(\gamma - 1)^2 u_i u_j \frac{f_{ij}}{P_i} v_{ij}^\alpha \partial_\alpha W_{ij}(h_i) + v_{ij}^\alpha q_{\text{acc}ij}^\alpha\right]$$

# $$q_{\text{acc}ij}^\alpha = \frac{1}{2} (\rho_i \Pi_i + \rho_j + \Pi_j) \frac{\partial_\alpha W_{ij}(h_i) + \partial_\alpha W_{ij}(h_j)}{\rho_i + \rho_j}$$

# $$f_{ij} = 1 - \left(\frac{h_i}{\nu(\gamma -1)n_i m_j u_j}\frac{\partial P_i}{\partial h_i}\right)\left(1 + \frac{h_i}{\nu n_i}\frac{\partial n_i}{\partial h_i}\right)^{-1}$$

# $$\frac{\partial n_i}{\partial h_i} = - \sum_j h_i^{-1} \left(\nu W_{ij}(h_i) + \eta_i \frac{\partial W}{\partial\eta}(\eta_i)\right)$$

# $$\frac{\partial P_i}{\partial h_i} = - \sum_j (\gamma -1) m_j u_j h_i^{-1}\left(\nu W_{ij}(h_i) + \eta_i \frac{\partial W}{\partial\eta}(\eta_i)\right)$$

# with $\eta_i = \frac{x_{ij}}{h_i}$, $x_{ij} = x_i - x_j$ for $\nu$ dimensions, also add artificial conductivity

# $$\frac{DE_i}{Dt} = \alpha_c \sum_j m_i m_j \alpha_{ij}\tilde{v}_s (u_i - u_j) \frac{|P_i - P_j|}{P_i + P_j} \frac{\partial_\alpha W_{ij}(h_i) + \partial_\alpha W_{ij}(h_j)}{\rho_i + \rho_j}$$

# $$\tilde{v}_s = c_i + c_j - 3 \frac{v_{ij}^\alpha x_{ij}^\alpha}{|x_{ij}}$$

# $$\alpha_{ij} = \frac{1}{2} (\alpha_i + \alpha_j)$$

# When $\tilde{v}_s > 0$ with $\alpha_c = 0.25$, $\gamma$ being the adiabtic index and $c_i$ being obtained from a gas EOS $c_i = \sqrt{u\gamma(\gamma-1)}$

# This uses the cullen-dehnen viscosity for the monaghan-gingold viscosity for the linear and quadratic (richtmyer) viscosities $C_l$ and $C_q$:

# $$\Pi_i = \frac{1}{\rho_i}\left(-C_l c_i \mu_i + C_q \mu_i^2\right);\quad \mu_i = \operatorname{min}\left(0, \frac{v_{ij}^\alpha \nu_i^\alpha}{\nu_i^\alpha \nu_i^\alpha}\right)$$

# For the cullen-dehnen model we have

# $$C_{lij} = \frac{1}{2}(\alpha_i + \alpha_j)C_j,\quad C_{qij} = \frac{1}{2}(\alpha_i + \alpha_j)C_q$$

# $$\alpha_i = \operatorname{max}\left(\alpha_{\text{min}}, \frac{|\beta_\xi \xi^4 \partial_\theta v_i^\theta|^2}{|\beta_\xi \xi^4 \partial_\theta v_i^\theta|^2 + S^{\theta\psi}S^{\psi\theta}}\alpha_{0i}(t)\right)$$

# $$\xi_i = 1 - \frac{1}{\rho_i}\sum_j \operatorname{sgn\partial_\theta v_i^\theta}m_j W_{ij}(h_i)$$

# $$\alpha_{\text{tmp}i} = \begin{cases}
# 0,&\partial_t (\partial_\theta v_i^\theta) \geq 0 \text{or} \partial_\theta v_i^\theta \geq 0\\
# \frac{\alpha_{\text{max}} | \partial_t \partial_\theta v_i^\theta|}{\alpha_\text{max} |\partial_t (\partial_\theta v_i^\theta)| + \beta_c c_i^2 (f_\text{kern}h_i)^{-2}},&^\text{otherwise}\end{cases}$$

# $$\alpha_{0i}(t+\Delta t) = \begin{cases}
# \alpha_{\text{tmp}i} & \alpha_{\text{tmp}i} \geq \alpha_{0i}(t)\\
# \alpha_{\text{tmp}i} + \left(\alpha_{0i}(t) - \alpha_{\text{tmp}i}\right)e^{-\beta_d \Delta t v_{\text{sig}_i}}/(2f_\text{kern}h_i), & \text{otherwise}\end{cases}$$

# from [12] $ v_{\text{sig}_i} = \operatorname{max}\left\{\bar{c}_{ij} - \operatorname{min}(0, v_{ij} \cdot \hat{x}_{ij})\right\}$

# with $\alpha_\text{min} = 0.02, \alpha_\text{max} = 2, \beta_c = 0.7, \beta_d = 0.05, f_\text{kern} = 3$ (might need to adjust $f_\text{kern}$ to 1 over kernel scale) and $S$ being the shear tensor.

# $$\partial_\beta v_i^\alpha = -(M_i^{-1})^{\theta\beta}\sum_j v_{ij}^\alpha \partial_\phi W_{ij}(h_i), \quad M_i^{\alpha\beta} = -\sum_j m_j x_{ij}^\alpha \partial\beta W_{ij}(h_i)$$

# for the shear tensor we have $V_i = \nabla_i \otimes v$ which is split into the isotropic part $\nabla\cdot v\mathbf{I}$, the shear (symmetric traceless) $S$ and vorticity (antisymmetric) part $R$ as (symmetric being $1/2 (A+A^T)$ (with diagonal elements!) and antisymmetric being $1/2(A-A^T)$)
# $$V_i = \frac{1}{\nu}\nabla\cdot v\mathbf{I} + \mathbf{S} + \mathbf{R}$$

# We can then also get $\partial_t \partial_\theta v_i^\theta$ as the trace of $\dot{\mathbf{V}}$  which is either obtained via finite difference or
# $$\dot{\mathbf{V}} = \nabla \otimes \frac{dv}{dt} - \mathbf{V}^2$$

from sphMath.modules.gravity import computeGravity
from sphMath.enums import *
def PressureEnergyScheme(SPHSystem, dt, config, verbose = False):
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

    if 'diffusionSwitch' in config and config['diffusionSwitch']['scheme'] is not None:
        with record_function("[CompSPH] - 05.5 - Cullen Dehnen Viscosity Terms"):
            verbosePrint(verbose, '[CompSPH]\tComputing Cullen Terms')
            particles.alphas, switchState = computeViscositySwitch(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config, dt)   
    
    
    with record_function("[PESPH] - 06 - Compute Pressure Terms"):
        verbosePrint(verbose, '[PESPH]\tComputing Pressure Terms')
        particles.P = computePressure_PSH(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
        particles.dPdh = compute_dPdh(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
        particles.pressures = particles.P

            
    with record_function("[PESPH] - 07 - Compute Kernel Moments"):
        verbosePrint(verbose, '[PESPH]\tComputing Kernel Moments')
        particles.n = computeKernelMoment(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)
        particles.dndh = compute_dndh(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)

    with record_function("[PESPH] - 08 - Compute SPH Update"):
        verbosePrint(verbose, '[PESPH]\tComputing SPH Update')
        dvdt = compute_dvdt(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config)
        dkdt = particles.masses * torch.einsum('ij,ij->i', particles.velocities, dvdt)


    with record_function("[PESPH] - 08 - Compute Energy Update"):
        verbosePrint(verbose, '[PESPH]\tComputing Energy Update')
        dEdt = compute_dEdt(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config, dvdt)

        dEdt += compute_ArtificialConductivity(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, config)
        dudt = dEdt / particles.masses
        dEdt += dkdt

    with record_function("[CompSPH] - 08 - Compute Density Update"):
        verbosePrint(verbose, '[CompSPH]\tComputing Density Update')
        drhodt = computeMomentumConsistent(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config)

    if 'diffusionSwitch' in config and config['diffusionSwitch']['scheme'] is not None:
        with record_function("[CompSPH] - 09 - Compute Cullen Update"):
            verbosePrint(verbose, '[CompSPH]\tComputing Cullen Update')
            particles.alpha0s, switchState = updateViscositySwitch(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config, dt, dvdt, switchState)

    with record_function("[CompSPH] - 10 - Compute Divergence"):
        verbosePrint(verbose, '[CompSPH]\tComputing Divergence')
        particles.divergence = drhodt
    forcing = applyForcing(particles, config, SPHSystem.t, dt)
    forcing += computeGravity(particles, config)

    # print('\n')
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

def getPressureEnergyConfig(gamma, kernel, targetNeighbors, domain, verletScale):
    
    return {
        # 'gamma': gamma,
        'targetNeighbors': targetNeighbors,
        'domain': domain,
        'kernel': kernel,
        # 'supportIter': 4,
        # 'verletScale': 1.4,
        # 'supportScheme': 'Monaghan', # Could also use Owen
        'correctiveOmega': False, # Use Omega terms to correct for adaptive support, seems to not be used in the comp SPH paper
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
          'LUT': None#computeOwen(kernel, dim = domain.dim, nMin = 2.0, nMax = 10.0, nLUT = 1024),  
        },
        'fluid':{
            'gamma': gamma,
            'backgroundPressure': 0.0,
        },
        # 'owenSupport': computeOwen(kernel, dim = domain.dim, nMin = 2.0, nMax = 6.0, nLUT = 1024),

        'diffusion':{
            'C_l': 1,
            'C_q': 2,
            # 'Cu_l': 1,
            # 'Cu_q': 2,
            'monaghanSwitch': True,
            'viscosityTerm': 'Monaghan',
            'correctXi': True,
            
            'viscosityFormulation': 'Monaghan1992',
            'use_cbar': False,
            'use_rho_bar': False,
            'use_h_bar': False,
            'scaleBeta': False,
            'K': 1.0,
            
            'thermalConductivity' : 0.5,
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
        
        # 'C_l': 1, # Linear and quadratic viscosity terms
        # 'C_q': 2,
        # 'Cu_l: 1, # Linear and quadratic viscosity terms for the internal energy
        # 'Cu_q: 2, # However, PESPH does not use internal energy dissipation

        # 'use_cbar': True, # Use the average speed of sound
        # 'use_rho_bar': False, # Use the average density

        # 'viscositySwitch': 'hopkins',
        # 'monaghanSwitch': True, # Use the viscosity switch (required)
        # 'viscosityTerm': 'Monaghan', # Use the standard viscosity term
        # 'correctXi': True, # Correct the xi term in the viscosity
        # 'signalTerm': 'Monaghan1997', # Not required for this scheme
        # 'thermalConductivity' : 0.5, # No explicit thermal conductivity
        # 'K': 1.0, # Scaling factor of viscosity

        # 'viscosityFormulation': 'Monaghan1992',
        # Possible energySchemes = ['equalWork', 'PdV', 'diminishing', 'monotonic', 'hybrid', 'CRK']
        'energyScheme': EnergyScheme.CRK,
        'schemeName': 'PESPH'
    }