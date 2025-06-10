import torch
from sphMath.sampling import buildDomainDescription, sampleRegularParticles
import numpy as np
from sphMath.util import ParticleSet
from sphMath.modules.adaptiveSmoothing import computeOmega, evaluateOptimalSupport
from sphMath.util import volumeToSupport
from sphMath.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr
from sphMath.modules.eos import idealGasEOS
from sphMath.schemes.gasDynamics import CompressibleState, CompressibleSystem
from sphMath.modules.density import computeDensity
import torch
from sphMath.sampling import buildDomainDescription, sampleRegularParticles
import numpy as np
from sphMath.util import ParticleSet
from sphMath.modules.adaptiveSmoothing import computeOmega, evaluateOptimalSupport
from sphMath.util import volumeToSupport
from sphMath.neighborhood import PointCloud, DomainDescription, buildNeighborhood, filterNeighborhood, coo_to_csrsc, coo_to_csr
from sphMath.modules.eos import idealGasEOS
from sphMath.schemes.gasDynamics import CompressibleState, CompressibleSystem
from sphMath.modules.density import computeDensity
from sphMath.enums import *
from sphMath.neighborhood import evaluateNeighborhood, filterNeighborhood,filterNeighborhoodByKind, SupportScheme
from sphMath.schema import getSimulationScheme

def buildLinearWaveSimulation(
        nx = 200,
        l = 1,

        A = 1e-3,
        lamda = 1,
        rho0 = 1,

        c_s = 1,
        gamma = 5/3,

        targetNeighbors = 9,
        device = torch.device('cpu'),
        dtype = torch.float64,
        kernel = KernelType.Wendland4,
        verbose = False,
        simulationScheme : SimulationScheme = SimulationScheme.CompSPH,
        integrationScheme: IntegrationSchemeType = IntegrationSchemeType.rungeKutta2,
        viscositySwitch: ViscositySwitch = ViscositySwitch.NoneSwitch,
        supportScheme: AdaptiveSupportScheme = AdaptiveSupportScheme.MonaghanScheme
):
    dim = 1
    domain = buildDomainDescription(l, dim, periodic = True, device = device, dtype = dtype)
    scheme, SimulationSystem, config, integrator = getSimulationScheme(simulationScheme, kernel, integrationScheme, gamma, targetNeighbors, domain, viscositySwitch, supportScheme, verletScale=1.4)
    print('nx', nx)
    particles = sampleRegularParticles(nx, domain, targetNeighbors, jitter = 0.0)
    print(particles.positions.shape[0])
    wrappedKernel = kernel

    if verbose:
        print(f'Sampled Domain: {domain.min} - {domain.max} [{domain.min.dtype}, {domain.max.dtype}]')
        print(f'Sampled Particles: {particles.positions.shape[0]} [{particles.positions.dtype}]')
        print(f'Sampled Supports: {particles.supports[0]} [{particles.supports.dtype}]')
        print(f'Sampled Masses: {particles.masses[0]} [{particles.masses.dtype}]')
        print(f'Sampled Densities: {particles.densities[0]} [{particles.densities.dtype}]')


    delta_i = A * torch.sin(2 * np.pi * particles.positions[:, 0] / lamda)
    P0 = c_s**2 * rho0 / gamma

    pressures = P0 + delta_i
    supports = volumeToSupport(l / nx, dim = 1, targetNeighbors = targetNeighbors)

    if verbose:
        print(f'Initial State: Number of particles: {particles.positions.shape[0]}, target neighbors: {targetNeighbors}')
        # print(f'Initial Velocity: {v_.min():8.5g} - {v_.max():8.5g} [shape: {v_.shape}, dtype: {v_.dtype}]')
        print(f'Initial Pressure: {pressures.min():8.5g} - {pressures.max():8.5g} [P0: {P0}, shape: {pressures.shape}, dtype: {pressures.dtype}]')
        print(f'Initial Support: {supports}')

    particles = CompressibleState(
        positions = particles.positions,
        supports = torch.ones_like(particles.positions[:, 0]) * supports,
        masses = particles.masses + delta_i,
        densities = torch.ones_like(particles.masses) + delta_i,
        velocities = torch.zeros_like(particles.positions),

        kinds = torch.zeros_like(particles.masses),
        materials=torch.zeros_like(particles.masses),
        UIDs = torch.arange(particles.masses.shape[0], device = particles.masses.device),

        internalEnergies=None,
        totalEnergies=None,
        entropies=None,
        pressures=None,
        soundspeeds=None,
        divergence=torch.zeros_like(densities),
        alpha0s= torch.ones_like(densities),
        alphas= torch.ones_like(densities),
    )
    m0 = particles.masses.clone()
    h = particles.supports.clone()
    neighborhood = None
    for i in range(16):    
        neighborhood, neighbors = evaluateNeighborhood(particles, domain, wrappedKernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=neighborhood)
        numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries
        rho = computeDensity(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather)
        
        samplingError = torch.mean(rho - rho0)
        error = rho - (rho0 + delta_i)



        masses = particles.masses*( 1 - error)
        print(f'\tIteration {i} - Sampling Error: {(error**2).mean():.2e}')

        particles = CompressibleState(
            positions = particles.positions,
            supports = h,
            masses = masses,
            densities = torch.ones_like(particles.masses) + delta_i,
            velocities = torch.zeros_like(particles.positions),

            kinds = torch.zeros_like(particles.masses),
            materials=torch.zeros_like(particles.masses),
            UIDs = torch.arange(particles.masses.shape[0], device = particles.masses.device),

            internalEnergies=None,
            totalEnergies=None,
            entropies=None,
            pressures=None,
            soundspeeds=None,
        divergence=torch.zeros_like(densities),
        alpha0s= torch.ones_like(densities),
        alphas= torch.ones_like(densities),
        )
        # masses = particles.masses*( 1 +  delta_i)
# evaluateOptimalSupport(
#         particles: Union[CompressibleState, WeaklyCompressibleState],
#         kernel_: SPHKernel,
#         neighborhood: NeighborhoodInformation = None,
#         supportScheme: SupportScheme = SupportScheme.Scatter,
#         config: Dict = {}
#         ):

        rho, h, *_ = evaluateOptimalSupport(particles, wrappedKernel, neighborhood, supportScheme=SupportScheme.Gather, config = config)
        # particles = particles._replace(supports = h)
        

    A_, u_, P_, c_ = idealGasEOS(A = None, u = None, P = pressures, rho = rho, gamma = gamma)

    # c_[:] = c_s
    v_ = (c_s * delta_i).view(-1,1)
    u_[:] = 1/(gamma - 1) - 1 / (gamma * (delta_i + 1))


    if verbose:
        adjacency = filterNeighborhood(sparseNeighborhood)
        csr = coo_to_csr(adjacency)
        nneighs = csr.rowEntries
        print(f'Initial State: Number of particles: {particles.positions.shape[0]}, target neighbors: {targetNeighbors} [Actual: {nneighs.min()} | {nneighs.median()} | {nneighs.max()}]')
        print(f'Initial Density: {rho.min() - 1:8.5g} - {rho.max() - 1:8.5g} [shape: {rho.shape}, dtype: {rho.dtype}]')
        print(f'Initial Velocity: {v_.min():8.5g} - {v_.max():8.5g} [shape: {v_.shape}, dtype: {v_.dtype}]')
        print(f'Initial Pressure: {pressures.min():8.5g} - {pressures.max():8.5g} [P0: {P0}, shape: {pressures.shape}, dtype: {pressures.dtype}]')
        print(f'Initial Support: {supports}')

        print(f'Initial Entropy: {A_.min():8.5g} - {A_.max():8.5g} [shape: {A_.shape}, dtype: {A_.dtype}]')
        print(f'Initial Internal Energy: {u_.min():8.5g} - {u_.max():8.5g} [shape: {u_.shape}, dtype: {u_.dtype}]')
        print(f'Initial Sound Speed: {c_.min():8.5g} - {c_.max():8.5g} [shape: {c_.shape}, dtype: {c_.dtype}]')
        print(f'Sampling Error: {samplingError:.2e}')



    internalEnergy = u_
    kineticEnergy = torch.linalg.norm(v_, dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) 

    psphState = CompressibleState(
        positions=particles.positions,
        supports=h,
        masses=particles.masses,
        densities=rho,
        velocities=v_,
        # soundspeeds=simulationState.soundspeeds,
        
        kinds = torch.zeros_like(particles.masses),
        materials=torch.zeros_like(particles.masses),
        UIDs = torch.arange(particles.masses.shape[0], device = particles.masses.device),

        UIDcounter=particles.positions.shape[0],
        internalEnergies=u_,
        totalEnergies=totalEnergy,
        entropies = A_,
        pressures=P_,
        soundspeeds=c_,
        
        alphas = torch.ones_like(rho),
        alpha0s = torch.ones_like(rho),
        divergence=torch.zeros_like(rho),
    )
    neighborhood, sparseNeighborhood = buildNeighborhood(psphState, psphState, domain, 1.4, 'superSymmetric')
    actualNeighbors = filterNeighborhood(sparseNeighborhood)

    simulationSystem = SimulationSystem(
        systemState = psphState,
        domain = domain,
        neighborhoodInfo = neighborhood,
        t = 0
    )

    if verbose:
        print(f'Positions:        min: {psphState.positions.min():8.3g}, max: {psphState.positions.max():8.3g} has nan: {torch.isnan(psphState.positions).any()} has inf: {torch.isinf(psphState.positions).any()} dtype: {psphState.positions.dtype}, device: {psphState.positions.device}')
        print(f'Supports:         min: {psphState.supports.min():8.3g}, max: {psphState.supports.max():8.3g} has nan: {torch.isnan(psphState.supports).any()} has inf: {torch.isinf(psphState.supports).any()} dtype: {psphState.supports.dtype}, device: {psphState.supports.device}')
        print(f'Masses:           min: {psphState.masses.min():8.3g}, max: {psphState.masses.max():8.3g} has nan: {torch.isnan(psphState.masses).any()} has inf: {torch.isinf(psphState.masses).any()} dtype: {psphState.masses.dtype}, device: {psphState.masses.device}')
        print(f'Densities:        min: {psphState.densities.min():8.3g}, max: {psphState.densities.max():8.3g} has nan: {torch.isnan(psphState.densities).any()} has inf: {torch.isinf(psphState.densities).any()} dtype: {psphState.densities.dtype}, device: {psphState.densities.device}')
        print(f'Velocities:       min: {psphState.velocities.min():8.3g}, max: {psphState.velocities.max():8.3g} has nan: {torch.isnan(psphState.velocities).any()} has inf: {torch.isinf(psphState.velocities).any()} dtype: {psphState.velocities.dtype}, device: {psphState.velocities.device}')
        print(f'Internal Energies: min: {psphState.internalEnergies.min():8.3g}, max: {psphState.internalEnergies.max():8.3g} has nan: {torch.isnan(psphState.internalEnergies).any()} has inf: {torch.isinf(psphState.internalEnergies).any()} dtype: {psphState.internalEnergies.dtype}, device: {psphState.internalEnergies.device}')
        print(f'Total Energies:   min: {psphState.totalEnergies.min():8.3g}, max: {psphState.totalEnergies.max():8.3g} has nan: {torch.isnan(psphState.totalEnergies).any()} has inf: {torch.isinf(psphState.totalEnergies).any()} dtype: {psphState.totalEnergies.dtype}, device: {psphState.totalEnergies.device}')
        print(f'Entropies:        min: {psphState.entropies.min():8.3g}, max: {psphState.entropies.max():8.3g} has nan: {torch.isnan(psphState.entropies).any()} has inf: {torch.isinf(psphState.entropies).any()} dtype: {psphState.entropies.dtype}, device: {psphState.entropies.device}')
        print(f'Pressures:        min: {psphState.pressures.min():8.3g}, max: {psphState.pressures.max():8.3g} has nan: {torch.isnan(psphState.pressures).any()} has inf: {torch.isinf(psphState.pressures).any()} dtype: {psphState.pressures.dtype}, device: {psphState.pressures.device}')
        print(f'Soundspeeds:      min: {psphState.soundspeeds.min():8.3g}, max: {psphState.soundspeeds.max():8.3g} has nan: {torch.isnan(psphState.soundspeeds).any()} has inf: {torch.isinf(psphState.soundspeeds).any()} dtype: {psphState.soundspeeds.dtype}, device: {psphState.soundspeeds.device}')
        print(f'Alphas:           min: {psphState.alphas.min():8.3g}, max: {psphState.alphas.max():8.3g} has nan: {torch.isnan(psphState.alphas).any()} has inf: {torch.isinf(psphState.alphas).any()} dtype: {psphState.alphas.dtype}, device: {psphState.alphas.device}')
        print(f'Alpha0s:          min: {psphState.alpha0s.min():8.3g}, max: {psphState.alpha0s.max():8.3g} has nan: {torch.isnan(psphState.alpha0s).any()} has inf: {torch.isinf(psphState.alpha0s).any()} dtype: {psphState.alpha0s.dtype}, device: {psphState.alpha0s.device}')

    return scheme, simulationSystem, domain, P0, config, integrator

import copy
from tqdm.autonotebook import tqdm
from sphMath.kernels import Kernel_xi

def runLinearWaveTest(
        nx = 200,
        l = 1,

        A = 1e-3,
        lamda = 1,
        rho0 = 1,

        c_s = 1,
        gamma = 5/3,

        targetNeighbors = 9,
        device = torch.device('cpu'),
        dtype = torch.float64,
        kernel = 'CubicSpline',
        verbose = False,
        periods = 5.0,
        CFLNumber = 0.1,
        simulationScheme : SimulationScheme = SimulationScheme.CompSPH,
        integrationScheme: IntegrationSchemeType = IntegrationSchemeType.rungeKutta2,
        viscositySwitch: ViscositySwitch = ViscositySwitch.NoneSwitch,
        supportScheme: AdaptiveSupportScheme = AdaptiveSupportScheme.MonaghanScheme):
    wrappedKernel = kernel
    scheme, simulationSystem, domain, P0, config, integrator = buildLinearWaveSimulation(
    nx = nx, l = l, A = A, lamda = lamda, rho0 = rho0, c_s = c_s, gamma = gamma, device = device, dtype = dtype, kernel = kernel, targetNeighbors=targetNeighbors, verbose=verbose, simulationScheme=simulationScheme, integrationScheme=integrationScheme, viscositySwitch=viscositySwitch, supportScheme=supportScheme)

    particles = simulationSystem.systemState

    neighborhood, neighbors = evaluateNeighborhood(particles, config['domain'], wrappedKernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
    particles.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries

    simulationSystem.systemState.densities = computeDensity(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather)

    A_, u_, P_, c_s = idealGasEOS(A = None, u = simulationSystem.systemState.internalEnergies, P = None, rho = simulationSystem.systemState.densities, gamma = gamma)

    timeLimit = lamda * periods#* c_s.mean() * 5

    dt = (0.4 * simulationSystem.systemState.supports / simulationSystem.systemState.soundspeeds / Kernel_xi(wrappedKernel, simulationSystem.systemState.positions.shape[1])).min().item() / 4
    dx = l / nx
    dt = CFLNumber * dx #/ c_s.max()
    timesteps = int(timeLimit / dt) + 1

    dt = timeLimit / timesteps

    if verbose:
        print(f'Time Limit: {timeLimit}, dt: {dt}, Timesteps: {timesteps}')
        print(f'dx: {dx}, c_s: {c_s.mean()}')
        print(f'Total Time: {timesteps * dt}')

    copiedSystem = copy.deepcopy(simulationSystem)
    systemStates = []
    systemStates.append(copy.deepcopy(simulationSystem))
    psphState = copiedSystem.systemState
    totalEnergy = (psphState.internalEnergies * psphState.masses).sum() + 0.5 * (torch.linalg.norm(psphState.velocities, dim = -1) **2 * psphState.masses).sum()
    energies = []
    energies.append({
        'Kinetic Energy': 0.5 * (torch.linalg.norm(psphState.velocities, dim = -1) **2 * psphState.masses).sum(),
        'Thermal Energy': (psphState.internalEnergies * psphState.masses).sum(),
        'Total Energy': totalEnergy
    })

    L1s = []
    L2s = []
    Linfs = []
    rhos = []

    for i in (tq:= tqdm(range(timesteps), leave=False)):
        copiedSystem, currentState, updates  = integrator.function(copiedSystem, dt, scheme, config, verbose = False, priorStep = copiedSystem.priorStep)
        copiedSystem.priorStep = [updates[-1], currentState[-1]]
        
        # psphState = psphStates[-1][0]
        systemStates.append(copy.deepcopy(copiedSystem.to(device = 'cpu', dtype = dtype)))


        kineticEnergy = 0.5 * (torch.linalg.norm(copiedSystem.systemState.velocities, dim = -1) **2 * copiedSystem.systemState.masses).sum()
        thermalEnergy = (copiedSystem.systemState.internalEnergies * copiedSystem.systemState.masses).sum()
        totalEnergy = kineticEnergy + thermalEnergy

        # tq.set_postfix({
        #     'time': copiedSystem.t,
        #     # 'Kinetic Energy': kineticEnergy.item(),
        #     # 'Thermal Energy': thermalEnergy.item(),
        #     # 'Total Energy': totalEnergy.item(),
        #     'dE': totalEnergy - energies[0]['Total Energy'],
        #     'rho': (psphState.densities.max().item(), psphState.densities.min().item())
        # })
        energies.append({
            'Kinetic Energy': kineticEnergy,
            'Thermal Energy': thermalEnergy,
            'Total Energy': totalEnergy
        })
        time = copiedSystem.t
        currentPositions = systemStates[-1].systemState.positions
        reference = (rho0 +  A * torch.sin(2 * np.pi * currentPositions[:,0] / l - time * c_s / lamda * 2 * np.pi)).cpu().numpy()
        
        particles = systemStates[-1].systemState
        neighborhood, neighbors = evaluateNeighborhood(particles, config['domain'], wrappedKernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
        particles.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries

        systemStates[-1].systemState.densities = computeDensity(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather)

        rhos.append(systemStates[-1].systemState.densities.cpu().numpy())
        
        current = systemStates[-1].systemState.densities
        
        L1 = (current - reference).abs().mean().item()
        L2 = ((current - reference)**2).mean().sqrt().item()
        Linf = (current - reference).abs().max().item()
        tq.set_postfix({
            'time': copiedSystem.t,
            'L1': L1,
            'L2': L2,
            'Linf': Linf
        })
        L1s.append(L1)
        L2s.append(L2)
        Linfs.append(Linf)
        # break

    i0 = 1
    i1 = -1
    rho0 = systemStates[i0].systemState.densities.cpu().numpy().mean()
    rho_reference = (rho0 +  A * torch.sin(2 * np.pi * systemStates[i1].systemState.positions[:,0] / l)).cpu().numpy()
    rho0 = systemStates[i0].systemState.densities.cpu().numpy()
    rho1 = systemStates[i1].systemState.densities.cpu().numpy()

    tdiff = systemStates[i1].t - systemStates[i0].t
    # print(f'Time Difference: {tdiff}, c_s: {systemStates[i0].systemState.soundspeeds.mean()} | {systemStates[i1].systemState.soundspeeds.mean()}')
    c_s = systemStates[i0].systemState.soundspeeds.mean()
    # print(f'Wave Lengths: {tdiff / c_s}')


    AE = np.abs((rho1 - rho_reference)).mean()

    rho0 = systemStates[1].systemState.densities.cpu().numpy()
    rho1 = systemStates[-1].systemState.densities.cpu().numpy()
    x0 = systemStates[1].systemState.positions.cpu().numpy()
    x1 = systemStates[-1].systemState.positions.cpu().numpy()

    rho_0 = systemStates[1].systemState.densities.cpu().numpy().mean()
    rho_reference = (rho_0 +  A * torch.sin(2 * np.pi * systemStates[-1].systemState.positions[:,0] / l)).cpu().numpy()

    MAE = np.abs(rho0 - rho_reference).mean()

    return L1, (x0, rho0), (x1, rho1), systemStates, energies, L1s, L2s, Linfs


# Values from CRKSPH paper
refDataQ0 = '''20,11538.4346
31,5286.4465
50,2045.1938
80,822.7241
126,326.6813
200,129.7162
316,54.2579
500,20.991
800,8.6667
1250,3.5783
2000,1.8916'''
refDataCullen = '''20,33968.2714
31,21265.8926
50,10953.3767
80,5496.8387
126,2329.3333
200,949.2949
316,367.2584
500,145.8283
800,57.9044
1250,27.5853
2000,16.3942'''

refDataCompSPH = '''20,40753.9297
31,31417.7545
50,22695.1054
80,15562.8953
126,10672.0681
200,7130.2925
316,4702.3655
500,3021.5236
800,1843.0483
1250,1280.397
2000,770.9135'''

def parseRefData(refData):
    lines = refData.split('\n')
    refX = []
    refY = []
    for line in lines:
        x, y = line.split(',')
        refX.append(float(x))
        refY.append(float(y)*1e-11)
    refX = np.array(refX)
    refY = np.array(refY)
    # for x,y in zip(refX, refY):
        # print(f'Reference Data: nx = {x}, L1 = {y}')
    return refX, refY

compSPHData = parseRefData(refDataCompSPH)
compSPHQ0Data = parseRefData(refDataQ0)
compSPHCullenData = parseRefData(refDataCullen)