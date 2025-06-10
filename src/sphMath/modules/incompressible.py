from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.neighborhood import filterNeighborhoodByKind, buildNeighborhood, filterNeighborhood, computeDistanceTensor, coo_to_csr, DomainDescription, SparseCOO
from sphMath.operations import sph_op
import torch
from typing import Union, Tuple, Dict
from torch.profiler import record_function
from sphMath.operations import sph_op, SPHOperation, SPHOperationCompiled, access_optional
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from sphMath.boundary import evalGhostQuantity
from sphMath.kernels import SPHKernel
from sphMath.neighborhood import evalKernelGradient, evalKernel
from sphMath.sphOperations.shared import scatter_sum


def computeAlpha(stateA, stateB, config, neighborhood, density = True):
    dt = config['timestep']['dt']**2 if density else config['timestep']['dt']

    fluidNeighbors = neighborhood# simulationState['fluid']['neighborhood']
    (i, j) = fluidNeighbors['indices']

    grad = fluidNeighbors['gradients']
    grad2 = torch.einsum('nd, nd -> n', grad, grad)

    term1 = stateB['actualArea'][j][:,None] * grad
    term2 = (stateB['actualArea']**2 / (stateB['areas'] * config['fluid']['rho0']))[j] * grad2

    kSum1 = scatter_sum(term1, i, dim=0, dim_size=stateA['areas'].shape[0])
    kSum2 = scatter_sum(term2, i, dim=0, dim_size=stateA['areas'].shape[0])

    fac = - dt * stateA['actualArea']
    mass = stateA['areas'] * config['fluid']['rho0']
    alpha = fac / mass * torch.einsum('nd, nd -> n', kSum1, kSum1) + fac * kSum2
    # alpha = torch.clamp(alpha, -1, -1e-7)

    return alpha


def computeAlpha(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    i, j        = neighborhood[0].row, neighborhood[0].col
    gradW       = evalKernelGradient(neighborhood[1], supportScheme, True)
    gradW2      = torch.einsum('nd, nd -> n', gradW, gradW)

    term1 = particles.apparentArea[j].view(-1,1) * gradW
    term2 = (particles.apparentArea**2 / (particles.masses))[j] * gradW2

    kSum1 = scatter_sum(term1, i, dim=0, dim_size=particles.apparentArea.shape[0])
    kSum2 = scatter_sum(term2, i, dim=0, dim_size=particles.apparentArea.shape[0])

    firstTerm  = - particles.apparentArea / particles.masses * torch.einsum('nd, nd -> n', kSum1, kSum1) 
    secondTerm = - particles.apparentArea * kSum2
    

    return firstTerm + secondTerm

def computeDivergence(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        predictedVelocities: torch.Tensor,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    
    return SPHOperation(
        particles,
        quantity = predictedVelocities,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation= Operation.Divergence,
        gradientMode= GradientMode.Difference,
        supportScheme = supportScheme
    )

def computePressureAccel(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        pressure: torch.Tensor,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    return - SPHOperation(
        particles,
        quantity = pressure,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation= Operation.Gradient,
        gradientMode= GradientMode.Summation,
        supportScheme = supportScheme) / particles.densities.view(-1,1)

def updatePressure(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        pressureA: torch.Tensor,
        pressureAccel: torch.Tensor,
        sourceTerm: torch.Tensor,
        alpha: torch.Tensor,
        dt: float,
        omega: float,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    kernelSum = -dt**2 * SPHOperation(
        particles,
        quantity = pressureAccel, # * particles.densities.view(-1,1),
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation= Operation.Divergence,
        gradientMode= GradientMode.Difference,
        supportScheme = supportScheme)

    residual = kernelSum - sourceTerm
    pressure = pressureA + omega * (sourceTerm - kernelSum) / alpha
    return pressure, residual


from sphMath.util import getSetConfig
def divergenceSolve(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        dt: float,
        dvdt: torch.Tensor,    
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {},  
):
    predictedVelocities = particles.velocities + dt * dvdt
    particles.apparentArea = particles.masses / particles.densities    
    divergence = computeDivergence(particles, kernel, predictedVelocities, neighborhood, supportScheme, config)

    # correctly this should be rho instead of rho0 but its supposed to be incompressible so we 'can' use rho0
    drhodt = - config['fluid']['rho0'] * divergence
    sourceTerm = -drhodt * dt

    alphas = dt**2 * computeAlpha(particles, kernel, neighborhood, supportScheme, config)
    
    pressureA = particles.pressures.clone() * 0.75
    pressureB = particles.pressures.clone() * 0.75

    errors = []
    pressures = []
    i = 0
    error = 0.
    minIters = getSetConfig(config, 'divergenceSolver', 'minIters', 2)
    maxIters = getSetConfig(config, 'divergenceSolver', 'maxIters', 32)
    errorThreshold = getSetConfig(config, 'divergenceSolver', 'errorThreshold', 1e-3* config['fluid']['rho0'])
    omega = getSetConfig(config, 'divergenceSolver', 'omega', 0.3)

    # print(f'minIters: {minIters} maxIters: {maxIters} errorThreshold: {errorThreshold} omega: {omega}')

    while i < maxIters and (i < minIters or error > errorThreshold):
        pressureAccel = computePressureAccel(particles, kernel, pressureB, neighborhood, supportScheme, config)
        pressureA = pressureB.clone()
        pressureB, residual = updatePressure(particles, kernel, pressureA, pressureAccel, sourceTerm, alphas, dt, omega, neighborhood, supportScheme, config)

        # need to remultiply with the timestep term here
        error = torch.mean(torch.clamp(residual / config['fluid']['rho0'], min = - errorThreshold))
        errors.append(error.detach().cpu().item())
        pressures.append(pressureB.mean().detach().cpu().item())
        i += 1

    pressureAccel = computePressureAccel(particles, kernel, pressureB, neighborhood, supportScheme, config)

    config['divergenceSolver']['convergence'] = (errors, pressures)

    return pressureAccel, errors, pressures, pressureA, pressureB


def densitySolve(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        dt: float,
        dvdt: torch.Tensor,    
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {},  
):
    predictedVelocities = particles.velocities + dt * dvdt
    particles.apparentArea = particles.masses / particles.densities    
    divergence = computeDivergence(particles, kernel, predictedVelocities, neighborhood, supportScheme, config)

    # correctly this should be rho instead of rho0 but its supposed to be incompressible so we 'can' use rho0
    drhodt = - config['fluid']['rho0'] * divergence
    rhoStar = particles.densities + dt * drhodt
    sourceTerm = (config['fluid']['rho0'] - rhoStar) #/ dt
    
    # print(f'Densities: {particles.densities.min()} {particles.densities.max()} {particles.densities.mean()}')
    # print(f'Source Term: {sourceTerm.min()} {sourceTerm.max()} {sourceTerm.mean()}')

    # note that we are precomputing the source term with the delta t term so it vanishes for alpha
    alphas = dt**2 * computeAlpha(particles, kernel, neighborhood, supportScheme, config)
    
    pressureA = particles.pressures.clone() * 0.
    pressureB = particles.pressures.clone() * 0.

    errors = []
    pressures = []
    i = 0
    error = 0.
    minIters = getSetConfig(config, 'densitySolver', 'minIters', 2)
    maxIters = getSetConfig(config, 'densitySolver', 'maxIters', 64)
    errorThreshold = getSetConfig(config, 'densitySolver', 'errorThreshold', 1e-3 * config['fluid']['rho0'])
    omega = getSetConfig(config, 'densitySolver', 'omega', 0.3)

    while i < maxIters and (i < minIters or error > errorThreshold):
        pressureAccel = computePressureAccel(particles, kernel, pressureB, neighborhood, supportScheme, config)
        pressureA = pressureB.clone()

        pressureB, residual = updatePressure(particles, kernel, pressureA, pressureAccel, sourceTerm, alphas, dt, omega, neighborhood, supportScheme, config)

        # pressureA = pressureA.clamp(min = 0)

        # need to remultiply with the timestep term here
        error = torch.mean(torch.clamp(residual, min = - errorThreshold))# * dt#**2
        errors.append(error.detach().cpu().item())
        pressures.append(pressureB.mean().detach().cpu().item())
        i += 1

    pressureAccel = computePressureAccel(particles, kernel, pressureB, neighborhood, supportScheme, config)

    config['densitySolver']['convergence'] = (errors, pressures)

    return pressureAccel, errors, pressures, pressureA, pressureB