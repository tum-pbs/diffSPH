from sphMath.modules.particleShifting import solveShifting
import torch
from typing import Union
from sphMath.util import ParticleSet, ParticleSetWithQuantity
from sphMath.neighborhood import DomainDescription, SparseCOO
from sphMath.kernels import SPHKernel, Kernel_Scale
from sphMath.operations import sph_op

from torch.profiler import record_function

# See Sun et al: The delta SPH-model: Simple procedures for a further improvement of the SPH scheme
def computeTimestepWCSPH(dt, state, systemUpdate, config, verbose = False):
    if not config.get('timestep', {}).get('active', True):
        print('[SPH] - Adaptive Timestep Update: Disabled')
        return config.get('timestep', {}).get('dt', 1e-4)
    
    timestepConfig = config.get('timestep', {})
    timestepCFL = timestepConfig.get('CFL', 0.3)
    maxDt = timestepConfig.get('maxDt', 1e-3)
    minDt = timestepConfig.get('minDt', 1e-6)
    
    diffusionAlpha = config.get('diffusion', {}).get('alpha', 0.01)
    diffusionScheme = config.get('diffusion', {}).get('velocityScheme', 'deltaSPH_inviscid')
    fluidCs = config.get('fluid', {}).get('c_s', 10)
    particleSupport = config.get('particle', {}).get('support', 1)
    kernelScale = Kernel_Scale(config['kernel'], state.positions.shape[1])

    dtype = config['domain'].min.dtype
    device = config['domain'].min.device

    if verbose:
        print(f'[SPH] - Adaptive Timestep Update')
        print(f'\tCFL: {timestepCFL}, maxDt: {maxDt}, minDt: {minDt}')
        print(f'\tDiffusion: {diffusionAlpha}, {diffusionScheme}')
        print(f'\tFluid: {fluidCs}, {particleSupport}, {kernelScale}')


    with record_function("[SPH] - Adaptive Timestep Update"):
        dim = state.positions.shape[1]

        nu = diffusionAlpha * fluidCs * particleSupport / (2 * (dim +2))
        nu = config.get('diffusion', {}).get('nu', nu) if diffusionScheme == 'deltaSPH_viscid' else nu
        dt_v = 0.125 * config.get('particle', {}).get('support', 1)**2 / nu / kernelScale
        dt_v = torch.tensor(dt_v, dtype = dtype, device = device) if not isinstance(dt_v, torch.Tensor) else dt_v

        dt_c = timestepCFL * particleSupport / fluidCs / kernelScale
        dt_c = torch.tensor(dt_c, dtype = dtype, device = device) if not isinstance(dt_c, torch.Tensor) else dt_c

        # acceleration timestep condition
        if systemUpdate is not None and hasattr(systemUpdate, 'velocities'):
            dudt = systemUpdate.velocities
            max_accel = torch.max(torch.linalg.norm(dudt[~torch.isnan(dudt)], dim = -1))
            dt_a = 0.25 * torch.sqrt(particleSupport / (max_accel + 1e-7)) / kernelScale
        else:
            dt_a = torch.tensor(maxDt, dtype = dtype, device = device)
        dt_a = torch.tensor(dt_a, dtype = dtype, device = device) if not isinstance(dt_a, torch.Tensor) else dt_a
        if verbose:
            print(f'\tViscosity: {dt_v}, Acoustic: {dt_c}, Acceleration: {dt_a}')

        # dt = config['timestep']['dt']
        dt = torch.tensor(dt, dtype = dtype, device = device) if not isinstance(dt, torch.Tensor) else dt
        new_dt = dt
        if config.get('timestep', {}).get('viscosityConstraint', True):
            new_dt = dt_v
        if config.get('timestep', {}).get('accelerationConstraint', True):
            new_dt = torch.min(new_dt, dt_a)
        if config.get('timestep', {}).get('acousticConstraint', True):
            new_dt = torch.min(new_dt, dt_c)
        new_dt = torch.min(new_dt, torch.tensor(maxDt, dtype = dtype, device = device))
        new_dt = torch.max(new_dt, torch.tensor(minDt, dtype = dtype, device = device))
        return new_dt


from sphMath.kernels import Kernel_xi
from sphMath.modules.eos import idealGasEOS
def computeDTCSPH(dt, particles, update, solverConfig):
    if not solverConfig.get('timestep', {}).get('active', True):
        return solverConfig.get('timestep', {}).get('dt', 1e-4)
    
    timestepConfig = solverConfig.get('timestep', {})
    timestepCFL = timestepConfig.get('CFL', 0.3)
    maxDt = timestepConfig.get('maxDt', 1e-3)
    minDt = timestepConfig.get('minDt', 1e-6)
    
    c_s = idealGasEOS(A=None, P = None, u = particles.internalEnergies, rho = particles.densities, gamma = solverConfig['fluid']['gamma'])[-1]
    c_s_max = c_s.max()
    h_min = particles.supports.min()
    xi = Kernel_xi(solverConfig['kernel'], particles.positions.shape[1])
    
    # xi = solverConfig['kernel'].xi(particles.positions.shape[1])
    # CFL = 0.3
    dt_cfl = timestepCFL * h_min / (c_s_max * xi)
    # print(dt_cfl)
    dt = 1e-5

    return dt_cfl

from sphMath.enums import *
from typing import Dict
def computeTimestep(SimulationScheme :SimulationScheme, currentDt, particles,  config: Dict, systemUpdate= None):    
    if SimulationScheme in [SimulationScheme.CompSPH, SimulationScheme.PESPH, SimulationScheme.CRKSPH, SimulationScheme.Price2007, SimulationScheme.Monaghan1997, SimulationScheme.Monaghan1992, SimulationScheme.MonaghanGingold1983]:
        return computeDTCSPH(currentDt, particles, systemUpdate, config)
    elif SimulationScheme in [SimulationScheme.DeltaSPH]:
        return computeTimestepWCSPH(currentDt, particles, systemUpdate, config)
    else:
        raise ValueError(f"Unknown simulation scheme: {SimulationScheme}")