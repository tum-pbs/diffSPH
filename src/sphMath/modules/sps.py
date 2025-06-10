from sphMath.neighborhood import computeDistanceTensor, NeighborhoodInformation, DomainDescription, SparseCOO
from sphMath.operations import sph_op, scatter_sum
from sphMath.util import ParticleSet, ParticleSetWithQuantity
from typing import Union
from sphMath.kernels import SPHKernel
from sphMath.math import pinv2x2
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
import torch
import numpy as np
from sphMath.sphOperations.shared import getTerms, compute_xij
from sphMath.modules.viscosity import compute_Pi
from sphMath.schemes.baseScheme import verbosePrint
from sphMath.operations import sph_op
from torch.profiler import record_function
from sphMath.operations import sph_op, SPHOperation
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode, SPHOperationCompiled
from typing import Dict, Tuple
from sphMath.util import getSetConfig

def computeSPSTurbulence_(
        particles: WeaklyCompressibleState,
        dx: float,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather):
    
    velGrad = -SPHOperationCompiled(
        particles,
        quantity = particles.velocities,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation=Operation.Gradient,
        supportScheme = supportScheme,
        gradientMode = GradientMode.Difference)
    
    u_xx = velGrad[:,0,0]
    u_xy = velGrad[:,0,1] + velGrad[:,1,0]
    u_yy = velGrad[:,1,1]
    
    C_smagorinsky = 0.12
    C_blinn = 0.0066
    
    dim = particles.positions.shape[1]
    dx_sps = dx * np.sqrt(dim) / dim
    
    # Sps_smag = C_smagorinsky**2 * dx_sps**2
    # Sps_blinn = 2/3 * C_blinn * dx_sps**2
    Sps_smag = C_smagorinsky**2 * dx_sps**2
    Sps_blinn = 2/3 * C_blinn * dx_sps**2
    
    pow1 = u_xx**2 + u_yy**2
    prr = pow1 + pow1 + u_xy**2
    
    visc_sps = Sps_smag * torch.sqrt(prr)
    div_u = velGrad.diagonal(dim1 = -2, dim2 = -1).sum(dim = -1)
    sps_k = 2/3 * visc_sps * div_u
    sps_blinn = Sps_blinn * prr
    sumsps = - (sps_k + sps_blinn)
    
    onerho = 1.0 / particles.densities
    tau_xx = (2 * visc_sps * u_xx + sumsps)
    tau_xy = (visc_sps * u_xy)
    tau_yy = (2 * visc_sps * u_yy + sumsps)
    tau = torch.zeros_like(velGrad)
    tau[:,0,0] = tau_xx
    tau[:,0,1] = tau_xy
    tau[:,1,0] = tau_xy
    tau[:,1,1] = tau_yy
    
    # m = particles.masses
    # i,j = neighborhood[0].row, neighborhood[1].col
    # x_ij = neighborhood[1].x_ij
    # hij = particles.supports[j]
    
    return SPHOperationCompiled(
        particles,
        tau,
        neighborhood= neighborhood[0],
        kernelValues= neighborhood[1],
        operation= Operation.Divergence,
        gradientMode= GradientMode.Symmetric,
        supportScheme= supportScheme,
        divergenceMode=DivergenceMode.div
    )
    
    # gradWij = kernel.jacobian(xij, hij)
    
    # dudt_x = m[j] * ((tau_xx[i] + tau_xx[j]) * gradWij[:,0] + (tau_xy[i] + tau_xy[j]) * gradWij[:,1])
    # dudt_y = m[j] * ((tau_xy[i] + tau_xy[j]) * gradWij[:,0] + (tau_yy[i] + tau_yy[j]) * gradWij[:,1])
    # dudt = torch.stack([dudt_x, dudt_y], dim = -1)
    
    # return scatter_sum(dudt, i, dim = 0, dim_size = particles.positions.shape[0])


def computeSPSTurbulence(
        particles: WeaklyCompressibleState,
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather,
        config: Dict = {}):
    neighborhood = (neighborhood[0](neighborhood[0], kernel)) if neighborhood[1] is None else neighborhood
    dx = config['particle']['dx']
    return computeSPSTurbulence_(
        particles,
        dx,
        neighborhood,
        supportScheme)