from diffSPH.util import ParticleSet, ParticleSetWithQuantity
from diffSPH.neighborhood import DomainDescription, SparseCOO
from typing import Union, Tuple
from diffSPH.kernels import SPHKernel
import torch
from diffSPH.operations import sph_op, SPHOperation
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState
from diffSPH.schemes.gasDynamics import CompressibleState
from diffSPH.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from diffSPH.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict

def computeMomentum(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather,
        config: Dict = {}):
    densities = particles.densities if isinstance(particles, WeaklyCompressibleState) else particles.densities
    omega = particles.omega if hasattr(particles, 'omega') else torch.ones_like(densities)
    if omega is None:
        omega = torch.ones_like(densities)
    
    return - densities / omega * SPHOperation(
        particles,
        quantity = particles.velocities,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation= Operation.Divergence,
        gradientMode = GradientMode.Difference,
        supportScheme = supportScheme,
        consistentDivergence=False
    )

def computeMomentumConsistent(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Gather,
        config: Dict = {}):
    densities = particles.densities if isinstance(particles, WeaklyCompressibleState) else particles.densities
    omega = particles.omega if hasattr(particles, 'omega') else torch.ones_like(densities)
    if omega is None:
        omega = torch.ones_like(densities)

    return - densities / omega * SPHOperation(
        particles,
        quantity = particles.velocities,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation= Operation.Divergence,
        gradientMode = GradientMode.Difference,
        supportScheme = supportScheme,
        consistentDivergence=True
    )
