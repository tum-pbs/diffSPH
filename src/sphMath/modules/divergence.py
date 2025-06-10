from sphMath.util import ParticleSet, ParticleSetWithQuantity
from sphMath.neighborhood import DomainDescription, SparseCOO
from typing import Union, Tuple
from sphMath.kernels import SPHKernel
import torch
from sphMath.operations import sph_op


def computeDivergence(
        particles_a: Union[ParticleSet, ParticleSetWithQuantity],
        particles_b: Union[ParticleSet, ParticleSetWithQuantity],
        domain: DomainDescription,
        kernel: SPHKernel,
        neighborhood: SparseCOO,
        supportScheme: str = 'scatter'):
    return sph_op(particles_a, particles_b, domain, kernel, neighborhood, supportScheme, operation = 'divergence', gradientMode = 'difference')