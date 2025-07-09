from sphMath.util import ParticleSet, ParticleSetWithQuantity
from sphMath.neighborhood import DomainDescription, SparseCOO
from typing import Union, Tuple
from sphMath.kernels import SPHKernel
import torch
from sphMath.operations import sph_op, SPHOperation
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict

def computeDensity(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    return SPHOperation(
        particles,
        quantity = None,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation=Operation.Density,
        supportScheme = supportScheme,
    )
from sphMath.enums import KernelCorrectionScheme

def computeGradRhoL(
    particles: WeaklyCompressibleState,
    kernel: SPHKernel,
    neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
    supportScheme: SupportScheme = SupportScheme.Scatter,
    config: Dict = {}):
    return SPHOperation(
        particles,
        quantity = particles.densities,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation=Operation.Gradient,
        supportScheme = supportScheme,
        gradientMode = GradientMode.Difference,
        correctionTerms=[KernelCorrectionScheme.gradientRenorm]
        )
def computeGradRho(
    particles: Union[CompressibleState, WeaklyCompressibleState],
    kernel: SPHKernel,
    neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
    supportScheme: SupportScheme = SupportScheme.Scatter,
    config: Dict = {}):
    return SPHOperation(
        particles,
        quantity = particles.densities,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation=Operation.Gradient,
        supportScheme = supportScheme,
        gradientMode = GradientMode.Naive)


def computeDensityGradient(
        particles_a: Union[ParticleSet, ParticleSetWithQuantity],
        particles_b: Union[ParticleSet, ParticleSetWithQuantity],
        domain: DomainDescription,
        kernel: SPHKernel,
        neighborhood: SparseCOO,
        supportScheme: str = 'scatter'):
    return sph_op(particles_a, particles_b, domain, kernel, neighborhood, supportScheme, 'gradient', gradientMode = 'difference')


def computeRenormalizedDensityGradient(
        particles_a: Union[ParticleSet, ParticleSetWithQuantity],
        particles_b: Union[ParticleSet, ParticleSetWithQuantity],
        L: torch.Tensor,
        domain: DomainDescription,
        kernel: SPHKernel,
        neighborhood: SparseCOO,
        supportScheme: str = 'scatter'):
    return sph_op(particles_a, particles_b, domain, kernel, neighborhood, supportScheme, 'gradient', gradientMode = 'difference',
        correctionTerms=[KernelCorrectionScheme.gradientRenorm]
        )



def computeDensityDeltaTerm(
        particles_a: Union[ParticleSet, ParticleSetWithQuantity],
        particles_b: Union[ParticleSet, ParticleSetWithQuantity],
        gradRho: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        gradRho_L: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        domain: DomainDescription,
        kernel: SPHKernel,
        neighborhood: SparseCOO,
        supportScheme: str = 'scatter',
        diffusionScheme = 'deltaSPH',
        diffusionDelta = 0.1,
        c_s = 10.0):
    
    gradRhoTerm = torch.zeros_like(particles_a.densities)
    densityTerm = torch.zeros_like(particles_a.densities)

    # if diffusionScheme == 'denormalizedOnly':
    gradRhoTerm = sph_op(particles_a, particles_b, domain, kernel, neighborhood, supportScheme, 'divergence', gradientMode = 'summation', quantity=gradRho)
    # elif diffusionScheme == 'deltaOnly':
    #     gradRhoTerm = sph_op(particles_a, particles_b, domain, kernel, neighborhood, supportScheme, 'divergence', gradientMode = 'summation', quantity=gradRho_L)
    # elif diffusionScheme == 'densityOnly':
    #     densityTerm = sph_op(particles_a, particles_b, domain, kernel, neighborhood, supportScheme, operation = 'laplacian', laplaceMode = 'other', quantity=particles_a.densities)
    # elif diffusionScheme == 'denormalized':
    #     gradRhoTerm = sph_op(particles_a, particles_b, domain, kernel, neighborhood, supportScheme, 'divergence', gradientMode = 'summation', quantity=gradRho)
    #     densityTerm = sph_op(particles_a, particles_b, domain, kernel, neighborhood, supportScheme, operation = 'laplacian', laplaceMode = 'other', quantity=particles_a.densities)
    # elif diffusionScheme == 'deltaSPH':
    #     gradRhoTerm = sph_op(particles_a, particles_b, domain, kernel, neighborhood, supportScheme, 'divergence', gradientMode = 'summation', quantity=gradRho_L)
    #     densityTerm = sph_op(particles_a, particles_b, domain, kernel, neighborhood, supportScheme, operation = 'laplacian', laplaceMode = 'other', quantity=particles_a.densities)
    # else:
    #     raise NotImplementedError(f'Diffusion scheme {diffusionScheme} not implemented')
    
    scalingFactor = diffusionDelta * particles_a.supports / kernel.kernelScale(domain.dim) * c_s

    return scalingFactor * (gradRhoTerm + densityTerm)
    