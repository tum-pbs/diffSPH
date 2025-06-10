from sphMath.neighborhood import computeDistanceTensor, NeighborhoodInformation, DomainDescription, SparseCOO
from sphMath.operations import sph_op
from sphMath.util import ParticleSet, ParticleSetWithQuantity
from typing import Union
from sphMath.kernels import SPHKernel
from sphMath.math import pinv2x2
import torch
from sphMath.operations import sph_op, SPHOperation
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple

from sphMath.schemes.states.common import BasicState
def computeCovarianceMatrix(
        particles: Union[BasicState, CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    neighborhood = (neighborhood[0](neighborhood[0], kernel)) if neighborhood[1] is None else neighborhood
    
    x_ij = neighborhood[1].x_ij
    C = SPHOperation(
        particles,
        quantity = -x_ij,
        kernel = kernel,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation=Operation.Gradient,
        supportScheme = supportScheme,
        gradientMode = GradientMode.Naive,
    )

    num_nbrs = particles.numNeighbors
    C[num_nbrs < 4,:,:] = torch.eye(particles.positions.shape[1], dtype = particles.positions.dtype, device = particles.positions.device)[None,:,:]
    
    return C

def computeKernelNormalization(
        particles: Union[BasicState, CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    C = computeCovarianceMatrix(particles, kernel, neighborhood, supportScheme, config)

    if particles.positions.shape[1] == 2:
        L, eigVals = pinv2x2(C)
    else:
        L = torch.linalg.pinv(C)
        eigVals = torch.linalg.eigvals(C).real

        if particles.positions.shape[1] == 3:
            eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:],[1])
            eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,1]),:] = torch.flip(eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,1]),:],[1])
            eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,0]),:],[1])
        


    return L, eigVals


from sphMath.schemes.weaklyCompressible import checkTensor
from sphMath.neighborhood import coo_to_csr
from sphMath.operations import SPHOperationCompiled
from typing import Optional
from torch.utils.checkpoint import checkpoint

# @torch.jit.script
def computeCovarianceMatrices_(
        particles: Union[BasicState, CompressibleState, WeaklyCompressibleState],
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter):
    num_nbrs : Optional[torch.Tensor] = None
    if isinstance(particles, BasicState):
        num_nbrs = particles.numNeighbors
    elif isinstance(particles, WeaklyCompressibleState):
        num_nbrs = particles.numNeighbors
    elif isinstance(particles, CompressibleState):
        num_nbrs = particles.numNeighbors
    else:
        raise ValueError("Particles must be of type BasicState, CompressibleState or WeaklyCompressibleState")
    if num_nbrs is None:
        raise ValueError("Particles must Number of Neighbors computed")  
    x_ij = neighborhood[1].x_ij
    dim = x_ij.shape[1]
    
    dtype = x_ij.dtype
    device = x_ij.device
    
    C = SPHOperationCompiled(
        particles,
        quantity = -x_ij,
        neighborhood = neighborhood[0],
        kernelValues = neighborhood[1],
        operation=Operation.Gradient,
        supportScheme = supportScheme,
        gradientMode = GradientMode.Naive,
    )
    C[num_nbrs < 4,:,:] = torch.eye(dim, dtype = dtype, device = device)[None,:,:]
    
    if x_ij.shape[1] == 2:
        L, eigVals = pinv2x2(C)
    else:
        L = torch.linalg.pinv(C)
        eigVals = torch.linalg.eigvals(C).real

        if x_ij.shape[1] == 3:
            eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,1]) > torch.abs(eigVals[:,0]),:],[1])
            eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,1]),:] = torch.flip(eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,1]),:],[1])
            eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,0]),:] = torch.flip(eigVals[torch.abs(eigVals[:,2]) > torch.abs(eigVals[:,0]),:],[1])
        
    return C, L, eigVals
    
def computeCovarianceMatrices(
        particles: Union[BasicState, CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    neighborhood = (neighborhood[0](neighborhood[0], kernel)) if neighborhood[1] is None else neighborhood
    return computeCovarianceMatrices_(particles, neighborhood, supportScheme)