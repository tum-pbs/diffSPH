# from sphMath.util import PointCloud, DomainDescription, SparseCOO, SparseCSR

from dataclasses import dataclass
import torch

try:
    # raise ImportError("Debug Test")
    import torchCompactRadius
    from torchCompactRadius.compactHashing.datastructure import CompactHashMap
    from torchCompactRadius import radiusSearch, neighborSearchExisting
    from torchCompactRadius.util import PointCloud, DomainDescription, SparseCOO, SparseCSR

except ImportError as e:
    # raise e
    print("torchCompactRadius not found, using fallback implementations.")

    from sphMath.neighborhoodFallback.fallback import CompactHashMap, radiusSearch, PointCloud, DomainDescription, SparseCOO, SparseCSR

from typing import Union, Optional
import torch
from sphMath.util import ParticleSet, mod
from torch.utils.checkpoint import checkpoint
import numpy as np

@torch.jit.script
def volumeToSupport(volume : float, targetNeighbors : int, dim : int):
    """
    Calculates the support radius based on the given volume, target number of neighbors, and dimension.

    Parameters:
    volume (float): The volume of the support region.
    targetNeighbors (int): The desired number of neighbors.
    dim (int): The dimension of the space.

    Returns:
    torch.Tensor: The support radius.
    """
    if dim == 1:
        # N_h = 2 h / v -> h = N_h * v / 2
        return targetNeighbors * volume / 2
    elif dim == 2:
        # N_h = \pi h^2 / v -> h = \sqrt{N_h * v / \pi}
        return torch.sqrt(targetNeighbors * volume / np.pi)
    else:
        # N_h = 4/3 \pi h^3 / v -> h = \sqrt[3]{N_h * v / \pi * 3/4}
        return torch.pow(targetNeighbors * volume / np.pi * 3 /4, 1/3)

# from sphMath.operations import mod

# @torch.jit.script
# def mod(x, min : float, max : float):
#     h = max - min
#     return ((x + h / 2.0) - torch.floor((x + h / 2.0) / h) * h) - h / 2.0
#     return torch.where(torch.abs(x) > (max - min) / 2, torch.sgn(x) * ((torch.abs(x) + min) % (max - min) + min), x)
    

from typing import NamedTuple, Union
class NeighborhoodInformation(NamedTuple):
    points_a : PointCloud
    points_b : PointCloud

    domain : DomainDescription

    accelerationStructure : CompactHashMap

    mode : str
    verletScale : float
    fullAdjacency : SparseCOO

from enum import Enum
from sphMath.enums import SupportScheme

def convertSet(particleSet):
    if not isinstance(particleSet, PointCloud):
        return PointCloud(particleSet.positions, particleSet.supports)
    else:
        return particleSet

def searchNeighbors(input_a: Union[ParticleSet,PointCloud], input_b: Optional[Union[Union[ParticleSet,PointCloud], CompactHashMap]], domain : DomainDescription, mode : str = 'symmetric', algorithm = 'compact'):
    if input_b is None:
        input_b = input_a
    if isinstance(input_b, PointCloud) or isinstance(input_b, ParticleSet):
        cooAdjacency, datastructure = radiusSearch(convertSet(input_a), convertSet(input_b), domain = domain, mode = mode, returnStructure=True, algorithm=algorithm)
    elif isinstance(input_b, CompactHashMap):
        if algorithm == 'compact':
            raise NotImplementedError('Compact algorithm not implemented for existing datastructure')
            row, col = neighborSearchExisting(convertSet(input_a), convertSet(input_b), domain = domain, mode = mode, returnStructure=True)
        else:
            raise ValueError('Cannot perform neighbor search with existing datastructure for non-compact algorithm')
        cooAdjacency = SparseCOO(row, col, numRows = input_a.shape[0], numCols = input_b.shape[0])
    return cooAdjacency, datastructure

from torch.profiler import record_function

from dataclasses import dataclass
from typing import Tuple

@torch.jit.script
@dataclass(slots = True)
class PrecomputedNeighborhood:
    r_ij: torch.Tensor # [n]
    x_ij: torch.Tensor # [n, d]
    
    W_i: torch.Tensor # [n], uses h_i
    W_j: torch.Tensor # [n], uses h_j
    gradW_i: torch.Tensor # [n, d], uses h_i
    gradW_j: torch.Tensor # [n, d], uses h_j
    
    H_i: Optional[torch.Tensor] # [n, d, d], uses h_i
    H_j: Optional[torch.Tensor] # [n, d, d], uses h_j
    
    ddh_W_i: Optional[torch.Tensor] # [n], uses h_i
    ddh_W_j: Optional[torch.Tensor] # [n], uses h_j

    
    
@torch.jit.script
@dataclass#(slots = True)
class SparseNeighborhood:
    row: torch.Tensor
    col: torch.Tensor
    
    numRows: int
    numCols: int
    
    points_a: 'PointCloud'
    points_b: 'PointCloud'
    
    domain: 'DomainDescription'
    # values: Optional[PrecomputedNeighborhood] = None



def computeDistanceTensor(neighborhood: Union[NeighborhoodInformation, SparseNeighborhood], normalize: bool = True, mode: str = 'superSymmetric'):
    with record_function(f"[Neighborhood] computeDistanceTensor"):
        if isinstance(neighborhood, NeighborhoodInformation):
            adjacency = neighborhood.fullAdjacency
            
            pos_ai = neighborhood.points_a.positions[adjacency.row]
            pos_bi = neighborhood.points_b.positions[adjacency.col]

            if neighborhood.mode == 'symmetric':
                support_ai = neighborhood.points_a.supports[adjacency.row]
                support_bi = neighborhood.points_b.supports[adjacency.col]
                hij = (support_ai + support_bi) / 2
            elif neighborhood.mode == 'scatter':
                support_bi = neighborhood.points_b.supports[adjacency.col]
                hij = support_bi
            elif neighborhood.mode == 'gather':
                support_ai = neighborhood.points_a.supports[adjacency.row]
                hij = support_ai
            elif neighborhood.mode == 'superSymmetric':
                support_ai = neighborhood.points_a.supports[adjacency.row]
                support_bi = neighborhood.points_b.supports[adjacency.col]
                hij = torch.maximum(support_ai, support_bi)
            else:
                raise ValueError('Invalid mode')

            minD = neighborhood.domain.min
            maxD = neighborhood.domain.max
            periodicity = neighborhood.domain.periodic
            xij = pos_ai - pos_bi
            xij = torch.stack([xij[:,i] if not periodic_i else mod(xij[:,i], minD[i], maxD[i]) for i, periodic_i in enumerate(periodicity)], dim = -1)

            distance = torch.linalg.norm(xij, dim = -1)
            if normalize:
                xij = torch.nn.functional.normalize(xij, dim = -1)
                rij = distance / hij
            else:
                rij = distance

            return rij, xij
        else:
            pos_ai = neighborhood.points_a.positions[neighborhood.row]
            pos_bi = neighborhood.points_b.positions[neighborhood.col]

            minD = neighborhood.domain.min
            maxD = neighborhood.domain.max
            periodicity = neighborhood.domain.periodic
            if mode == 'symmetric':
                support_ai = neighborhood.points_a.supports[neighborhood.row]
                support_bi = neighborhood.points_b.supports[neighborhood.col]
                hij = (support_ai + support_bi) / 2
            elif mode == 'scatter':
                support_bi = neighborhood.points_b.supports[neighborhood.col]
                hij = support_bi
            elif mode == 'gather':
                support_ai = neighborhood.points_a.supports[neighborhood.row]
                hij = support_ai
            elif mode == 'superSymmetric':
                support_ai = neighborhood.points_a.supports[neighborhood.row]
                support_bi = neighborhood.points_b.supports[neighborhood.col]
                hij = torch.maximum(support_ai, support_bi)
            else:
                raise ValueError('Invalid mode')

            xij = pos_ai - pos_bi
            xij = torch.stack([xij[:,i] if not periodic_i else mod(xij[:,i], minD[i], maxD[i]) for i, periodic_i in enumerate(periodicity)], dim = -1)

            distance = torch.linalg.norm(xij, dim = -1)
            if normalize:
                xij = torch.nn.functional.normalize(xij, dim = -1)
                rij = distance / hij
            else:
                rij = distance

            return rij, xij
    
from sphMath.kernels import KernelType, SPHKernel, Kernel, Kernel_Gradient, Kernel_Hessian, Kernel_DkDh
from torch.profiler import record_function



# @torch.jit.script
def evalSparseNeighborhood(
        neighborhood: SparseNeighborhood,
        kernel: KernelType,
        gradientKernel: Optional[KernelType] = None,
        computeHessian: bool = True, 
        computeDkDh: bool = True, 
        only_j: bool = False
):
    with record_function(f"[Neighborhood] evalSparseNeighborhood"):
        i, j = neighborhood.row, neighborhood.col
        h_i = neighborhood.points_a.supports[i]
        h_j = neighborhood.points_b.supports[j]
        
        r_ij, x_ij = computeDistanceTensor(neighborhood, normalize = False, mode = 'gather')

        with record_function(f"[Neighborhood] kernel"):
            W_j = Kernel(kernel, x_ij, h_j)
            W_i = Kernel(kernel, x_ij, h_i) if not only_j else W_j
        if gradientKernel is None:
            with record_function(f"[Neighborhood] kernel grad"):
                gradW_j = Kernel_Gradient(kernel, x_ij, h_j)
                gradW_i = Kernel_Gradient(kernel, x_ij, h_i) if not only_j else -gradW_j
        else:
            with record_function(f"[Neighborhood] kernel grad"):
                gradW_j = Kernel_Gradient(gradientKernel, x_ij, h_j)
                gradW_i = Kernel_Gradient(gradientKernel, x_ij, h_i) if not only_j else -gradW_j
        
        if computeHessian:
            with record_function(f"[Neighborhood] kernel hessian"):
                H_i = Kernel_Hessian(kernel, x_ij, h_i)
                H_j = Kernel_Hessian(kernel, x_ij, h_j)
        else:
            H_i = None
            H_j = None
        if computeDkDh:
            with record_function(f"[Neighborhood] kernel ddh"):
                ddh_W_i = Kernel_DkDh(kernel, x_ij, h_i)
                ddh_W_j = Kernel_DkDh(kernel, x_ij, h_j)
        else:
            ddh_W_i = None
            ddh_W_j = None
        
        precomputed = PrecomputedNeighborhood(
            r_ij = r_ij,
            x_ij = x_ij,
            
            W_i = W_i,
            W_j = W_j,
            gradW_i = gradW_i,
            gradW_j = gradW_j,
            
            H_i = H_i,
            H_j = H_j,
            
            ddh_W_i = ddh_W_i,
            ddh_W_j = ddh_W_j
        )
        return precomputed
    

def evalDistanceTensor_2(
        row: torch.Tensor, col: torch.Tensor,
        minD: torch.Tensor, maxD: torch.Tensor, periodicity: torch.Tensor,

        positions_a: torch.Tensor, positions_b: torch.Tensor,
        support_a: torch.Tensor, support_b: torch.Tensor,

        mode: SupportScheme,
        normalize: bool = False):
    # print('evalDistanceTensor Begin')
    pos_ai = positions_a[row]
    pos_bi = positions_b[col]
    h_i = support_a[row]
    h_j = support_b[col] 

    # minD = neighborhood.domain.min
    # maxD = neighborhood.domain.max
    # periodicity = neighborhood.domain.periodic
    # if mode == SupportScheme.Symmetric:
    #     support_ai = support_a[row]
    #     support_bi = support_b[col]
    #     hij = (support_ai + support_bi) / 2
    # elif mode == SupportScheme.Scatter:
    #     support_bi = support_b[col]
    #     hij = support_bi
    # elif mode == SupportScheme.Gather:
    #     support_ai = support_a[row]
    #     hij = support_ai 
    # elif mode == SupportScheme.SuperSymmetric:
    #     support_ai = support_a[row]
    #     support_bi = support_b[col]
    #     hij = torch.maximum(support_ai, support_bi)
    # else:
    #     raise ValueError('Invalid mode')

    xij = pos_ai - pos_bi
    xij_ = torch.stack([xij[:,i] if not periodic_i else mod(xij[:,i], minD[i], maxD[i]) for i, periodic_i in enumerate(periodicity)], dim = -1)

    distance = torch.linalg.norm(xij_, dim = -1)
    # if normalize:
    #     xij = torch.nn.functional.normalize(xij, dim = -1)
    #     rij = distance / hij
    # else:
    rij = distance
    # print('evalDistanceTensor End')
    return rij, xij_, h_i, h_j



def evalSparseNeighborhood2(
        row: torch.Tensor, col: torch.Tensor,

        positions_a: torch.Tensor, positions_b: torch.Tensor,
        support_a: torch.Tensor, support_b: torch.Tensor,

        minD: torch.Tensor, maxD: torch.Tensor, periodicity: torch.Tensor,

        kernel: KernelType,
        gradientKernel: Optional[KernelType] = None,
        computeHessian: bool = True, 
        computeDkDh: bool = True, 
        only_j: bool = False,
        use_checkpoint: bool = True
):
    # print(f'eval: {row.shape}, {col.shape}, {positions_a.shape}, {positions_b.shape}, {support_a.shape}, {support_b.shape}, {minD}, {maxD}, {periodicity}, {kernel}, {gradientKernel}, {computeHessian}, {computeDkDh}, {only_j}')
    with record_function(f"[Neighborhood] evalSparseNeighborhood"):
        # h_i = support_a[row]
        # h_j = support_b[col] 
        if use_checkpoint:
            r_ij, x_ij, h_i, h_j = checkpoint(evalDistanceTensor_2, 
                row, col,
                minD, maxD, periodicity,
                positions_a, positions_b,
                support_a, support_b,
                SupportScheme.Gather, False, use_reentrant=False)
        else:
            r_ij, x_ij, h_i, h_j = evalDistanceTensor_2( 
                row, col,
                minD, maxD, periodicity,
                positions_a, positions_b,
                support_a, support_b,
                SupportScheme.Gather, False)

        with record_function(f"[Neighborhood] kernel"):
            if use_checkpoint:
                W_j = checkpoint(Kernel, kernel, x_ij, h_j, use_reentrant=False)
                W_i = checkpoint(Kernel, kernel, x_ij, h_i, use_reentrant=False)
            else:
                W_j = Kernel(kernel, x_ij, h_j)
                W_i = Kernel(kernel, x_ij, h_i)

        if gradientKernel is None:
            with record_function(f"[Neighborhood] kernel grad"):
                if use_checkpoint:
                    gradW_j = checkpoint(Kernel_Gradient, kernel, x_ij, h_j, use_reentrant=False)
                    gradW_i = checkpoint(Kernel_Gradient, kernel, x_ij, h_i, use_reentrant=False) # if not only_j else -gradW_j
                else:
                    gradW_j = Kernel_Gradient(kernel, x_ij, h_j)
                    gradW_i = Kernel_Gradient(kernel, x_ij, h_i) # if not only_j else -gradW_j
        else:
            with record_function(f"[Neighborhood] kernel grad"):
                if use_checkpoint:
                    gradW_j = checkpoint(Kernel_Gradient, gradientKernel, x_ij, h_j, use_reentrant=False)
                    gradW_i = checkpoint(Kernel_Gradient, gradientKernel, x_ij, h_i, use_reentrant=False) # if not only_j else -gradW_j
                else:
                    gradW_j = Kernel_Gradient(gradientKernel, x_ij, h_j)
                    gradW_i = Kernel_Gradient(gradientKernel, x_ij, h_i) # if not only_j else -gradW_j
        
        if computeHessian:
            with record_function(f"[Neighborhood] kernel hessian"):
                if use_checkpoint:
                    H_i = checkpoint(Kernel_Hessian, kernel, x_ij, h_i, use_reentrant=False)
                    H_j = checkpoint(Kernel_Hessian, kernel, x_ij, h_j, use_reentrant=False)
                else:
                    H_i = Kernel_Hessian(kernel, x_ij, h_i)
                    H_j = Kernel_Hessian(kernel, x_ij, h_j)
        else:
            H_i = None
            H_j = None
        if computeDkDh:
            with record_function(f"[Neighborhood] kernel ddh"):
                if use_checkpoint:
                    ddh_W_i = checkpoint(Kernel_DkDh, kernel, x_ij, h_i, use_reentrant=False)
                    ddh_W_j = checkpoint(Kernel_DkDh, kernel, x_ij, h_j, use_reentrant=False)
                else:
                    ddh_W_i = Kernel_DkDh(kernel, x_ij, h_i)
                    ddh_W_j = Kernel_DkDh(kernel, x_ij, h_j)
        else:
            ddh_W_i = None
            ddh_W_j = None
        
        return r_ij, x_ij, W_i, W_j, gradW_i, gradW_j, H_i, H_j, ddh_W_i, ddh_W_j
    


def buildNeighborhood(points_a : Union[ParticleSet,PointCloud], points_b : Union[ParticleSet,PointCloud], domain : DomainDescription, verletScale : float = 1.0, mode : str = 'symmetric', priorNeighborhood : Optional[NeighborhoodInformation] = None, verbose : bool = False, neighborhoodAlgorithm : str = 'compact'):
    with record_function(f"[Neighborhood] {mode}"):
        if priorNeighborhood is None:
            if verbose:
                print('Building neighborhood from scratch [no prior neighborhood]')
            with record_function(f"[Neighborhood] no prior"):
                adjacency, datastructure = searchNeighbors(
                    PointCloud(points_a.positions, points_a.supports * verletScale), 
                    PointCloud(points_b.positions, points_b.supports * verletScale), domain, mode, algorithm=neighborhoodAlgorithm)
        else:
            if points_a.positions.shape[0] != priorNeighborhood.points_a.positions.shape[0] or points_b.positions.shape[0] != priorNeighborhood.points_b.positions.shape[0]:
                if verbose:
                    print('Building neighborhood from scratch [different number of particles]')
                with record_function(f"[Neighborhood] mismatch"):
                        adjacency, datastructure = searchNeighbors(
                            PointCloud(points_a.positions, points_a.supports * verletScale), 
                            PointCloud(points_b.positions, points_b.supports * verletScale), domain, mode, algorithm=neighborhoodAlgorithm)
            else:
                distance_a = torch.linalg.norm(points_a.positions - priorNeighborhood.points_a.positions, dim = -1)
                distance_b = torch.linalg.norm(points_b.positions - priorNeighborhood.points_b.positions, dim = -1)
                supports_a = priorNeighborhood.points_a.supports
                supports_b = priorNeighborhood.points_b.supports
                
                supportDistance_a = (supports_a - points_a.supports) #/ supports_a
                supportDistance_b = (supports_b - points_b.supports) #/ supports_b
                maxDistance = distance_a.max() + distance_b.max()
                minSupport = None
                if mode == 'symmetric':
                    supportFactor = max(supportDistance_a.max(), supportDistance_b.max())
                    minSupport = min(supports_a.min(), supports_b.min())
                elif mode == 'scatter':
                    supportFactor = supportDistance_b.max()
                    minSupport = supports_b.min()
                elif mode == 'gather':
                    supportFactor = supportDistance_a.max()
                    minSupport = supports_a.min()
                elif mode == 'superSymmetric':
                    supportFactor = max(supportDistance_a.max(), supportDistance_b.max())
                    minSupport = min(supports_a.min(), supports_b.min())
                else:
                    raise ValueError('Invalid mode')
                
                hFactor = 1 + supportFactor / minSupport
                dFactor = 1 + maxDistance / minSupport

                if verbose:
                    print(f'Distance a: min: {distance_a.min()}, max: {distance_a.max()}, avg: {distance_a.mean()}')
                    print(f'Distance b: min: {distance_b.min()}, max: {distance_b.max()}, avg: {distance_b.mean()}')
                    print(f'Support a: min: {supports_a.min()}, max: {supports_a.max()}, avg: {supports_a.mean()}')
                    print(f'Support b: min: {supports_b.min()}, max: {supports_b.max()}, avg: {supports_b.mean()}')
                    print(f'Support Factor: {supportFactor}, Minimum Support: {minSupport}')
                    print(f'Max Distance: {maxDistance}')
                    print(f'Distance Factor: {maxDistance / minSupport}')



                # print(maxDistance, minSupport * verletScale)
                if hFactor * dFactor > verletScale:
                    if verbose:
                # if verbose:
                        print('Support Factor: ', supportFactor / minSupport, 'Minimum Support: ', minSupport)
                        print('Distance Factor: ', maxDistance / minSupport)
                        print('Ratio: ', hFactor * dFactor)
                        print('Building neighborhood from scratch [distance larger than support]')
                    with record_function(f"[Neighborhood] verlet"):
                        adjacency, datastructure = searchNeighbors(
                            PointCloud(points_a.positions, points_a.supports * verletScale), 
                            PointCloud(points_b.positions, points_b.supports * verletScale), domain, mode, algorithm=neighborhoodAlgorithm)
                        
                        priorNeighborhood = NeighborhoodInformation(
                            PointCloud(points_a.positions.clone(), points_a.supports.clone()),
                            PointCloud(points_b.positions.clone(), points_b.supports.clone()),
                            domain,
                            datastructure,
                            mode,
                            verletScale,
                            adjacency
                        )
                else:
                    if verbose:
                        print('Reusing neighborhood')
                    adjacency = priorNeighborhood.fullAdjacency
                    datastructure = priorNeighborhood.accelerationStructure

                return priorNeighborhood, SparseNeighborhood(
                    row = adjacency.row,
                    col = adjacency.col,

                    numRows= adjacency.numRows,
                    numCols = adjacency.numCols,

                    points_a = PointCloud(points_a.positions, points_a.supports),
                    points_b = PointCloud(points_b.positions, points_b.supports),

                    domain = domain
                )
        return NeighborhoodInformation(
            PointCloud(points_a.positions.clone(), points_a.supports.clone()), 
            PointCloud(points_b.positions.clone(), points_b.supports.clone()), domain, datastructure, mode, verletScale, adjacency), SparseNeighborhood(
                    row = adjacency.row,
                    col = adjacency.col,

                    numRows= adjacency.numRows,
                    numCols = adjacency.numCols,

                    points_a = PointCloud(points_a.positions, points_a.supports),
                    points_b = PointCloud(points_b.positions, points_b.supports),

                    domain = domain
                )
    
    
@torch.jit.script
def computeSparseDistanceTensor(neighborhood: SparseNeighborhood, normalize: bool = True, mode : str = 'superSymmetric'):
    with record_function(f"[Neighborhood] computeSparseDistanceTensor"):
        pos_ai = neighborhood.points_a.positions[neighborhood.row]
        pos_bi = neighborhood.points_b.positions[neighborhood.col]

        minD = neighborhood.domain.min
        maxD = neighborhood.domain.max
        periodic = neighborhood.domain.periodic
        if mode == 'symmetric':
            support_ai = neighborhood.points_a.supports[neighborhood.row]
            support_bi = neighborhood.points_b.supports[neighborhood.col]
            hij = (support_ai + support_bi) / 2
        elif mode == 'scatter':
            support_bi = neighborhood.points_b.supports[neighborhood.col]
            hij = support_bi
        elif mode == 'gather':
            support_ai = neighborhood.points_a.supports[neighborhood.row]
            hij = support_ai
        elif mode == 'superSymmetric':
            support_ai = neighborhood.points_a.supports[neighborhood.row]
            support_bi = neighborhood.points_b.supports[neighborhood.col]
            hij = torch.maximum(support_ai, support_bi)
        else:
            raise ValueError('Invalid mode')

        xij = pos_ai - pos_bi
        xij = torch.stack([xij[:,i] if not periodic_i else mod(xij[:,i], minD[i], maxD[i]) for i, periodic_i in enumerate(periodic)], dim = -1)

        distance = torch.linalg.norm(xij, dim = -1)
        if normalize:
            xij = torch.nn.functional.normalize(xij, dim = -1)
            rij = distance / hij
        else:
            rij = distance

        return rij, xij

@torch.jit.script
def filterNeighborhood(neighborhood: SparseNeighborhood, mode: str = 'superSymmetric'):
    if isinstance(neighborhood, NeighborhoodInformation):
        raise ValueError('Cannot filter neighborhood with NeighborhoodInformation')

    # adjacency = neighborhood.fullAdjacency
    row, col = neighborhood.row, neighborhood.col
    rij, xij = computeSparseDistanceTensor(neighborhood, normalize = True, mode = mode)
    mask = rij <= 1
    return SparseNeighborhood(row[mask], col[mask], numRows = neighborhood.numRows, numCols = neighborhood.numCols,
                                points_a = neighborhood.points_a, points_b = neighborhood.points_b,
                                domain = neighborhood.domain)

# from sphMath.neighborhood import SparseNeighborhood
# @torch.jit.script
def filterNeighborhoodByKind(particleState, sparseNeighborhood : SparseNeighborhood, which : str = 'normal'):
    if which == 'all':
        return sparseNeighborhood
    i = sparseNeighborhood.row
    j = sparseNeighborhood.col
    
    if which == 'fluid':
        maskA = particleState.kinds[i] == 0
        maskB = particleState.kinds[j] == 0
    elif which == 'boundary':
        maskA = particleState.kinds[i] == 1
        maskB = particleState.kinds[j] == 1
    elif which == 'boundaryToFluid':
        maskA = particleState.kinds[i] == 0
        maskB = particleState.kinds[j] == 1
    elif which == 'fluidToBoundary':
        maskA = particleState.kinds[i] == 1
        maskB = particleState.kinds[j] == 0
    elif which == 'ghostToFluid':
        maskA = particleState.kinds[i] == 0
        maskB = particleState.kinds[j] == 2
    elif which == 'fluidToGhost':
        maskA = particleState.kinds[i] == 2
        maskB = particleState.kinds[j] == 0
    elif which == 'normal':
        maskA = torch.ones_like(i, dtype = torch.bool)
        maskB = particleState.kinds[j] != 2
    elif which == 'fluidWBoundary':
        maskA = particleState.kinds[i] == 0
        maskB = torch.logical_or(particleState.kinds[j] == 1, particleState.kinds[j] == 0)
    elif which == 'boundaryWFluid':
        maskA = particleState.kinds[i] == 1
        maskB = torch.logical_or(particleState.kinds[j] == 1, particleState.kinds[j] == 0)
    elif which == 'noghost':
        maskA = particleState.kinds[i] != 2
        maskB = particleState.kinds[j] != 2
    else:
        raise ValueError(f'which = {which} not recognized')
        
    mask = torch.logical_and(maskA, maskB)
    
    i_filtered = i[mask]
    j_filtered = j[mask]
    # newNumRows = torch.sum(maskA)
    # newNumCols = torch.sum(maskB)
    
    return SparseNeighborhood(
        row = i_filtered,
        col = j_filtered,
        numRows = sparseNeighborhood.numRows,
        numCols = sparseNeighborhood.numCols,
        
        points_a = sparseNeighborhood.points_a,
        points_b = sparseNeighborhood.points_b,
        
        domain = sparseNeighborhood.domain,
    )

import warnings
try:
    # raise ImportError("Debug Test")
    import torchCompactRadius
    from torchCompactRadius.util import SparseCOO, SparseCSR, PointCloud, DomainDescription, coo_to_csr, csr_to_coo
except ImportError:
    warnings.warn('Using fallback implementation for radius search.')
    from sphMath.neighborhoodFallback.fallback import SparseCOO, SparseCSR, PointCloud, DomainDescription, coo_to_csr, csr_to_coo


# from sphMath.util import coo_to_csr
def coo_to_csrsc(coo: SparseCOO):
    return coo_to_csr(coo, isSorted=True), coo_to_csr(SparseCOO(coo.col, coo.row, numRows = coo.numCols, numCols = coo.numRows), isSorted=False)


import torch
# from sphMath.kernels import SPHKernel
from typing import Union, Tuple, Optional
# from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product
# from sphMath.neighborhood import DomainDescription, SparseCOO

# def buildSuperSymmetricNeighborhood(particles, domain, verletScale = 1.0):
#     neighborhood_scatter = buildNeighborhood(particles, particles, domain, verletScale= verletScale, mode ='scatter')
#     neighborhood_gather = buildNeighborhood(particles, particles, domain, verletScale= verletScale, mode ='gather')

#     coo_scatter = neighborhood_scatter.fullAdjacency
#     coo_gather = neighborhood_gather.fullAdjacency

#     coo_scatter_row = coo_scatter.row
#     coo_scatter_col = coo_scatter.col

#     coo_gather_row = coo_gather.row
#     coo_gather_col = coo_gather.col

#     # print(coo_scatter_row.shape, coo_scatter_col.shape)
#     # print(coo_gather_row.shape, coo_gather_col.shape)

#     merged_row = torch.cat([coo_scatter_row, coo_gather_row], dim = 0)
#     merged_col = torch.cat([coo_scatter_col, coo_gather_col], dim = 0)

#     numParticles = particles.positions.shape[0]

#     key = merged_row * numParticles + merged_col

#     key, indices = torch.sort(key)

#     sorted_row = merged_row[indices]
#     sorted_col = merged_col[indices]

#     unique_key, unique_counts = torch.unique(key, return_counts = True)
#     # print(unique_key.shape, unique_counts.shape, unique_key, unique_counts)

#     offsets = torch.cat([torch.tensor([0]), torch.cumsum(unique_counts, dim = 0)])[:-1]

#     sorted_row = sorted_row[offsets]
#     sorted_col = sorted_col[offsets]

#     return SparseCOO(sorted_row, sorted_col, numParticles, numParticles)

# def buildSuperSymmetricNeighborhood(particles, domain, verletScale = 1.0):
#     neighborhood_scatter = buildNeighborhood(particles, particles, domain, verletScale= verletScale, mode ='scatter')
#     neighborhood_gather = buildNeighborhood(particles, particles, domain, verletScale= verletScale, mode ='gather')

#     coo_scatter = neighborhood_scatter.fullAdjacency
#     coo_gather = neighborhood_gather.fullAdjacency

#     coo_scatter_row = coo_scatter.row
#     coo_scatter_col = coo_scatter.col

#     coo_gather_row = coo_gather.row
#     coo_gather_col = coo_gather.col

#     # print(coo_scatter_row.shape, coo_scatter_col.shape)
#     # print(coo_gather_row.shape, coo_gather_col.shape)

#     merged_row = torch.cat([coo_scatter_row, coo_gather_row], dim = 0)
#     merged_col = torch.cat([coo_scatter_col, coo_gather_col], dim = 0)

#     numParticles = particles.positions.shape[0]

#     key = merged_row * numParticles + merged_col

#     key, indices = torch.sort(key)

#     sorted_row = merged_row[indices]
#     sorted_col = merged_col[indices]

#     unique_key, unique_counts = torch.unique(key, return_counts = True)
#     # print(unique_key.shape, unique_counts.shape, unique_key, unique_counts)

#     offsets = torch.cat([torch.tensor([0]), torch.cumsum(unique_counts, dim = 0)])[:-1]

#     sorted_row = sorted_row[offsets]
#     sorted_col = sorted_col[offsets]

#     return SparseCOO(sorted_row, sorted_col, numParticles, numParticles)

@torch.jit.script
def access_optional(tensor: Optional[torch.Tensor], i: torch.Tensor) -> torch.Tensor:
    if tensor is not None:
        return tensor[i]
    else:
        raise ValueError("Tensor is None and cannot be accessed.")
@torch.jit.script
def access_optional_range(tensor: Optional[torch.Tensor], begin: int, end: int) -> torch.Tensor:
    if tensor is not None:
        return tensor[begin:end]
    else:
        raise ValueError("Tensor is None and cannot be accessed.")


@torch.jit.script
def filterNeighborsByKind(kinds : torch.Tensor, sparseNeighborhood : SparseNeighborhood, kernelValues : PrecomputedNeighborhood, which : str = 'normal'):
    if which == 'all':
        return sparseNeighborhood, kernelValues
    i = sparseNeighborhood.row
    j = sparseNeighborhood.col
    
    if which == 'fluid':
        maskA = kinds[i] == 0
        maskB = kinds[j] == 0
    elif which == 'boundary':
        maskA = kinds[i] == 1
        maskB = kinds[j] == 1
    elif which == 'boundaryToFluid':
        maskA = kinds[i] == 0
        maskB = kinds[j] == 1
    elif which == 'fluidToBoundary':
        maskA = kinds[i] == 1
        maskB = kinds[j] == 0
    elif which == 'ghostToFluid':
        maskA = kinds[i] == 0
        maskB = kinds[j] == 2
    elif which == 'fluidToGhost':
        maskA = kinds[i] == 2
        maskB = kinds[j] == 0
    elif which == 'normal':
        maskA = torch.ones_like(i, dtype = torch.bool)
        maskB = kinds[j] != 2
    elif which == 'noghost':
        maskA = kinds[i] != 2
        maskB = kinds[j] != 2
    elif which == 'fluidWBoundary':
        maskA = kinds[i] == 0
        maskB = torch.logical_or(kinds[j] == 1, kinds[j] == 0)
    elif which == 'boundaryWFluid':
        maskA = kinds[i] == 1
        maskB = torch.logical_or(kinds[j] == 1, kinds[j] == 0)    
    else:
        raise ValueError(f'which = {which} not recognized')
        
    mask = torch.logical_and(maskA, maskB)
    
    i_filtered = i[mask]
    j_filtered = j[mask]
    # newNumRows = torch.sum(maskA)
    # newNumCols = torch.sum(maskB)
    
    return SparseNeighborhood(
        row = i_filtered,
        col = j_filtered,
        numRows = sparseNeighborhood.numRows,
        numCols = sparseNeighborhood.numCols,
        
        points_a = sparseNeighborhood.points_a,
        points_b = sparseNeighborhood.points_b,
        
        domain = sparseNeighborhood.domain,
    ), PrecomputedNeighborhood(
        r_ij = kernelValues.r_ij[mask],
        x_ij= kernelValues.x_ij[mask],
        W_i= kernelValues.W_i[mask],
        W_j= kernelValues.W_j[mask],
        H_i = access_optional(kernelValues.H_i, mask) if kernelValues.H_i is not None else None,
        H_j = access_optional(kernelValues.H_j, mask) if kernelValues.H_j is not None else None,
        gradW_i= kernelValues.gradW_i[mask],
        gradW_j= kernelValues.gradW_j[mask],
        ddh_W_i= access_optional(kernelValues.ddh_W_i, mask) if kernelValues.ddh_W_i is not None else None,
        ddh_W_j= access_optional(kernelValues.ddh_W_j, mask) if kernelValues.ddh_W_j is not None else None,
    )

def sliceNeighbors(
        neighborhood: SparseNeighborhood,
        kernelValues: PrecomputedNeighborhood,
        interval: Optional[Tuple[int, int]] = None,
):
    if interval is None:
        return None, None
    begin, end = interval

    return SparseNeighborhood(
        row=neighborhood.row[begin:end],
        col=neighborhood.col[begin:end],
        numRows=neighborhood.numRows,
        numCols=neighborhood.numCols,
        points_a=neighborhood.points_a,
        points_b=neighborhood.points_b,
        domain=neighborhood.domain
    ), PrecomputedNeighborhood(
        r_ij=kernelValues.r_ij[begin:end],
        x_ij=kernelValues.x_ij[begin:end],
        W_i=kernelValues.W_i[begin:end],
        W_j=kernelValues.W_j[begin:end],
        gradW_i=kernelValues.gradW_i[begin:end],
        gradW_j=kernelValues.gradW_j[begin:end],
        H_i=access_optional_range(kernelValues.H_i, begin, end) if kernelValues.H_i is not None else None,
        H_j=access_optional_range(kernelValues.H_j, begin, end) if kernelValues.H_j is not None else None,
        ddh_W_i=access_optional_range(kernelValues.ddh_W_i, begin, end) if kernelValues.ddh_W_i is not None else None,
        ddh_W_j=access_optional_range(kernelValues.ddh_W_j, begin, end) if kernelValues.ddh_W_j is not None else None,
    )

@dataclass(slots = True)
class NeighborhoodStates:
    neighbors: SparseNeighborhood
    kernelValues: PrecomputedNeighborhood

            # normal = (normalBegin, normalEnd),
            # fluid = (fluidBegin, fluidEnd),
            # boundaryToFluid = (boundaryToFluidBegin, boundaryToFluidEnd),
            # fluidToBoundary = (fluidToBoundaryBegin, fluidToBoundaryEnd),
            # boundaryToBoundary = (boundaryToBoundaryBegin, boundaryToBoundaryEnd),
            # fluidToGhost = (fluidToGhostBegin, fluidToGhostEnd)
    
    normal: Tuple[int, int]
    fluid: Tuple[int, int]
    boundaryToFluid: Optional[Tuple[int, int]]
    fluidToBoundary: Optional[Tuple[int, int]]
    boundaryToBoundary: Optional[Tuple[int, int]]
    fluidToGhost: Optional[Tuple[int, int]]

    # fluid_to_fluid_neighbors: SparseNeighborhood
    # fluid_to_fluid_kernelValues: PrecomputedNeighborhood

    # fluid_to_boundary_neighbors: Optional[SparseNeighborhood] = None
    # fluid_to_boundary_kernelValues: Optional[PrecomputedNeighborhood] = None

    # boundary_to_fluid_neighbors: Optional[SparseNeighborhood] = None
    # boundary_to_fluid_kernelValues: Optional[PrecomputedNeighborhood] = None

    # boundary_to_boundary_neighbors: Optional[SparseNeighborhood] = None
    # boundary_to_boundary_kernelValues: Optional[PrecomputedNeighborhood] = None

    # fluid_to_ghost_neighbors: Optional[SparseNeighborhood] = None
    # fluid_to_ghost_kernelValues: Optional[PrecomputedNeighborhood] = None

    def get(self, which: str = 'all'):
        if which == 'fluid':
            return sliceNeighbors(self.neighbors, self.kernelValues, self.fluid)
        elif which == 'boundary':
            return sliceNeighbors(self.neighbors, self.kernelValues, self.fluidToBoundary)
        elif which == 'boundaryToFluid':
            return sliceNeighbors(self.neighbors, self.kernelValues, self.boundaryToFluid)
        elif which == 'fluidToBoundary':
            return sliceNeighbors(self.neighbors, self.kernelValues, self.fluidToBoundary)
        elif which == 'boundaryToBoundary':
            return sliceNeighbors(self.neighbors, self.kernelValues, self.boundaryToBoundary)
        elif which == 'fluidToGhost':
            return sliceNeighbors(self.neighbors, self.kernelValues, self.fluidToGhost)
        elif which == 'all' or which == 'normal' or which == 'noghost':
            return self.neighbors, self.kernelValues
        else:
            raise ValueError(f'which = {which} not recognized')
        

def evalEdgeKinds(kinds : torch.Tensor, i : torch.Tensor, j: torch.Tensor):
    out = i.new_ones(i.shape, dtype = torch.int32) * 10

    ki = kinds[i]
    kj = kinds[j]

    mask_ftf = torch.logical_and(ki == 0, kj == 0)
    mask_btf = torch.logical_and(ki == 0, kj == 1)
    mask_ftb = torch.logical_and(ki == 1, kj == 0)
    mask_btb = torch.logical_and(ki == 1, kj == 1)
    mask_ftg = torch.logical_and(ki == 2, kj == 0)

    out = torch.where(mask_ftf, 0, out)
    out = torch.where(mask_btf, 1, out)
    out = torch.where(mask_ftb, 2, out)
    out = torch.where(mask_btb, 3, out)
    out = torch.where(mask_ftg, 4, out)

    outSorted = torch.argsort(out, dim = 0)

    return out, outSorted, torch.sum(mask_ftf), torch.sum(mask_btf), torch.sum(mask_ftb), torch.sum(mask_btb), torch.sum(mask_ftg)




def computeNeighborhoodStates(particles, sparseNeighborhood_, mode_str, kernel, gradientKernel, computeHessian = True, computeDkDh = True, only_j = False):
    # print(f'Computing neighborhood states for {mode_str}')
    # Filter based on verlet
    with record_function(f"[Neighborhood] filter [verlet]"):
        actualNeigbors = filterNeighborhood(sparseNeighborhood_, mode_str)
    with record_function(f"[Neighborhood] evaluate kernels"):
        positions_a = actualNeigbors.points_a.positions.clone()
        positions_b = actualNeigbors.points_b.positions.clone()
        supports_a = actualNeigbors.points_a.supports.clone()
        supports_b = actualNeigbors.points_b.supports.clone()

        use_checkpoint = False
        if not use_checkpoint:
            r_ij, x_ij, W_i, W_j, gradW_i, gradW_j, H_i, H_j, ddh_W_i, ddh_W_j = evalSparseNeighborhood2(
            actualNeigbors.row, actualNeigbors.col,
            positions_a, positions_b,
            supports_a, supports_b,
            actualNeigbors.domain.min, actualNeigbors.domain.max, actualNeigbors.domain.periodic,
            kernel, gradientKernel, computeHessian, computeDkDh, only_j, True)
        else:
            r_ij, x_ij, W_i, W_j, gradW_i, gradW_j, H_i, H_j, ddh_W_i, ddh_W_j = checkpoint(evalSparseNeighborhood2,
                actualNeigbors.row, actualNeigbors.col,
                positions_a, positions_b,
                supports_a, supports_b,
                actualNeigbors.domain.min, actualNeigbors.domain.max, actualNeigbors.domain.periodic,
                kernel, gradientKernel, computeHessian, computeDkDh, only_j, False, use_reentrant=False)
        kernelValues = PrecomputedNeighborhood(
            r_ij = r_ij,
            x_ij = x_ij,
            W_i = W_i,
            W_j = W_j,
            gradW_i = gradW_i,
            gradW_j = gradW_j,
            H_i = H_i,
            H_j = H_j,
            ddh_W_i = ddh_W_i,
            ddh_W_j = ddh_W_j
        )

        # kernelValues = evalSparseNeighborhood(actualNeigbors, kernel, gradientKernel, computeHessian, computeDkDh, only_j = only_j)

    if not torch.any(particles.kinds > 0):
        sortedNeighbors = actualNeigbors
        sortedKernelValues = kernelValues
        normalIndices = (0, actualNeigbors.row.shape[0])
        fluidIndices = (0, actualNeigbors.row.shape[0])
        boundaryToFluidIndices = None
        fluidToBoundaryIndices = None
        boundaryToBoundaryIndices = None
        fluidToGhostIndices = None
    else:
        with record_function(f"[Neighborhood] filter [kinds]"):
            with record_function(f"[Neighborhood] filter [kinds] - sort"):
                indices, indicesSorted, numFluidToFluid, numBoundaryToFluid, numFluidToBoundary, numBoundaryToBoundary, numFluidToGhost = evalEdgeKinds(particles.kinds, actualNeigbors.row, actualNeigbors.col)

                numNormal = numFluidToFluid + numBoundaryToFluid + numFluidToBoundary + numBoundaryToBoundary #+ numFluidToGhost

                sortedNeighbors = SparseNeighborhood(actualNeigbors.row[indicesSorted], actualNeigbors.col[indicesSorted], numRows = actualNeigbors.numRows, numCols = actualNeigbors.numCols, 
                                                    points_a = actualNeigbors.points_a, points_b = actualNeigbors.points_b, domain = actualNeigbors.domain)
                r_ij_ = kernelValues.r_ij[indicesSorted]
                x_ij_ = kernelValues.x_ij[indicesSorted]
                if only_j:
                    W_i_ = kernelValues.W_j[indicesSorted]
                    gradW_i_ = kernelValues.gradW_j[indicesSorted]
                else:
                    W_i_ = kernelValues.W_i[indicesSorted]
                    gradW_i_ = kernelValues.gradW_i[indicesSorted]
                W_j_ = kernelValues.W_j[indicesSorted]
                gradW_j_ = kernelValues.gradW_j[indicesSorted]
                if computeHessian:
                    H_i_ = kernelValues.H_i[indicesSorted]
                    H_j_ = kernelValues.H_j[indicesSorted]
                else:
                    H_i_ = None
                    H_j_ = None
                if computeDkDh:
                    ddh_W_i_ = kernelValues.ddh_W_i[indicesSorted]
                    ddh_W_j_ = kernelValues.ddh_W_j[indicesSorted]
                else:   
                    ddh_W_i_ = None
                    ddh_W_j_ = None

                sortedKernelValues = PrecomputedNeighborhood(
                    r_ij = r_ij_,
                    x_ij = x_ij_,
                    W_i = W_i_,
                    W_j = W_j_,
                    gradW_i = gradW_i_,
                    gradW_j = gradW_j_,
                    H_i = H_i_,
                    H_j = H_j_,
                    ddh_W_i = ddh_W_i_,
                    ddh_W_j = ddh_W_j_
                )

            normalIndices = (0, numNormal)
            fluidIndices = (0, numFluidToFluid)
            boundaryToFluidIndices = (fluidIndices[-1], fluidIndices[-1] + numBoundaryToFluid)
            fluidToBoundaryIndices = (boundaryToFluidIndices[-1], boundaryToFluidIndices[-1] + numFluidToBoundary)
            boundaryToBoundaryIndices = (fluidToBoundaryIndices[-1], fluidToBoundaryIndices[-1] + numBoundaryToBoundary)
            fluidToGhostIndices = (boundaryToBoundaryIndices[-1], boundaryToBoundaryIndices[-1] + numFluidToGhost)
    return NeighborhoodStates(
            neighbors = sortedNeighbors,
            kernelValues = sortedKernelValues,

            normal = normalIndices,
            fluid = fluidIndices,
            boundaryToFluid = boundaryToFluidIndices,
            fluidToBoundary = fluidToBoundaryIndices,
            boundaryToBoundary = boundaryToBoundaryIndices,
            fluidToGhost = fluidToGhostIndices
        )

def evaluateNeighborhood(particles,
                         domain : DomainDescription,
                         kernel: SPHKernel,
                         verletScale : float = 1.0,
                         mode : SupportScheme = SupportScheme.SuperSymmetric,
                         priorNeighborhood: Optional[NeighborhoodInformation] = None,
                         gradientKernel : Optional[SPHKernel] = None,
                         useCheckpoint : bool = True,
                         computeHessian : bool = True,
                         computeDkDh : bool = True,
                         only_j : bool = False):
    with record_function(f"[Neighborhood] evaluateNeighborhood"):
        mode_str = mode.name
        mode_str = mode_str[0].lower() + mode_str[1:]
        neighborhood, sparseNeighborhood_ = buildNeighborhood(particles, particles, domain, verletScale = verletScale, mode = mode_str, priorNeighborhood=priorNeighborhood)
    if useCheckpoint:
        return neighborhood, checkpoint(computeNeighborhoodStates, particles, sparseNeighborhood_, mode_str, kernel, gradientKernel, computeHessian, computeDkDh, only_j, use_reentrant=False)
    else:
        return neighborhood, computeNeighborhoodStates(particles, sparseNeighborhood_, mode_str, kernel, gradientKernel, computeHessian, computeDkDh, only_j)


def evalSupport(supports: torch.Tensor, i: torch.Tensor, j: torch.Tensor, supportScheme : SupportScheme):
    if supportScheme == SupportScheme.Symmetric:
        return (supports[i] + supports[j]) / 2
    elif supportScheme == SupportScheme.Scatter:
        return supports[j]
    elif supportScheme == SupportScheme.Gather:
        return supports[i]
    elif supportScheme == SupportScheme.SuperSymmetric:
        return torch.maximum(supports[i], supports[j])
    else:
        raise ValueError('Invalid support scheme')
    
def evalKernel(PrecomputedKernel: PrecomputedNeighborhood, supportScheme: SupportScheme, combined = True):
    if combined:
        if supportScheme == SupportScheme.Scatter:
            return PrecomputedKernel.W_j
        elif supportScheme == SupportScheme.Gather:
            return PrecomputedKernel.W_i
        elif supportScheme == SupportScheme.Symmetric:
            return (PrecomputedKernel.W_i + PrecomputedKernel.W_j) / 2
        elif supportScheme == SupportScheme.SuperSymmetric:
            return (PrecomputedKernel.W_i + PrecomputedKernel.W_j) / 2
        else:
            raise ValueError('Invalid support scheme')
    else:
        if supportScheme == SupportScheme.Scatter:
            return PrecomputedKernel.W_j, PrecomputedKernel.W_j
        elif supportScheme == SupportScheme.Gather:
            return PrecomputedKernel.W_i, PrecomputedKernel.W_i
        elif supportScheme == SupportScheme.Symmetric:
            W_ij = (PrecomputedKernel.W_i + PrecomputedKernel.W_j) / 2
            return W_ij, W_ij
        elif supportScheme == SupportScheme.SuperSymmetric:
            return PrecomputedKernel.W_i, PrecomputedKernel.W_j
        else:
            raise ValueError('Invalid support scheme')
def evalKernelGradient(PrecomputedKernel: PrecomputedNeighborhood, supportScheme: SupportScheme, combined = True):
    if combined:
        if supportScheme == SupportScheme.Scatter:
            return PrecomputedKernel.gradW_j
        elif supportScheme == SupportScheme.Gather:
            return PrecomputedKernel.gradW_i
        elif supportScheme == SupportScheme.Symmetric:
            return (PrecomputedKernel.gradW_i + PrecomputedKernel.gradW_j) / 2
        elif supportScheme == SupportScheme.SuperSymmetric:
            return (PrecomputedKernel.gradW_i + PrecomputedKernel.gradW_j) / 2
        else:
            raise ValueError('Invalid support scheme')
    else:
        if supportScheme == SupportScheme.Scatter:
            return PrecomputedKernel.gradW_j, PrecomputedKernel.gradW_j
        elif supportScheme == SupportScheme.Gather:
            return PrecomputedKernel.gradW_i, PrecomputedKernel.gradW_i
        elif supportScheme == SupportScheme.Symmetric:
            gradW_ij = (PrecomputedKernel.gradW_i + PrecomputedKernel.gradW_j) / 2
            return gradW_ij, gradW_ij
        elif supportScheme == SupportScheme.SuperSymmetric:
            return PrecomputedKernel.gradW_i, PrecomputedKernel.gradW_j
        else:
            raise ValueError('Invalid support scheme')
def evalKernelHessian(PrecomputedKernel: PrecomputedNeighborhood, supportScheme: SupportScheme, combined = True):
    if combined:
        if supportScheme == SupportScheme.Scatter:
            return PrecomputedKernel.H_j
        elif supportScheme == SupportScheme.Gather:
            return PrecomputedKernel.H_i
        elif supportScheme == SupportScheme.Symmetric:
            return (PrecomputedKernel.H_i + PrecomputedKernel.H_j) / 2
        elif supportScheme == SupportScheme.SuperSymmetric:
            return (PrecomputedKernel.H_i + PrecomputedKernel.H_j) / 2
        else:
            raise ValueError('Invalid support scheme')
    else:
        if supportScheme == SupportScheme.Scatter:
            return PrecomputedKernel.H_j, PrecomputedKernel.H_j
        elif supportScheme == SupportScheme.Gather:
            return PrecomputedKernel.H_i, PrecomputedKernel.H_i
        elif supportScheme == SupportScheme.Symmetric:
            H_ij = (PrecomputedKernel.H_i + PrecomputedKernel.H_j) / 2
            return H_ij, H_ij
        elif supportScheme == SupportScheme.SuperSymmetric:
            return PrecomputedKernel.H_i, PrecomputedKernel.H_j
        else:
            raise ValueError('Invalid support scheme')
def evalKernelDkDh(PrecomputedKernel: PrecomputedNeighborhood, supportScheme: SupportScheme, combined = True):
    if combined:
        if supportScheme == SupportScheme.Scatter:
            return PrecomputedKernel.ddh_W_j
        elif supportScheme == SupportScheme.Gather:
            return PrecomputedKernel.ddh_W_i
        elif supportScheme == SupportScheme.Symmetric:
            return (PrecomputedKernel.ddh_W_i + PrecomputedKernel.ddh_W_j) / 2
        elif supportScheme == SupportScheme.SuperSymmetric:
            return (PrecomputedKernel.ddh_W_i + PrecomputedKernel.ddh_W_j) / 2
        else:
            raise ValueError('Invalid support scheme')
    else:
        if supportScheme == SupportScheme.Scatter:
            return PrecomputedKernel.ddh_W_j, PrecomputedKernel.ddh_W_j
        elif supportScheme == SupportScheme.Gather:
            return PrecomputedKernel.ddh_W_i, PrecomputedKernel.ddh_W_i
        elif supportScheme == SupportScheme.Symmetric:
            ddh_ij = (PrecomputedKernel.ddh_W_i + PrecomputedKernel.ddh_W_j) / 2
            return ddh_ij, ddh_ij
        elif supportScheme == SupportScheme.SuperSymmetric:
            return PrecomputedKernel.ddh_W_i, PrecomputedKernel.ddh_W_j
        else:
            raise ValueError('Invalid support scheme')
        


def correctedKernel(A, B, kernelValues : PrecomputedNeighborhood, supportScheme: SupportScheme = SupportScheme.SuperSymmetric, xji: bool = False):
    fac = -1 if xji else 1
    prod = torch.einsum('...i, ...j -> ...', B, fac * kernelValues.x_ij)
    W_ij = evalKernel(kernelValues, supportScheme=supportScheme, combined=True)
    return A * (1 + prod) * W_ij

def correctedGradientKernel(A, B, gradA, gradB, kernelValues : PrecomputedNeighborhood, supportScheme: SupportScheme = SupportScheme.SuperSymmetric, xji: bool = False):
    fac = -1 if xji else 1
    W_ij = evalKernel(kernelValues, supportScheme=supportScheme, combined=True)
    gradW_ij = fac * evalKernelGradient(kernelValues, supportScheme=supportScheme, combined=True)
    
    dotProd = lambda a, b: torch.einsum('...i, ...j -> ...', a, b)
    
    firstTerm = (A * (1 + dotProd(B, fac * kernelValues.x_ij))).view(-1,1) * gradW_ij
    secondTerm = gradA * ((1 + dotProd(B, fac * kernelValues.x_ij)) * W_ij).view(-1,1)
    thirdTerm = (torch.einsum('...ca,...a -> ...c', gradB, fac * kernelValues.x_ij) + B) * (W_ij * A).view(-1,1)

    return firstTerm + secondTerm + thirdTerm


def correctedKernel_CRK(i, j, A, B, x_ij, W_ij, xji: bool = False):
    index = j if xji else i
    fac = -1 if xji else 1
    prod = torch.einsum('...i, ...j -> ...', B[index], fac * x_ij)
    return A[index] * (1 + prod) * W_ij
def correctedKernelGradient_CRK(i, j, A, B, gradA, gradB, x_ij, W_ij, gradW_ij, xji: bool = False):
    index = j if xji else i
    fac = -1 if xji else 1
    gradW_ij_ = fac * gradW_ij
    
    dotProd = lambda a, b: torch.einsum('...i, ...j -> ...', a[index], b[index])
    
    firstTerm = (A[index] * (1 + dotProd(B[index], fac * x_ij))).view(-1,1) * gradW_ij_
    secondTerm = gradA * ((1 + dotProd(B[index], fac * x_ij)) * W_ij).view(-1,1)
    thirdTerm = (torch.einsum('...ca,...a -> ...c', gradB[index], fac * x_ij) + B[index]) * (W_ij * A[index]).view(-1,1)

    return firstTerm + secondTerm + thirdTerm