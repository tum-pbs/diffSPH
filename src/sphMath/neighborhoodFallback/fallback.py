import torch
from typing import Optional, Tuple, Union, List
from typing import NamedTuple
# from sphMath.util import PointCloud, DomainDescription, SparseCOO

from dataclasses import dataclass
@torch.jit.script
@dataclass(slots=True)
class DomainDescription:
    """
    A named tuple containing the minimum and maximum domain values.
    """
    min: torch.Tensor
    max: torch.Tensor
    periodic: torch.Tensor
    dim: int

    def __ne__(self, other: 'DomainDescription') -> bool:
        return not self.__eq__(other)
    
@torch.jit.script
@dataclass#(slots=True)
class PointCloud:
    """
    A named tuple containing the positions of the particles and the number of particles.
    """
    positions: torch.Tensor
    supports: torch.Tensor

    def __ne__(self, other: 'PointCloud') -> bool:
        return not self.__eq__(other)


    
@torch.jit.script
@dataclass#(slots=True)
class SparseCOO:
    """
    A named tuple containing the neighbor list in coo format and the number of neighbors for each particle.
    """
    row: torch.Tensor
    col: torch.Tensor

    numRows: int
    numCols: int


@torch.jit.script    
@dataclass#(slots=True)
class SparseCSR:
    """
    A named tuple containing the neighbor list in csr format and the number of neighbors for each particle.
    """
    indices: torch.Tensor
    indptr: torch.Tensor

    rowEntries: torch.Tensor

    numRows: int
    numCols: int

def coo_to_csr(coo: SparseCOO, isSorted: bool = False) -> SparseCSR:
    if not isSorted:
        neigh_order = torch.argsort(coo.row)
        row = coo.row[neigh_order]
        col = coo.col[neigh_order]
    else:
        row = coo.row
        col = coo.col

    # print(f'Converting COO To CSR for matrix of shape {coo.numRows} x {coo.numCols}')
    # print(f'Number of Entries: {row.shape[0]}/{col.shape[0]}')
    jj, nit = torch.unique(row, return_counts=True)
    nj = torch.zeros(coo.numRows, dtype=coo.row.dtype, device=coo.row.device)
    nj[jj] = nit
    # print(f'Number of neighbors: {nj} ({nj.sum()} total, shape {nj.shape})')

    indptr = torch.zeros(coo.numRows + 1, dtype=torch.int64, device=coo.row.device)
    indptr[1:] = torch.cumsum(nj, 0)
    # print(f'Row pointers: {indptr} ({indptr.shape})')
    indptr = indptr.int()
    indices = col
    rowEntries = nj

    return SparseCSR(indices, indptr, rowEntries, coo.numRows, coo.numCols)

def csr_to_coo(csr: SparseCSR) -> SparseCOO:
    row = torch.zeros(csr.rowEntries.sum(), dtype=csr.indices.dtype, device=csr.indices.device)
    col = torch.zeros_like(row)
    rowStart = 0
    for i in range(csr.numRows):
        rowEnd = rowStart + csr.rowEntries[i]
        row[rowStart:rowEnd] = i
        col[rowStart:rowEnd] = csr.indices[csr.indptr[i]:csr.indptr[i+1]]
        rowStart = rowEnd
    return SparseCOO(row, col, csr.numRows, csr.numCols)




class CompactHashMap(NamedTuple):
    sortedPositions : torch.Tensor
    referencePositions : torch.Tensor


    hashTable : torch.Tensor
    hashMapLength : int

    sortedCellTable : torch.Tensor
    numCells : int

    qMin : torch.Tensor
    qMax : torch.Tensor
    minD : torch.Tensor
    maxD : torch.Tensor

    sortIndex : torch.Tensor
    hCell : torch.Tensor
    periodicity : torch.Tensor
    searchRadius : int

    sortedSupports : Optional[torch.Tensor] = None
    referenceSupports : Optional[torch.Tensor]  = None
    fixedSupport : Optional[float] = None

def getPeriodicPointCloud(
        queryPointCloud: PointCloud,
        domain: Optional[DomainDescription] = None,
):
    if domain is None:
        return queryPointCloud
    else:
        domainMin = domain.min
        domainMax = domain.max
        periodic = domain.periodic
        # if isinstance(periodic, bool):
            # periodic = [periodic] * queryPointCloud.positions.shape[1]
        return PointCloud(torch.stack([queryPointCloud.positions[:,i] if not periodic_i else torch.remainder(queryPointCloud.positions[:,i] - domainMin[i], domainMax[i] - domainMin[i]) + domainMin[i] for i, periodic_i in enumerate(periodic)], dim = 1), queryPointCloud.supports)

@torch.jit.script
def mod(x, min : float, max : float):
    h = max - min
    return ((x + h / 2.0) - torch.floor((x + h / 2.0) / h) * h) - h / 2.0
# def moduloDistance(xij, periodic, min, max):
#     return torch.stack([xij[:,i] if not periodic else mod(xij[:,i], min[i], max[i]) for i, periodic in enumerate(periodicity)], dim = -1)


@torch.jit.script
def radiusNaive(x, y, hx, hy, periodic : Optional[torch.Tensor] = None, minDomain = None, maxDomain = None, mode : str = 'gather'):
    periodicity = torch.tensor([False] * x.shape[1], dtype = torch.bool, device = x.device) if periodic is None else periodic
    
    pos_x = torch.stack([x[:,i] if not periodic_i else torch.remainder(x[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)
    pos_y = torch.stack([y[:,i] if not periodic_i else torch.remainder(y[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)
    
    distanceMatrices = torch.stack([pos_x[:,i] - pos_y[:,i,None] if not periodic_i else mod(pos_x[:,i] - pos_y[:,i,None], minDomain[i], maxDomain[i]) for i, periodic_i in enumerate(periodicity)], dim = -1)
    distanceMatrix = torch.sqrt(torch.sum(distanceMatrices**2, dim = -1))
    
    indexI, indexJ = torch.meshgrid(torch.arange(x.shape[0]).to(x.device), torch.arange(y.shape[0]).to(y.device), indexing = 'xy')
    if mode == 'gather':        
        gatherMatrix = hx.repeat(y.shape[0],1)
        adjacencyDense = distanceMatrix <= gatherMatrix
        # supports = gatherMatrix[adjacencyDense]
    elif mode == 'scatter':        
        scatterMatrix = hy.repeat(x.shape[0],1).mT
        adjacencyDense = distanceMatrix <= scatterMatrix
        # supports = scatterMatrix[adjacencyDense]
    elif mode == 'symmetric':
        symmetricMatrix = (hx + hy[:,None]) / 2
        adjacencyDense = distanceMatrix <= symmetricMatrix
    elif mode == 'superSymmetric':
        symmetricMatrix = torch.max(hx, hy)
        adjacencyDense = distanceMatrix <= symmetricMatrix
    else:
        raise ValueError('mode must be one of gather, scatter, symmetric, or superSymmetric')
        # supports = symmetricMatrix[adjacencyDense]
    
    ii = indexI[adjacencyDense]
    jj = indexJ[adjacencyDense]

    return ii, jj#, distanceMatrix[adjacencyDense], distanceMatrices[adjacencyDense], supports


def radiusSearch( 
        queryPointCloud: PointCloud,
        referencePointCloud: Optional[PointCloud],
        supportOverride : Optional[float] = None,

        mode : str = 'gather',
        domain : Optional[DomainDescription] = None,
        hashMapLength = 4096,
        algorithm: str = 'naive',
        verbose: bool = False,
        format: str = 'coo',
        returnStructure : bool = False,
        useDenseMLM: bool = False,
        hashMapLengthAlgorithm: str = 'primes'
        ):
    with torch.no_grad():
        # print(f'Using {algorithm} algorithm for radius search', flush = True)
        numQueryPoints = queryPointCloud.positions.shape[0]
        if referencePointCloud is not None:
            numReferencePoints = referencePointCloud.positions.shape[0]
        else:
            numReferencePoints = numQueryPoints
        dimensionality = queryPointCloud.positions.shape[1]
        if referencePointCloud is None:
            referencePointCloud = queryPointCloud
        support = (queryPointCloud.supports if queryPointCloud.supports is not None else None, referencePointCloud.supports if referencePointCloud.supports is not None else None)
        domainInformation = None
        if domain is not None:
            domainInformation = DomainDescription(domain.min, domain.max, domain.periodic, queryPointCloud.positions.shape[0])
        else:
            domainInformation = DomainDescription(
                torch.tensor([0.0] * dimensionality, device = queryPointCloud.positions.device), 
                torch.tensor([1.0] * dimensionality, device = queryPointCloud.positions.device), 
                torch.tensor([False] * dimensionality, dtype = torch.bool, device = queryPointCloud.positions.device), queryPointCloud.positions.shape[0])
        # if torch.any(periodicTensor):
            # if algorithm == 'cluster':
                # raise ValueError(f'algorithm = {algorithm} not supported for periodic search')
        
        x = getPeriodicPointCloud(queryPointCloud, domain)
        y = getPeriodicPointCloud(referencePointCloud, domain)
        i, j =  radiusNaive(x.positions, y.positions, x.supports, y.supports, domainInformation.periodic, domainInformation.min, domainInformation.max, mode)

        coo = SparseCOO(
            row=i,
            col=j,
            numRows=numQueryPoints,
            numCols=numReferencePoints
        )

        return coo, None
    


# @torch.jit.script
# @dataclass(slots = True)
# class SparseNeighborhood:
#     row: torch.Tensor
#     col: torch.Tensor
    
#     numRows: int
#     numCols: int
    
#     points_a: PointCloud
#     points_b: PointCloud
    
#     domain: DomainDescription
#     # values: Optional[PrecomputedNeighborhood] = None
