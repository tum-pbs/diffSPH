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

    from diffSPH.dataLoaderUtils.fallback import CompactHashMap, radiusSearch, PointCloud, DomainDescription, SparseCOO, SparseCSR
    
from dataclasses import dataclass
from typing import List

@torch.jit.script
@dataclass(slots=True)
class AugmentedDomainDescription:
    """
    A named tuple containing the minimum and maximum domain values.
    """
    min: torch.Tensor
    max: torch.Tensor
    periodic: torch.Tensor
    dim: int

    angles: List[float] = None,
    device: torch.device = None,
    dtype: torch.dtype = None

    def __ne__(self, other: 'AugmentedDomainDescription') -> bool:
        return not self.__eq__(other)
    
    def __init__(self, min, max, periodic, dim, angles=None, device=None, dtype=None):
        self.min = min
        self.max = max
        self.periodic = periodic
        self.dim = dim
        self.angles = angles if angles is not None else [0.0] * (dim - 1)
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32

from typing import Union
def domainToAugmented(dom : Union[DomainDescription, AugmentedDomainDescription]) -> AugmentedDomainDescription:
    if isinstance(dom, AugmentedDomainDescription):
        return dom
    elif isinstance(dom, DomainDescription):
        return AugmentedDomainDescription(
            min = dom.min,
            max = dom.max,
            periodic = dom.periodic,
            dim = dom.dim,
            angles = [0.0] * (dom.dim - 1),
            device = dom.min.device,
            dtype = dom.min.dtype
        )
    else:
        raise TypeError(f"Unsupported domain type: {type(dom)}")

@dataclass(slots=True)
class SparseNeighborhood:
    row: torch.Tensor
    col: torch.Tensor
    
    numRows: int
    numCols: int
    
    points_a: 'PointCloud'
    points_b: 'PointCloud'
    
    domain: 'AugmentedDomainDescription'

@dataclass(slots=True)
class SparseBatchedNeighborhood:
    row: torch.Tensor
    col: torch.Tensor
    batch: torch.Tensor

    numRows: List[int]
    numCols: List[int]
    
    points_a: List['PointCloud']
    points_b: List['PointCloud']
    
    domain: List['AugmentedDomainDescription']

import numpy as np
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
    
    if isinstance(sparseNeighborhood, SparseBatchedNeighborhood):
        return SparseBatchedNeighborhood(
            row = i_filtered,
            col = j_filtered,
            batch = sparseNeighborhood.batch[mask],
            numRows = sparseNeighborhood.numRows,
            numCols = sparseNeighborhood.numCols,
            points_a = sparseNeighborhood.points_a,
            points_b = sparseNeighborhood.points_b,
            domain = sparseNeighborhood.domain
        )

    return SparseNeighborhood(
        row = i_filtered,
        col = j_filtered,
        numRows = sparseNeighborhood.numRows,
        numCols = sparseNeighborhood.numCols,
        
        points_a = sparseNeighborhood.points_a,
        points_b = sparseNeighborhood.points_b,
        
        domain = sparseNeighborhood.domain,
    )

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



from diffSPH.dataLoaderUtils.util import buildRotationMatrix

def neighborSearch(state, domain, config):
    points = PointCloud(state.positions, state.supports)
    if isinstance(domain, AugmentedDomainDescription):
        # print(f'Using angles: {domain.angles}, rotation matrix: {buildRotationMatrix(torch.tensor(domain.angles, device=domain.device, dtype=domain.dtype), domain.dim, device=domain.device, dtype=domain.dtype)}')
        rotationMatrix = buildRotationMatrix(torch.tensor(domain.angles, device=domain.device, dtype=domain.dtype), domain.dim, device=domain.device, dtype=domain.dtype)
        invRotationMatrix = torch.linalg.inv(rotationMatrix)
        # print(f'Using angles: {domain.angles}, rotation matrix: {rotationMatrix}')
        points = PointCloud(
            positions=torch.einsum('ij, ni->nj', invRotationMatrix, points.positions),
            supports=points.supports
        )


    fullAdjacency = radiusSearch(
        points,
        points,
        domain = DomainDescription(
            domain.min, 
            domain.max, 
            domain.periodic, 
            points.positions.shape[0]
        ),
        mode = 'superSymmetric',
        returnStructure=False,
        algorithm='compact'
    )

    # print(f'Domain: {domain.min} - {domain.max}, periodic: {domain.periodic}, dim: {domain.dim}')

    sparseNeighborhood = SparseNeighborhood(
        row = fullAdjacency.row,
        col = fullAdjacency.col,
        numRows = fullAdjacency.numRows,
        numCols = fullAdjacency.numCols,
        points_a = PointCloud(state.positions, state.supports),
        points_b = PointCloud(state.positions, state.supports),
        domain = domain
    )
    indices, indicesSorted, numFluidToFluid, numBoundaryToFluid, numFluidToBoundary, numBoundaryToBoundary, numFluidToGhost = evalEdgeKinds(state.kinds, sparseNeighborhood.row, sparseNeighborhood.col)

    numNormal = numFluidToFluid + numBoundaryToFluid + numFluidToBoundary + numBoundaryToBoundary #+ numFluidToGhost

    sortedNeighbors = SparseNeighborhood(sparseNeighborhood.row[indicesSorted], sparseNeighborhood.col[indicesSorted], numRows = sparseNeighborhood.numRows, numCols = sparseNeighborhood.numCols, 
                                        points_a = sparseNeighborhood.points_a, points_b = sparseNeighborhood.points_b, domain = sparseNeighborhood.domain)



    normalIndices = (0, numNormal)
    fluidIndices = (0, numFluidToFluid)
    boundaryToFluidIndices = (fluidIndices[-1], fluidIndices[-1] + numBoundaryToFluid)
    fluidToBoundaryIndices = (boundaryToFluidIndices[-1], boundaryToFluidIndices[-1] + numFluidToBoundary)
    boundaryToBoundaryIndices = (fluidToBoundaryIndices[-1], fluidToBoundaryIndices[-1] + numBoundaryToBoundary)
    fluidToGhostIndices = (boundaryToBoundaryIndices[-1], boundaryToBoundaryIndices[-1] + numFluidToGhost)

    return sortedNeighbors


def coo_to_csr(coo: SparseCOO, isSorted: bool = False) -> SparseCSR:
    if not isSorted:
        neigh_order = torch.argsort(coo.row)
        row = coo.row[neigh_order]
        col = coo.col[neigh_order]
    else:
        row = coo.row
        col = coo.col

    # if isinstance(coo : SparseBatchedNeighborhood)

    # print(f'Converting COO To CSR for matrix of shape {coo.numRows} x {coo.numCols}')
    # print(f'Number of Entries: {row.shape[0]}/{col.shape[0]}')
    jj, nit = torch.unique(row, return_counts=True)
    if isinstance(coo.numRows, int):
        nj = torch.zeros(coo.numRows, dtype=coo.row.dtype, device=coo.row.device)
    else:
        nj = torch.zeros(np.sum(coo.numRows), dtype=coo.row.dtype, device=coo.row.device)
    nj[jj] = nit
    # print(f'Number of neighbors: {nj} ({nj.sum()} total, shape {nj.shape})')

    if isinstance(coo.numRows, int):
        indptr = torch.zeros(coo.numRows + 1, dtype=torch.int64, device=coo.row.device)
    else:
        indptr = torch.zeros(np.sum(coo.numRows) + 1, dtype=torch.int64, device=coo.row.device)

    indptr[1:] = torch.cumsum(nj, 0)
    # print(f'Row pointers: {indptr} ({indptr.shape})')
    indptr = indptr.int()
    indices = col
    rowEntries = nj

    return SparseCSR(indices, indptr, rowEntries, coo.numRows, coo.numCols)


from typing import Optional
@torch.jit.script
def mod(x, min : float, max : float):
    h = max - min
    return ((x + h / 2.0) - torch.floor((x + h / 2.0) / h) * h) - h / 2.0
def evalDistanceTensor_(
        row: torch.Tensor, col: torch.Tensor,
        minD: torch.Tensor, maxD: torch.Tensor, periodicity: torch.Tensor,

        positions_a: torch.Tensor, positions_b: torch.Tensor,
        support_a: torch.Tensor, support_b: torch.Tensor,
        angles: Optional[torch.Tensor] = None,):
    
    # print(
    #     f'row: {row.shape}, col: {col.shape}, minD: {minD.shape}, maxD: {maxD.shape}, periodicity: {periodicity.shape}, positions_a: {positions_a.shape}, positions_b: {positions_b.shape}, support_a: {support_a.shape}, support_b: {support_b.shape}, angles: {angles.shape if angles is not None else None}'
    # )

    # print('evalDistanceTensor Begin')
    pos_ai = positions_a[row]
    pos_bi = positions_b[col]
    h_i = support_a[row]
    h_j = support_b[col] 
    if angles is not None:
        rotMat = buildRotationMatrix(angles, positions_a.shape[-1], device=positions_a.device, dtype=positions_a.dtype)

        pos_ai = torch.matmul(pos_ai, rotMat.T)
        pos_bi = torch.matmul(pos_bi, rotMat.T)


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

    if angles is not None:
        xij_ = torch.matmul(xij_, rotMat)
        # print(f'Using angles: {angles}, rotation matrix: {rotMat}')

    distance = torch.linalg.norm(xij_, dim = -1)
    # if normalize:
    #     xij = torch.nn.functional.normalize(xij, dim = -1)
    #     rij = distance / hij
    # else:
    rij = distance
    # print('evalDistanceTensor End')
    return rij, xij_, h_i, h_j

def evalDistanceTensor(neighborhood):
    if isinstance(neighborhood, SparseNeighborhood) and not isinstance(neighborhood.domain, list):
        # print('evalDistanceTensor Begin')
        # print(f'Evaluating distance tensor for neighborhood with {neighborhood.row.shape[0]} entries')
        return evalDistanceTensor_(
            neighborhood.row, neighborhood.col,
            neighborhood.domain.min, neighborhood.domain.max,
            neighborhood.domain.periodic,
            neighborhood.points_a.positions, neighborhood.points_b.positions,
            neighborhood.points_a.supports, neighborhood.points_b.supports,
            torch.tensor(neighborhood.domain.angles, dtype=neighborhood.points_a.positions.dtype, device=neighborhood.points_a.positions.device) if isinstance(neighborhood.domain, AugmentedDomainDescription) else None
        )
    elif isinstance(neighborhood, SparseBatchedNeighborhood) or isinstance(neighborhood.domain, list):
        # print(f'Evaluating distance tensor for batched neighborhood with {len(neighborhood.points_a)} batches')
        rijs, xijs, his, hjs = [], [], [], []
        offset = 0
        for i in range(len(neighborhood.points_a)):
            rij, xij, hi, hj = evalDistanceTensor_(
                neighborhood.row[neighborhood.batch == i] - offset,
                neighborhood.col[neighborhood.batch == i] - offset,
                neighborhood.domain[i].min, neighborhood.domain[i].max,
                neighborhood.domain[i].periodic,
                neighborhood.points_a[i].positions, neighborhood.points_b[i].positions,
                neighborhood.points_a[i].supports, neighborhood.points_b[i].supports,
                torch.tensor(neighborhood.domain[i].angles, dtype=neighborhood.points_a[i].positions.dtype, device=neighborhood.points_a[i].positions.device) if isinstance(neighborhood.domain[i], AugmentedDomainDescription) else None
            )
            offset += neighborhood.numRows[i]
            rijs.append(rij)
            xijs.append(xij)
            his.append(hi)
            hjs.append(hj)
        rijs = torch.cat(rijs, dim=0)
        xijs = torch.cat(xijs, dim=0)
        his = torch.cat(his, dim=0)
        hjs = torch.cat(hjs, dim=0)
        return rijs, xijs, his, hjs
        # # print('evalDistanceTensor Begin')
        # return evalDistanceTensor_(
        #     neighborhood.row, neighborhood.col,
        #     neighborhood.domain.min, neighborhood.domain.max,
        #     neighborhood.domain.periodic,
        #     neighborhood.points_a.positions, neighborhood.points_b.positions,
        #     neighborhood.points_a.supports, neighborhood.points_b.supports,
        #     torch.tensor(neighborhood.domain.angles, dtype = neighborhood.points_a.positions.dtype, device = neighborhood.points_a.positions.device) if isinstance(neighborhood.domain, AugmentedDomainDescription) else None
        # )
    else:
        raise TypeError(f'Unsupported neighborhood type: {type(neighborhood)}. Expected SparseNeighborhood or SparseBatchedNeighborhood.')


from diffSPH.dataLoaderUtils.state import PointCloudWithKinds

def batchNeighborsearch(mergedState, domains, configs):
    neighborhoods = []
    if len(domains) != len(configs):
        raise ValueError("The number of domains and configs must match.")
    if len(domains) == 1:
        return neighborSearch(mergedState, domains[0], configs[0]), None
    for ib, (domain, config) in enumerate(zip(domains, configs)):
        cloud = PointCloudWithKinds(positions=mergedState.positions[mergedState.batches == ib],
                                     supports=mergedState.supports[mergedState.batches == ib],
                                     kinds=mergedState.kinds[mergedState.batches == ib])
        neighborhood = neighborSearch(cloud, domain, config)
        neighborhoods.append(neighborhood)

    rows = []
    cols = []
    offset = 0
    for i, neighborhood in enumerate(neighborhoods):
        rows.append(neighborhood.row + offset)
        cols.append(neighborhood.col + offset)
        offset += neighborhood.numRows
    mergedNeighborhood = SparseBatchedNeighborhood(
        row = torch.cat(rows, dim=0),
        col = torch.cat(cols, dim=0),
        batch= mergedState.batches[torch.cat(rows, dim=0)],

        numRows = [n.numRows for n in neighborhoods],
        numCols = [n.numCols for n in neighborhoods],

        points_a = [n.points_a for n in neighborhoods],
        points_b = [n.points_b for n in neighborhoods],

        domain = domains
    )
    return mergedNeighborhood, neighborhoods

    # return neighborhoods