from typing import Optional
import numpy as np
import torch

@torch.jit.script
def compute_h(qMin, qMax, referenceSupport): 
    """
    Compute the smoothing length (h) based on the given minimum and maximum coordinates (qMin and qMax)
    and the reference support value. The smoothing length is used for grid operations and is determined
    by dividing the domain into cells based on the reference support value such that h > referenceSupport.

    Args:
        qMin (torch.Tensor): The minimum coordinates.
        qMax (torch.Tensor): The maximum coordinates.
        referenceSupport (float): The reference support value.

    Returns:
        torch.Tensor: The computed smoothing length (h).
    """
    qExtent = qMax - qMin
    numCells = torch.floor(qExtent / referenceSupport)
    h = qExtent / numCells
    return torch.max(h)
@torch.jit.script
def getDomainExtents(positions, minDomain : Optional[torch.Tensor], maxDomain : Optional[torch.Tensor]):
    """
    Calculates the domain extents based on the given positions and optional minimum and maximum domain values.

    Args:
        positions (torch.Tensor): The positions of the particles.
        minDomain (Optional[torch.Tensor]): Optional minimum domain values.
        maxDomain (Optional[torch.Tensor]): Optional maximum domain values.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the minimum and maximum domain extents.
    """
    if minDomain is not None and isinstance(minDomain, list):
        minD = torch.tensor(minDomain).to(positions.device).type(positions.dtype)
    elif minDomain is not None:
        minD = minDomain
    else:
        minD = torch.min(positions, dim = 0)[0]
    if maxDomain is not None and isinstance(minDomain, list):
        maxD = torch.tensor(maxDomain).to(positions.device).type(positions.dtype)
    elif maxDomain is not None:
        maxD = maxDomain
    else:
        maxD = torch.max(positions, dim = 0)[0]
    return minD, maxD



@torch.jit.script
def hashCellIndices(cellIndices, hashMapLength):
    """
    Hashes the cell indices using a hash function.

    Args:
        cellIndices (torch.Tensor): Tensor containing the cell indices.
        hashMapLength (int): Length of the hash map.

    Returns:
        torch.Tensor: Hashed cell indices.

    Raises:
        ValueError: If the dimension of cellIndices is not 1, 2, or 3.
    """
    primes = [73856093, 19349663, 83492791] # arbitrary primes but they should be large and different and these have been used in literature before
    if cellIndices.shape[1] == 1:
        return cellIndices[:,0] % hashMapLength
    elif cellIndices.shape[1]  == 2:
        return (cellIndices[:,0] * primes[0] + cellIndices[:,1] * primes[1]) % hashMapLength
    elif cellIndices.shape[1]  == 3:
        return (cellIndices[:,0] * primes[0] + cellIndices[:,1] * primes[1] + cellIndices[:,2] * primes[2]) % hashMapLength
    else: 
        raise ValueError('Only 1D, 2D and 3D supported')
    
@torch.jit.script
def linearIndexing(cellIndices, cellCounts):
    """
    Compute the linear index based on the given cell indices and cell counts.

    Args:
        cellIndices (torch.Tensor): Tensor containing the cell indices.
        cellCounts (torch.Tensor): Tensor containing the cell counts.

    Returns:
        torch.Tensor: Tensor containing the linear indices.
    """
    dim = cellIndices.shape[1]
    linearIndex = torch.zeros(cellIndices.shape[0], dtype=cellIndices.dtype, device=cellIndices.device)
    product = 1
    for i in range(dim):
        linearIndex += cellIndices[:, i] * product
        product = product * cellCounts[i].item()
    return linearIndex

@torch.jit.script
def queryCell(cellIndex, hashTable, hashMapLength, numCells, cellTable):
    """
    Queries a cell in the hash table and returns the indices of particles in that cell.

    Args:
        cellIndex (Tensor): The index of the cell to query.
        hashTable (Tensor): The hash table containing cell information.
        hashMapLength (int): The length of the hash map.
        numCells: The number of cells in the hash table.
        cellTable: The table containing cell information.

    Returns:
        Tensor: The indices of particles in the queried cell. If the cell is empty, returns an empty tensor.
    """

    linearIndex = linearIndexing(cellIndex.view(-1,cellIndex.shape[0]), numCells)# * cellIndex[1]
    hashedIndex = hashCellIndices(cellIndex.view(-1,cellIndex.shape[0]), hashMapLength)

    tableEntry = hashTable[hashedIndex,:]
    hBegin = tableEntry[:,0][0]
    hLength = tableEntry[:,1][0]

    if hBegin != -1:
        cell = cellTable[hBegin:hBegin + hLength]
        for c in range(cell.shape[0]):
            if cell[c,0] == linearIndex:
                cBegin = cell[c,1]
                cLength = cell[c,2]
                particlesInCell = torch.arange(cBegin, cBegin + cLength, device = hashTable.device, dtype = hashTable.dtype)
                return particlesInCell

    return torch.empty(0, dtype = hashTable.dtype, device = hashTable.device)



@torch.jit.script
def iPower(x: int, n: int):
    """
    Calculates the power of an integer.

    Args:
        x (int): The base number.
        n (int): The exponent.

    Returns:
        int: The result of x raised to the power of n.
    """
    res : int = 1
    for i in range(n):
        res *= x
    return res

@torch.jit.script
def getOffsets(searchRange: int, dim: int):
    """
    Generates a tensor of offsets based on the search range and dimension.

    Args:
        searchRange (int): The range of values to generate offsets from.
        dim (int): The dimension of the offsets tensor.

    Returns:
        torch.Tensor: A tensor of offsets with shape [iPower(1 + 2 * searchRange, dim), dim].
    """
    offsets = torch.zeros([iPower(1 + 2 * searchRange, dim), dim], dtype=torch.int32)
    for d in range(dim):
        itr = -searchRange
        ctr = 0
        for o in range(offsets.size(0)):
            c = o % pow(1 + 2 * searchRange, d)
            if c == 0 and ctr > 0:
                itr += 1
            if itr > searchRange:
                itr = -searchRange
            offsets[o][dim - d - 1] = itr
            ctr += 1
    return offsets
