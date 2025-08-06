import torch

def plotDomain(axis, domain):
    axis.set_xlim(domain.min[0].cpu(), domain.max[0].cpu())
    axis.set_ylim(domain.min[1].cpu(), domain.max[1].cpu())
    axis.set_aspect('equal')

    axis.plot([domain.min[0].cpu(), domain.max[0].cpu()], [domain.min[1].cpu(), domain.min[1].cpu()], 'k-')
    axis.plot([domain.min[0].cpu(), domain.max[0].cpu()], [domain.max[1].cpu(), domain.max[1].cpu()], 'k-')
    axis.plot([domain.min[0].cpu(), domain.min[0].cpu()], [domain.min[1].cpu(), domain.max[1].cpu()], 'k-')
    axis.plot([domain.max[0].cpu(), domain.max[0].cpu()], [domain.min[1].cpu(), domain.max[1].cpu()], 'k-')

def splitDomain(split_x, split_y, domain, nx):
    lx = domain.max[0] - domain.min[0]
    dx = lx / nx
    ly = domain.max[1] - domain.min[1]
    dy = ly / nx

    splitx_ = (split_x - domain.min[0]) / dx
    splity_ = (split_y - domain.min[1]) / dy

    rounded_splitx_ = torch.round(splitx_).long()
    rounded_splity_ = torch.round(splity_).long()

    splitx = rounded_splitx_ * dx + domain.min[0]
    splity = rounded_splity_ * dy + domain.min[1]

    return splitx.cpu().item(), splity.cpu().item()

def maskParticles(particles, split_x, split_y, domain, nx):
    if isinstance(split_x, list):
        splitXLower, splitYLower = splitDomain(split_x[0], split_y[0], domain, nx)
        splitXUpper, splitYUpper = splitDomain(split_x[1], split_y[1], domain, nx)

        A_1 = (particles.positions[:, 0] < splitXLower) & (particles.positions[:, 1] < splitYLower)
        A_2 = (particles.positions[:, 0] >= splitXUpper) & (particles.positions[:, 1] < splitYLower)
        A_3 = (particles.positions[:, 0] < splitXLower) & (particles.positions[:, 1] >= splitYUpper)
        A_4 = (particles.positions[:, 0] >= splitXUpper) & (particles.positions[:, 1] >= splitYUpper)
        maskA = torch.logical_or(torch.logical_or(A_1, A_2), torch.logical_or(A_3, A_4))

        B_1 = (particles.positions[:, 0] < splitXLower) & (particles.positions[:, 1] >= splitYLower)
        B_2 = (particles.positions[:, 0] < splitXLower) & (particles.positions[:, 1] < splitYUpper)
        B_3 = (particles.positions[:, 0] >= splitXUpper) & (particles.positions[:, 1] >= splitYLower)
        B_4 = (particles.positions[:, 0] >= splitXUpper) & (particles.positions[:, 1] < splitYUpper)
        maskB = torch.logical_or(torch.logical_and(B_1, B_2), torch.logical_and(B_3, B_4))

        C_1 = (particles.positions[:, 0] >= splitXLower) & (particles.positions[:, 1] < splitYLower)
        C_2 = (particles.positions[:, 0] < splitXUpper) & (particles.positions[:, 1] < splitYLower)
        C_3 = (particles.positions[:, 0] >= splitXLower) & (particles.positions[:, 1] >= splitYUpper)
        C_4 = (particles.positions[:, 0] < splitXUpper) & (particles.positions[:, 1] >= splitYUpper)
        maskC = torch.logical_or(torch.logical_and(C_1, C_2), torch.logical_and(C_3, C_4))

        mask = torch.ones_like(particles.positions[:, 0], dtype=torch.int64)*3
        mask[maskA] = 0
        mask[maskB] = 2
        mask[maskC] = 1

        return mask, [splitXLower, splitXUpper], [splitYLower, splitYUpper]

    else:

        splitx, splity = splitDomain(split_x, split_y, domain, nx)

        maskA = (particles.positions[:, 0] < splitx) & (particles.positions[:, 1] < splity)
        maskB = (particles.positions[:, 0] >= splitx) & (particles.positions[:, 1] < splity)
        maskC = (particles.positions[:, 0] < splitx) & (particles.positions[:, 1] >= splity)
        maskD = (particles.positions[:, 0] >= splitx) & (particles.positions[:, 1] >= splity)

        mask = torch.zeros_like(particles.positions[:, 0], dtype=torch.int64)
        mask[maskA] = 0
        mask[maskB] = 1
        mask[maskC] = 2
        mask[maskD] = 3

        return mask, splitx, splity

def maskParticlesSymmetric(particles, split_x, split_y, domain, nx):
    return maskParticles(particles, [split_x, domain.max[0] - split_x], [split_y, domain.max[1] - split_y], domain, nx)


from diffSPH.sampling import buildDomainDescription, sampleRegularParticles
from diffSPH.util import ParticleSet 
import numpy as np

def sampleRegions(domain, nx, targetNeighbors, splitX, splitY, jitter = 0.0):
    print(f'Sampling regions with nx = {nx}, splitX = {splitX}, splitY = {splitY}, jitter = {jitter}')
    if isinstance(nx, int):
        particles = sampleRegularParticles(nx, domain, targetNeighbors, jitter = jitter)

        # split_x, split_y = splitDomain(splitX, splitY, domain, nx)
        mask, split_x, split_y = maskParticles(particles, splitX, splitY, domain, nx)

        particlesA = ParticleSet(
            positions = particles.positions[mask == 0],
            supports = particles.supports[mask == 0],
            masses = particles.masses[mask == 0],
            densities = particles.densities[mask == 0]
        ) 
        particlesB = ParticleSet(
            positions = particles.positions[mask == 1],
            supports = particles.supports[mask == 1],
            masses = particles.masses[mask == 1],
            densities = particles.densities[mask == 1]
        ) 
        particlesC = ParticleSet(
            positions = particles.positions[mask == 2],
            supports = particles.supports[mask == 2],
            masses = particles.masses[mask == 2],
            densities = particles.densities[mask == 2]
        ) 
        particlesD = ParticleSet(
            positions = particles.positions[mask == 3],
            supports = particles.supports[mask == 3],
            masses = particles.masses[mask == 3],
            densities = particles.densities[mask == 3]
        )

        return particlesA, particlesB, particlesC, particlesD, split_x, split_y
    else:
        particlesA = sampleRegularParticles(nx[0], domain, targetNeighbors, jitter = jitter)
        particlesB = sampleRegularParticles(nx[1], domain, targetNeighbors, jitter = jitter)
        particlesC = sampleRegularParticles(nx[2], domain, targetNeighbors, jitter = jitter)
        particlesD = sampleRegularParticles(nx[3], domain, targetNeighbors, jitter = jitter)

        # splitx, splity = splitDomain(splitX, splitY, domain, np.min(nx))

        maskA, splitx, splity = maskParticles(particlesA, splitX, splitY, domain, np.min(nx))
        maskB, *_ = maskParticles(particlesB, splitX, splitY, domain, np.min(nx))
        maskC, *_ = maskParticles(particlesC, splitX, splitY, domain, np.min(nx))
        maskD, *_ = maskParticles(particlesD, splitX, splitY, domain, np.min(nx))

        particlesA = ParticleSet(
            positions = particlesA.positions[maskA == 0],
            supports = particlesA.supports[maskA == 0],
            masses = particlesA.masses[maskA == 0],
            densities = particlesA.densities[maskA == 0]
        ) 
        particlesB = ParticleSet(
            positions = particlesB.positions[maskB == 1],
            supports = particlesB.supports[maskB == 1],
            masses = particlesB.masses[maskB == 1],
            densities = particlesB.densities[maskB == 1]
        ) 
        particlesC = ParticleSet(
            positions = particlesC.positions[maskC == 2],
            supports = particlesC.supports[maskC == 2],
            masses = particlesC.masses[maskC == 2],
            densities = particlesC.densities[maskC == 2]
        )
        particlesD = ParticleSet(
            positions = particlesD.positions[maskD == 3],
            supports = particlesD.supports[maskD == 3],
            masses = particlesD.masses[maskD == 3],
            densities = particlesD.densities[maskD == 3]
        )

        return particlesA, particlesB, particlesC, particlesD, splitx, splity
    
def sampleRegionsSymmetric(domain, nx, targetNeighbors, splitX, splitY, jitter = 0.0):
    lx = domain.max[0].item() - domain.min[0].item()
    ly = domain.max[1].item() - domain.min[1].item()

    minX = domain.min[0].item()
    minY = domain.min[1].item()
    maxX = domain.max[0].item()
    maxY = domain.max[1].item()

    splitXUpper = minX + (maxX - minX) * (1 - (splitX - minX) / (maxX - minX))
    splitYUpper = minY + (maxY - minY) * (1 - (splitY - minY) / (maxY - minY))

    return sampleRegions(domain, nx, targetNeighbors, [splitX, splitXUpper], [splitY, splitYUpper], jitter = jitter)


def mergeParticles(particles_list):
    positions = torch.cat([p.positions for p in particles_list], dim=0)
    supports = torch.cat([p.supports for p in particles_list], dim=0)
    masses = torch.cat([p.masses for p in particles_list], dim=0)
    densities = torch.cat([p.densities for p in particles_list], dim=0)
    index = torch.cat([torch.ones_like(p.masses) * i for i, p in enumerate(particles_list)], dim=0)

    return ParticleSet(
        positions = positions,
        supports = supports,
        masses = masses,
        densities = densities
    ), index


def recreateUIDs(particles, domain, verbose = False):
    """
    Recreate unique UIDs for particles based on their positions and the domain.
    """
    hMin = particles.supports.min().item()
    hMin_2 = hMin / 2

    ix = ((particles.positions[:, 0] - domain.min[0]) / hMin_2).long()
    iy = ((particles.positions[:, 1] - domain.min[1]) / hMin_2).long()
    nCellsx = (domain.max[0] - domain.min[0]) / hMin_2
    nCellsy = (domain.max[1] - domain.min[1]) / hMin_2

    ilin = ix + iy * nCellsx

    # Sort particles by cell index to get a consistent ordering
    iisort = torch.argsort(ilin)
    sorted_ilin = ilin[iisort]

    # Create unique UIDs that preserve cell-based ordering
    # We'll use the cell index as the primary component and add a counter within each cell
    unique_UIDs = torch.zeros_like(ilin, dtype=torch.long)

    # For each unique cell, assign sequential UIDs
    unique_cells, inverse_indices = torch.unique(sorted_ilin, return_inverse=True)

    # Count particles in each cell and create offsets
    cell_counts = torch.bincount(inverse_indices)
    cell_offsets = torch.cumsum(torch.cat([torch.tensor([0], device=cell_counts.device, dtype=torch.long), cell_counts[:-1]]), dim=0)

    # Create within-cell indices
    within_cell_indices = torch.zeros_like(inverse_indices, dtype=torch.long)
    for i, count in enumerate(cell_counts):
        mask = inverse_indices == i
        within_cell_indices[mask] = torch.arange(count, device=inverse_indices.device, dtype=torch.long)

    # Generate unique UIDs: cell_offset + within_cell_index
    unique_UIDs[iisort] = cell_offsets[inverse_indices] + within_cell_indices
    if verbose:
        print(f"Original particle order: {torch.arange(len(particles.positions))[:10]}...")
        print(f"Cell indices: {ilin[:10]}...")
        print(f"Sorted order: {iisort[:10]}...")
        print(f"Unique UIDs: {unique_UIDs[:10]}...")
        print(f"Number of unique cells: {len(unique_cells)}")
        print(f"UID range: {unique_UIDs.min()} - {unique_UIDs.max()}")
    return unique_UIDs


from diffSPH.sdf import getSDF

def buildSDF(positions, function: str, overrideArguments = None):
    sdf = getSDF(function)
    defaultArguments = sdf['sample']
    if function == 'circle':
        defaultArguments = [torch.tensor([0.5], device=positions.device, dtype=positions.dtype)]
    elif function == 'box':
        defaultArguments = [torch.tensor([0.5], device=positions.device, dtype=positions.dtype)]
    elif function == 'roundedBox':
        defaultArguments = [torch.tensor([0.5], device=positions.device, dtype=positions.dtype), torch.tensor([0.1, 0.1, 0.1, 0.1], device=positions.device, dtype=positions.dtype)]
    elif function == 'rhombus':
        defaultArguments = [torch.tensor([0.5, 0.5], device=positions.device, dtype=positions.dtype)]
    elif function == 'parallelogram':
        defaultArguments = [0.5, 0.25, 0.3]
    elif function == 'triangle':
        defaultArguments = [torch.tensor([-0.3, 0.5], device=positions.device, dtype=positions.dtype), torch.tensor([0.3, 0.5], device=positions.device, dtype=positions.dtype), torch.tensor([0., -0.5], device=positions.device, dtype=positions.dtype)]
    elif function == 'hexagon':
        defaultArguments = [0.5]
    elif function == 'star5':
        defaultArguments = [0.5, 0.5]
    elif function == 'vesica':
        defaultArguments = [0.75, 0.25]

    convertedArguments = []

    for arg in defaultArguments:
        if isinstance(arg, torch.Tensor):
            convertedArguments.append(arg.to(device=positions.device, dtype=positions.dtype))
        else:
            convertedArguments.append(arg)    
    if overrideArguments is not None:
        for i, arg in enumerate(overrideArguments):
            if isinstance(arg, torch.Tensor):
                convertedArguments[i] = arg.to(device=positions.device, dtype=positions.dtype)
            else:
                convertedArguments[i] = arg

    return sdf['function'](positions, *convertedArguments)

def maskParticles_2(particles, numberOfRegions, domain, nx, sdf = None, sdfParameters = None, split_x = 0.0, split_y = 0.0):
    splitx, splity = splitDomain(split_x, split_y, domain, nx)

    mask = torch.zeros_like(particles.positions[:,0], dtype=torch.int64)
    if numberOfRegions == 1:
        pass
    else:
        ul = torch.logical_and(particles.positions[:, 0] < splitx, particles.positions[:, 1] >= splity)
        ur = torch.logical_and(particles.positions[:, 0] >= splitx, particles.positions[:, 1] >= splity)
        ll = torch.logical_and(particles.positions[:, 0] < splitx, particles.positions[:, 1] < splity)
        lr = torch.logical_and(particles.positions[:, 0] >= splitx, particles.positions[:, 1] < splity)
        if numberOfRegions == 2:
            mask[ul] = 0
            mask[ll] = 0
            mask[ur] = 1
            mask[lr] = 1
        elif numberOfRegions == 3:
            mask[ul] = 0
            mask[ur] = 1
            mask[ll] = 2
            mask[lr] = 2
        elif numberOfRegions == 4:
            mask[ul] = 0
            mask[ur] = 1
            mask[ll] = 2
            mask[lr] = 3
        else:
            raise ValueError(f'Unsupported number of regions: {numberOfRegions}')
    if sdf is not None:
        sdf_ = buildSDF(particles.positions, sdf, sdfParameters).flatten()
        print(sdf_)
        mask[sdf_ <= 0] = numberOfRegions
        return mask, sdf_

    return mask, torch.zeros(mask.shape, dtype=particles.positions.dtype, device=particles.positions.device)