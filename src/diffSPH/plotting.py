from typing import Union, Optional
from diffSPH.kernels import SPHKernel
from matplotlib import patches
from matplotlib import colors
from diffSPH.sampling import mapToGrid
from diffSPH.util import ParticleSet, ParticleSetWithQuantity, DomainDescription
import torch
import matplotlib as mpl
import numpy as np


def scatterPlot(axis,
                particles: Union[ParticleSet, ParticleSetWithQuantity],
                domain: DomainDescription,
                quantity : Optional[torch.Tensor] = None,  
                cbar                : bool = True,
                cmap                : str = 'viridis',
                scaling             : str = 'linear',
                linthresh           : float = 1e-3,
                vmin                : Optional[float] = None,
                vmax                : Optional[float] = None,
                midPoint            : Optional[Union[float,str]] = None,  
                marker              : str = 'o',         
                markerSize          : float = 4,   
                ):
    positions = particles.positions
    minD = domain.min.cpu().detach()
    maxD = domain.max.cpu().detach()
    periodicity = domain.periodic

    pos = [(torch.remainder(positions[:, i] - minD[i], maxD[i] - minD[i]) + minD[i]) if periodicity[i] else positions[:,i] for i in range(domain.dim)]
    modPos = torch.stack(pos, dim = -1).detach().cpu().numpy()

    if quantity is None:
        sc = axis.scatter(modPos[:, 0], modPos[:, 1], s=markerSize, c='black', marker = marker)
        return sc
    else:
        q = quantity.detach().cpu().numpy()
        minScale = np.min(q) if vmin is None else vmin
        maxScale = np.max(q) if vmax is None else vmax
        if 'sym' in scaling:
            minScale = -np.max(np.abs(q)) if vmin is None else vmin
            maxScale = np.max(np.abs(q)) if vmax is None else vmax
            if midPoint is not None:
                if isinstance(midPoint, str):
                    midPoint = np.median(q)
                minScale = -np.max(np.abs(q - midPoint)) if vmin is None else vmin
                maxScale = np.max(np.abs(q - midPoint)) if vmax is None else vmax
            if 'log'in scaling:
                minElement = np.min(np.abs(q)[np.abs(q)>0.0]) if np.sum(np.abs(q) > 0.0) > 0 else 1.0
                maxElement = np.max(np.abs(q)) if np.sum(np.abs(q) > 0.0) > 0 else 10.0
                maxScale = maxScale if maxScale > 0.0 else maxElement
                minScale = minScale if minScale > 0.0 else minElement
                norm = colors.SymLogNorm(linthresh=linthresh, linscale=0.03, vmin=minScale, vmax=maxScale)
            else:
                norm = colors.CenteredNorm(vcenter=midPoint, halfrange = maxScale)
        else:
            if 'log' in scaling:
                minElement = np.min(np.abs(q)[np.abs(q)>0.0]) if np.sum(np.abs(q) > 0.0) > 0 else 1.0
                maxElement = np.max(np.abs(q)) if np.sum(np.abs(q) > 0.0) > 0 else 10.0
                maxScale = maxScale if maxScale > 0.0 else maxElement
                minScale = minScale if minScale > 0.0 else minElement
                norm = colors.LogNorm(vmin=minScale, vmax=maxScale)
            else:
                norm = colors.Normalize(vmin=minScale, vmax=maxScale)
        
        qs = q.clip(minScale, maxScale)
        
        sc = axis.scatter(modPos[:, 0], modPos[:, 1], s=markerSize, c=qs, cmap = cmap, norm = norm, marker = marker)
        return sc
    

def scatterPlotUpdate(sc, axis,
                particles: Union[ParticleSet, ParticleSetWithQuantity],
                domain: DomainDescription,
                quantity : Optional[torch.Tensor] = None,  
                cbar                : bool = True,
                cmap                : str = 'viridis',
                scaling             : str = 'linear',
                linthresh           : float = 1e-3,
                vmin                : Optional[float] = None,
                vmax                : Optional[float] = None,
                midPoint            : Optional[Union[float,str]] = None,  
                marker              : str = 'o',         
                markerSize          : float = 4,   
                ):
    positions = particles.positions
    minD = domain.min.cpu().detach()
    maxD = domain.max.cpu().detach()
    periodicity = domain.periodic

    pos = [(torch.remainder(positions[:, i] - minD[i], maxD[i] - minD[i]) + minD[i]) if periodicity[i] else positions[:,i] for i in range(domain.dim)]
    modPos = torch.stack(pos, dim = -1).detach().cpu().numpy()

    # print(modPos, quantity)

    if quantity is None:
        sc.set_offsets(modPos)
        # sc = axis.scatter(modPos[:, 0], modPos[:, 1], s=markerSize, c='black', marker = marker)
        return sc
    else:
        q = quantity.detach().cpu().numpy()
        minScale = np.min(q) if vmin is None else vmin
        maxScale = np.max(q) if vmax is None else vmax
        if 'sym' in scaling:
            minScale = -np.max(np.abs(q)) if vmin is None else vmin
            maxScale = np.max(np.abs(q)) if vmax is None else vmax
            if midPoint is not None:
                if isinstance(midPoint, str):
                    midPoint = np.median(q)
                minScale = -np.max(np.abs(q - midPoint)) if vmin is None else vmin
                maxScale = np.max(np.abs(q - midPoint)) if vmax is None else vmax
            if 'log'in scaling:
                minElement = np.min(np.abs(q)[np.abs(q)>0.0]) if np.sum(np.abs(q) > 0.0) > 0 else 1.0
                maxElement = np.max(np.abs(q)) if np.sum(np.abs(q) > 0.0) > 0 else 10.0
                maxScale = maxScale if maxScale > 0.0 else maxElement
                minScale = minScale if minScale > 0.0 else minElement
                norm = colors.SymLogNorm(linthresh=linthresh, linscale=0.03, vmin=minScale, vmax=maxScale)
            else:
                norm = colors.CenteredNorm(vcenter=midPoint, halfrange = maxScale)
        else:
            if 'log' in scaling:
                minElement = np.min(np.abs(q)[np.abs(q)>0.0]) if np.sum(np.abs(q) > 0.0) > 0 else 1.0
                maxElement = np.max(np.abs(q)) if np.sum(np.abs(q) > 0.0) > 0 else 10.0
                maxScale = maxScale if maxScale > 0.0 else maxElement
                minScale = minScale if minScale > 0.0 else minElement
                norm = colors.LogNorm(vmin=minScale, vmax=maxScale)
            else:
                norm = colors.Normalize(vmin=minScale, vmax=maxScale)
        
        qs = q.clip(minScale, maxScale)
        
        # print(modPos.shape)
        sc.set_offsets(modPos)
        sc.set_array(qs)
        sc.set_norm(norm)
        sc.set_cmap(cmap)
        # sc = axis.scatter(modPos[:, 0], modPos[:, 1], s=markerSize, c=q, cmap = cmap, norm = norm, marker = marker)
        return sc
    
def gridPlot(axis, grid, quantity, 
                cmap                : str = 'viridis',
                scaling             : str = 'linear',
                linthresh           : float = 1e-3,
                vmin                : Optional[float] = None,
                vmax                : Optional[float] = None,
                midPoint            : Optional[Union[float,str]] = None,  ):
    
        q = quantity.detach().cpu().numpy()
        minScale = np.min(q) if vmin is None else vmin
        maxScale = np.max(q) if vmax is None else vmax
        if 'sym' in scaling:
            minScale = -np.max(np.abs(q)) if vmin is None else vmin
            maxScale = np.max(np.abs(q)) if vmax is None else vmax
            if midPoint is not None:
                if isinstance(midPoint, str):
                    midPoint = np.median(q)
                minScale = -np.max(np.abs(q - midPoint)) if vmin is None else vmin
                maxScale = np.max(np.abs(q - midPoint)) if vmax is None else vmax
            if 'log'in scaling:
                minElement = np.min(np.abs(q)[np.abs(q)>0.0]) if np.sum(np.abs(q) > 0.0) > 0 else 1.0
                maxElement = np.max(np.abs(q)) if np.sum(np.abs(q) > 0.0) > 0 else 10.0
                maxScale = maxScale if maxScale > 0.0 else maxElement
                minScale = minScale if minScale > 0.0 else minElement
                norm = colors.SymLogNorm(linthresh=linthresh, linscale=0.03, vmin=minScale, vmax=maxScale)
            else:
                norm = colors.CenteredNorm(vcenter=midPoint, halfrange = maxScale)
        else:
            if 'log' in scaling:
                minElement = np.min(np.abs(q)[np.abs(q)>0.0]) if np.sum(np.abs(q) > 0.0) > 0 else 1.0
                maxElement = np.max(np.abs(q)) if np.sum(np.abs(q) > 0.0) > 0 else 10.0
                maxScale = maxScale if maxScale > 0.0 else maxElement
                minScale = minScale if minScale > 0.0 else minElement
                norm = colors.LogNorm(vmin=minScale, vmax=maxScale)
            else:
                norm = colors.Normalize(vmin=minScale, vmax=maxScale)
        sc = axis.pcolormesh(grid[0].cpu().numpy(), grid[1].cpu().numpy(), q, cmap = cmap, norm = norm, shading = 'auto')
        return sc
    
def gridPlotUpdate(axis, sc, grid, quantity, 
                cmap                : str = 'viridis',
                scaling             : str = 'linear',
                linthresh           : float = 1e-3,
                vmin                : Optional[float] = None,
                vmax                : Optional[float] = None,
                midPoint            : Optional[Union[float,str]] = None,  ):
    
        q = quantity.detach().cpu().numpy()
        minScale = np.min(q) if vmin is None else vmin
        maxScale = np.max(q) if vmax is None else vmax
        if 'sym' in scaling:
            minScale = -np.max(np.abs(q)) if vmin is None else vmin
            maxScale = np.max(np.abs(q)) if vmax is None else vmax
            if midPoint is not None:
                if isinstance(midPoint, str):
                    midPoint = np.median(q)
                minScale = -np.max(np.abs(q - midPoint)) if vmin is None else vmin
                maxScale = np.max(np.abs(q - midPoint)) if vmax is None else vmax
            if 'log'in scaling:
                minElement = np.min(np.abs(q)[np.abs(q)>0.0]) if np.sum(np.abs(q) > 0.0) > 0 else 1.0
                maxElement = np.max(np.abs(q)) if np.sum(np.abs(q) > 0.0) > 0 else 10.0
                maxScale = maxScale if maxScale > 0.0 else maxElement
                minScale = minScale if minScale > 0.0 else minElement
                norm = colors.SymLogNorm(linthresh=linthresh, linscale=0.03, vmin=minScale, vmax=maxScale)
            else:
                norm = colors.CenteredNorm(vcenter=midPoint, halfrange = maxScale)
        else:
            if 'log' in scaling:
                minElement = np.min(np.abs(q)[np.abs(q)>0.0]) if np.sum(np.abs(q) > 0.0) > 0 else 1.0
                maxElement = np.max(np.abs(q)) if np.sum(np.abs(q) > 0.0) > 0 else 10.0
                maxScale = maxScale if maxScale > 0.0 else maxElement
                minScale = minScale if minScale > 0.0 else minElement
                norm = colors.LogNorm(vmin=minScale, vmax=maxScale)
            else:
                norm = colors.Normalize(vmin=minScale, vmax=maxScale)
        
        sc.set_array(q)
        sc.set_norm(norm)

        # sc = axis.pcolormesh(grid[0], grid[1], q, cmap = cmap, norm = norm, shading = 'auto')
        return sc
    
            
def mapQuantity(inputQuantity : torch.Tensor, mapping: str):
    quantity = None
    if len(inputQuantity.shape) == 2:
        # Non scalar quantity
        if mapping == '.x' or mapping == '[0]':
            quantity = inputQuantity[:,0]
        if mapping == '.y' or mapping == '[1]':
            quantity = inputQuantity[:,1]
        if mapping == '.z' or mapping == '[2]':
            quantity = inputQuantity[:,2]
        if mapping == '.w' or mapping == '[3]':
            quantity = inputQuantity[:,3]
        if mapping == 'Linf':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = float('inf'))
        if mapping == 'L-inf':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = -float('inf'))
        if mapping == 'L0':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = 0)
        if mapping == 'L1':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = 1)
        if mapping == 'L2' or mapping == 'norm' or mapping == 'magnitude':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = 2)
        if mapping == 'theta':
            quantity = torch.atan2(inputQuantity[:,1], inputQuantity[:,0])
    else:
        quantity = inputQuantity
    return quantity


def filterParticles(particles: Union[ParticleSet, ParticleSetWithQuantity], which:str = 'fluid', batch: Optional[int] = None):
    if hasattr(particles, 'kinds'):
        if batch is not None:
            if which == 'fluid':
                return ParticleSetWithQuantity(
                    particles.positions[(particles.kinds == 0) & (particles.batches == batch)],
                    particles.supports[(particles.kinds == 0) & (particles.batches == batch)],
                    particles.masses[(particles.kinds == 0) & (particles.batches == batch)],
                    particles.densities[(particles.kinds == 0) & (particles.batches == batch)],
                    particles.quantities[(particles.kinds == 0) & (particles.batches == batch)] if hasattr(particles, 'quantities') else None,
                    # batches = particles.batches[(particles.kinds == 0) & (particles.batches == batch)]
                )
            elif which == 'boundary':
                return ParticleSetWithQuantity(
                    particles.positions[(particles.kinds == 1) & (particles.batches == batch)],
                    particles.supports[(particles.kinds == 1) & (particles.batches == batch)],
                    particles.masses[(particles.kinds == 1) & (particles.batches == batch)],
                    particles.densities[(particles.kinds == 1) & (particles.batches == batch)],
                    particles.quantities[(particles.kinds == 1) & (particles.batches == batch)] if hasattr(particles, 'quantities') else None,
                    # batches = particles.batches[(particles.kinds == 1) & (particles.batches == batch)]
                )
        else:
            if which == 'fluid':
                return ParticleSetWithQuantity(
                    particles.positions[particles.kinds == 0],
                    particles.supports[particles.kinds == 0],
                    particles.masses[particles.kinds == 0],
                    particles.densities[particles.kinds == 0],
                    particles.quantities[particles.kinds == 0] if hasattr(particles, 'quantities') else None
                )
            elif which == 'boundary':
                return ParticleSetWithQuantity(
                    particles.positions[particles.kinds == 1],
                    particles.supports[particles.kinds == 1],
                    particles.masses[particles.kinds == 1],
                    particles.densities[particles.kinds == 1],
                    particles.quantities[particles.kinds == 1] if hasattr(particles, 'quantities') else None
                )
            else:
                return particles
    else:
        if batch is not None:
            return ParticleSetWithQuantity(
                particles.positions[particles.batches == batch],
                particles.supports[particles.batches == batch],
                particles.masses[particles.batches == batch],
                particles.densities[particles.batches == batch],
                particles.quantities[particles.batches == batch] if hasattr(particles, 'quantities') else None,
                # batches = particles.batches[particles.batches == batch]
            )
        if which == 'fluid':
            return particles
        else:
            return None

from diffSPH.kernels import KernelType, SPHKernel, getSPHKernelv2
from diffSPH.operations import sph_op
from diffSPH.neighborhood import buildNeighborhood, filterNeighborhoodByKind
from typing import Union, Optional, Tuple, List

def buildRotationMatrix(angles : List[float], dim: int, device: torch.device = None, dtype: torch.dtype = None):
    if dim == 1:
        return torch.tensor([[1.0]], device=device, dtype=dtype)
    elif dim == 2:
        return torch.tensor([[torch.cos(angles), -torch.sin(angles)],
                             [torch.sin(angles), torch.cos(angles)]], device=device, dtype=dtype)
    elif dim == 3:
        angle_phi = angles[0]
        angle_theta = angles[1]
        return torch.tensor([
            [torch.cos(angle_phi) * torch.sin(angle_theta), -torch.sin(angle_phi), torch.cos(angle_phi) * torch.cos(angle_theta)],
            [torch.sin(angle_phi) * torch.sin(angle_theta), torch.cos(angle_phi), torch.sin(angle_phi) * torch.cos(angle_theta)],
            [torch.cos(angle_theta), 0, -torch.sin(angle_theta)]
        ], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    
def filterBatchFluid(quant, kinds, particles, batch: Optional[int] = None):
    if batch is not None:
        return quant[particles.batches[kinds == 0] == batch]
    return quant
def filterBatchBoundary(quant, kinds, particles, batch: Optional[int] = None):
    if batch is not None:
        return quant[particles.batches[kinds == 1] == batch]
    return quant

def filterBatchParticles(particles, batch: Optional[int] = None):
    if batch is not None:
        return ParticleSetWithQuantity(
            particles.positions[particles.batches == batch],
            particles.supports[particles.batches == batch],
            particles.masses[particles.batches == batch],
            particles.densities[particles.batches == batch],
            particles.quantities[particles.batches == batch] if hasattr(particles, 'quantities') else None,
            batches = particles.batches[particles.batches == batch]
        )
    return particles

def visualizeParticles(    
    fig, axis,
    particles           : Union[ParticleSet, ParticleSetWithQuantity],
    domain              : DomainDescription,
    quantity            : Optional[torch.Tensor] = None,
    kernel              : SPHKernel = getSPHKernelv2('Wendland4'),

    # boundaryParticles   : Optional[Union[ParticleSet, ParticleSetWithQuantity]] = None,

    which               : str = 'fluid',
    visualizeBoth       : bool = False,

    cbar                : bool = True,
    cmap                : str = 'viridis',
    scaling             : str = 'linear',
    linthresh           : float = 1e-3,
    vmin                : Optional[float] = None,
    vmax                : Optional[float] = None,
    midPoint            : Optional[Union[float,str]] = None,
    domainEpsilon       : float = 0.05,
    markerSize          : float = 4,

    mapping             : str = '.x',
    gridVisualization   : bool = False,
    gridResolution      : int = 128,
    streamLines         : bool = False,

    operation           : Optional[str] = None,
    gradientMode = 'naive',
    laplaceMode = 'naive',
    divergenceMode = 'div',
    consistentDivergence = False,

    streamLineOperation : Optional[str] = None,
    streamLineGradientMode = 'naive',
    streamLineLaplaceMode = 'naive',
    streamLineDivergenceMode = 'div',
    streamLineConsistentDivergence = False,

    plotDomain          : bool = True,

    title               : Optional[str] = None,
    batch: Optional[int] = None,
):
    domain_ = DomainDescription(
        min = domain.min.cpu().detach(),
        max = domain.max.cpu().detach(),
        periodic = domain.periodic.cpu().detach(),
        dim = domain.dim
    )
    if hasattr(domain, 'angles'):
        rotMat = buildRotationMatrix(torch.tensor(domain.angles, dtype = domain.min.dtype, device = domain.min.device), domain.dim, device=domain.min.device, dtype=domain.min.dtype)
        invRotMat = rotMat.inverse()
    else:
        rotMat = None
        invRotMat = None

    # Set up the axis
    eps = (domain_.max - domain_.min) * domainEpsilon
    axis.set_xlim(domain_.min[0] - eps[0], domain_.max[0] + eps[0])
    axis.set_ylim(domain_.min[1] - eps[1], domain_.max[1] + eps[1])
    if plotDomain:
        square = patches.Rectangle((domain_.min[0], domain_.min[1]), domain_.max[0] - domain_.min[0], domain_.max[1] - domain_.min[1],    linewidth=1, edgecolor='b', facecolor='none',ls='--')
        axis.add_patch(square)
    axis.set_aspect('equal')

    if title is not None:
        axis.set_title(title)

    fluidParticles = filterParticles(particles, which = 'fluid', batch = batch)
    boundaryParticles = filterParticles(particles, which = 'boundary', batch = batch)

    # fluidParticles = filterBatchParticles(fluidParticles, batch)   
    # if boundaryParticles is not None:
    #     boundaryParticles = filterBatchParticles(boundaryParticles, batch)

    # fluidParticles = filterBatch(fluidParticles, particles, batch)

    if rotMat is not None:
        fluidParticles = fluidParticles._replace(positions = torch.einsum('ij, ni->nj', invRotMat, fluidParticles.positions))
    if boundaryParticles is not None and rotMat is not None:
        boundaryParticles = boundaryParticles._replace(positions = torch.einsum('ij, ni->nj', invRotMat, boundaryParticles.positions))

    fluidQuantity = fluidParticles.quantities if isinstance(fluidParticles, ParticleSetWithQuantity) else None
    boundaryQuantity = boundaryParticles.quantities if isinstance(boundaryParticles, ParticleSetWithQuantity) else None


    if quantity is not None:
        numParticlesTotal = particles.positions.shape[0]
        numFluidParticles = particles.positions[particles.kinds == 0].shape[0]
        numBoundaryParticles = numParticlesTotal - numFluidParticles

        numFluidParticlesInBatch = fluidParticles.positions.shape[0]
        numBoundaryParticlesInBatch = boundaryParticles.positions.shape[0] if boundaryParticles is not None else 0
        numParticlesInBatch = numFluidParticlesInBatch + numBoundaryParticlesInBatch


        if isinstance(quantity, tuple):
            # print('Quantity is a tuple, separating fluid and boundary quantities')
            fluidQuantity = quantity[0]
            boundaryQuantity = quantity[1] if boundaryParticles is not None else None

            if batch is not None:
                if fluidQuantity.shape[0] == numFluidParticles:
                    # print(f'Filtering fluid quantity for batch {batch} with {numFluidParticles} particles')
                    fluidQuantity = fluidQuantity[particles.kinds[particles.batches == batch] == 0]
                if boundaryQuantity is not None and boundaryQuantity.shape[0] == numBoundaryParticles:
                    # print(f'Filtering boundary quantity for batch {batch} with {numBoundaryParticles} particles')
                    boundaryQuantity = boundaryQuantity[particles.kinds[particles.batches == batch] == 1]

        else:
            # print('Quantity is a single tensor, filtering based on particle kinds')
            if hasattr(particles, 'kinds'):
                # print('Particles have kinds attribute, filtering based on kinds')
                if batch is not None:
                    if quantity.shape[0] == numParticlesInBatch:
                        # print(f'Filtering quantity for batch [quantity is batch sized]')
                        fluidQuantity = quantity[particles.kinds[particles.batches == batch] == 0]
                        boundaryQuantity = quantity[particles.kinds[particles.batches == batch] == 1] if boundaryParticles is not None else None
                    elif quantity.shape[0] == numParticlesTotal:
                        # print(f'Filtering quantity for batch [quantity is total sized]')
                        fluidQuantity = quantity[torch.logical_and(particles.kinds == 0, particles.batches == batch)]
                        boundaryQuantity = quantity[torch.logical_and(particles.kinds == 1, particles.batches == batch)] if boundaryParticles is not None else None
                else:
                    fluidQuantity = quantity[particles.kinds == 0]
                    boundaryQuantity = quantity[particles.kinds == 1] if boundaryParticles is not None else None
            else:
                fluidQuantity = quantity
                boundaryQuantity = quantity if boundaryParticles is not None else None
                if batch is not None:
                    if fluidQuantity.shape[0] == numParticlesTotal:
                        fluidQuantity = fluidQuantity[particles.batches == batch]
                    if boundaryQuantity is not None and boundaryQuantity.shape[0] == numParticlesTotal:
                        boundaryQuantity = boundaryQuantity[particles.batches == batch]
                

    if rotMat is not None and len(fluidQuantity.shape) == 2:
        if fluidQuantity is not None and fluidQuantity.shape[1] == fluidParticles.positions.shape[1]:
            fluidQuantity = torch.einsum('ij, ni->nj', invRotMat, fluidQuantity)
        if boundaryQuantity is not None and boundaryQuantity.shape[1] == boundaryParticles.positions.shape[1]:
            boundaryQuantity = torch.einsum('ij, ni->nj', invRotMat, boundaryQuantity)

    if not gridVisualization:
        if which == 'fluid' and fluidQuantity is None:
            scatterPlot(axis, fluidParticles, domain, None, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'o', markerSize)
            return
        if which == 'boundary' and boundaryQuantity is None:
            scatterPlot(axis, boundaryParticles, domain, None, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'x', markerSize)
            return
        if which == 'both' and fluidQuantity is None and boundaryQuantity is None:
            scatterPlot(axis, fluidParticles, domain, None, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'o', markerSize)
            scatterPlot(axis, boundaryParticles, domain, None, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'x', markerSize)
            return
        if which == 'both' and ((fluidQuantity is not None and boundaryQuantity is None) or (fluidQuantity is None and boundaryQuantity is not None)):
            raise ValueError('Both fluid and boundary particles must have quantities to visualize both')
        
    # print('...')
    if operation is not None:
        # print(fluidParticles)
        neighborhood, sparseNeighborhood_ = buildNeighborhood(fluidParticles, fluidParticles, domain, verletScale= 1.0, mode = 'superSymmetric')
        
        # neighborhood = filterNeighborhoodByKind(fluidParticles, sparseNeighborhood_, which = 'normal')
        
        currentFluidParticles = ParticleSetWithQuantity(fluidParticles.positions, fluidParticles.supports, fluidParticles.masses, fluidParticles.densities, fluidQuantity)

        q = sph_op(currentFluidParticles, currentFluidParticles, domain, getSPHKernelv2(kernel), sparseNeighborhood_, operation = operation, gradientMode = gradientMode, laplaceMode = laplaceMode, divergenceMode = divergenceMode, consistentDivergence = consistentDivergence)

        if boundaryParticles is not None and isinstance(boundaryParticles, ParticleSetWithQuantity) and boundaryParticles.positions.shape[0] > 0:
            boundaryParticles = boundaryParticles._replace(quantities = boundaryQuantity)
            neighborhood, sparseNeighborhood_ = buildNeighborhood(fluidParticles, boundaryParticles, domain, verletScale= 1.0)
            q += sph_op(currentFluidParticles, boundaryParticles, domain, getSPHKernelv2(kernel), sparseNeighborhood_, operation = operation, gradientMode = gradientMode, laplaceMode = laplaceMode, divergenceMode = divergenceMode, consistentDivergence = consistentDivergence)

        fluidQuantity = q

    preMapping = fluidQuantity
    if fluidQuantity is not None:
        mappedQuantity = mapQuantity(fluidQuantity, mapping)
    if boundaryQuantity is not None:
        boundaryQuantity = mapQuantity(boundaryQuantity, mapping)
    sc = None
    cb = None
    scBoundary = None

    if not gridVisualization:
        if which == 'fluid':
            sc = scatterPlot(axis, fluidParticles, domain, mappedQuantity, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'o', markerSize)
            if cbar:
                cb = fig.colorbar(sc, ax=axis)
            if visualizeBoth:
                scBoundary = scatterPlot(axis, boundaryParticles, domain, None, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'x', markerSize)
        elif which == 'boundary':
            sc = scatterPlot(axis, boundaryParticles, domain, boundaryQuantity, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'x', markerSize)
            if cbar:
                cb = fig.colorbar(sc, ax=axis)
            if visualizeBoth:
                scBoundary = scatterPlot(axis, fluidParticles, domain, None, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'o', markerSize)
        elif which == 'both':
            mergedParticles = ParticleSetWithQuantity(
                torch.vstack([fluidParticles.positions, boundaryParticles.positions],).view(-1, fluidParticles.positions.shape[-1]),
                torch.hstack([fluidParticles.supports, boundaryParticles.supports]),
                torch.hstack([fluidParticles.masses, boundaryParticles.masses]),
                torch.hstack([fluidParticles.densities, boundaryParticles.densities]),
                torch.cat([mappedQuantity, boundaryQuantity], dim = 0)
            )
            sc = scatterPlot(axis, mergedParticles, domain, mergedParticles.quantities, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'o', markerSize)
            if cbar:
                cb = fig.colorbar(sc, ax=axis)
        # return
    else:
        if which == 'fluid':
            grid, interpolated = mapToGrid(fluidParticles, mappedQuantity, domain, kernel, gridResolution)
        elif which == 'boundary':
            grid, interpolated = mapToGrid(boundaryParticles, boundaryQuantity, domain, kernel, gridResolution)
        else:
            mergedParticles = ParticleSetWithQuantity(
                positions = torch.cat([fluidParticles.positions, boundaryParticles.positions], dim = 0),
                supports = torch.cat([fluidParticles.supports, boundaryParticles.supports], dim = 0),
                masses = torch.cat([fluidParticles.masses, boundaryParticles.masses], dim = 0),
                densities = torch.cat([fluidParticles.densities, boundaryParticles.densities], dim = 0),
                quantities = torch.cat([mappedQuantity, boundaryQuantity], dim = 0)
            )
            grid, interpolated = mapToGrid(mergedParticles, mergedParticles.quantities, domain, kernel, gridResolution)

        sc = gridPlot(axis, grid, interpolated, cmap, scaling, linthresh, vmin, vmax, midPoint)
        if cbar:
            cb = fig.colorbar(sc, ax=axis)
        # return


    streamPlot = None
    if streamLines:
        if len(mappedQuantity.shape) == 2:
            # print('Using mapped quantity')
            inputQuantity = mappedQuantity
        elif len(preMapping.shape) == 2:
            # print('Using pre mapped quantity')
            inputQuantity = preMapping
        # elif len(fluidParticles.quantities.shape) == 2:
            # print('Using fluid particles quantity')
            # inputQuantity = fluidParticles.quantities
        elif quantity is not None and len(quantity.shape) == 2:
            # print('Using input quantity')
            inputQuantity = quantity
        else:
            # print('Using mapped quantity pre op')
            inputQuantity = mappedQuantity
        
        if streamLineOperation is not None:
            neighborhood = buildNeighborhood(fluidParticles, fluidParticles, domain, verletScale= 1.0).fullAdjacency

            currentFluidParticles = ParticleSetWithQuantity(fluidParticles.positions, fluidParticles.supports, fluidParticles.masses, fluidParticles.densities, inputQuantity)

            inputQuantity = sph_op(currentFluidParticles, currentFluidParticles, domain, kernel, neighborhood, operation = streamLineOperation, gradientMode = streamLineGradientMode, laplaceMode = streamLineLaplaceMode, divergenceMode = streamLineDivergenceMode, consistentDivergence = streamLineConsistentDivergence)
        
        if len(inputQuantity.shape) != 2:
            raise ValueError('Streamlines require a 2D quantity')

        grid_ux, quant_ux = mapToGrid(fluidParticles, inputQuantity[:, 0], domain, kernel, gridResolution)
        grid_uy, quant_uy = mapToGrid(fluidParticles, inputQuantity[:, 1], domain, kernel, gridResolution)

        # print(grid_ux.shape, quant_ux.shape)

        streamPlot = axis.streamplot(grid_ux[0].T.detach().cpu().numpy(), grid_ux[1].T.detach().cpu().numpy(), quant_ux.T.detach().cpu().numpy(), quant_uy.T.detach().cpu().numpy(), color='black', linewidth=1, arrowstyle='->')

    # print(sc, cb)

    return {
        'fig': fig,
        'axis': axis,
        'domain': domain,
        'kernel': kernel,

        'which': which,
        'visualizeBoth': visualizeBoth,

        'cbar': cbar,
        'cmap': cmap,
        'scaling': scaling,
        'linthresh': linthresh,
        'vmin': vmin,
        'vmax': vmax,
        'midPoint': midPoint,
        'domainEpsilon': domainEpsilon,
        'markerSize': markerSize,

        'mapping': mapping,
        'gridVisualization': gridVisualization,
        'gridResolution': gridResolution,
        'streamLines': streamLines,

        'operation': operation,
        'gradientMode': gradientMode,
        'laplaceMode': laplaceMode,
        'divergenceMode': divergenceMode,
        'consistentDivergence': consistentDivergence,

        'streamLineOperation': streamLineOperation,
        'streamLineGradientMode': streamLineGradientMode,
        'streamLineLaplaceMode': streamLineLaplaceMode,
        'streamLineDivergenceMode': streamLineDivergenceMode,
        'streamLineConsistentDivergence': streamLineConsistentDivergence,

        'plotDomain': plotDomain,

        'title': title,

        'streamPlot': streamPlot,
        'scatterPlot': sc,
        'boundaryScatterPlot': scBoundary,
        'colorBar': cb,
        'batch': batch

    }


def updatePlot(plotState, particles: Union[ParticleSet, ParticleSetWithQuantity],quantity: Optional[torch.Tensor] = None, **kwargs):
    for key, value in kwargs.items():
        plotState[key] = value
    fig = plotState['fig']
    axis = plotState['axis']
    domain = plotState['domain']
    kernel = plotState['kernel']
    which = plotState['which']
    visualizeBoth = plotState['visualizeBoth']
    
    cbar = plotState['cbar']
    cmap = plotState['cmap']
    scaling = plotState['scaling']
    linthresh = plotState['linthresh']
    vmin = plotState['vmin']
    vmax = plotState['vmax']
    midPoint = plotState['midPoint']
    domainEpsilon = plotState['domainEpsilon']
    markerSize = plotState['markerSize']

    mapping = plotState['mapping']
    gridVisualization = plotState['gridVisualization']
    gridResolution = plotState['gridResolution']
    streamLines = plotState['streamLines']

    operation = plotState['operation']
    gradientMode = plotState['gradientMode']
    laplaceMode = plotState['laplaceMode']
    divergenceMode = plotState['divergenceMode']
    consistentDivergence = plotState['consistentDivergence']

    streamLineOperation = plotState['streamLineOperation']
    streamLineGradientMode = plotState['streamLineGradientMode']
    streamLineLaplaceMode = plotState['streamLineLaplaceMode']
    streamLineDivergenceMode = plotState['streamLineDivergenceMode']
    streamLineConsistentDivergence = plotState['streamLineConsistentDivergence']

    plotDomain = plotState['plotDomain']
    title = plotState['title']

    sc = plotState['scatterPlot']
    scBoundary = plotState['boundaryScatterPlot']
    cb = plotState['colorBar']
    streamPlot = plotState['streamPlot']
    batch = plotState['batch']

    domain_ = DomainDescription(
        min = domain.min.cpu().detach(),
        max = domain.max.cpu().detach(),
        periodic = domain.periodic.cpu().detach(),
        dim = domain.dim
    )
    if hasattr(domain, 'angles'):
        rotMat = buildRotationMatrix(torch.tensor(domain.angles, dtype = domain.min.dtype, device = domain.min.device), domain.dim, device=domain.min.device, dtype=domain.min.dtype)
        invRotMat = rotMat.inverse()
    else:
        rotMat = None
        invRotMat = None

    # Set up the axis
    eps = (domain_.max - domain_.min) * domainEpsilon
    axis.set_xlim(domain_.min[0] - eps[0], domain_.max[0] + eps[0])
    axis.set_ylim(domain_.min[1] - eps[1], domain_.max[1] + eps[1])
    if plotDomain:
        square = patches.Rectangle((domain_.min[0], domain_.min[1]), domain_.max[0] - domain_.min[0], domain_.max[1] - domain_.min[1],    linewidth=1, edgecolor='b', facecolor='none',ls='--')
        axis.add_patch(square)
    axis.set_aspect('equal')

    if title is not None:
        axis.set_title(title)

    fluidParticles = filterParticles(particles, which = 'fluid', batch = batch)
    boundaryParticles = filterParticles(particles, which = 'boundary', batch = batch)

    # fluidParticles = filterBatchParticles(fluidParticles, batch)   
    # if boundaryParticles is not None:
    #     boundaryParticles = filterBatchParticles(boundaryParticles, batch)

    # fluidParticles = filterBatch(fluidParticles, particles, batch)

    if rotMat is not None:
        fluidParticles = fluidParticles._replace(positions = torch.einsum('ij, ni->nj', invRotMat, fluidParticles.positions))
    if boundaryParticles is not None and rotMat is not None:
        boundaryParticles = boundaryParticles._replace(positions = torch.einsum('ij, ni->nj', invRotMat, boundaryParticles.positions))

    fluidQuantity = fluidParticles.quantities if isinstance(fluidParticles, ParticleSetWithQuantity) else None
    boundaryQuantity = boundaryParticles.quantities if isinstance(boundaryParticles, ParticleSetWithQuantity) else None


    if quantity is not None:
        numParticlesTotal = particles.positions.shape[0]
        numFluidParticles = particles.positions[particles.kinds == 0].shape[0]
        numBoundaryParticles = numParticlesTotal - numFluidParticles

        numFluidParticlesInBatch = fluidParticles.positions.shape[0]
        numBoundaryParticlesInBatch = boundaryParticles.positions.shape[0] if boundaryParticles is not None else 0
        numParticlesInBatch = numFluidParticlesInBatch + numBoundaryParticlesInBatch


        if isinstance(quantity, tuple):
            # print('Quantity is a tuple, separating fluid and boundary quantities')
            fluidQuantity = quantity[0]
            boundaryQuantity = quantity[1] if boundaryParticles is not None else None

            if batch is not None:
                if fluidQuantity.shape[0] == numFluidParticles:
                    # print(f'Filtering fluid quantity for batch {batch} with {numFluidParticles} particles')
                    fluidQuantity = fluidQuantity[particles.kinds[particles.batches == batch] == 0]
                if boundaryQuantity is not None and boundaryQuantity.shape[0] == numBoundaryParticles:
                    # print(f'Filtering boundary quantity for batch {batch} with {numBoundaryParticles} particles')
                    boundaryQuantity = boundaryQuantity[particles.kinds[particles.batches == batch] == 1]

        else:
            # print('Quantity is a single tensor, filtering based on particle kinds')
            if hasattr(particles, 'kinds'):
                # print('Particles have kinds attribute, filtering based on kinds')
                if batch is not None:
                    if quantity.shape[0] == numParticlesInBatch:
                        # print(f'Filtering quantity for batch [quantity is batch sized]')
                        fluidQuantity = quantity[particles.kinds[particles.batches == batch] == 0]
                        boundaryQuantity = quantity[particles.kinds[particles.batches == batch] == 1] if boundaryParticles is not None else None
                    elif quantity.shape[0] == numParticlesTotal:
                        # print(f'Filtering quantity for batch [quantity is total sized]')
                        fluidQuantity = quantity[torch.logical_and(particles.kinds == 0, particles.batches == batch)]
                        boundaryQuantity = quantity[torch.logical_and(particles.kinds == 1, particles.batches == batch)] if boundaryParticles is not None else None
                else:
                    fluidQuantity = quantity[particles.kinds == 0]
                    boundaryQuantity = quantity[particles.kinds == 1] if boundaryParticles is not None else None
            else:
                fluidQuantity = quantity
                boundaryQuantity = quantity if boundaryParticles is not None else None
                if batch is not None:
                    if fluidQuantity.shape[0] == numParticlesTotal:
                        fluidQuantity = fluidQuantity[particles.batches == batch]
                    if boundaryQuantity is not None and boundaryQuantity.shape[0] == numParticlesTotal:
                        boundaryQuantity = boundaryQuantity[particles.batches == batch]
                

    if rotMat is not None and len(fluidQuantity.shape) == 2:
        if fluidQuantity is not None and fluidQuantity.shape[1] == fluidParticles.positions.shape[1]:
            fluidQuantity = torch.einsum('ij, ni->nj', invRotMat, fluidQuantity)
        if boundaryQuantity is not None and boundaryQuantity.shape[1] == boundaryParticles.positions.shape[1]:
            boundaryQuantity = torch.einsum('ij, ni->nj', invRotMat, boundaryQuantity)


    if not gridVisualization:
        # print(fluidParticles, fluidQuantity)
        
        if which == 'fluid' and fluidQuantity is None:
            scatterPlotUpdate(sc, axis, fluidParticles, domain, None, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'o', markerSize)
            if visualizeBoth:
                scBoundary.set_offsets(boundaryParticles.positions.cpu().detach().numpy())
            return
        if which == 'boundary' and boundaryQuantity is None:
            scatterPlotUpdate(sc, axis, boundaryParticles, domain, None, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'x', markerSize)
            if visualizeBoth:
                scBoundary.set_offsets(fluidParticles.positions.cpu().detach().numpy())
            return
        if which == 'both' and fluidQuantity is None and boundaryQuantity is None:
            scatterPlotUpdate(sc, axis, fluidParticles, domain, None, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'o', markerSize)
            scatterPlotUpdate(sc, axis, boundaryParticles, domain, None, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'x', markerSize)
            return
        if which == 'both' and ((fluidQuantity is not None and boundaryQuantity is None) or (fluidQuantity is None and boundaryQuantity is not None)):
            raise ValueError('Both fluid and boundary particles must have quantities to visualize both')
        
    # print('...')
    if operation is not None:
        neighborhood, sparseNeighborhood_ = buildNeighborhood(fluidParticles, fluidParticles, domain, verletScale= 1.0, mode = 'superSymmetric')
        
        currentFluidParticles = ParticleSetWithQuantity(fluidParticles.positions, fluidParticles.supports, fluidParticles.masses, fluidParticles.densities, fluidQuantity)

        q = sph_op(currentFluidParticles, currentFluidParticles, domain, getSPHKernelv2(kernel), sparseNeighborhood_, operation = operation, gradientMode = gradientMode, laplaceMode = laplaceMode, divergenceMode = divergenceMode, consistentDivergence = consistentDivergence)

        if boundaryParticles is not None and isinstance(boundaryParticles, ParticleSetWithQuantity) and boundaryParticles.positions.shape[0] > 0:
            boundaryParticles = boundaryParticles._replace(quantities = boundaryQuantity)
            neighborhood, sparseNeighborhood_ = buildNeighborhood(fluidParticles, boundaryParticles, domain, verletScale= 1.0)
            q += sph_op(currentFluidParticles, boundaryParticles, domain, getSPHKernelv2(kernel), sparseNeighborhood_, operation = operation, gradientMode = gradientMode, laplaceMode = laplaceMode, divergenceMode = divergenceMode, consistentDivergence = consistentDivergence)

        fluidQuantity = q

    preMapping = fluidQuantity
    if fluidQuantity is not None:
        mappedQuantity = mapQuantity(fluidQuantity, mapping)
    if boundaryQuantity is not None:
        boundaryQuantity = mapQuantity(boundaryQuantity, mapping)
    # sc = None
    # cb = None

    if not gridVisualization:
        if which == 'fluid':
            # print('Updating fluid', mappedQuantity)
            sc = scatterPlotUpdate(sc, axis, fluidParticles, domain, mappedQuantity, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'o', markerSize)
            if cbar:
                # cb = fig.colorbar(sc, ax=axis)
                # cb.mappable.set_clim(sc.get_clim())
                # cb.mappable.set_norm(sc.norm)
                cb.update_normal(sc)    

            if visualizeBoth:
                scBoundary.set_offsets(boundaryParticles.positions.cpu().detach().numpy())
                # sc = scatterPlotUpdate(sc, axis, boundaryParticles, domain, None, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'x', markerSize)
        elif which == 'boundary':
            sc = scatterPlotUpdate(sc, axis, boundaryParticles, domain, boundaryQuantity, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'x', markerSize)
            if cbar:
                # cb = fig.colorbar(sc, ax=axis)
                # cb.mappable.set_clim(sc.get_clim())
                # cb.mappable.set_norm(sc.norm)
                cb.update_normal(sc)
            if visualizeBoth:
                scBoundary.set_offsets(fluidParticles.positions.cpu().detach().numpy())
                # sc = scatterPlotUpdate(sc, axis, fluidParticles, domain, None, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'o', markerSize)
        elif which == 'both':
            mergedParticles = ParticleSetWithQuantity(
                torch.vstack([fluidParticles.positions, boundaryParticles.positions],).view(-1, fluidParticles.positions.shape[-1]),
                torch.hstack([fluidParticles.supports, boundaryParticles.supports]),
                torch.hstack([fluidParticles.masses, boundaryParticles.masses]),
                torch.hstack([fluidParticles.densities, boundaryParticles.densities]),
                torch.cat([mappedQuantity, boundaryQuantity], dim = 0)
            )
            sc = scatterPlotUpdate(sc, axis, mergedParticles, domain, mergedParticles.quantities, cbar, cmap, scaling, linthresh, vmin, vmax, midPoint, 'o', markerSize)
        # return
    else:
        if which == 'fluid':
            grid, interpolated = mapToGrid(fluidParticles, mappedQuantity, domain, kernel, gridResolution)
        elif which == 'boundary':
            grid, interpolated = mapToGrid(boundaryParticles, boundaryQuantity, domain, kernel, gridResolution)
        else:
            mergedParticles = ParticleSetWithQuantity(
                positions = torch.cat([fluidParticles.positions, boundaryParticles.positions], dim = 0),
                supports = torch.cat([fluidParticles.supports, boundaryParticles.supports], dim = 0),
                masses = torch.cat([fluidParticles.masses, boundaryParticles.masses], dim = 0),
                densities = torch.cat([fluidParticles.densities, boundaryParticles.densities], dim = 0),
                quantities = torch.cat([mappedQuantity, boundaryQuantity], dim = 0)
            )
            grid, interpolated = mapToGrid(mergedParticles, mergedParticles.quantities, domain, kernel, gridResolution)
        # grid, interpolated = mapToGrid(fluidParticles, mappedQuantity, domain, kernel, gridResolution)

        sc = gridPlotUpdate(axis, sc, grid, interpolated, cmap, scaling, linthresh, vmin, vmax, midPoint)
        if cbar:
            # cb = fig.colorbar(sc, ax=axis)
            # cb.mappable.set_clim(sc.get_clim())
            # cb.mappable.set_norm(sc.norm)
            cb.update_normal(sc)
        # return


    if streamLines:
        if len(mappedQuantity.shape) == 2:
            # print('Using mapped quantity')
            inputQuantity = mappedQuantity
        elif len(preMapping.shape) == 2:
            # print('Using pre mapped quantity')
            inputQuantity = preMapping
        elif len(fluidParticles.quantities.shape) == 2:
            # print('Using fluid particles quantity')
            inputQuantity = fluidParticles.quantities
        elif quantity is not None and len(quantity.shape) == 2:
            # print('Using input quantity')
            inputQuantity = quantity
        else:
            # print('Using mapped quantity pre op')
            inputQuantity = mappedQuantity
        
        if streamLineOperation is not None:
            neighborhood = buildNeighborhood(fluidParticles, fluidParticles, domain, verletScale= 1.0).fullAdjacency

            currentFluidParticles = ParticleSetWithQuantity(fluidParticles.positions, fluidParticles.supports, fluidParticles.masses, fluidParticles.densities, inputQuantity)

            inputQuantity = sph_op(currentFluidParticles, currentFluidParticles, domain, kernel, neighborhood, operation = streamLineOperation, gradientMode = streamLineGradientMode, laplaceMode = streamLineLaplaceMode, divergenceMode = streamLineDivergenceMode, consistentDivergence = streamLineConsistentDivergence)
        
        if len(inputQuantity.shape) != 2:
            raise ValueError('Streamlines require a 2D quantity')

        grid_ux, quant_ux = mapToGrid(fluidParticles, inputQuantity[:, 0], domain, kernel, gridResolution)
        grid_uy, quant_uy = mapToGrid(fluidParticles, inputQuantity[:, 1], domain, kernel, gridResolution)

        # print(grid_ux.shape, quant_ux.shape)
        keep = lambda x: not isinstance(x, mpl.patches.FancyArrowPatch)
        axis.patches = [patch for patch in axis.patches if keep(patch)]
        # raise ValueError('Streamlines not implemented yet')


        streamPlot = axis.streamplot(grid_ux[0].T.detach().cpu().numpy(), grid_ux[1].T.detach().cpu().numpy(), quant_ux.T.detach().cpu().numpy(), quant_uy.T.detach().cpu().numpy(), color='black', linewidth=1, arrowstyle='->')



    # fig, axis,
    # fluidParticles      : Union[ParticleSet, ParticleSetWithQuantity],
    # domain              : DomainDescription,
    # quantity            : Optional[torch.Tensor] = None,
    

    # boundaryParticles   : Optional[Union[ParticleSet, ParticleSetWithQuantity]] = None,

    # which               : str = 'fluid',
    # visualizeBoth       : bool = False,

    # cbar                : bool = True,
    # cmap                : str = 'viridis',
    # scaling             : str = 'linear',
    # linthresh           : float = 1e-3,
    # vmin                : Optional[float] = None,
    # vmax                : Optional[float] = None,
    # midPoint            : Optional[Union[float,str]] = None,
    # domainEpsilon       : float = 0.05,
    # markerSize          : float = 4,

    # mapping             : str = '.x',
    # gridVisualization   : bool = False,
    # gridResolution      : int = 128,
    # streamLines         : bool = False,

    # operation           : Optional[str] = None,
    # gradientMode = 'naive',
    # laplaceMode = 'naive',
    # divergenceMode = 'div',
    # consistentDivergence = False,

    # streamLineOperation : Optional[str] = None,
    # streamLineGradientMode = 'naive',
    # streamLineLaplaceMode = 'naive',
    # streamLineDivergenceMode = 'div',
    # streamLineConsistentDivergence = False,

    # plotDomain          : bool = True,

    # title               : Optional[str] = None,


# from util import getModPosition
import torch
from diffSPH.sphOperations.shared import scatter_sum
from matplotlib.colors import LogNorm

def plotDistribution(fig, axis, particleState, neighborhood, logNorm = True, nnx = 63):
    ddx = 2 / (nnx)
    hij = (particleState.supports[neighborhood[0].row] + particleState.supports[neighborhood[0].col]) / 2
    # print(hij.shape)
    # print(neighborhood[0].row, neighborhood[0].col)

    positions = neighborhood[1].x_ij / hij.view(-1,1)
    # positions = positions[neighborhood['indices'][0] != neighborhood['indices'][1]]

    index = ((positions + 1) / ddx).to(torch.int64)
    linIdx = index[:,0] * nnx + index[:,1]
    # print(linIdx, linIdx.min(), linIdx.max())

    counter = scatter_sum(torch.ones_like(linIdx), dim = 0, index = linIdx, dim_size = nnx**2).reshape(nnx,nnx).cpu().numpy()
    if logNorm:
        sc = axis.imshow(counter, norm=LogNorm(), extent=(-1, 1, -1, 1))
    else:
        sc = axis.imshow(counter, extent=(-1, 1, -1, 1))

    # print(counter.min(), counter.max()) 

    # neighGrid = torch.zeros(64,64)
    cbar = fig.colorbar(sc, ax=axis)
    axis.set_aspect('equal')
    axis.set_title('Neighbor Distribution')
    return sc, cbar