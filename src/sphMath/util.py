
from scipy.interpolate import RegularGridInterpolator
# from sphMath.noise import generateNoise 
from torch.autograd import gradcheck
import torch
import warnings

try:
    # raise ImportError("Debug Test")
    import torchCompactRadius
    from torchCompactRadius.util import SparseCOO, SparseCSR, PointCloud, DomainDescription, coo_to_csr, csr_to_coo
except ImportError:
    # warnings.warn('Using fallback implementation for radius search.')
    from sphMath.neighborhoodFallback.fallback import SparseCOO, SparseCSR, PointCloud, DomainDescription, coo_to_csr, csr_to_coo



from typing import Optional, Tuple

from typing import Union
import warnings

scatter_max_op = None
try:
    # raise ImportError("Debug Test")
    import torchCompactRadius
    import torchCompactRadius_cuda
    scatter_max_op = torch.ops.torchCompactRadius_cuda.scatter_max
except ImportError:
    # warnings.warn('No cuda version of the neighbor search is available.')
    pass
except Exception as e:
    raise e
if scatter_max_op is None:
    try: 
        # raise ImportError("Debug Test")
        import torchCompactRadius
        import torchCompactRadius_cpu
        # raise ImportError("Debug Test")
        scatter_max_op = torch.ops.torchCompactRadius_cpu.scatter_max
    except Exception as e:
        # raise e
        warnings.warn("No Implementation of scatter_max is available. CRKSPH and the Cullen Dehnen Viscosity Switch won't work.")
    
# from sphMath.boundary import RigidBody
def cloneOptionalTensor(T : Optional[torch.Tensor]):
    if T is None:
        return None
    else:
        return T.clone()

def checkTensor(T : torch.Tensor, dtype, device, label, verbose = False):
    if T is None:
        return
    if verbose:
        print(f'Tensor {label:16s}: {T.min().item():10.3e} - {T.max().item():10.3e} - {T.median().item():10.3e} - {T.std().item():10.3e} - \t {T.dtype} - {T.device} - {T.shape}')
    
    if T.dtype != dtype:
        raise ValueError(f'{label} has incorrect dtype {T.dtype} != {dtype}')
    if T.device != device:
        raise ValueError(f'{label} has incorrect device {T.device} != {device}')
    # if not T.is_contiguous():
        # raise ValueError(f'{label} is not contiguous')
    if torch.isnan(T).any():
        raise ValueError(f'{label} has NaN values')
    if torch.isinf(T).any():
        raise ValueError(f'{label} has Inf values')
# def scatter_min(
#         src: torch.Tensor,
#         index: torch.Tensor,
#         dim: int = -1,
#         out: Optional[torch.Tensor] = None,
#         dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#     return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)


def scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if scatter_max_op is None:
        raise RuntimeError('No scatter_max operation available. Please install torchCompactRadius.')
    return scatter_max_op(src, index, dim, out, dim_size)


from typing import NamedTuple
class ParticleSet(NamedTuple):
    positions: torch.Tensor
    supports: torch.Tensor

    masses: torch.Tensor
    densities: torch.Tensor

class ParticleSetWithQuantity(NamedTuple):
    positions: torch.Tensor
    supports: torch.Tensor

    masses: torch.Tensor
    densities: torch.Tensor

    quantities: torch.Tensor

@torch.jit.script
def mod(x, min : float, max : float):
    h = max - min
    return ((x + h / 2.0) - torch.floor((x + h / 2.0) / h) * h) - h / 2.0
# @torch.jit.script
# def mod(x, min : float, max : float):
    # return torch.where(torch.abs(x) > (max - min) / 2, torch.sgn(x) * ((torch.abs(x) + min) % (max - min) + min), x)

def moduloDistance(xij, periodicity, min, max):
    return torch.stack([xij[:,i] if not periodic else mod(xij[:,i], min[i], max[i]) for i, periodic in enumerate(periodicity)], dim = -1)

def mod_distance(xi, xj, 
                 periodicity : Union[bool, torch.Tensor], 
                 minExtent : torch.Tensor, maxExtent : torch.Tensor):
    xij = xi - xj
    periodic = periodicity if isinstance(periodicity, torch.Tensor) else torch.tensor([periodicity for _ in range(xij.shape[1])]).to(xij.device)
    return moduloDistance(xij, periodic, minExtent, maxExtent)


def mergeParticles(particles_l, particles_r):
    positions = torch.cat([particles_l.positions, particles_r.positions], dim = 0)
    supports = torch.cat([particles_l.supports, particles_r.supports], dim = 0)
    masses = torch.cat([particles_l.masses, particles_r.masses], dim = 0)
    densities = torch.cat([particles_l.densities, particles_r.densities], dim = 0)

    sortedIndices = torch.argsort(positions[:, 0])
    positions = positions[sortedIndices]
    supports = supports[sortedIndices]
    masses = masses[sortedIndices]
    densities = densities[sortedIndices]

    if hasattr(particles_l, 'quantities'):
        quantities = torch.cat([particles_l.quantities, particles_r.quantities], dim = 0)
        quantities = quantities[sortedIndices]
        return ParticleSetWithQuantity(positions, supports, masses, densities, quantities)
    return ParticleSet(positions, supports, masses, densities)



import numpy as np
from typing import Union
# @torch.jit.script
def volumeToSupport(volume : float, targetNeighbors : Union[int, float], dim : int):
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
        if isinstance(volume, torch.Tensor):
            return torch.sqrt(targetNeighbors * volume / np.pi)
        else:
            return np.sqrt(targetNeighbors * volume / np.pi)
    else:
        # N_h = 4/3 \pi h^3 / v -> h = \sqrt[3]{N_h * v / \pi * 3/4}
        if isinstance(volume, torch.Tensor):
            return torch.pow(targetNeighbors * volume / np.pi * 3 /4, 1/3)
        else:
            return np.pow(targetNeighbors * volume / np.pi * 3 /4, 1/3)


def domainSDF(p, domain, invert = False):
    center = (domain.min + domain.max) / 2
    halfExtents = (domain.max - domain.min) / 2
    d = torch.abs(p - center) - halfExtents
    return (1 if invert else -1) * (torch.linalg.norm(d.clamp(min = 0), dim = -1) + torch.max(d, dim = -1)[0].clamp(max=0))

def sampleDomainSDF(x, domain, invert = False):
    x_ = x.clone()
    x_.requires_grad = True
    d = domainSDF(x_, domain, invert = invert)
    grad = torch.autograd.grad(outputs = d, inputs = x_, grad_outputs = torch.ones_like(d), create_graph = True, retain_graph = True)

    return d.detach() if x.requires_grad == False else d, grad[0].detach() if x.requires_grad == False else grad[0]

def getPeriodicPositions(x, domain):
    minD = domain.min.cpu().detach()
    maxD = domain.max.cpu().detach()
    periodicity = domain.periodic
    pos = [(torch.remainder(x[:, i] - minD[i], maxD[i] - minD[i]) + minD[i]) if periodicity[i] else x[:,i] for i in range(domain.dim)]
    modPos = torch.stack(pos, dim = -1)
    return modPos


def getSetConfig(config, namespace, key, default):
    if namespace in config:
        if key in config[namespace]:
            return config[namespace][key]
        else:
            print(f'Key {key} not found in config["{namespace}"]. Setting {key} to {default}')
            config[namespace][key] = default
            return default
    else:
        print(f'Namespace {namespace} not found in config. Setting {key} to {default}')
        config[namespace] = {}
        config[namespace][key] = default
        return default
    
    
import os
import subprocess
import shlex
import imageio.v2 as imageio
from skimage.transform import resize, rescale
from skimage.io import imread

def postProcess(imagePrefix, fps, exportName, targetLongEdge = 600):
    fileList = os.listdir(imagePrefix)
    fileList = [f for f in fileList if f.endswith('.png') and f.startswith('frame_')]
    fileList = sorted(fileList)

    writer = imageio.get_writer(f'{imagePrefix}/output.mp4', fps=fps, bitrate='10M')
    for image in fileList:
        writer.append_data(imageio.imread(imagePrefix + image))
    writer.close()

    images = []
    for image in fileList:
        images.append(imread(imagePrefix + image))

    currentImageSize = images[0].shape
    currentLongEdge = max(images[0].shape[0], images[0].shape[1])
    ratio = targetLongEdge / currentLongEdge

    images = [(rescale(image, ratio, anti_aliasing=True, channel_axis=-1)*255).astype('uint8') for image in images]

    # for image in images:
        # print(image.shape)
        # break

    # images = [resizeImageLongEdge(i, 600) for i in images]

    imageio.mimsave(f'{imagePrefix}/output.gif', images, fps=fps, loop = 0)
    
    print('Copying video to videos folder')
    os.makedirs(f'./videos/', exist_ok= True)
    # subprocess.run(shlex.split(f'cp {imagePrefix}/output.mp4 ./videos/{exportName}.mp4'))
    subprocess.run(shlex.split(f'cp {imagePrefix}/output.gif ./videos/{exportName}.gif'))
    lastFrameFile = imagePrefix + fileList[-1]
    subprocess.run(shlex.split(f'cp {lastFrameFile} ./videos/{exportName}.png'))
    print('Done!')



from dataclasses import dataclass
from typing import Optional
@torch.jit.script
@dataclass(slots = True)
class KernelTerms:
    # CRK-SPH Terms:
    A: Optional[torch.Tensor] = None
    B: Optional[torch.Tensor] = None
    gradA: Optional[torch.Tensor] = None
    gradB: Optional[torch.Tensor] = None

    # Kernel Gradient Renormalization
    gradCorrectionMatrices: Optional[torch.Tensor] = None

    # grad-H Term:
    omega: Optional[torch.Tensor] = None
