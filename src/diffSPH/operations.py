import torch

from typing import Optional
from typing import Tuple
from typing import Union

from diffSPH.util import mod, moduloDistance, mod_distance


@torch.jit.script
def access_optional(tensor: Optional[torch.Tensor], i: torch.Tensor) -> torch.Tensor:
    if tensor is not None:
        return tensor[i]
    else:
        raise ValueError("Tensor is None and cannot be accessed.")

def product(a,  b):
    if len(a.shape) == 1 and len(b.shape) == 1:
        return a * b
    elif len(a.shape) == 1 and len(b.shape) != 1:
        return torch.einsum('n, n... -> n...', a, b)
    elif len(a.shape) != 1 and len(b.shape) == 1:
        return torch.einsum('n..., n -> n...', a, b)
    else:
        raise ValueError(f"Invalid shapes {a.shape} and {b.shape}")
    
# ------ Beginning of scatter functionality ------ #
# Scatter summation functionality based on pytorch geometric scatter functionality
# This is included here to make the code independent of pytorch geometric for portability
# Note that pytorch geometric is licensed under an MIT licenses for the PyG Team <team@pyg.org>
@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

@torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out_ = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out_.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)
# ------ End of scatter functionality ------ #

from typing import Union, Tuple

def get_i(q : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], i):
    if isinstance(q, torch.Tensor):
        return q[i]
    else:
        return q[0][i]
def get_j(q : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], j):
    if isinstance(q, torch.Tensor):
        return q[j]
    else:
        return q[1][j]
    
def getSupport(h : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], i , j, mode = 'scatter'):
    if isinstance(h, torch.Tensor):
        return h[i]
    else:
        if mode == 'scatter':
            return get_j(h, j)
        elif mode == 'gather':
            return get_i(h, i)
        elif mode == 'symmetric':
            return (get_i(h, i) + get_j(h, j))/2
        else:
            raise ValueError(f"Unknown mode {mode}")
        

def sph_density(masses      : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                positions   : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                supports    : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                kernel, 
                i : torch.Tensor, j : torch.Tensor,
                support : str = 'scatter'
                ):
    mj = get_j(masses, j)
    Wij = kernel(get_i(positions, i) - get_j(positions, j), getSupport(supports, i, j, mode = support))

    ni = positions[0].shape[0] if isinstance(positions, tuple) else positions.shape[0]
    

    return scatter_sum(mj * Wij, i, dim_size=ni, dim = 0)

# from kernels import SPHKernel
# def SPHKernel(fn, 
#                   masses : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
#                   densities : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
#                   quantities : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
#                   positions : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
#                   supports : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
#                   kernel : SPHKernel,
#                   i : torch.Tensor, j : torch.Tensor,
#                   support : str = 'symmetric',
#                   **kwargs):
#     return fn(
#         masses if isinstance(masses, torch.Tensor) else masses[0],
#         masses if isinstance(masses, torch.Tensor) else masses[1],
#         densities if isinstance(densities, torch.Tensor) else densities[0],
#         densities if isinstance(densities, torch.Tensor) else densities[1],
#         quantities if isinstance(quantities, torch.Tensor) else quantities[0],
#         quantities if isinstance(quantities, torch.Tensor) else quantities[1],
#         positions if isinstance(positions, torch.Tensor) else positions[0],
#         positions if isinstance(positions, torch.Tensor) else positions[1],
#         supports if isinstance(supports, torch.Tensor) else supports[0],
#         supports if isinstance(supports, torch.Tensor) else supports[1],        
#         kernel,
#         i, j,
#         support,
#         **kwargs
#     )

from diffSPH.kernels import SPHKernel


from torch.profiler import profile, record_function, ProfilerActivity


from diffSPH.sphOperations.shared import wrapper
from diffSPH.sphOperations.interpolate import SPHInterpolation
from diffSPH.sphOperations.gradient import SPHGradient
from diffSPH.sphOperations.divergence import SPHDivergence
from diffSPH.sphOperations.curl import SPHCurl
from diffSPH.sphOperations.laplacian import SPHLaplacian

def sph_operation(
        masses : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        densities : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        quantities : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        positions : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        supports : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        kernel: SPHKernel,
        i, j,
        minExtent : torch.Tensor, maxExtent : torch.Tensor,
        periodicity : torch.Tensor,
        supportScheme = 'scatter',
        operation = 'interpolate',
        gradientMode = 'naive',
        laplaceMode = 'naive',
        divergenceMode = 'div',
        consistentDivergence = False,
        gradientRenormalizationMatrix = None,
        gradHTerms = None,
        piSwitch = False,
        rotationMatrix : Optional[torch.Tensor] = None,
        batches : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None
):
    if operation == 'density':
        with record_function(f"[SPH rho] {supportScheme}"):
            return wrapper(SPHInterpolation.apply, masses, masses, masses, positions, supports, kernel, i, j, supportScheme, periodicity, minExtent, maxExtent, rotationMatrix, batches)
    if operation == 'interpolate':
        with record_function(f"[SPH op] {supportScheme}"):
            return wrapper(SPHInterpolation.apply, masses, densities, quantities, positions, supports, kernel, i, j,  supportScheme, periodicity, minExtent, maxExtent, rotationMatrix, batches)
    elif operation == 'gradient':
        with record_function(f"[SPH grad] {gradientMode} {supportScheme}{' ReNorm' if gradientRenormalizationMatrix is not None else ' '}{' gradH' if gradHTerms is not None else ''}"):
            return wrapper(SPHGradient.apply, masses, densities, quantities, positions, supports, kernel, i, j, supportScheme, gradientMode, periodicity, minExtent, maxExtent, gradientRenormalizationMatrix, gradHTerms, rotationMatrix, batches)
    elif operation == 'divergence':
        with record_function(f"[SPH div] {divergenceMode} {supportScheme}{' consistent' if consistentDivergence else ''}"):
            return wrapper(SPHDivergence.apply, masses, densities, quantities, positions, supports, kernel, i, j, supportScheme, gradientMode, divergenceMode, consistentDivergence, periodicity, minExtent,  maxExtent, rotationMatrix, batches)
    elif operation == 'curl':
        with record_function(f"[SPH curl] {supportScheme}"):
            return wrapper(SPHCurl.apply, masses, densities, quantities, positions, supports, kernel, i, j, supportScheme, gradientMode, periodicity, minExtent, maxExtent, rotationMatrix, batches)
    elif operation == 'laplacian':
        with record_function(f"[SPH laplacian] {laplaceMode} {supportScheme}"):
            return wrapper(SPHLaplacian.apply, masses, densities, quantities, positions, supports, kernel, i, j, supportScheme, laplaceMode, periodicity, minExtent, maxExtent, piSwitch, rotationMatrix, batches)
    else:
        raise ValueError(f"Unknown operation {operation}")
    

from diffSPH.util import ParticleSetWithQuantity, ParticleSet, DomainDescription
from diffSPH.neighborhood import NeighborhoodInformation, SparseCOO, SparseCSR


from typing import List
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

def sph_op(
    particles_a : Union[ParticleSet, ParticleSetWithQuantity],
    particles_b : Union[ParticleSet, ParticleSetWithQuantity],
    domain : DomainDescription,
    kernel: SPHKernel,
    neighborhood: SparseCOO,
    supportScheme = 'scatter',
    operation = 'interpolate',
    gradientMode = 'naive',
    laplaceMode = 'naive',
    divergenceMode = 'div',
    consistentDivergence = False,
    quantity : Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    gradientRenormalizationMatrix :Optional[torch.Tensor] = None,
    gradHTerms : Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    piSwitch = False,
    rotationMatrix : Optional[torch.Tensor] = None,
    batchTensor: Optional[torch.Tensor] = None
):
    quantities = (particles_a.quantities if hasattr(particles_a, 'quantities') else None, particles_b.quantities if hasattr(particles_b, 'quantities') else None)
    if quantity is not None:
        if isinstance(quantity, torch.Tensor):
            if quantity.shape[0] == neighborhood.row.shape[0]:
                # print('using single quantity as prescattered')
                quantities = quantity
            else:
                if quantity.shape[0] == particles_a.positions.shape[0] and (particles_b is None or quantity.shape[0] != particles_b.positions.shape[0]):
                    quantities = (quantity, None)
                elif quantity.shape[0] == particles_b.positions.shape[0] and (particles_a is None or quantity.shape[0] != particles_a.positions.shape[0]):
                    quantities = (None, quantity)
                elif quantity.shape[0] == particles_a.positions.shape[0] and quantity.shape[0] == particles_b.positions.shape[0]:
                    quantities = (quantity, quantity)
                else:
                    raise ValueError(f"Invalid quantity shape {quantity.shape} for particles_a {particles_a.positions.shape} and particles_b {particles_b.positions.shape}")
        elif isinstance(quantity, tuple):
            quantities = quantity

    rotationMatrices = rotationMatrix

    if rotationMatrices is None:
        if isinstance(domain, List):
            if hasattr(domain[0], 'angles'):
                rotationMatrices = [buildRotationMatrix(torch.tensor(d.angles), d.dim, device = d.device, dtype = d.dtype) for d in domain]
        else:
            if hasattr(domain, 'angles'):
                rotationMatrices = buildRotationMatrix(torch.tensor(domain.angles), domain.dim, device = domain.device, dtype = domain.dtype)
            else:
                rotationMatrices = None


    return sph_operation(
        (particles_a.masses if hasattr(particles_a, 'masses') else None, particles_b.masses),
        (particles_a.densities if hasattr(particles_a, 'densities') else None, particles_b.densities),
        quantities,
        (particles_a.positions, particles_b.positions),
        (particles_a.supports, particles_b.supports),
        kernel,
        neighborhood.row, neighborhood.col,
        [d.min for d in domain] if isinstance(domain, List) else domain.min, [d.max for d in domain] if isinstance(domain, List) else domain.max,
        [d.periodic for d in domain] if isinstance(domain, List) else domain.periodic,
        supportScheme,
        operation,
        gradientMode,
        laplaceMode,
        divergenceMode,
        consistentDivergence,
        gradientRenormalizationMatrix,
        gradHTerms,
        piSwitch,
        rotationMatrices,
        batchTensor if batchTensor is not None else particles_a.batches if hasattr(particles_a, 'batches') else None
)

from diffSPH.schemes.states.compressiblesph import CompressibleState
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState
from diffSPH.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme




from diffSPH.sphOperations.preComputed import invokeOperation
from diffSPH.enums import *
from typing import List
from diffSPH.util import KernelTerms

from dataclasses import dataclass
@torch.jit.script
@dataclass(slots = True)
class ParticleState:
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    densities: torch.Tensor
    apparentArea: Optional[torch.Tensor]
from diffSPH.schemes.states.common import BasicState

def getParticleState(particles : Union[BasicState, CompressibleState, WeaklyCompressibleState]):
    if isinstance(particles, BasicState):
        return ParticleState(
            positions = particles.positions,
            supports = particles.supports,
            masses = particles.masses,
            densities = particles.densities,
            apparentArea = particles.apparentArea
        )
    elif isinstance(particles, CompressibleState):
        return ParticleState(
            positions = particles.positions,
            supports = particles.supports,
            masses = particles.masses,
            densities = particles.densities,
            apparentArea = particles.apparentArea
        )
    elif isinstance(particles, WeaklyCompressibleState):
        return ParticleState(
            positions = particles.positions,
            supports = particles.supports,
            masses = particles.masses,
            densities = particles.densities,
            apparentArea = particles.apparentArea
        )
    else:
        raise ValueError(f'Unknown type for argument {particles}')
    
def getCorrectionTerms(particles : Union[BasicState, CompressibleState, WeaklyCompressibleState]):
    if isinstance(particles, BasicState):
        return KernelTerms(
            A = particles.A,
            B = particles.B,
            gradA = particles.gradA,
            gradB = particles.gradB,
            gradCorrectionMatrices=particles.gradCorrectionMatrices,
            omega = particles.omega
        )
    elif isinstance(particles, CompressibleState):
        return KernelTerms(
            A = particles.A,
            B = particles.B,
            gradA = particles.gradA,
            gradB = particles.gradB,
            gradCorrectionMatrices=particles.gradCorrectionMatrices,
            omega = particles.omega
        )
    elif isinstance(particles, WeaklyCompressibleState):
        return KernelTerms(
            A = particles.A,
            B = particles.B,
            gradA = particles.gradA,
            gradB = particles.gradB,
            gradCorrectionMatrices=particles.gradCorrectionMatrices,
            omega = particles.omega
        )
    else:
        raise ValueError(f'Unknown type for argument {particles}')



# @torch.jit.script
def SPHOperationCompiled(
    particles_a : Union[BasicState, CompressibleState, WeaklyCompressibleState],
    quantity : Optional[Union[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]],

    neighborhood: SparseNeighborhood,
    kernelValues: PrecomputedNeighborhood,

    operation : Operation = Operation.Interpolate,
    supportScheme : SupportScheme = SupportScheme.Scatter,
    gradientMode : GradientMode = GradientMode.Naive,
    divergenceMode : DivergenceMode = DivergenceMode.div,
    laplacianMode : LaplacianMode = LaplacianMode.default,
    consistentDivergence : bool = False,
    useApparentArea: bool = False,

    # gradientRenormalizationMatrix : Optional[torch.Tensor] = None,
    # gradHTerms : Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    correctionTerms: Optional[List[KernelCorrectionScheme]] = None,

    particles_b : Optional[Union[BasicState, CompressibleState, WeaklyCompressibleState]] = None,
    positiveDivergence : bool = False
):
    ni = neighborhood.numRows
    nj = neighborhood.numCols

    stateA = getParticleState(particles_a)
    stateB = getParticleState(particles_b) if particles_b is not None else stateA
    correctionTermsA = getCorrectionTerms(particles_a)
    correctionTermsB = getCorrectionTerms(particles_b) if particles_b is not None else correctionTermsA


    positions_a = stateA.positions 
    supports_a = stateA.supports
    masses_a = stateA.masses
    densities_a = stateA.densities
    apparentArea_a = stateA.apparentArea
    
    if particles_b is not None:
        positions_b = stateB.positions
        supports_b = stateB.supports
        masses_b = stateB.masses 
        densities_b = stateB.densities
        apparentArea_b = stateB.apparentArea
    else:
        positions_b = positions_a
        supports_b = supports_a
        masses_b = masses_a
        densities_b = densities_a
        apparentArea_b = apparentArea_a

    if particles_b is None:
        particles_b = particles_a

    quantities : Union[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = (None, None)
    
    if quantity is None:
        if operation == Operation.Density:
            quantities = (torch.ones_like(masses_a), torch.ones_like(masses_b))
        else:
            raise ValueError(f"Quantity is None for operation {operation}")
    else:
        if isinstance(quantity, torch.Tensor):
            if quantity.shape[0] == neighborhood.row.shape[0]:
                quantities = quantity
            else:
                if quantity.shape[0] == ni and (particles_b is None or quantity.shape[0] != nj):
                    quantities = (quantity, None)
                elif quantity.shape[0] == nj and (particles_a is None or quantity.shape[0] != ni):
                    quantities = (None, quantity)
                elif quantity.shape[0] == ni and quantity.shape[0] == nj:
                    quantities = (quantity, quantity)
                else:
                    raise ValueError(f"Invalid quantity shape {quantity.shape} for particles_a {ni} and particles_b {nj}")
        elif isinstance(quantity, tuple):
            quantities = quantity
        else:
            raise ValueError(f"Invalid quantity {quantity}")
    
    return invokeOperation(
            positions_a = positions_a,
            positions_b = positions_b,
            supports_a = supports_a,
            supports_b = supports_b,
            masses_a = masses_a,
            masses_b = masses_b,
            densities_a = densities_a,
            densities_b = densities_b,
            apparentArea_a = apparentArea_a,
            apparentArea_b = apparentArea_b,

            quantity_a= quantities[0] if isinstance(quantities, tuple) else None,
            quantity_b= quantities[1] if isinstance(quantities, tuple) else None,
            quantity_ab = quantities if not isinstance(quantities, tuple) else None,

            i = neighborhood.row,
            j = neighborhood.col,
            numRows = neighborhood.numRows,
            numCols = neighborhood.numCols,

            r_ij = kernelValues.r_ij,
            x_ij = kernelValues.x_ij,

            W_i = kernelValues.W_i,
            W_j = kernelValues.W_j,
            gradW_i = kernelValues.gradW_i,
            gradW_j = kernelValues.gradW_j,

            H_i = kernelValues.H_i,
            H_j = kernelValues.H_j,

            gradH_i= kernelValues.ddh_W_i,
            gradH_j= kernelValues.ddh_W_j,

            operation = operation,
            supportScheme = supportScheme,
            gradientMode = gradientMode,
            divergenceMode = divergenceMode,
            laplacianMode = laplacianMode,
            consistentDivergence = consistentDivergence,
            useApparentArea = useApparentArea,
            correctionTerms = correctionTerms,
            kernelCorrectionValues = (correctionTermsA, correctionTermsB),
            positiveDivergence = positiveDivergence,

        )    

from diffSPH.neighborhood import evalSparseNeighborhood
from diffSPH.kernels import KernelType

def SPHOperation(
    particles : Union[BasicState, CompressibleState, WeaklyCompressibleState],
    quantity : Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],

    kernel: KernelType,
    neighborhood: SparseNeighborhood,
    kernelValues: Optional[PrecomputedNeighborhood] = None,

    operation : Operation = Operation.Interpolate,
    supportScheme : SupportScheme = SupportScheme.Scatter,
    gradientMode : GradientMode = GradientMode.Naive,
    divergenceMode : DivergenceMode = DivergenceMode.div,
    laplacianMode : LaplacianMode = LaplacianMode.default,
    consistentDivergence : bool = False,
    useApparentArea: bool = False,

    # gradientRenormalizationMatrix : Optional[torch.Tensor] = None,
    # gradHTerms : Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    correctionTerms: Optional[List[KernelCorrectionScheme]] = None,

    particles_b : Optional[Union[BasicState, CompressibleState, WeaklyCompressibleState]] = None,
    positiveDivergence : bool = False
):
    kernelValues = kernelValues if kernelValues is not None else evalSparseNeighborhood(neighborhood, kernel)

    return SPHOperationCompiled(
        particles,
        quantity,
        neighborhood,
        kernelValues,
        operation,
        supportScheme,
        gradientMode,
        divergenceMode,
        laplacianMode,
        consistentDivergence,

        useApparentArea,
        correctionTerms,
        particles_b if particles_b is not None else particles,
        positiveDivergence=positiveDivergence
    )
