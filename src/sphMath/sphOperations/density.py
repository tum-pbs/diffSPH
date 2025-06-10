import torch
from sphMath.kernels import SPHKernel
from typing import Union, Tuple, Optional
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product

from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.states.compressiblesph import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, DomainDescription, PrecomputedNeighborhood
from sphMath.enums import Operation, SupportScheme, GradientMode, DivergenceMode, LaplacianMode
from sphMath.enums import KernelCorrectionScheme
from typing import List
from sphMath.util import KernelTerms



from sphMath.enums import KernelCorrectionScheme
from typing import List
from sphMath.util import KernelTerms
# from sphMath.neighborhood 
from sphMath.sphOperations.opUtil import evalSupport, evalPrecomputed, get_qj, get_qs, correctedKernel_CRK, correctedKernelGradient_CRK

def interpolate_precomputed(
        positions_a : torch.Tensor,
        positions_b : torch.Tensor,

        supports_a : torch.Tensor,
        supports_b : torch.Tensor,

        masses_a : torch.Tensor,
        masses_b : torch.Tensor,

        densities_a : torch.Tensor,
        densities_b : torch.Tensor,
        apparentArea_a: Optional[torch.Tensor],
        apparentArea_b: Optional[torch.Tensor],

        quantity_a : Optional[torch.Tensor],
        quantity_b : Optional[torch.Tensor],
        quantity_ab : Optional[torch.Tensor],
    
        i: torch.Tensor,
        j: torch.Tensor,
        numRows: int,
        numCols: int,

        r_ij: torch.Tensor,
        x_ij: torch.Tensor,

        W_i: torch.Tensor,
        W_j: torch.Tensor,
        gradW_i: torch.Tensor,
        gradW_j: torch.Tensor,
        H_i: torch.Tensor,
        H_j: torch.Tensor,
        gradH_i: torch.Tensor,
        gradH_j: torch.Tensor,

        operation : Operation = Operation.Interpolate,
        supportScheme : SupportScheme = SupportScheme.Scatter,
        gradientMode : GradientMode = GradientMode.Naive,
        divergenceMode : DivergenceMode = DivergenceMode.div,
        laplacianMode : LaplacianMode = LaplacianMode.naive,
        consistentDivergence : bool = False,
        useApparentArea: bool = False,

        correctionTerms: Optional[List[KernelCorrectionScheme]] = None,
        correctionTerm_A: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        correctionTerm_B: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        correctionTerm_gradA: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        correctionTerm_gradB: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        correctionTerm_gradMatrix: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        correctionTerm_omega: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        positiveDivergence:bool = False
    ):
        # rename variables for ease of usage
        positions = (positions_a, positions_b)
        supports = (supports_a, supports_b)
        masses = (masses_a, masses_b)
        densities = (densities_a, densities_b)
        quantity = (quantity_a, quantity_b)
        
        W = (W_i, W_j) 
        gradW = (gradW_i, gradW_j)
        Hessian = (H_i, H_j)
        gradH = (gradH_i, gradH_j)
                    
        # compute relative positions and support radii
        x_ij = x_ij
        h_ij = evalSupport(supports, i, j, mode = supportScheme)

        # compute ancillary variables
        ni = numRows
        nj = numCols

        ## -------------------------------------------------------------- ##
        ## ---------------------- End of preamble ----------------------- ##
        ## -------------------------------------------------------------- ##

        m_j = masses[1][j]
        rho_j = densities[1][j]
        W_ij = evalPrecomputed(W, mode = supportScheme)
        if correctionTerms is not None and KernelCorrectionScheme.CRKSPH.value in [c.value for c in correctionTerms]:
            W_ij = correctedKernel_CRK(i, j, correctionTerm_A[0], correctionTerm_B[0], x_ij, W_ij, False)

        k = m_j / rho_j * W_ij
        if useApparentArea and apparentArea_b is not None:
            k = apparentArea_b[j] * W_ij

        q_j = quantity_ab if quantity_ab is not None else get_qj(quantity, i, j, (None, None))
        kq = torch.einsum('n..., n -> n...', q_j, k)

        return scatter_sum(kq, i, dim_size=ni, dim = 0)
    
def density_precomputed(
        positions_a : torch.Tensor,
        positions_b : torch.Tensor,

        supports_a : torch.Tensor,
        supports_b : torch.Tensor,

        masses_a : torch.Tensor,
        masses_b : torch.Tensor,

        densities_a : torch.Tensor,
        densities_b : torch.Tensor,
        apparentArea_a: Optional[torch.Tensor],
        apparentArea_b: Optional[torch.Tensor],

        quantity_a : Optional[torch.Tensor],
        quantity_b : Optional[torch.Tensor],
        quantity_ab : Optional[torch.Tensor],
    
        i: torch.Tensor,
        j: torch.Tensor,
        numRows: int,
        numCols: int,

        r_ij: torch.Tensor,
        x_ij: torch.Tensor,

        W_i: torch.Tensor,
        W_j: torch.Tensor,
        gradW_i: torch.Tensor,
        gradW_j: torch.Tensor,
        H_i: Optional[torch.Tensor],
        H_j: Optional[torch.Tensor],
        gradH_i: Optional[torch.Tensor],
        gradH_j: Optional[torch.Tensor],

        operation : Operation = Operation.Interpolate,
        supportScheme : SupportScheme = SupportScheme.Scatter,
        gradientMode : GradientMode = GradientMode.Naive,
        divergenceMode : DivergenceMode = DivergenceMode.div,
        laplacianMode : LaplacianMode = LaplacianMode.naive,
        consistentDivergence : bool = False,
        useApparentArea: bool = False,

        correctionTerms: Optional[List[KernelCorrectionScheme]] = None,
        correctionTerm_A: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        correctionTerm_B: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        correctionTerm_gradA: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        correctionTerm_gradB: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        correctionTerm_gradMatrix: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        correctionTerm_omega: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
        positiveDivergence:bool = False
):
        # rename variables for ease of usage
        positions = (positions_a, positions_b)
        supports = (supports_a, supports_b)
        masses = (masses_a, masses_b)
        densities = (densities_a, densities_b)
        quantity = (quantity_a, quantity_b)
        
        W = (W_i, W_j)
        gradW = (gradW_i, gradW_j)
        Hessian = (H_i, H_j)
        gradH = (gradH_i, gradH_j)
        
        # compute relative positions and support radii
        x_ij = x_ij
        h_ij = evalSupport(supports, i, j, mode = supportScheme)

        # compute ancillary variables
        ni = numRows
        nj = numCols

        ## -------------------------------------------------------------- ##
        ## ---------------------- End of preamble ----------------------- ##
        ## -------------------------------------------------------------- ##

        m_j = masses[1][j]
        W_ij = evalPrecomputed(W, mode = supportScheme)
        if correctionTerms is not None and KernelCorrectionScheme.CRKSPH.value in [c.value for c in correctionTerms]:
            W_ij = correctedKernel_CRK(i, j, correctionTerm_A[0], correctionTerm_B[0], x_ij, W_ij, False)

        k = m_j * W_ij
        if useApparentArea and apparentArea_b is not None:
            k = apparentArea_b[j] /densities_b[j] * W_ij

        return scatter_sum(k, i, dim_size=ni, dim = 0)
    

from sphMath.sphOperations.opUtil import custom_forwards, custom_backwards, evaluateKernel_, evaluateKernelGradient_

def density_fn(
        supportScheme: SupportScheme,
        useApparentArea: bool,
        crkCorrection: bool,
        i: torch.Tensor, j: torch.Tensor,
        numRows: int, numCols: int,

        A_i: Optional[torch.Tensor], B_i: Optional[torch.Tensor],
        m_j: torch.Tensor,
        
        x_ij: torch.Tensor,
        W_i: torch.Tensor, W_j: torch.Tensor):    
    W_ij = evaluateKernel_(W_i, W_j, supportScheme, crkCorrection, i, j, x_ij, A_i, B_i)

    k = m_j * W_ij
    return scatter_sum(k, i, dim_size=numRows, dim = 0)

# counter = 0
class Density(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
        positions_a : torch.Tensor,
        positions_b : torch.Tensor,

        supports_a : torch.Tensor,
        supports_b : torch.Tensor,

        masses_a : torch.Tensor,
        masses_b : torch.Tensor,

        apparentArea_a: Optional[torch.Tensor],
        apparentArea_b: Optional[torch.Tensor],

        quantity_a : Optional[torch.Tensor],
        quantity_b : Optional[torch.Tensor],
        quantity_ab : Optional[torch.Tensor],
    
        i: torch.Tensor,
        j: torch.Tensor,
        numRows: int,
        numCols: int,

        r_ij: torch.Tensor,
        x_ij: torch.Tensor,

        W_i: torch.Tensor,
        W_j: torch.Tensor,
        gradW_i: torch.Tensor,
        gradW_j: torch.Tensor,
        H_i: Optional[torch.Tensor],
        H_j: Optional[torch.Tensor],
        gradH_i: Optional[torch.Tensor],
        gradH_j: Optional[torch.Tensor],

        operation : Operation,
        supportScheme : SupportScheme,
        gradientMode : GradientMode,
        divergenceMode : DivergenceMode,
        laplacianMode : LaplacianMode,
        consistentDivergence : bool,
        useApparentArea: bool,

        correctionTerms: Optional[List[KernelCorrectionScheme]],
        correctionTerm_A_i: Optional[torch.Tensor],
        correctionTerm_A_j: Optional[torch.Tensor],
        correctionTerm_B_i: Optional[torch.Tensor],
        correctionTerm_B_j: Optional[torch.Tensor],
        
        correctionTerm_gradA_i: Optional[torch.Tensor],
        correctionTerm_gradA_j: Optional[torch.Tensor],
        correctionTerm_gradB_i: Optional[torch.Tensor] ,
        correctionTerm_gradB_j: Optional[torch.Tensor],
        correctionTerm_gradMatrix_i: Optional[torch.Tensor],
        correctionTerm_gradMatrix_j: Optional[torch.Tensor],
        correctionTerm_omega_i: Optional[torch.Tensor],
        correctionTerm_omega_j: Optional[torch.Tensor],
        positiveDivergence:bool):
        # global counter
        # print("[SPH] - [Density] Forward pass, counter:", counter)
        # Store state for backwards pass
        ctx.save_for_backward(masses_b, W_i, W_j, correctionTerm_A_i, correctionTerm_B_i, x_ij, i, j)
        ctx.correctionTerms = correctionTerms
        ctx.supportScheme = supportScheme
        ctx.useApparentArea = useApparentArea
        ctx.numRows = numRows
        ctx.numCols = numCols
        ctx.crkCorrection = correctionTerms is not None and KernelCorrectionScheme.CRKSPH.value in [c.value for c in correctionTerms]
        # ctx.counter = counter
        # counter += 1

        inputs_i = [correctionTerm_A_i, correctionTerm_B_i]
        inputs_j = [masses_b]
        inputs_ij = [x_ij, W_i, W_j]

        return custom_forwards(
            density_fn, 
            i, j, ctx.numRows, ctx.numCols,
            inputs_i, inputs_j, inputs_ij,
            supportScheme,
            useApparentArea,
            ctx.crkCorrection
        )
    
    @staticmethod
    def backward(ctx, grad_output):
        # print("[SPH] - [Density] Backward pass, counter:", ctx.counter)
        # Load saved tensors
        masses_b, W_i, W_j, correctionTerm_A_i, correctionTerm_B_i, x_ij, i, j = ctx.saved_tensors

        # Load saved variables
        correctionTerms = ctx.correctionTerms
        supportScheme = ctx.supportScheme
        useApparentArea = ctx.useApparentArea
        crkCorrection = ctx.crkCorrection

        inputs_i = [correctionTerm_A_i, correctionTerm_B_i]
        inputs_j = [masses_b]
        inputs_ij = [x_ij, W_i, W_j]

        grad_A_i, grad_B_i, \
            grad_m_j, \
            grad_x_ij, grad_W_i, grad_W_j = custom_backwards(
            density_fn,
            grad_output, i, j, ctx.numRows, ctx.numCols,
            inputs_i, inputs_j, inputs_ij,
            supportScheme,
            useApparentArea,
            ctx.crkCorrection
            )

        return (
            None, None, # positions_a, positions_b, 
            None, None, # supports_a, supports_b, 
            None, grad_m_j, # masses_a, masses_b,
            None, None, # apparentArea_a, apparentArea_b,
            None, None, None, # quantity_a, quantity_b, quantity_ab,
            None, None, # i, j,
            None, None, # numRows, numCols,
            None, grad_x_ij, # r_ij, x_ij,
            grad_W_i, grad_W_j, # W_i, W_j,
            None, None, # gradW_i, gradW_j,
            None, None, # H_i, H_j,
            None, None, # gradH_i, gradH_j,
            None, # operation,
            None, # supportScheme,
            None, # gradientMode,
            None, # divergenceMode,
            None, # laplacianMode,
            None, # consistentDivergence,
            None, # useApparentArea,
            None, # correctionTerms,
            grad_A_i, None, # correctionTerm_A_i, correctionTerm_A_j,
            grad_B_i, None, # correctionTerm_B_i, correctionTerm_B_j,
            None, None, # correctionTerm_gradA_i, correctionTerm_gradA_j,
            None, None, # correctionTerm_gradB_i, correctionTerm_gradB_j,
            None, None, # correctionTerm_gradMatrix_i, correctionTerm_gradMatrix_j,
            None, None, # correctionTerm_omega_i, correctionTerm_omega_j,
            None # positiveDivergence
        )
            
            
density_precomputed_op = Density.apply