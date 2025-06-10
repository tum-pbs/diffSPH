import torch
from sphMath.kernels import SPHKernel
from typing import Union, Tuple, Optional
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product


class SPHInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                masses_i : torch.Tensor, 
                masses_j : torch.Tensor,
                densities_i : torch.Tensor,
                densities_j : torch.Tensor,
                quantities_i : torch.Tensor,
                quantities_j : torch.Tensor,
                positions_i : torch.Tensor,
                positions_j : torch.Tensor,
                supports_i : torch.Tensor,
                supports_j : torch.Tensor,
                kernel : SPHKernel,
                i : torch.Tensor, j : torch.Tensor,
                support : str = 'scatter',
                periodicity : Union[bool, torch.Tensor] = False,
                minExtent : torch.Tensor = torch.zeros(3),
                maxExtent : torch.Tensor = torch.ones(3)
                ):
        ## -------------------------------------------------------------- ##
        ## ---------------------- Start of preamble --------------------- ##
        ## -------------------------------------------------------------- ##
        # Store state for backwards pass
        ctx.save_for_backward(masses_i, masses_j, densities_i, densities_j, quantities_i, quantities_j, positions_i, positions_j, supports_i, supports_j, i, j)
        ctx.kernel = kernel
        ctx.support = support
        ctx.periodicity = periodicity
        ctx.minExtent = minExtent
        ctx.maxExtent = maxExtent

        # rename variables for ease of usage
        masses = (masses_i, masses_j)
        densities = (densities_i, densities_j)
        quantities = (quantities_i, quantities_j)
        positions = (positions_i, positions_j)
        supports = (supports_i, supports_j)
            
        # compute relative positions and support radii
        x_ij = mod_distance(get_i(positions, i), get_j(positions, j), periodicity, minExtent, maxExtent)
        h_ij = getSupport(supports, i, j, mode = support)

        # compute ancillary variables
        ni = positions[0].shape[0]
        nj = positions[1].shape[0]

        ## -------------------------------------------------------------- ##
        ## ---------------------- End of preamble ----------------------- ##
        ## -------------------------------------------------------------- ##

        m_j = get_j(masses, j)
        rho_j = get_j(densities, j)
        W_ij = kernel.eval(x_ij, h_ij)

        k = m_j / rho_j * W_ij

        q_j = get_j(quantities, j) if quantities[1].shape[0] == positions[1].shape[0] else quantities[1]
        kq = torch.einsum('n..., n -> n...', q_j, k)

        return scatter_sum(kq, i, dim_size=ni, dim = 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        ## -------------------------------------------------------------- ##
        ## ---------------------- Start of preamble --------------------- ##
        ## -------------------------------------------------------------- ##
        # Load saved tensors
        masses_i, masses_j, densities_i, densities_j, quantities_i, quantities_j, positions_i, positions_j, supports_i, supports_j, i, j = ctx.saved_tensors

        # Load saved variables
        wrappedKernel = ctx.kernel
        support = ctx.support
        periodicity = ctx.periodicity
        minExtent = ctx.minExtent
        maxExtent = ctx.maxExtent

        # rename variables for ease of usage
        masses = (masses_i, masses_j)
        densities = (densities_i, densities_j)
        quantities = (quantities_i, quantities_j)
        positions = (positions_i, positions_j)
        supports = (supports_i, supports_j)
        
        # compute relative positions and support radii
        x_ij = mod_distance(get_i(positions, i), get_j(positions, j), periodicity, minExtent, maxExtent)
        h_ij = getSupport(supports, i, j, mode = support)

        # compute ancillary variables
        ni = positions[0].shape[0] if isinstance(positions, tuple) else positions.shape[0]
        nj = positions[1].shape[0] if isinstance(positions, tuple) else positions.shape[0]
        
        

        ## -------------------------------------------------------------- ##
        ## ---------------------- End of preamble ----------------------- ##
        ## -------------------------------------------------------------- ##

        W_ij = wrappedKernel.eval(x_ij, h_ij)
        gradW_ij = wrappedKernel.jacobian(x_ij, h_ij)
        ddhW_ij = wrappedKernel.dkdh(x_ij, h_ij)

        q_j = get_j(quantities, j) if quantities[1].shape[0] == positions[1].shape[0] else quantities[1]
        grad_i = get_j(grad_output, i)
        m_j = get_j(masses, j)
        rho_j = get_j(densities, j)

        # Initialize gradients as None for all variables
        m_i_grad = rho_i_grad = q_i_grad = x_i_grad = h_i_grad = None
        m_j_grad = rho_j_grad = q_j_grad = x_j_grad = h_j_grad = None

        # Premultiply the incoming gradient with the outgoing particle quantities
        qji = torch.einsum('n..., n...->n', q_j, grad_i)

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]: # masses
            m_j_grad = scatter_sum(product(qji, 1 / rho_j * W_ij), j, dim_size=nj, dim = 0)

        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]: # densities
            rho_j_grad = -scatter_sum(product(qji, m_j /rho_j**2 * W_ij), j, dim_size=nj, dim = 0)

        if ctx.needs_input_grad[4] or ctx.needs_input_grad[5]: # quantities
            if quantities[1].shape[0] == positions[1].shape[0]:
                q_j_grad = scatter_sum(product(grad_i, m_j / rho_j * W_ij), j, dim_size=nj, dim = 0)
            else:
                q_j_grad = product(grad_i, m_j / rho_j * W_ij)

        if ctx.needs_input_grad[6] or ctx.needs_input_grad[7]: # positions
            x_j_grad = -scatter_sum(product(qji, product(m_j / rho_j, gradW_ij)), j, dim_size=nj, dim = 0)
            x_i_grad = scatter_sum(product(qji, product(m_j / rho_j, gradW_ij)), i, dim_size=ni, dim = 0)

        if ctx.needs_input_grad[8] or ctx.needs_input_grad[9]: # supports
            support_grad = scatter_sum(product(qji, m_j / rho_j * ddhW_ij), j, dim_size=nj, dim = 0) * (1/2 if support == 'symmetric' else 1)

            h_i_grad = support_grad if support == 'gather'  or support == 'symmetric' else None
            h_j_grad = support_grad if support == 'scatter' or support == 'symmetric' else None

        return  \
            m_i_grad, m_j_grad, \
            rho_i_grad, rho_j_grad, \
            q_i_grad, q_j_grad, \
            x_i_grad, x_j_grad, \
            h_i_grad, h_j_grad, \
            None, \
            None, None,\
            None, \
            None, None, None
            
            
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
from sphMath.sphOperations.opUtil import evalSupport, evalPrecomputed,get_qi,  get_qj, get_qs, correctedKernel_CRK, correctedKernelGradient_CRK

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

        correctionTerm_A_i: Optional[torch.Tensor] = None,
        correctionTerm_A_j: Optional[torch.Tensor] = None,
        correctionTerm_B_i: Optional[torch.Tensor] = None,
        correctionTerm_B_j: Optional[torch.Tensor] = None,
        
        correctionTerm_gradA_i: Optional[torch.Tensor] = None,
        correctionTerm_gradA_j: Optional[torch.Tensor] = None,
        correctionTerm_gradB_i: Optional[torch.Tensor] = None,
        correctionTerm_gradB_j: Optional[torch.Tensor] = None,
        correctionTerm_gradMatrix_i: Optional[torch.Tensor] = None,
        correctionTerm_gradMatrix_j: Optional[torch.Tensor] = None,
        correctionTerm_omega_i: Optional[torch.Tensor] = None,
        correctionTerm_omega_j: Optional[torch.Tensor] = None,
        positiveDivergence:bool = False
    ):
        m_j = masses_b[j]
        rho_j = densities_b[j]
        W_ij = evalPrecomputed((W_i, W_j), mode = supportScheme)
        if correctionTerms is not None and KernelCorrectionScheme.CRKSPH.value in [c.value for c in correctionTerms]:
            W_ij = correctedKernel_CRK(i, j, correctionTerm_A_i, correctionTerm_B_i, x_ij, W_ij, False)

        k = m_j / rho_j * W_ij
        if useApparentArea and apparentArea_b is not None:
            k = apparentArea_b[j] * W_ij

        q_j = quantity_ab if quantity_ab is not None else get_qj((quantity_a, quantity_b), i, j, (None, None))
        kq = torch.einsum('n..., n -> n...', q_j, k)

        return scatter_sum(kq, i, dim_size=numRows, dim = 0)
    



from sphMath.sphOperations.opUtil import custom_forwards, custom_backwards, evaluateKernel_, evaluateKernelGradient_

def interpolate_fn(
        supportScheme: SupportScheme,
        useApparentArea: bool,
        crkCorrection: bool,
        i: torch.Tensor, j: torch.Tensor,
        numRows: int, numCols: int,

        q_i: Optional[torch.Tensor], A_i: Optional[torch.Tensor], B_i: Optional[torch.Tensor],
        q_j: Optional[torch.Tensor], m_j: torch.Tensor, rho_j: torch.Tensor, apparentArea_b: Optional[torch.Tensor],
        
        q_ij: Optional[torch.Tensor], x_ij: torch.Tensor,
        W_i: torch.Tensor, W_j: torch.Tensor):    
    W_ij = evaluateKernel_(W_i, W_j, supportScheme, crkCorrection, i, j, x_ij, A_i, B_i)
    q_j_ = q_j if q_ij is None else q_ij

    k = m_j / rho_j * W_ij
    if useApparentArea and apparentArea_b is not None:
        k = apparentArea_b[j] * W_ij

    kq = torch.einsum('n..., n -> n...', q_j_, k)

    return scatter_sum(kq, i, dim_size=numRows, dim = 0)

class Interpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
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
        # Store state for backwards pass
        ctx.save_for_backward(masses_b, densities_b, W_i, W_j, apparentArea_b, correctionTerm_A_i, correctionTerm_B_i, x_ij, i, j, quantity_a, quantity_b, quantity_ab)
        ctx.correctionTerms = correctionTerms
        ctx.supportScheme = supportScheme
        ctx.useApparentArea = useApparentArea
        ctx.numRows = numRows
        ctx.numCols = numCols
        ctx.crkCorrection = correctionTerms is not None and KernelCorrectionScheme.CRKSPH.value in [c.value for c in correctionTerms]

        inputs_i = [quantity_a, correctionTerm_A_i, correctionTerm_B_i]
        inputs_j = [quantity_b, masses_b, densities_b, apparentArea_b]
        inputs_ij = [quantity_ab, x_ij, W_i, W_j]

        return custom_forwards(
            interpolate_fn, 
            i, j, ctx.numRows, ctx.numCols,
            inputs_i, inputs_j, inputs_ij,
            supportScheme,
            useApparentArea,
            ctx.crkCorrection
        )
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        masses_b, densities_b, W_i, W_j, apparentArea_b, correctionTerm_A_i, correctionTerm_B_i, x_ij, i, j, quantity_a, quantity_b, quantity_ab = ctx.saved_tensors

        # Load saved variables
        correctionTerms = ctx.correctionTerms
        supportScheme = ctx.supportScheme
        useApparentArea = ctx.useApparentArea
        crkCorrection = ctx.crkCorrection

        inputs_i = [quantity_a, correctionTerm_A_i, correctionTerm_B_i]
        inputs_j = [quantity_b, masses_b, densities_b, apparentArea_b]
        inputs_ij = [quantity_ab, x_ij, W_i, W_j]

        grad_q_i, grad_A_i, grad_B_i, \
            grad_q_j, grad_m_j, grad_rho_j, grad_apparentArea_j, \
            grad_q_ij, grad_x_ij, grad_W_i, grad_W_j = custom_backwards(
            interpolate_fn,
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
            None, grad_rho_j, # densities_a, densities_b,
            None, grad_apparentArea_j, # apparentArea_a, apparentArea_b,
            grad_q_i, grad_q_j, grad_q_ij, # quantity_a, quantity_b, quantity_ab,
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
            
            
interpolate_precomputed_op = Interpolate.apply


