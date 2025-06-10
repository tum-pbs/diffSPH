import torch
from sphMath.kernels import SPHKernel
from typing import Union, Tuple, Optional
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product

def flattened_sum(tensor, index, dim_size, dim):
    if tensor.dim() > 1:
        tensor = tensor.flatten(start_dim=1).sum(dim=1)
    return scatter_sum(tensor, index, dim_size=dim_size, dim=dim)

class SPHGradient(torch.autograd.Function):
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
                gradientMode : str = 'gradient',
                periodicity : Union[bool, torch.Tensor] = False,
                minExtent : torch.Tensor = torch.zeros(3),
                maxExtent : torch.Tensor = torch.ones(3),
                gradientRenormalizationMatrix : Optional[torch.Tensor] = None,
                gradHTerms : Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ):
        ## -------------------------------------------------------------- ##
        ## ---------------------- Start of preamble --------------------- ##
        ## -------------------------------------------------------------- ##
        # Store state for backwards pass
        ctx.save_for_backward(masses_i, masses_j, densities_i, densities_j, quantities_i, quantities_j, positions_i, positions_j, supports_i, supports_j, i, j, gradientRenormalizationMatrix, gradHTerms)
        ctx.kernel = kernel
        ctx.gradientMode = gradientMode
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

        # print(f'Support Radii [Scheme: {support}]: min: {h_ij.min().item()}, max: {h_ij.max().item()}, mean: {h_ij.mean().item()}')

        # compute ancillary variables
        ni = positions[0].shape[0]
        nj = positions[1].shape[0]

        ## -------------------------------------------------------------- ##
        ## ---------------------- End of preamble ----------------------- ##
        ## -------------------------------------------------------------- ##

        m_j = get_j(masses, j)
        rho_i = get_i(densities, i)
        rho_j = get_j(densities, j)
        gradW_ij = kernel.jacobian(x_ij, h_ij)
        # print(f'|Gradient| min: {torch.linalg.norm(gradW_ij, dim=-1).min().item()}, max: {torch.linalg.norm(gradW_ij, dim=-1).max().item()}, mean: {torch.linalg.norm(gradW_ij, dim=-1).mean().item()}')

        # if gradientRenormalizationMatrix is not None:
            # L_i = gradientRenormalizationMatrix[i]
            # gradW_ij = torch.bmm(L_i, gradW_ij.unsqueeze(-1)).squeeze(-1)

        k = (m_j / rho_j).view(-1,1) * gradW_ij if gradientMode != 'symmetric' else (m_j * rho_i).view(-1,1) * gradW_ij

        q = None
        if quantities[1].shape[0] == positions[1].shape[0]:
            if gradientMode == 'naive':
                q = get_j(quantities, j)
            elif gradientMode == 'difference':
                q = get_j(quantities, j) - get_i(quantities, i)
            elif gradientMode == 'summation':
                q = get_j(quantities, j) + get_i(quantities, i)
            elif gradientMode == 'symmetric':
                qi = torch.einsum('n..., n -> n...', get_i(quantities, i), 1.0 / rho_i**2)
                qj = torch.einsum('n..., n -> n...', get_j(quantities, j), 1.0 / rho_j**2)
                
                if gradHTerms is not None:
                    qi = qi / get_i(gradHTerms, i)
                    qj = qj / get_j(gradHTerms, j)
                
                q = qi + qj
                if support == 'superSymmetric':
                    gradW_ij_i = kernel.jacobian(x_ij, get_i(supports, i))
                    gradW_ij_j = kernel.jacobian(x_ij, get_j(supports, j))

                    kq_i = torch.einsum('n... , nd -> n...d', qi, gradW_ij_i)
                    kq_j = torch.einsum('n... , nd -> n...d', qj, gradW_ij_j)
                    kq = (kq_i + kq_j) * (m_j * rho_i).view(-1,1)
        else:
            if gradientMode != 'naive':
                raise ValueError('Gradient mode must be naive when quantities are provided as scattered')
            # print('Overriding quantities')
            q = quantities[1]

        if support != 'superSymmetric' or gradientMode != 'symmetric':
            kq = torch.einsum('n... , nd -> n...d', q, k)
        # kq = torch.einsum('n... , nd -> n...d', q, k)

        summed = scatter_sum(kq, i, dim_size=ni, dim = 0)

        if gradientRenormalizationMatrix is not None:
            L_i = gradientRenormalizationMatrix
            return torch.bmm(L_i, summed.unsqueeze(-1)).squeeze(-1)
            # summed = torch.bmm(L_i, summed.unsqueeze(-1)).squeeze(-1)

        return summed

    @staticmethod
    def backward(ctx, grad_output):
        ## -------------------------------------------------------------- ##
        ## ---------------------- Start of preamble --------------------- ##
        ## -------------------------------------------------------------- ##
        # Load saved tensors
        masses_i, masses_j, densities_i, densities_j, quantities_i, quantities_j, positions_i, positions_j, supports_i, supports_j, i, j, gradientRenormalizationMatrix = ctx.saved_tensors

        # Load saved variables
        wrappedKernel = ctx.kernel
        gradientMode = ctx.gradientMode
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
        ni = positions[0].shape[0]# if isinstance(positions, tuple) else positions.shape[0]
        nj = positions[1].shape[0]# if isinstance(positions, tuple) else positions.shape[0]

        ## -------------------------------------------------------------- ##
        ## ---------------------- End of preamble ----------------------- ##
        ## -------------------------------------------------------------- ##

        W_ij = wrappedKernel.eval(x_ij, h_ij)
        gradW_ij = wrappedKernel.jacobian(x_ij, h_ij)
        ddhW_ij = wrappedKernel.dkdh(x_ij, h_ij)

        if gradientRenormalizationMatrix is not None:
            L_i = gradientRenormalizationMatrix[i]
            gradLW_ij = torch.bmm(L_i, gradW_ij.unsqueeze(-1)).squeeze(-1)
        else:
            gradLW_ij = gradW_ij

        q_i = get_i(quantities, i) if gradientMode != 'naive' else None
        q_j = get_j(quantities, j) if quantities[1].shape[0] == positions[1].shape[0] else quantities[1]
        rho_i = get_i(densities, i) if gradientMode == 'symmetric' else None
        rho_j = get_j(densities, j)
        m_j = get_j(masses, j)
        grad_i = get_j(grad_output, i)


        if gradientRenormalizationMatrix is not None:
            L_i = gradientRenormalizationMatrix[i]
            # L_i_inv = torch.linalg.pinv(L_i)
            gradL_i = torch.einsum('ijk,ik->ij', L_i.mT, grad_i)
        else:
            gradL_i = grad_i

            # grad_i = torch.bmm(L_i.mT, grad_i.unsqueeze(-1)).squeeze(-1)

        # Initialize gradients as None for all variables
        m_i_grad = rho_i_grad = q_i_grad = x_i_grad = h_i_grad = None
        m_j_grad = rho_j_grad = q_j_grad = x_j_grad = h_j_grad = None
        L_grad = None

        # Premultiply the incoming gradient with the outgoing gradient
        gradientKernel = torch.einsum('n...d, nd -> n...', grad_i, gradLW_ij)

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]: # masses
            if gradientMode == 'naive':
                m_j_grad = flattened_sum(product(1 / rho_j, product(q_j, gradientKernel)), j, nj, 0)
            elif gradientMode == 'difference':
                m_j_grad = flattened_sum(product(1 / rho_j, product(q_j - q_i, gradientKernel)), j, nj, 0)
            elif gradientMode == 'summation':
                m_j_grad = flattened_sum(product(1 / rho_j, product(q_j + q_i, gradientKernel)), j, nj, 0)
            elif gradientMode == 'symmetric':
                m_j_grad = flattened_sum(product(product(q_j, rho_i/rho_j**2) + product(q_i, 1/rho_i), gradientKernel), j, nj, 0)
        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]: # densities
            if gradientMode == 'naive':
                rho_j_grad = -flattened_sum(product(m_j /rho_j**2, product(q_j, gradientKernel)), j, nj, 0)
            elif gradientMode == 'difference':
                rho_j_grad = -flattened_sum(product(m_j /rho_j**2, product(q_j - q_i, gradientKernel)), j, nj, 0)
            elif gradientMode == 'summation':
                rho_j_grad = -flattened_sum(product(m_j /rho_j**2, product(q_j + q_i, gradientKernel)), j, nj, 0)
            elif gradientMode == 'symmetric':
                rho_j_grad = flattened_sum(product(m_j, product(product(q_j, -2 * rho_i/rho_j**3), gradientKernel)), j, nj, 0)
                rho_i_grad = flattened_sum(product(m_j, product(product(q_j, 1/rho_j**2) + product(q_i, -1/rho_i**2), gradientKernel)), i, ni, 0)
        if ctx.needs_input_grad[4] or ctx.needs_input_grad[5]: # quantities
            if gradientMode == 'naive':
                if quantities[1].shape[0] == positions[1].shape[0]:
                    q_j_grad = scatter_sum(product(m_j / rho_j, gradientKernel), j, dim_size=nj, dim = 0)
                else:
                    q_j_grad = product(m_j / rho_j, gradientKernel)
            elif gradientMode == 'difference':
                q_j_grad = scatter_sum(product(m_j / rho_j, gradientKernel), j, dim_size=nj, dim = 0)
                q_i_grad = scatter_sum(product(-m_j / rho_j, gradientKernel), i, dim_size=ni, dim = 0)
            elif gradientMode == 'summation':
                q_j_grad = scatter_sum(product(m_j / rho_j, gradientKernel), j, dim_size=nj, dim = 0)
                q_i_grad = scatter_sum(product(m_j / rho_j, gradientKernel), i, dim_size=ni, dim = 0)
            elif gradientMode == 'symmetric':
                q_j_grad = scatter_sum(product(rho_i * m_j / rho_j**2, gradientKernel), j, dim_size=nj, dim = 0)
                q_i_grad = scatter_sum(product(m_j / rho_i, gradientKernel), i, dim_size=ni, dim = 0)
        if ctx.needs_input_grad[6] or ctx.needs_input_grad[7]: # positions
            hessian = -wrappedKernel.hessian(x_ij, h_ij)
            # if gradientRenormalizationMatrix is not None:
            #     L_i = gradientRenormalizationMatrix[i]
            #     hessian = torch.bmm(L_i, hessian)

            gradProd = torch.einsum('nuv, n...v -> n...u', hessian, gradL_i)
            # if gradientRenormalizationMatrix is not None:
                # gradProd = torch.bmm(L_i, gradProd.unsqueeze(-1)).squeeze(-1)

            if gradientMode == 'naive':
                quantProd = product(q_j, m_j / rho_j)
                term = torch.einsum('n..., n...d -> nd', quantProd, gradProd)
            elif gradientMode == 'difference':
                quantProd = product(q_j - q_i, m_j / rho_j)
                term = torch.einsum('n..., n...d -> nd', quantProd, gradProd)

            elif gradientMode == 'summation':
                quantProd = product(q_j + q_i, m_j / rho_j)
                term = torch.einsum('n..., n...d -> nd', quantProd, gradProd)
            elif gradientMode == 'symmetric':
                quantProd = product(q_j, rho_i * m_j / rho_j**2) + product(q_i, m_j / rho_i)
                term = torch.einsum('n..., n...d -> nd', quantProd, gradProd)

            # if gradientRenormalizationMatrix is not None:
                # term = torch.bmm(gradientRenormalizationMatrix[i], term.unsqueeze(-1)).squeeze(-1)

            x_i_grad = scatter_sum(-term, i, dim_size=ni, dim = 0)
            x_j_grad = scatter_sum(term, j, dim_size=nj, dim = 0)



        if ctx.needs_input_grad[8] or ctx.needs_input_grad[9]:            
            kernelTerm = wrappedKernel.djacobiandh(x_ij, h_ij)
            gradientKernelTerm = torch.einsum('n...d, nd -> n...', gradL_i, kernelTerm)
            support_grad = None
            
            if gradientMode == 'naive':
                support_grad = flattened_sum(product(m_j / rho_j, product(q_j, gradientKernelTerm)), j, nj, 0)
            elif gradientMode == 'difference':
                support_grad = flattened_sum(product(m_j / rho_j, product(q_j - q_i, gradientKernelTerm)), j, nj, 0)
            elif gradientMode == 'summation':
                support_grad = flattened_sum(product(m_j / rho_j, product(q_j + q_i, gradientKernelTerm)), j, nj, 0)
            elif gradientMode == 'symmetric':
                support_grad = flattened_sum(product(product(q_j, m_j * rho_i/rho_j**2) + product(q_i, m_j/rho_i), gradientKernelTerm), j, nj, 0)

            support_grad = support_grad * (1/2 if support == 'symmetric' else 1)
            h_i_grad = support_grad if support == 'gather'  or support == 'symmetric' else None
            h_j_grad = support_grad if support == 'scatter' or support == 'symmetric' else None

        if gradientRenormalizationMatrix is not None:
            if ctx.needs_input_grad[18]:

                m_j = get_j(masses, j)
                rho_i = get_i(densities, i)
                rho_j = get_j(densities, j)
                # gradW_ij = kernel.jacobian(x_ij, h_ij)

                # if gradientRenormalizationMatrix is not None:
                    # L_i = gradientRenormalizationMatrix[i]
                    # gradW_ij = torch.bmm(L_i, gradW_ij.unsqueeze(-1)).squeeze(-1)

                k = (m_j / rho_j).view(-1,1) * gradW_ij if gradientMode != 'symmetric' else (m_j * rho_i).view(-1,1) * gradW_ij

                q = None
                if quantities[1].shape[0] == positions[1].shape[0]:
                    if gradientMode == 'naive':
                        q = get_j(quantities, j)
                    elif gradientMode == 'difference':
                        q = get_j(quantities, j) - get_i(quantities, i)
                    elif gradientMode == 'summation':
                        q = get_j(quantities, j) + get_i(quantities, i)
                    elif gradientMode == 'symmetric':
                        qi = torch.einsum('n..., n -> n...', get_i(quantities, i), 1.0 / rho_i**2)
                        qj = torch.einsum('n..., n -> n...', get_j(quantities, j), 1.0 / rho_j**2)
                        q = qi + qj
                else:
                    if gradientMode != 'naive':
                        raise ValueError('Gradient mode must be naive when quantities are provided as scattered')
                    q = quantities[1]

                kq = torch.einsum('n... , nd -> n...d', q, k)

                summed = scatter_sum(kq, i, dim_size=ni, dim = 0)

                L_grad = grad_output.unsqueeze(-1) * summed.unsqueeze(-2)

        return  \
            m_i_grad, m_j_grad, \
            rho_i_grad, rho_j_grad, \
            q_i_grad, q_j_grad, \
            x_i_grad, x_j_grad, \
            h_i_grad, h_j_grad, \
            None, \
            None, None,\
            None, None,\
            None, None, None, L_grad
            
            
            
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
from sphMath.sphOperations.opUtil import evalSupport, evalPrecomputed, get_qi, get_qj, get_qs, correctedKernel_CRK, correctedKernelGradient_CRK

def gradient_op(
        q: torch.Tensor,
        grad: torch.Tensor
):
    return torch.einsum('n..., nd -> n...d', q, grad)

def gradient_precomputed(
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

        # Helper functions
        op = gradient_op
        ## -------------------------------------------------------------- ##
        ## ---------------------- End of preamble ----------------------- ##
        ## -------------------------------------------------------------- ##


        m_j = masses[1][j]
        rho_j = densities[1][j]
        rho_i = densities[0][i]
        gradW_ij = evalPrecomputed(gradW, mode = supportScheme)
        if correctionTerms is not None and KernelCorrectionScheme.CRKSPH.value in [c.value for c in correctionTerms]:
            gradW_ij = correctedKernelGradient_CRK(i, j, correctionTerm_A[0], correctionTerm_B[0], correctionTerm_gradA[0], correctionTerm_gradB[0], x_ij, (W_i + W_j)/2, (gradW_i + gradW_j)/2, False)
            gradW_ji = correctedKernelGradient_CRK(i, j, correctionTerm_A[1], correctionTerm_B[1], correctionTerm_gradA[1], correctionTerm_gradB[1], x_ij, (W_i + W_j)/2, (gradW_i + gradW_j)/2, True)
            gradW_ij = (gradW_ij - gradW_ji)/2
        omegas = correctionTerm_omega if correctionTerms is not None and KernelCorrectionScheme.gradH.value in [c.value for c in correctionTerms] else (None, None)
        gradientRenormalizationMatrix = correctionTerm_gradMatrix if correctionTerms is not None and KernelCorrectionScheme.gradientRenorm.value in [c.value for c in correctionTerms] else (None, None)
        if correctionTerms is not None and KernelCorrectionScheme.gradientRenorm.value in [c.value for c in correctionTerms]:
            L_i = gradientRenormalizationMatrix[0]
            if L_i is None:
                raise ValueError("Gradient renormalization matrix is None")
            else:
                gradW_ij = torch.bmm(L_i[i], gradW_ij.unsqueeze(-1)).squeeze(-1)
                
        factor = m_j / rho_j if gradientMode != GradientMode.Symmetric else m_j * rho_i
        if useApparentArea and apparentArea_b is not None:
            if gradientMode != GradientMode.Symmetric:
                factor = apparentArea_b[j]
            else:
                factor = apparentArea_b[j] * rho_i * rho_j


        if quantity_ab is None:
            if gradientMode == GradientMode.Naive:
                  q_j = get_qj(quantity, i, j, omegas)
                  kq = torch.einsum('n..., nd -> n...d', q_j, gradW_ij)
            else:
                q_i, q_j = get_qs(quantity, i, j, omegas)
                if gradientMode == GradientMode.Difference:                    
                    if supportScheme == SupportScheme.SuperSymmetric:
                        kq_i = op(q_i, gradW_i)
                        kq_j = op(q_j, gradW_j)
                        kq = kq_j - kq_i
                    else:
                        kq = op(q_j - q_i, gradW_ij)
                elif gradientMode == GradientMode.Summation:
                    if supportScheme == SupportScheme.SuperSymmetric:
                        kq_i = op(q_i, gradW_i)
                        kq_j = op(q_j, gradW_j)
                        kq = kq_j + kq_i
                    else:
                        kq = op(q_j + q_i, gradW_ij)
                elif gradientMode == GradientMode.Symmetric:
                    qi = torch.einsum('n..., n -> n...', q_i, 1/ rho_i**2)
                    qj = torch.einsum('n..., n -> n...', q_j, 1/ rho_j**2)
                    if supportScheme == SupportScheme.SuperSymmetric:
                        kq_i = op(qi, gradW_i)
                        kq_j = op(qj, gradW_j)
                        kq = kq_j + kq_i
                    else:
                        kq = op(qi + qj, gradW_ij)
                else:
                    raise ValueError(f"Unknown gradient mode {gradientMode}")
        else:
            kq = op(quantity_ab, gradW_ij)

        fkq = torch.einsum('n..., n -> n...', kq, factor)
        
        summed = scatter_sum(fkq, i, dim_size=ni, dim = 0)
        # if correctionTerms is not None and KernelCorrectionScheme.gradientRenorm.value in [c.value for c in correctionTerms]:
        #     L_i = gradientRenormalizationMatrix[0]
        #     if L_i is None:
        #         raise ValueError("Gradient renormalization matrix is None")
        #     else:
        #         return torch.bmm(L_i, summed.unsqueeze(-1)).squeeze(-1)
        # else:
        return summed
        
from sphMath.sphOperations.opUtil import custom_forwards, custom_backwards, evaluateKernel_, evaluateKernelGradient_, get_q


def gradient_fn(
        supportScheme: SupportScheme,
        gradientMode: GradientMode,
        useApparentArea: bool,
        crkCorrection: bool,
        gradientCorrection: bool,
        omegaCorrection: bool,
        i: torch.Tensor, j: torch.Tensor,
        numRows: int, numCols: int,

        q_i_: Optional[torch.Tensor], rho_i: Optional[torch.Tensor], omega_i: Optional[torch.Tensor], A_i: Optional[torch.Tensor], B_i: Optional[torch.Tensor], gradA_i: Optional[torch.Tensor], gradB_i: Optional[torch.Tensor], L_i: Optional[torch.Tensor],

        q_j_: Optional[torch.Tensor], m_j: torch.Tensor, rho_j: torch.Tensor, apparentArea_b: Optional[torch.Tensor], omega_j: Optional[torch.Tensor], A_j: Optional[torch.Tensor], B_j: Optional[torch.Tensor], gradA_j: Optional[torch.Tensor], gradB_j: Optional[torch.Tensor], 

        q_ij: Optional[torch.Tensor], x_ij: torch.Tensor,
        W_i: torch.Tensor, W_j: torch.Tensor,
        gradW_i: torch.Tensor, gradW_j: torch.Tensor,

):
        # compute relative positions and support radii
        x_ij = x_ij
        # h_ij = evalSupport(supports, i, j, mode = supportScheme)

        # compute ancillary variables
        ni = numRows
        nj = numCols

        # Helper functions
        op = gradient_op
        ## -------------------------------------------------------------- ##
        ## ---------------------- End of preamble ----------------------- ##
        ## -------------------------------------------------------------- ##


        # m_j = masses[1][j]
        # rho_j = densities[1][j]
        # rho_i = densities[0][i]
        gradW_ij = evaluateKernelGradient_(
            W_i, W_j, gradW_i, gradW_j, supportScheme, crkCorrection, i, j, x_ij, A_i, B_i, gradA_i, gradB_i, A_j, B_j, gradA_j, gradB_j
        )

        if gradientCorrection:
            if L_i is None:
                raise ValueError("Gradient renormalization matrix is None")
            else:
                gradW_ij = torch.bmm(L_i, gradW_ij.unsqueeze(-1)).squeeze(-1)

                
        factor = m_j / rho_j if gradientMode != GradientMode.Symmetric else m_j * rho_i
        if useApparentArea and apparentArea_b is not None:
            if gradientMode != GradientMode.Symmetric:
                factor = apparentArea_b[j]
            else:
                factor = apparentArea_b[j] * rho_i * rho_j


        if q_ij is None:
            if gradientMode == GradientMode.Naive:
                  q_j = get_q(q_j_, omega_j, omegaCorrection)
                  kq = torch.einsum('n..., nd -> n...d', q_j, gradW_ij)
            else:
                q_i = get_q(q_i_, omega_i, omegaCorrection)
                q_j = get_q(q_j_, omega_j, omegaCorrection)
                if gradientMode == GradientMode.Difference:                    
                    if supportScheme == SupportScheme.SuperSymmetric:
                        kq_i = op(q_i, gradW_i)
                        kq_j = op(q_j, gradW_j)
                        kq = kq_j - kq_i
                    else:
                        kq = op(q_j - q_i, gradW_ij)
                elif gradientMode == GradientMode.Summation:
                    if supportScheme == SupportScheme.SuperSymmetric:
                        kq_i = op(q_i, gradW_i)
                        kq_j = op(q_j, gradW_j)
                        kq = kq_j + kq_i
                    else:
                        kq = op(q_j + q_i, gradW_ij)
                elif gradientMode == GradientMode.Symmetric:
                    qi = torch.einsum('n..., n -> n...', q_i, 1/ rho_i**2)
                    qj = torch.einsum('n..., n -> n...', q_j, 1/ rho_j**2)
                    if supportScheme == SupportScheme.SuperSymmetric:
                        kq_i = op(qi, gradW_i)
                        kq_j = op(qj, gradW_j)
                        kq = kq_j + kq_i
                    else:
                        kq = op(qi + qj, gradW_ij)
                else:
                    raise ValueError(f"Unknown gradient mode {gradientMode}")
        else:
            kq = op(q_ij, gradW_ij)

        fkq = torch.einsum('n..., n -> n...', kq, factor)
        
        summed = scatter_sum(fkq, i, dim_size=ni, dim = 0)

        return summed






class Gradient(torch.autograd.Function):
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
        ctx.save_for_backward(
            quantity_a, densities_a, correctionTerm_omega_i, correctionTerm_A_i, correctionTerm_B_i, correctionTerm_gradA_i, correctionTerm_gradB_i, correctionTerm_gradMatrix_i,
            quantity_b, masses_b, densities_b, apparentArea_b, correctionTerm_omega_j, correctionTerm_A_j, correctionTerm_B_j, correctionTerm_gradA_j, correctionTerm_gradB_j,
            quantity_ab, x_ij, W_i, W_j, gradW_i, gradW_j, i,  j)
        ctx.correctionTerms = correctionTerms
        ctx.supportScheme = supportScheme
        ctx.useApparentArea = useApparentArea
        ctx.numRows = numRows
        ctx.numCols = numCols
        ctx.gradientMode = gradientMode
        ctx.crkCorrection = correctionTerms is not None and KernelCorrectionScheme.CRKSPH.value in [c.value for c in correctionTerms]
        ctx.gradientCorrection = correctionTerms is not None and KernelCorrectionScheme.gradientRenorm.value in [c.value for c in correctionTerms]
        ctx.omegaCorrection = correctionTerms is not None and KernelCorrectionScheme.gradH.value in [c.value for c in correctionTerms]

        inputs_i = [quantity_a, densities_a, correctionTerm_omega_i, correctionTerm_A_i, correctionTerm_B_i, correctionTerm_gradA_i, correctionTerm_gradB_i, correctionTerm_gradMatrix_i]
        inputs_j = [quantity_b, masses_b, densities_b, apparentArea_b, correctionTerm_omega_j, correctionTerm_A_j, correctionTerm_B_j, correctionTerm_gradA_j, correctionTerm_gradB_j]
        inputs_ij = [quantity_ab, x_ij, W_i, W_j, gradW_i, gradW_j]

        return custom_forwards(
            gradient_fn, 
            i, j, ctx.numRows, ctx.numCols,
            inputs_i, inputs_j, inputs_ij,
            supportScheme,
            gradientMode,
            useApparentArea,
            ctx.crkCorrection,
            ctx.gradientCorrection,
            ctx.omegaCorrection,
        )
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        quantity_a, densities_a, correctionTerm_omega_i, correctionTerm_A_i, correctionTerm_B_i, correctionTerm_gradA_i, correctionTerm_gradB_i, correctionTerm_gradMatrix_i, quantity_b, masses_b, densities_b, apparentArea_b, correctionTerm_omega_j, correctionTerm_A_j, correctionTerm_B_j, correctionTerm_gradA_j, correctionTerm_gradB_j, quantity_ab, x_ij, W_i, W_j, gradW_i, gradW_j, i,  j = ctx.saved_tensors

        # Load saved variables

        inputs_i = [quantity_a, densities_a, correctionTerm_omega_i, correctionTerm_A_i, correctionTerm_B_i, correctionTerm_gradA_i, correctionTerm_gradB_i, correctionTerm_gradMatrix_i]
        inputs_j = [quantity_b, masses_b, densities_b, apparentArea_b, correctionTerm_omega_j, correctionTerm_A_j, correctionTerm_B_j, correctionTerm_gradA_j, correctionTerm_gradB_j]
        inputs_ij = [quantity_ab, x_ij, W_i, W_j, gradW_i, gradW_j]


        grad_quantity_a, grad_densities_a, grad_correctionTerm_omega_i, grad_correctionTerm_A_i, grad_correctionTerm_B_i, grad_correctionTerm_gradA_i, grad_correctionTerm_gradB_i, grad_correctionTerm_gradMatrix_i, grad_quantity_b, grad_masses_b, grad_densities_b, grad_apparentArea_b, grad_correctionTerm_omega_j, grad_correctionTerm_A_j, grad_correctionTerm_B_j, grad_correctionTerm_gradA_j, grad_correctionTerm_gradB_j, grad_quantity_ab, grad_x_ij, grad_W_i, grad_W_j, grad_gradW_i, grad_gradW_j  = custom_backwards(
            gradient_fn,
            grad_output,
            i, j, ctx.numRows, ctx.numCols,
            inputs_i, inputs_j, inputs_ij,
            ctx.supportScheme,
            ctx.gradientMode,
            ctx.useApparentArea,
            ctx.crkCorrection,
            ctx.gradientCorrection,
            ctx.omegaCorrection,
            )

        return (
            None, None, # positions_a, positions_b, 
            None, None, # supports_a, supports_b, 
            None, grad_masses_b, # masses_a, masses_b,
            grad_densities_a, grad_densities_b, # densities_a, densities_b,
            None, grad_apparentArea_b, # apparentArea_a, apparentArea_b,
            grad_quantity_a, grad_quantity_b, grad_quantity_ab, # quantity_a, quantity_b, quantity_ab,
            None, None, # i, j,
            None, None, # numRows, numCols,
            None, grad_x_ij, # r_ij, x_ij,
            grad_W_i, grad_W_j, # W_i, W_j,
            grad_gradW_i, grad_gradW_j, # gradW_i, gradW_j,
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
            grad_correctionTerm_A_i, grad_correctionTerm_A_j, # correctionTerm_A_i, correctionTerm_A_j,
            grad_correctionTerm_B_i, grad_correctionTerm_B_j, # correctionTerm_B_i, correctionTerm_B_j,
            grad_correctionTerm_gradA_i, grad_correctionTerm_gradA_j, # correctionTerm_gradA_i, correctionTerm_gradA_j,
            grad_correctionTerm_gradB_i, grad_correctionTerm_gradB_j, # correctionTerm_gradB_i, correctionTerm_gradB_j,
            grad_correctionTerm_gradMatrix_i, None, # correctionTerm_gradMatrix_i, correctionTerm_gradMatrix_j,
            grad_correctionTerm_omega_i, grad_correctionTerm_omega_j, # correctionTerm_omega_i, correctionTerm_omega_j,
            None # positiveDivergence
        )
            
            
gradient_precomputed_op = Gradient.apply