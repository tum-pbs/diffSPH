import torch
from sphMath.kernels import SPHKernel
from typing import Union, Tuple, Optional
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product

def flattened_sum(tensor, index, dim_size, dim):
    if tensor.dim() > 1:
        tensor = tensor.flatten(start_dim=1).sum(dim=1)
    return scatter_sum(tensor, index, dim_size=dim_size, dim=dim)

class SPHLaplacian(torch.autograd.Function):
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
                laplacianMode : str = 'direct',
                periodicity : Union[bool, torch.Tensor] = False,
                minExtent : torch.Tensor = torch.zeros(3),
                maxExtent : torch.Tensor = torch.ones(3),
                piSwitch : bool = False
                ):
        ## -------------------------------------------------------------- ##
        ## ---------------------- Start of preamble --------------------- ##
        ## -------------------------------------------------------------- ##
        # Store state for backwards pass
        ctx.save_for_backward(masses_i, masses_j, densities_i, densities_j, quantities_i, quantities_j, positions_i, positions_j, supports_i, supports_j, i, j)
        ctx.kernel = kernel
        ctx.laplacianMode = laplacianMode
        ctx.support = support
        ctx.periodicity = periodicity
        ctx.minExtent = minExtent
        ctx.maxExtent = maxExtent
        ctx.piSwitch = piSwitch

        # rename variables for ease of usage
        masses = (masses_i, masses_j)
        densities = (densities_i, densities_j)
        quantities = (quantities_i, quantities_j)
        positions = (positions_i, positions_j)
        supports = (supports_i, supports_j)
            
        # compute relative positions and support radii
        x_ij = -mod_distance(get_i(positions, i), get_j(positions, j), periodicity, minExtent, maxExtent)
        r_ij = torch.linalg.norm(x_ij, dim = -1)
        h_ij = getSupport(supports, i, j, mode = support)

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
        laplaceW_ij = kernel.laplacian(x_ij, h_ij)

        quotient = (r_ij + 1e-7 * h_ij)
        kernelApproximation = torch.linalg.norm(gradW_ij, dim = -1) /  quotient

        q_ij = (get_j(quantities, j) - get_i(quantities, i)) if quantities[1].shape[0] == positions[1].shape[0] else quantities[1]
        
        fac_j = m_j / rho_j

        if laplacianMode == 'naive': # (eq 86)
            lk = fac_j * laplaceW_ij
            kq = torch.einsum('n, n... -> n...', lk, q_ij)

            return -scatter_sum(kq, i, dim = 0, dim_size = ni)
        elif laplacianMode == 'verynaive': # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 777 (eq 90) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
            lk = fac_j * laplaceW_ij
            q_ij = get_j(quantities, j)  if quantities[1].shape[0] == positions[1].shape[0] else quantities[1]
            kq = torch.einsum('n, n... -> n...', lk, q_ij)

            return -scatter_sum(kq, i, dim = 0, dim_size = ni)
        elif laplacianMode == 'non-conserving':  # Check this equation if the dim term is actually correct
            dim = x_ij.shape[1]
            fac = fac_j
            n_ij = torch.nn.functional.normalize(x_ij, dim = -1)

            dot = (dim + 2) * torch.einsum('nd, nd -> n', q_ij, n_ij).view(-1,1) * n_ij
            qx_ij = torch.einsum('ij,ij->i', q_ij, x_ij)
            if piSwitch:
                dot = torch.where(qx_ij < 0, dot, 0)

            q = fac * kernelApproximation

            kq = -q.view(-1,1) * (dot + q_ij)

            return scatter_sum(kq, i, dim = 0, dim_size = ni)
        elif laplacianMode == 'conserving': # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 778 (eq 97) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
            n_ij = torch.nn.functional.normalize(x_ij, dim = -1)

            dot = torch.einsum('nd, nd -> n', q_ij, n_ij)
            qx_ij = torch.einsum('ij,ij->i', q_ij, x_ij)
            if piSwitch:
                dot = torch.where(qx_ij < 0, dot, 0)


            kernelApproximation = gradW_ij /  quotient.view(-1,1) #**2 

            q = fac_j * dot
            kq = -q.view(-1, 1) * kernelApproximation

            return scatter_sum(kq, i, dim = 0, dim_size = ni)
        elif laplacianMode == 'divergenceFree': # https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf eq 26
            n_ij = torch.nn.functional.normalize(x_ij, dim = -1)

            dot = torch.einsum('nd, nd -> n', q_ij, x_ij) / quotient**2
            q = 2 * (x_ij.shape[1] + 2) * fac_j * dot
            
            kq = q.view(-1, 1) * gradW_ij

            return scatter_sum(kq, i, dim = 0, dim_size = ni)
        elif laplacianMode == 'dot': # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 778 (eq 96) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
            n_ij = torch.nn.functional.normalize(x_ij, dim = -1)
            
            term = (x_ij.shape[1] + 2) * torch.einsum('nd, nd -> n', q_ij, n_ij).view(-1,1) * n_ij + q_ij
            
            qx_ij = torch.einsum('ij,ij->i', q_ij, x_ij)
            if piSwitch: # price vab . rab <= 0 -> not zero
                term[qx_ij > 0,:] = 0
                # term = torch.where(qx_ij < 0, term, 0)

            kq = term * (fac_j * kernelApproximation).view(-1,1)

            return scatter_sum(kq, i, dim = 0, dim_size = ni)    
        elif laplacianMode == 'Monaghan1983': # Also alpha term in Monaghan1992

            rho_bar = (rho_i + rho_j) / 2
            qx_ij = torch.einsum('ij,ij->i', q_ij, x_ij)

            dot = qx_ij / quotient**2

            pi_ij = dot / rho_bar * m_j
            if piSwitch:
                pi_ij = torch.where(qx_ij < 0, pi_ij, 0)

            return -scatter_sum(pi_ij.view(-1,1) * gradW_ij, i, dim = 0, dim_size = ni)  
        elif laplacianMode == 'Monaghan1992beta':
            rho_bar = (rho_i + rho_j) / 2
            qx_ij = torch.einsum('ij,ij->i', q_ij, x_ij)

            dot = qx_ij / quotient**2

            mu_ij = (dot / rho_bar)**2
            
            pi_ij = mu_ij * m_j
            if piSwitch:
                pi_ij = torch.where(qx_ij < 0, pi_ij, 0)

            return scatter_sum(pi_ij.view(-1,1) * gradW_ij, i, dim = 0, dim_size = ni)  

        else: # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 777 (eq 91) in https://www.sciencedirect.com/science/article/pii/S0021999110006753 also equation 95
            lk = 2 * fac_j * kernelApproximation

            qx_ij = torch.einsum('ij,ij->i', q_ij, x_ij)
            if piSwitch:
                lk = torch.where(qx_ij < 0, lk, 0)

            kq = torch.einsum('n, n... -> n...', lk, q_ij)

            return scatter_sum(kq, i, dim = 0, dim_size = ni)


    @staticmethod
    def backward(ctx, grad_output):
        ## -------------------------------------------------------------- ##
        ## ---------------------- Start of preamble --------------------- ##
        ## -------------------------------------------------------------- ##
        # Load saved tensors
        masses_i, masses_j, densities_i, densities_j, quantities_i, quantities_j, positions_i, positions_j, supports_i, supports_j, i, j = ctx.saved_tensors

        # Load saved variables
        wrappedKernel = ctx.kernel
        laplacianMode = ctx.laplacianMode
        support = ctx.support
        periodicity = ctx.periodicity
        minExtent = ctx.minExtent
        maxExtent = ctx.maxExtent
        # print('Computing backwards for {laplacianMode}')

        # rename variables for ease of usage
        masses = (masses_i, masses_j)
        densities = (densities_i, densities_j)
        quantities = (quantities_i, quantities_j)
        positions = (positions_i, positions_j)
        supports = (supports_i, supports_j)
        
        # compute relative positions and support radii
        x_ij = mod_distance(get_i(positions, i), get_j(positions, j), periodicity, minExtent, maxExtent)
        r_ij = torch.linalg.norm(x_ij, dim = -1)
        h_ij = getSupport(supports, i, j, mode = support)

        # compute ancillary variables
        ni = positions[0].shape[0]# if isinstance(positions, tuple) else positions.shape[0]
        nj = positions[1].shape[0]# if isinstance(positions, tuple) else positions.shape[0]

        ## -------------------------------------------------------------- ##
        ## ---------------------- End of preamble ----------------------- ##
        ## -------------------------------------------------------------- ##

        # gradW_ij = wrappedKernel.jacobian(x_ij, h_ij)

        q_i = get_i(quantities, i) #if laplacianMode != 'naive' else None
        q_j = get_j(quantities, j)  if quantities[1].shape[0] == positions[1].shape[0] else quantities[1]
        rho_i = get_i(densities, i) #if gradientMode == 'symmetric' else None
        rho_j = get_j(densities, j)
        m_j = get_j(masses, j)
        grad_i = grad_output[i]

        # Initialize gradients as None for all variables
        m_i_grad = rho_i_grad = q_i_grad = x_i_grad = h_i_grad = None
        m_j_grad = rho_j_grad = q_j_grad = x_j_grad = h_j_grad = None

        # Premultiply the incoming gradient with the outgoing gradient
        # gradientKernel = torch.einsum('n...d, nd -> n...', grad_i, gradW_ij)

        m_j = get_j(masses, j)
        rho_i = get_i(densities, i)
        rho_j = get_j(densities, j)

        W_ij = wrappedKernel.eval(x_ij, h_ij)
        # print('W_ij', W_ij.shape)
        gradW_ij = wrappedKernel.jacobian(x_ij, h_ij)
        # print('gradW_ij', gradW_ij.shape)
        laplaceW_ij = wrappedKernel.laplacian(x_ij, h_ij)
        # print('laplaceW_ij', laplaceW_ij.shape)
        laplaceW_ijGrad = wrappedKernel.laplaciangrad(x_ij, h_ij)
        # print('laplaceW_ijGrad', laplaceW_ijGrad.shape)

        ddhW_ij = wrappedKernel.dkdh(x_ij, h_ij)
        ddhJacobian = wrappedKernel.djacobiandh(x_ij, h_ij)
        ddhLaplacian = wrappedKernel.dlaplaciandh(x_ij, h_ij)
        # print('ddhW_ij', ddhW_ij.shape)

        # quotient = (r_ij + 1e-7 * h_ij)
        laplacianApproximation = wrappedKernel.laplacianApproximation(x_ij, h_ij)
        # print('laplacianApproximation', laplacianApproximation.shape)
        laplacianApproximationGrad = wrappedKernel.laplacianApproximationGrad(x_ij, h_ij)
        # print('laplacianApproximationGrad', laplacianApproximationGrad.shape)
        laplacianApproximationGradH = wrappedKernel.ddhlaplacianApproximation(x_ij, h_ij)

        q_ij = (get_j(quantities, j) - get_i(quantities, i))
        fac_j = m_j / rho_j

        # print(f'grad_i: {grad_i.shape} | q_ij: {q_ij.shape}')

        if laplacianMode == 'naive': # (eq 86)
            lk = fac_j * laplaceW_ij
            kq = torch.einsum('n, n... -> n...', lk, q_ij)

            m_j_grad = -flattened_sum(product(laplaceW_ij / rho_j, product(q_ij, grad_i)), j, nj, 0)
            rho_j_grad = flattened_sum(product(laplaceW_ij * m_j / rho_j**2, product(q_ij, grad_i)), j, nj, 0)
            q_i_grad = scatter_sum(product(m_j / rho_j, product(grad_i, laplaceW_ij)), i, dim_size=ni, dim = 0)
            q_j_grad = -scatter_sum(product(m_j / rho_j, product(grad_i, laplaceW_ij)), j, dim_size=nj, dim = 0)

            gradProd = torch.einsum('nu, n... -> n...u', laplaceW_ijGrad, grad_i)
            prodTerm = torch.einsum('n...u, n... -> nu', gradProd, q_ij)
            x_i_grad = scatter_sum(-product(prodTerm, m_j / rho_j), i, dim_size=ni, dim = 0)
            x_j_grad = scatter_sum(product(prodTerm, m_j / rho_j), j, dim_size=nj, dim = 0)

            support_grad = -flattened_sum(product(m_j / rho_j, product(q_ij, product(ddhLaplacian, grad_i))), j, nj, 0)
            support_grad = support_grad * (1/2 if support == 'symmetric' else 1)

            h_i_grad = support_grad if support == 'gather'  or support == 'symmetric' else None
            h_j_grad = support_grad if support == 'scatter' or support == 'symmetric' else None
        elif laplacianMode == 'verynaive': # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 777 (eq 90) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
            lk = fac_j * laplaceW_ij
            q_ij = get_j(quantities, j)  if quantities[1].shape[0] == positions[1].shape[0] else quantities[1]
            kq = torch.einsum('n, n... -> n...', lk, q_ij)

            m_j_grad = -flattened_sum(product(laplaceW_ij / rho_j, product(q_ij, grad_i)), j, nj, 0)
            rho_j_grad = flattened_sum(product(laplaceW_ij * m_j / rho_j**2, product(q_ij, grad_i)), j, nj, 0)
            if quantities[1].shape[0] == positions[1].shape[0]:
                q_j_grad = -scatter_sum(product(m_j / rho_j, product(grad_i, laplaceW_ij)), j, dim_size=nj, dim = 0)
            else:
                q_j_grad = -product(m_j / rho_j, product(grad_i, laplaceW_ij))
            # print(f'q_j_grad: {q_j_grad.shape}')

            gradProd = torch.einsum('nu, n... -> n...u', laplaceW_ijGrad, grad_i)
            prodTerm = torch.einsum('n...u, n... -> nu', gradProd, q_ij)

            x_i_grad = scatter_sum(-product(prodTerm, m_j / rho_j), i, dim_size=ni, dim = 0)
            x_j_grad = scatter_sum(product(prodTerm, m_j / rho_j), j, dim_size=nj, dim = 0)

            support_grad = -flattened_sum(product(m_j / rho_j, product(q_ij, product(ddhLaplacian, grad_i))), j, nj, 0)
            support_grad = support_grad * (1/2 if support == 'symmetric' else 1)

            h_i_grad = support_grad if support == 'gather'  or support == 'symmetric' else None
            h_j_grad = support_grad if support == 'scatter' or support == 'symmetric' else None

        elif laplacianMode == 'non-conserving': 
            x_ij = mod_distance(get_i(positions, i), get_j(positions, j), periodicity, minExtent, maxExtent)
            r_ij = torch.linalg.norm(x_ij, dim = -1)

            dim = x_ij.shape[1]
            n_ij = torch.nn.functional.normalize(x_ij, dim = -1)
            
            term = -(dim + 2) * torch.einsum('nd, nd -> n', q_ij, n_ij).view(-1,1) * n_ij - q_ij
            # term = -(dim + 2) * torch.einsum('nd, nd -> n', q_ij, n_ij).view(-1,1) * n_ij
            # term = -q_ij
            kq = term * (fac_j * laplacianApproximation).view(-1,1)

            kernelTerm = product(laplacianApproximation, grad_i)

            m_j_grad = flattened_sum(product(1 / rho_j, product(term, kernelTerm)), j, nj, 0)
            rho_j_grad = -flattened_sum(product(m_j / rho_j**2, product(term, kernelTerm)), j, nj, 0)

            qTerm_i = (dim + 2) * torch.einsum('ni, nj -> nij', n_ij, n_ij) + torch.eye(dim).to(x_ij.device).to(x_ij.dtype)
            qTerm_j = -qTerm_i
            qGrad_i = torch.einsum('nij, nj -> ni', qTerm_i, grad_i)
            pst_i = product(m_j / rho_j * laplacianApproximation, qGrad_i)
            q_i_grad = scatter_sum(pst_i, i, dim_size=ni, dim = 0)
            qGrad_j = torch.einsum('nij, nj -> ni', qTerm_j, grad_i)
            pst_j = product(m_j / rho_j * laplacianApproximation, qGrad_j)
            q_j_grad = scatter_sum(pst_j, j, dim_size=nj, dim = 0)

            gradProd = torch.einsum('nu, n... -> n...u', laplacianApproximationGrad, grad_i)
            rightTerm = torch.einsum('n..., n...d -> nd', term, gradProd)

            r_eps = r_ij + 1e-7 * h_ij

            # We need to compute grad (-(d+2) dot(q_ij, x_ij) x_ij / |x_ij|**2)
            # This gives three terms:
            # a: dot(q_ij, x_ij)
            # b: x_ij
            # c: |x_ij|**2 = r_ij**2 + eps = r_eps**2
            # term = a * b / c
            a = -(dim + 2) * torch.einsum('nd, nd -> n', q_ij, x_ij)
            b = x_ij
            c = r_eps**2
            # use prod rule for f = ab and g = c, i.e., term' = (f'g - f g')/g**2
            f = a.view(-1,1) * b
            g = c

            # grad_a = - (d+2) * (grad q_ij . x_ij + q_ij . grad x_ij) # dot prod is distributive wrt nabla
            # grad q_ij =   0
            # grad x_ij = -Id
            # grad_a then is simply - (d+2) q_ij
            grad_a = - (dim + 2) * q_ij
            # build a batched Id matrix via einsum
            grad_b = -torch.einsum('nk, kx -> nxk', torch.ones_like(x_ij), torch.eye(x_ij.shape[1]))
            # straight forward grad of c
            grad_c = 2 * x_ij
            # apply prod rule ab = a'b + ab' to compute grad term
            # a' and b are both vectors and need to be broadcasted to a matrix
            # note the transpose in the einsum
            # for ab' we need to broadcast a to a matrix via view and b' is already a matrix
            # however, grad_b is simply -Id, so this is equivalent to substracting from the diagonals
            grad_f = torch.einsum('ni, nj -> nji', grad_a, x_ij) - a.view(-1,1,1) * grad_b
            # compute f'g - f g'
            nom = grad_f * g.view(-1,1,1) - torch.einsum('ni, nj -> nij', f, grad_c)
            # compute g**2
            denom = g.view(-1,1,1)**2
            leftTerm_imm = nom / denom
            leftTerm = torch.einsum('nki, nk -> ni', leftTerm_imm, kernelTerm)

            x_i_grad = scatter_sum(product(leftTerm + rightTerm, m_j / rho_j), i, dim_size=ni, dim = 0)
            x_j_grad = scatter_sum(-product(leftTerm + rightTerm, m_j / rho_j), j, dim_size=nj, dim = 0)



            q_grad = torch.einsum('n..., n... -> n', term, grad_i)
            support_grad = flattened_sum(product(m_j / rho_j, product(q_grad, laplacianApproximationGradH)), j, nj, 0)
            support_grad = support_grad * (1/2 if support == 'symmetric' else 1)

            h_i_grad = support_grad if support == 'gather'  or support == 'symmetric' else None
            h_j_grad = support_grad if support == 'scatter' or support == 'symmetric' else None


        elif laplacianMode == 'conserving': # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 778 (eq 97) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
            n_ij = torch.nn.functional.normalize(x_ij, dim = -1)
            r_ij = torch.linalg.norm(x_ij, dim = -1)
            r_eps = r_ij + 1e-7 * h_ij
            dim = x_ij.shape[1]

            dot = torch.einsum('nd, nd -> n', q_ij, n_ij)
            f = torch.einsum('nd, nd -> n', q_ij, x_ij)
            g = r_eps#**2
            f_prime = q_ij
            g_prime = x_ij / r_eps.view(-1,1)

            dot_grad = (f_prime * g.view(-1,1) - f.view(-1,1) * g_prime) / g.view(-1,1)**2

            
            # hessian = -laplacianApproximationGrad
            kernelApproximation = gradW_ij /  r_eps.view(-1,1) #**2 
            f = gradW_ij
            g = r_eps
            f_prime = wrappedKernel.hessian(x_ij, h_ij)
            g_prime = x_ij / r_eps.view(-1,1)
            kernelApproximationGrad = (f_prime * g.view(-1,1,1) - torch.einsum('ni, nk -> nik', f, g_prime)) / g.view(-1,1,1)**2
            # print(f'hessian: {kernelApproximationGrad.shape} | grad_i: {grad_i.shape} | dot {dot.shape} | gradW_ij: {gradW_ij.shape}')

            q = fac_j * dot
            kq = -q.view(-1, 1) * kernelApproximation

            # return scatter_sum(kq, i, dim = 0, dim_size = ni)

            m_j_grad = -flattened_sum(product(1 / rho_j, product(dot, product(kernelApproximation, grad_i))), j, nj, 0)
            rho_j_grad = flattened_sum(product(m_j / rho_j**2, product(dot, product(kernelApproximation, grad_i))), j, nj, 0)

            qTerm = torch.einsum('ni, nk -> nik', x_ij / r_eps.view(-1,1), grad_i)
            qTerm = torch.einsum('nik, nk -> ni', qTerm, kernelApproximation)
            q_i_grad = scatter_sum(product(m_j / rho_j, qTerm), i, dim_size=ni, dim = 0)
            q_j_grad = scatter_sum(-product(m_j / rho_j, qTerm), j, dim_size=nj, dim = 0)


            leftTerm = torch.einsum('ni, nj -> nij', dot_grad, kernelApproximation)
            rightTerm = kernelApproximationGrad * dot.view(-1,1,1)
            # print(f'leftTerm: {leftTerm.shape} | rightTerm: {rightTerm.shape} | grad_i: {grad_i.shape}')

            gradTerm = torch.einsum('nik, nk -> ni', leftTerm + rightTerm, grad_i)
            # print(f'gradTerm: {gradTerm.shape}')
            x_i_grad = scatter_sum(-product(m_j / rho_j, gradTerm), i, dim_size=ni, dim = 0)
            x_j_grad = scatter_sum(product(m_j / rho_j, gradTerm), j, dim_size=nj, dim = 0)

            gradHTerm = ddhJacobian / r_eps.view(-1,1)
            support_grad = -flattened_sum(product(m_j / rho_j, product(dot, product(gradHTerm, grad_i))), j, nj, 0)
            support_grad = support_grad * (1/2 if support == 'symmetric' else 1)

            h_i_grad = support_grad if support == 'gather'  or support == 'symmetric' else None
            h_j_grad = support_grad if support == 'scatter' or support == 'symmetric' else None

        elif laplacianMode == 'divergenceFree': # https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf eq 26
            n_ij = torch.nn.functional.normalize(x_ij, dim = -1)
            r_ij = torch.linalg.norm(x_ij, dim = -1)
            r_eps = r_ij + 1e-7 * h_ij
            dim = x_ij.shape[1]

            hessian = -wrappedKernel.hessian(x_ij, h_ij)

            dot = torch.einsum('nd, nd -> n', q_ij, x_ij) / r_eps**2

            f = torch.einsum('nd, nd -> n', q_ij, x_ij)
            g = r_eps**2
            f_prime = q_ij
            g_prime = 2 * x_ij

            dot_grad = (f_prime * g.view(-1,1) - f.view(-1,1) * g_prime) / g.view(-1,1)**2

            q = 2 * (dim + 2) * fac_j * dot
            
            kq = q.view(-1, 1) * gradW_ij

            kernelTerm = product(gradW_ij, grad_i)            

            m_j_grad = 2 * (dim + 2 ) * flattened_sum(product(dot / rho_j, kernelTerm), j, nj, 0)
            rho_j_grad = - 2 * (dim + 2 ) * flattened_sum(product(m_j * dot / rho_j**2, kernelTerm), j, nj, 0)

            qTerm = torch.einsum('ni, nk -> nik', x_ij / r_eps.view(-1,1)**2, grad_i)
            qTerm = torch.einsum('nik, nk -> ni', qTerm, gradW_ij)
            q_i_grad = 2 * (dim + 2 ) * scatter_sum(-product(m_j / rho_j, qTerm), i, dim_size=ni, dim = 0)
            q_j_grad = 2 * (dim + 2 ) * scatter_sum(product(m_j / rho_j, qTerm), j, dim_size=nj, dim = 0)

            leftTerm = torch.einsum('ni, nj -> nij', dot_grad, gradW_ij)
            rightTerm = hessian * dot.view(-1,1,1)

            gradTerm = torch.einsum('nik, nk -> ni', leftTerm - rightTerm, grad_i)

            x_i_grad = 2 * (dim + 2 ) * scatter_sum(product(m_j / rho_j, gradTerm), i, dim_size=ni, dim = 0)
            x_j_grad = 2 * (dim + 2 ) * scatter_sum(-product(m_j / rho_j, gradTerm), j, dim_size=nj, dim = 0)

            # ddhLaplacian = wrappedKernel.dlaplaciandh(x_ij, h_ij)
            # print(f'ddhLaplacian: {ddhLaplacian.shape} | grad_i: {grad_i.shape} | dot {dot.shape} | gradW_ij: {gradW_ij.shape}')

            support_grad = 2 * (dim + 2 ) * flattened_sum(product(m_j / rho_j, product(dot, product(ddhJacobian, grad_i))), j, nj, 0)
            support_grad = support_grad * (1/2 if support == 'symmetric' else 1)

            h_i_grad = support_grad if support == 'gather'  or support == 'symmetric' else None
            h_j_grad = support_grad if support == 'scatter' or support == 'symmetric' else None
            # return scatter_sum(kq, i, dim = 0, dim_size = ni)
        
        elif laplacianMode == 'dot': # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 778 (eq 96) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
            x_ij = mod_distance(get_i(positions, i), get_j(positions, j), periodicity, minExtent, maxExtent)
            r_ij = torch.linalg.norm(x_ij, dim = -1)

            dim = x_ij.shape[1]
            n_ij = torch.nn.functional.normalize(x_ij, dim = -1)
            
            term = -(dim + 2) * torch.einsum('nd, nd -> n', q_ij, n_ij).view(-1,1) * n_ij - q_ij
            # term = -(dim + 2) * torch.einsum('nd, nd -> n', q_ij, n_ij).view(-1,1) * n_ij
            # term = -q_ij
            kq = term * (fac_j * laplacianApproximation).view(-1,1)

            kernelTerm = product(laplacianApproximation, grad_i)

            m_j_grad = flattened_sum(product(1 / rho_j, product(term, kernelTerm)), j, nj, 0)
            rho_j_grad = -flattened_sum(product(m_j / rho_j**2, product(term, kernelTerm)), j, nj, 0)

            qTerm_i = (dim + 2) * torch.einsum('ni, nj -> nij', n_ij, n_ij) + torch.eye(dim).to(x_ij.device).to(x_ij.dtype)
            qTerm_j = -qTerm_i
            qGrad_i = torch.einsum('nij, nj -> ni', qTerm_i, grad_i)
            pst_i = product(m_j / rho_j * laplacianApproximation, qGrad_i)
            q_i_grad = scatter_sum(pst_i, i, dim_size=ni, dim = 0)
            qGrad_j = torch.einsum('nij, nj -> ni', qTerm_j, grad_i)
            pst_j = product(m_j / rho_j * laplacianApproximation, qGrad_j)
            q_j_grad = scatter_sum(pst_j, j, dim_size=nj, dim = 0)

            gradProd = torch.einsum('nu, n... -> n...u', laplacianApproximationGrad, grad_i)
            rightTerm = torch.einsum('n..., n...d -> nd', term, gradProd)


            r_eps = r_ij + 1e-7 * h_ij

            # We need to compute grad (-(d+2) dot(q_ij, x_ij) x_ij / |x_ij|**2)
            # This gives three terms:
            # a: dot(q_ij, x_ij)
            # b: x_ij
            # c: |x_ij|**2 = r_ij**2 + eps = r_eps**2
            # term = a * b / c
            a = -(dim + 2) * torch.einsum('nd, nd -> n', q_ij, x_ij)
            b = x_ij
            c = r_eps**2
            # use prod rule for f = ab and g = c, i.e., term' = (f'g - f g')/g**2
            f = a.view(-1,1) * b
            g = c

            # grad_a = - (d+2) * (grad q_ij . x_ij + q_ij . grad x_ij) # dot prod is distributive wrt nabla
            # grad q_ij =   0
            # grad x_ij = -Id
            # grad_a then is simply - (d+2) q_ij
            grad_a = - (dim + 2) * q_ij
            # build a batched Id matrix via einsum
            grad_b = -torch.einsum('nk, kx -> nxk', torch.ones_like(x_ij), torch.eye(x_ij.shape[1]))
            # straight forward grad of c
            grad_c = 2 * x_ij
            # apply prod rule ab = a'b + ab' to compute grad term
            # a' and b are both vectors and need to be broadcasted to a matrix
            # note the transpose in the einsum
            # for ab' we need to broadcast a to a matrix via view and b' is already a matrix
            # however, grad_b is simply -Id, so this is equivalent to substracting from the diagonals
            grad_f = torch.einsum('ni, nj -> nji', grad_a, x_ij) - a.view(-1,1,1) * grad_b
            # compute f'g - f g'
            nom = grad_f * g.view(-1,1,1) - torch.einsum('ni, nj -> nij', f, grad_c)
            # compute g**2
            denom = g.view(-1,1,1)**2
            leftTerm_imm = nom / denom

            leftTerm = torch.einsum('nki, nk -> ni', leftTerm_imm, kernelTerm)

            x_i_grad = scatter_sum(product(leftTerm + rightTerm, m_j / rho_j), i, dim_size=ni, dim = 0)
            x_j_grad = scatter_sum(-product(leftTerm + rightTerm, m_j / rho_j), j, dim_size=nj, dim = 0)



            q_grad = torch.einsum('n..., n... -> n', term, grad_i)
            support_grad = flattened_sum(product(m_j / rho_j, product(q_grad, laplacianApproximationGradH)), j, nj, 0)
            support_grad = support_grad * (1/2 if support == 'symmetric' else 1)

            h_i_grad = support_grad if support == 'gather'  or support == 'symmetric' else None
            h_j_grad = support_grad if support == 'scatter' or support == 'symmetric' else None

            


        else: # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 777 (eq 91) in https://www.sciencedirect.com/science/article/pii/S0021999110006753
            kernelTerm = product(laplacianApproximation, grad_i)

            m_j_grad = flattened_sum(product(2 / rho_j, product(q_ij, kernelTerm)), j, nj, 0)
            rho_j_grad = -flattened_sum(product(2 * m_j / rho_j**2, product(q_ij, kernelTerm)), j, nj, 0)
            q_i_grad = -scatter_sum(product(2 * m_j / rho_j, kernelTerm), i, dim_size=ni, dim = 0)
            q_j_grad = scatter_sum(product(2 * m_j / rho_j, kernelTerm), j, dim_size=nj, dim = 0)

            # qgrad = torch.einsum('n..., n... -> n', q_ij, grad_i)

            gradProd = torch.einsum('nu, n... -> n...u', laplacianApproximationGrad, grad_i)
            term = 2 * torch.einsum('n..., n...d -> nd', q_ij, gradProd)
            x_i_grad = -scatter_sum(-product(term, m_j / rho_j), i, dim_size=ni, dim = 0)
            x_j_grad = -scatter_sum(product(term, m_j / rho_j), j, dim_size=nj, dim = 0)

            q_grad = torch.einsum('n..., n... -> n', q_ij, grad_i)

            support_grad = flattened_sum(2 * product(m_j / rho_j, product(q_grad, laplacianApproximationGradH)), j, nj, 0)
            support_grad = support_grad * (1/2 if support == 'symmetric' else 1)

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
            None, None,\
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
from sphMath.sphOperations.opUtil import evalSupport, evalPrecomputed, get_qi, get_qj, get_qs, correctedKernel_CRK, correctedKernelGradient_CRK

def laplacian_precomputed(
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
        H_ij = evalPrecomputed(Hessian, mode = supportScheme)

        laplacian_ij = torch.diagonal(H_ij, dim1 = 1, dim2 = 2).sum(dim = 1)



        factor = m_j / rho_j if gradientMode != GradientMode.Symmetric else m_j * rho_i
        if useApparentArea and apparentArea_b is not None:
            if gradientMode != GradientMode.Symmetric:
                factor = apparentArea_b[j]
            else:
                factor = apparentArea_b[j] * rho_i * rho_j

        q_ij: Optional[torch.Tensor] = None
        if quantity_ab is not None:
            q_ij = quantity_ab
        else:
            q_i = get_qi(quantity, i, j, gradH = (None, None))
            q_j = get_qj(quantity, i, j, gradH = (None, None))
            if gradientMode == GradientMode.Naive:
                q_ij = q_j
            elif gradientMode == GradientMode.Difference:
                q_ij = q_j - q_i
            elif gradientMode == GradientMode.Summation:
                q_ij = q_j + q_i
            elif gradientMode == GradientMode.Symmetric:
                qi = torch.einsum('n..., n -> n...', q_i, 1/ rho_i**2)
                qj = torch.einsum('n..., n -> n...', q_j, 1/ rho_j**2)
                q_ij = qj + qi
            else:
                raise ValueError(f"Unknown gradient mode {gradientMode}")

        if q_ij is None:
            raise ValueError("q_ij is None")

        fq = torch.einsum('n..., n -> n...', q_ij, factor)

        fkq : Optional[torch.Tensor] = None
        if laplacianMode == LaplacianMode.naive:
            fkq = torch.einsum('n..., n -> n...', fq, laplacian_ij)
            # print("Something")
        elif laplacianMode == LaplacianMode.Brookshaw:
            F_ab = torch.einsum('nd, nd -> n', x_ij, gradW_ij) / (r_ij + 1e-8 * supports[0][i])**2
            fkq = torch.einsum('n..., n -> n...', fq, -2 * F_ab)
        elif laplacianMode == LaplacianMode.dot:
            # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 778 (eq 96) in https://www.sciencedirect.com/science/article/pii/S0021999110006753

            r_eps = r_ij + 1e-8 * supports[0][i]
            n_ij = x_ij / r_eps.view(-1,1)
            F_ab = torch.einsum('nd, nd -> n', n_ij, gradW_ij) / r_eps

            leftTerm = (x_ij.shape[1] + 2) * torch.einsum('n..., nd -> n...d', torch.einsum('n...d, nd -> n...', fq, n_ij), n_ij)
            rightTerm = - fq

            fkq = -(leftTerm + rightTerm) * F_ab.view(-1,1)

            dot = torch.einsum('n...d, nd -> n...d', fq, x_ij / (r_ij + 1e-8 * supports[0][i]).view(-1,1)**2)
        elif laplacianMode == LaplacianMode.default:
            dot = torch.einsum('n..., nd -> n...d', fq, x_ij / (r_ij + 1e-8 * supports[0][i]).view(-1,1)**2)
            fkq = torch.einsum('n...d, nd -> n...', dot, gradW_ij)
            fkq = -2 * fkq

            # if positiveDivergence:
                # fkq = torch.where(dot < 0, torch.zeros_like(fkq), fkq)
            # fkq = fkq * (2 + x_ij.shape[1])
        else:
            raise ValueError(f"Unknown laplacian mode {laplacianMode}")
    
        if fkq is None:
            raise ValueError("fkq is None")
    
        if positiveDivergence:
            dot = torch.einsum('n...d, nd -> n...d', fq, x_ij / (r_ij + 1e-8 * supports[0][i]).view(-1,1)**2)
            fkq = torch.where(dot < 0, torch.zeros_like(fkq), fkq)
            # print("Positive divergence")

        summed = scatter_sum(fkq, i, dim_size=ni, dim = 0)
        return summed
        
from sphMath.sphOperations.opUtil import custom_forwards, custom_backwards, evaluateKernel_, evaluateKernelGradient_, get_q



def laplacian_fn(
        supportScheme: SupportScheme,
        gradientMode: GradientMode,
        useApparentArea: bool,
        laplacianMode: LaplacianMode,
        positiveDivergence: bool,
        crkCorrection: bool,
        gradientCorrection: bool,
        omegaCorrection: bool,
        i: torch.Tensor, j: torch.Tensor,
        numRows: int, numCols: int,

        q_i_: Optional[torch.Tensor], rho_i: Optional[torch.Tensor], omega_i: Optional[torch.Tensor], h_i: Optional[torch.Tensor], A_i: Optional[torch.Tensor], B_i: Optional[torch.Tensor], gradA_i: Optional[torch.Tensor], gradB_i: Optional[torch.Tensor], L_i: Optional[torch.Tensor],

        q_j_: Optional[torch.Tensor], m_j: torch.Tensor, rho_j: torch.Tensor, apparentArea_b: Optional[torch.Tensor], omega_j: Optional[torch.Tensor], A_j: Optional[torch.Tensor], B_j: Optional[torch.Tensor], gradA_j: Optional[torch.Tensor], gradB_j: Optional[torch.Tensor], 

        q_ij: Optional[torch.Tensor], x_ij: torch.Tensor, r_ij: torch.Tensor,
        W_i: torch.Tensor, W_j: torch.Tensor,
        gradW_i: torch.Tensor, gradW_j: torch.Tensor,
        H_i: Optional[torch.Tensor], H_j: Optional[torch.Tensor],

):
        # compute relative positions and support radii
        x_ij = x_ij
        # h_ij = evalSupport(supports, i, j, mode = supportScheme)

        # compute ancillary variables
        ni = numRows
        nj = numCols

        # Helper functions
        # op = gradient_op
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

        if laplacianMode == LaplacianMode.naive:
            if H_i is None or H_j is None:
                raise ValueError("Hessian is None")
            else:
                H_ij = evalPrecomputed((H_i, H_j), mode = supportScheme)
            laplacian_ij = torch.diagonal(H_ij, dim1 = 1, dim2 = 2).sum(dim = 1)

                
        factor = m_j / rho_j if gradientMode != GradientMode.Symmetric else m_j * rho_i
        if useApparentArea and apparentArea_b is not None:
            if gradientMode != GradientMode.Symmetric:
                factor = apparentArea_b[j]
            else:
                factor = apparentArea_b[j] * rho_i * rho_j

        q_ij: Optional[torch.Tensor] = None
        if q_ij is not None:
            q_ij = q_ij
        else:
            if gradientMode == GradientMode.Naive:
                q_j = get_q(q_j_, omega_j, omegaCorrection)
                q_ij = q_j
            elif gradientMode == GradientMode.Difference:
                q_i = get_q(q_i_, omega_i, omegaCorrection)
                q_j = get_q(q_j_, omega_j, omegaCorrection)
                q_ij = q_j - q_i
            elif gradientMode == GradientMode.Summation:
                q_i = get_q(q_i_, omega_i, omegaCorrection)
                q_j = get_q(q_j_, omega_j, omegaCorrection)
                q_ij = q_j + q_i
            elif gradientMode == GradientMode.Symmetric:
                q_i = get_q(q_i_, omega_i, omegaCorrection)
                q_j = get_q(q_j_, omega_j, omegaCorrection)
                qi = torch.einsum('n..., n -> n...', q_i, 1/ rho_i**2)
                qj = torch.einsum('n..., n -> n...', q_j, 1/ rho_j**2)
                q_ij = qj + qi
            else:
                raise ValueError(f"Unknown gradient mode {gradientMode}")

        if q_ij is None:
            raise ValueError("q_ij is None")

        fq = torch.einsum('n..., n -> n...', q_ij, factor)

        fkq : Optional[torch.Tensor] = None
        if laplacianMode == LaplacianMode.naive:
            if laplacian_ij is None:
                fkq = torch.einsum('n..., n -> n...', fq, laplacian_ij)
            # print("Something")
        elif laplacianMode == LaplacianMode.Brookshaw:
            F_ab = torch.einsum('nd, nd -> n', x_ij, gradW_ij) / (r_ij + 1e-8 * h_i)**2
            fkq = torch.einsum('n..., n -> n...', fq, -2 * F_ab)
        elif laplacianMode == LaplacianMode.dot:
            # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 778 (eq 96) in https://www.sciencedirect.com/science/article/pii/S0021999110006753

            r_eps = r_ij + 1e-8 * h_i
            n_ij = x_ij / r_eps.view(-1,1)
            F_ab = torch.einsum('nd, nd -> n', n_ij, gradW_ij) / r_eps

            leftTerm = (x_ij.shape[1] + 2) * torch.einsum('n..., nd -> n...d', torch.einsum('n...d, nd -> n...', fq, n_ij), n_ij)
            rightTerm = - fq

            fkq = -(leftTerm + rightTerm) * F_ab.view(-1,1)

            dot = torch.einsum('n...d, nd -> n...d', fq, x_ij / (r_ij + 1e-8 * h_i).view(-1,1)**2)
        elif laplacianMode == LaplacianMode.default:
            dot = torch.einsum('n..., nd -> n...d', fq, x_ij / (r_ij + 1e-8 * h_i).view(-1,1)**2)
            fkq = torch.einsum('n...d, nd -> n...', dot, gradW_ij)
            fkq = -2 * fkq

            # if positiveDivergence:
                # fkq = torch.where(dot < 0, torch.zeros_like(fkq), fkq)
            # fkq = fkq * (2 + x_ij.shape[1])
        else:
            raise ValueError(f"Unknown laplacian mode {laplacianMode}")
    
        if fkq is None:
            raise ValueError("fkq is None")
    
        if positiveDivergence:
            dot = torch.einsum('n...d, nd -> n...d', fq, x_ij / (r_ij + 1e-8 * h_i).view(-1,1)**2)
            fkq = torch.where(dot < 0, torch.zeros_like(fkq), fkq)
            # print("Positive divergence")

        # fkq = torch.einsum('n..., n -> n...', foklq, factor)
        
        summed = scatter_sum(fkq, i, dim_size=ni, dim = 0)

        return summed





class Laplacian(torch.autograd.Function):
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
            quantity_a, densities_a, supports_a, correctionTerm_omega_i, correctionTerm_A_i, correctionTerm_B_i, correctionTerm_gradA_i, correctionTerm_gradB_i, correctionTerm_gradMatrix_i,
            quantity_b, masses_b, densities_b, apparentArea_b, correctionTerm_omega_j, correctionTerm_A_j, correctionTerm_B_j, correctionTerm_gradA_j, correctionTerm_gradB_j,
            quantity_ab, x_ij, r_ij, W_i, W_j, gradW_i, gradW_j, H_i, H_j, i,  j)
        ctx.correctionTerms = correctionTerms
        ctx.supportScheme = supportScheme
        ctx.useApparentArea = useApparentArea
        ctx.numRows = numRows
        ctx.numCols = numCols
        ctx.gradientMode = gradientMode
        ctx.crkCorrection = correctionTerms is not None and KernelCorrectionScheme.CRKSPH.value in [c.value for c in correctionTerms]
        ctx.gradientCorrection = correctionTerms is not None and KernelCorrectionScheme.gradientRenorm.value in [c.value for c in correctionTerms]
        ctx.omegaCorrection = correctionTerms is not None and KernelCorrectionScheme.gradH.value in [c.value for c in correctionTerms]
        ctx.laplacianMode = laplacianMode
        ctx.positiveDivergence = positiveDivergence

        inputs_i = [quantity_a, densities_a, supports_a, correctionTerm_omega_i, correctionTerm_A_i, correctionTerm_B_i, correctionTerm_gradA_i, correctionTerm_gradB_i, correctionTerm_gradMatrix_i]
        inputs_j = [quantity_b, masses_b, densities_b, apparentArea_b, correctionTerm_omega_j, correctionTerm_A_j, correctionTerm_B_j, correctionTerm_gradA_j, correctionTerm_gradB_j]
        inputs_ij = [quantity_ab, x_ij, r_ij, W_i, W_j, gradW_i, gradW_j]

        return custom_forwards(
            laplacian_fn, 
            i, j, ctx.numRows, ctx.numCols,
            inputs_i, inputs_j, inputs_ij,
            supportScheme,
            gradientMode,
            useApparentArea,
            ctx.laplacianMode,
            ctx.positiveDivergence,
            ctx.crkCorrection,
            ctx.gradientCorrection,
            ctx.omegaCorrection,
        )
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        quantity_a, densities_a, supports_a, correctionTerm_omega_i, correctionTerm_A_i, correctionTerm_B_i, correctionTerm_gradA_i, correctionTerm_gradB_i, correctionTerm_gradMatrix_i, quantity_b, masses_b, densities_b, apparentArea_b, correctionTerm_omega_j, correctionTerm_A_j, correctionTerm_B_j, correctionTerm_gradA_j, correctionTerm_gradB_j, quantity_ab, x_ij, r_ij, W_i, W_j, gradW_i, gradW_j, H_i, H_j, i,  j = ctx.saved_tensors

        # Load saved variables

        inputs_i = [quantity_a, densities_a, supports_a, correctionTerm_omega_i, correctionTerm_A_i, correctionTerm_B_i, correctionTerm_gradA_i, correctionTerm_gradB_i, correctionTerm_gradMatrix_i]
        inputs_j = [quantity_b, masses_b, densities_b, apparentArea_b, correctionTerm_omega_j, correctionTerm_A_j, correctionTerm_B_j, correctionTerm_gradA_j, correctionTerm_gradB_j]
        inputs_ij = [quantity_ab, x_ij, r_ij, W_i, W_j, H_i, H_j, gradW_i, gradW_j]


        grad_quantity_a, grad_densities_a, grad_supports_a, grad_correctionTerm_omega_i, grad_correctionTerm_A_i, grad_correctionTerm_B_i, grad_correctionTerm_gradA_i, grad_correctionTerm_gradB_i, grad_correctionTerm_gradMatrix_i, grad_quantity_b, grad_masses_b, grad_densities_b, grad_apparentArea_b, grad_correctionTerm_omega_j, grad_correctionTerm_A_j, grad_correctionTerm_B_j, grad_correctionTerm_gradA_j, grad_correctionTerm_gradB_j, grad_quantity_ab, grad_x_ij, grad_r_ij, grad_W_i, grad_W_j, grad_gradW_i, grad_gradW_j, grad_H_i, grad_H_j  = custom_backwards(
            laplacian_fn,
            grad_output,
            i, j, ctx.numRows, ctx.numCols,
            inputs_i, inputs_j, inputs_ij,
            ctx.supportScheme,
            ctx.gradientMode,
            ctx.useApparentArea,
            ctx.laplacianMode,
            ctx.positiveDivergence,
            ctx.crkCorrection,
            ctx.gradientCorrection,
            ctx.omegaCorrection,
            )

        return (
            None, None, # positions_a, positions_b, 
            grad_supports_a, None, # supports_a, supports_b, 
            None, grad_masses_b, # masses_a, masses_b,
            grad_densities_a, grad_densities_b, # densities_a, densities_b,
            None, grad_apparentArea_b, # apparentArea_a, apparentArea_b,
            grad_quantity_a, grad_quantity_b, grad_quantity_ab, # quantity_a, quantity_b, quantity_ab,
            None, None, # i, j,
            None, None, # numRows, numCols,
            grad_r_ij, grad_x_ij, # r_ij, x_ij,
            grad_W_i, grad_W_j, # W_i, W_j,
            grad_gradW_i, grad_gradW_j, # gradW_i, gradW_j,
            grad_H_i, grad_H_j, # H_i, H_j,
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
            
            
laplacian_precomputed_op = Laplacian.apply