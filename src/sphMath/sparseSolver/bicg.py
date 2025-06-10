from torch.profiler import profile, record_function, ProfilerActivity
from sphMath.neighborhood import filterNeighborhood, coo_to_csr
import torch
from sphMath.modules.renorm import computeCovarianceMatrices
from sphMath.neighborhood import computeDistanceTensor
from sphMath.sphOperations.shared import scatter_sum
from sphMath.operations import sph_op
# Maronne surface detection
from sphMath.modules.surfaceDetection import computeNormalsMaronne, detectFreeSurfaceMaronne, computeColorField, detectFreeSurfaceColorFieldGradient, detectFreeSurfaceBarecasco, expandFreeSurfaceMask, computeLambdaGrad, detectFreeSurfaceColorField
import numpy as np
from sphMath.modules.density import computeDensity
from sphMath.sparse import make_id, _get_atol_rtol, _get_tensor_eps, matvec_sparse_coo

from typing import Optional, Tuple, Callable
def bicgstab_shifting(A,  b, x0:Optional[torch.Tensor]=None, tol:float = 1e-5, rtol:float=1e-5, atol:float=0., maxiter:Optional[int]=None, M:Optional[ Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int]]=None, verbose: bool = False, threshold: float = 1.0):
    if verbose:
        print(f'BiCGStab Solver')
    
    if M is None:
        if verbose:
            print(f'No preconditioner')
        M = make_id(A)#(shape, device=b.device, dtype=b.dtype)
    xk = x0.clone() if x0 is not None else torch.zeros_like(b)

    bnrm2 = torch.linalg.norm(b)
    if verbose:
        print(f'Initial Residual: {bnrm2}, Threshold {tol}')
    atol, _ = _get_atol_rtol('bicgstab', bnrm2, atol, rtol)
    if verbose:
        print(f'atol: {atol} [{rtol}]')
    convergence = []

    if bnrm2 == 0:
        if verbose:
            print(f'Initial Residual is zero')
        return b, 0, convergence

    n = len(b)

    dotprod = torch.dot

    if maxiter is None:
        if verbose:
            print(f'No max iterations setting to {n*10}')
        maxiter = n*10

    # matvec = A.matvec
    # psolve = M.matvec

    # These values make no sense but coming from original Fortran code
    # sqrt might have been meant instead.
    rhotol = _get_tensor_eps(xk)**2
    omegatol = rhotol
    if verbose:
        print(f'rhotol: {rhotol} | omegatol: {omegatol}')

    # Dummy values to initialize vars, silence linter warnings
    rho_prev = torch.zeros_like(b)
    omega = 0.
    alpha = 0.
    pk = torch.zeros_like(b)
    apk = torch.zeros_like(b)
    # rho_prev, omega, alpha, p, v = None, None, None, None, None

    H, (i,j), numParticles = A
    numParticles = numParticles //2

    # xk = x

    def prod(H, xk, b):
        rk = torch.zeros_like(b)
        rk[::2]  = rk[::2] + scatter_sum(H[:,0,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
        rk[::2]  = rk[::2] + scatter_sum(H[:,0,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        rk[1::2] = rk[1::2] + scatter_sum(H[:,1,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
        rk[1::2] = rk[1::2] + scatter_sum(H[:,1,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)
        return rk
    
    matvec = lambda x: prod(H, x, b)
    psolve = lambda x: matvec_sparse_coo(M, x)

    rk = matvec(xk)

    rk = b -  rk if xk.any() else b.clone()
    # rk = r.clone()
    r0 = rk.clone() # rtilde
    pk = rk.clone() # p
    if verbose:
        print(f'Initial Residual: {torch.linalg.norm(r0)}')

    for iteration in range(maxiter):
        if torch.linalg.norm(rk) < atol:  # Are we done?
            if verbose:
                print(f'Converged after {iteration} iterations {torch.linalg.norm(rk)} | {atol}')
            return xk, iteration, convergence#, r0

        rho = dotprod(rk, r0)
        if torch.abs(rho) < rhotol:  # rho breakdown
            if verbose:
                print(f'\t[{iteration:3d}]\trho breakdown {rho} | {rhotol}')
            return xk, -10, convergence#, r0


        phat = psolve(pk) # v
        # phat = pk
        
        
        apk = matvec(phat)        
        # apk = torch.zeros_like(x0)
        # apk[::2]  = apk[::2] + scatter_sum(H[:,0,0] * phat[j * 2], i, dim=0, dim_size=numParticles)
        # apk[::2]  = apk[::2] + scatter_sum(H[:,0,1] * phat[j * 2 + 1], i, dim=0, dim_size=numParticles)

        # apk[1::2] = apk[1::2] + scatter_sum(H[:,1,0] * phat[j * 2], i, dim=0, dim_size=numParticles)
        # apk[1::2] = apk[1::2] + scatter_sum(H[:,1,1] * phat[j * 2 + 1], i, dim=0, dim_size=numParticles)
                
        
        # print(v)
        rv = dotprod(apk, r0)
        if rv == 0:
            if verbose:
                print(f'\t[{iteration:3d}]\trv breakdown {rv} | {torch.linalg.norm(apk)} | {torch.linalg.norm(phat)} | {torch.linalg.norm(pk)}')
            return xk, -11, convergence#, r0
        alpha = rho / rv
        sk = rk - alpha*apk
        # sk[:] = r[:]

        if torch.linalg.norm(sk) < atol:
            if verbose:
                print(f'\t[{iteration:3d}]\tConverged after {iteration} iterations {torch.linalg.norm(sk)} | {atol}')
            xk += alpha*pk
            return xk, 0, convergence#, r0

        shat = psolve(sk)
        # shat = sk
        ask = matvec(shat)# t
        # ask = torch.zeros_like(x0)
        # ask[::2]  = ask[::2] + scatter_sum(H[:,0,0] * shat[j * 2], i, dim=0, dim_size=numParticles)
        # ask[::2]  = ask[::2] + scatter_sum(H[:,0,1] * shat[j * 2 + 1], i, dim=0, dim_size=numParticles)

        # ask[1::2] = ask[1::2] + scatter_sum(H[:,1,0] * shat[j * 2], i, dim=0, dim_size=numParticles)
        # ask[1::2] = ask[1::2] + scatter_sum(H[:,1,1] * shat[j * 2 + 1], i, dim=0, dim_size=numParticles)

        # print(f'{torch.linalg.norm(pk)} -> {torch.linalg.norm(apk)} | {torch.linalg.norm(sk)} -> {torch.linalg.norm(ask)} | {torch.linalg.norm(xk)} | {torch.linalg.norm(rk)}')

        # t = A_fn(shat)
        omega = dotprod(ask, sk) / dotprod(ask, ask)
        xk = xk + alpha * phat + omega * shat
        rho_prev = torch.dot(rk, r0)
        rk = sk - omega * ask

        beta = (torch.dot(rk, r0) / rho_prev) * (alpha / omega)
        pk = rk + beta * (pk - omega * apk)


        # print(f'\t=>\t{torch.linalg.norm(pk)} | {torch.linalg.norm(sk)} | {torch.linalg.norm(xk)} | {torch.linalg.norm(rk)}')

        # rho_prev = rho

        if iteration > 0:
            if torch.abs(omega) < omegatol:  # omega breakdown
                if verbose:
                    print(f'\t[{iteration:3d}]\tomega breakdown {omega} | {omegatol}')
                return xk, -11, convergence#, r0

            # beta = (rho / rho_prev) * (alpha / omega)
            # pk -= omega*apk
            # pk *= beta
            # pk += rk



        # print(omega, alpha)
        residual = matvec(xk)
        # residual = torch.zeros_like(b)
        # residual[::2]  += scatter_sum(H[:,0,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
        # residual[::2]  += scatter_sum(H[:,0,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        # residual[1::2] += scatter_sum(H[:,1,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
        # residual[1::2] += scatter_sum(H[:,1,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        convergence.append(torch.linalg.norm(residual - b))
        if verbose:
            print(f'\t[{iteration:3d}]\tResidual: {torch.linalg.norm(residual - b)} | rho: {torch.abs(rho)} | alpha: {torch.linalg.norm(alpha)} | omega: {torch.linalg.norm(omega)}')

        dx = xk.view(-1,2)
        dist = torch.linalg.norm(dx, dim = -1)
        if torch.any(dist > threshold):
            if verbose:
                print(f'\t[{iteration:3d}]\txk breakdown: {xk}, dist: {dist.max()} | {threshold}')

            return xk, -12, convergence#, r0
            break

    # else:  # for loop exhausted
        # Return incomplete progress
    if verbose:
        print(f'Reached maximum iterations {maxiter}, returning with {torch.linalg.norm(rk)}')
    return xk, maxiter, convergence#, r0, H, i, j