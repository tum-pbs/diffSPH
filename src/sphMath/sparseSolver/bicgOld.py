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

# @torch.jit.script
### THIS FUNCTIONS IS BROKEN
### IT DOES NOT APPLY PRECONDITIONING CORRECTLY
### IT DOES NOT COMPUTE BETA CORRECTLY
### IT CAN STILL WORK
### USE AT YOUR OWN RISK
def BiCGStab_wJacobi(H, B, x0, i, j, tol : float =1e-5, maxIter : int = 32):
    with record_function("[Shifting] - BiCGStab Solver w/ Jacobi Preconditioner"):
        # print(f'BiCGStab Solver (gpt)')
        xk = x0
        rk = torch.zeros_like(x0)
        numParticles = rk.shape[0] // 2
        ii = torch.unique(i)
        # Calculate the Jacobi preconditioner
        diag = torch.zeros_like(B).view(-1, 2)
        diag[ii,0] = H[i == j, 0, 0]
        diag[ii,1] = H[i == j, 1, 1]
        diag = diag.flatten()

        # diag = torch.vstack((H[i == j, 0, 0], H[i == j, 1, 1])).flatten()
        # diag[diag < 1e-8] = 1
        M_inv = 1 / diag
        M_inv[diag.abs() < 1e-8] = 0
        M_inv[:] = 1
        # M_inv[torch.isnan(M_inv)] = 1

        rk[::2]  += scatter_sum(H[:,0,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
        rk[::2]  += scatter_sum(H[:,0,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)

        rk[1::2] += scatter_sum(H[:,1,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
        rk[1::2] += scatter_sum(H[:,1,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)
        
        rk = B - rk
        r0 = rk.clone()

        # Apply the preconditioner
        # zk = torch.bmm(M_inv, rk.unsqueeze(-1)).squeeze(-1)
        # print(f'i: {i.shape} x j: {j.shape} -> {i[i==j]}, {i}, {j}')
        # print(f'rk: {rk}, M_inv: {M_inv}, @ {M_inv.shape} * {rk.shape}')
        zk = M_inv * rk
        pk = zk.clone()
        
        num_iter = 0
        convergence = []
        rk_norm = torch.linalg.norm(rk)
        # print(f'Initial Residual: {torch.linalg.norm(rk)}')

        while (torch.abs(torch.linalg.norm(rk) / rk_norm - 1) > 1e-3 or num_iter == 0) and num_iter < maxIter and torch.linalg.norm(rk) > tol:
            rk_norm = torch.linalg.norm(rk)
            apk = torch.zeros_like(x0)

            apk[::2]  += scatter_sum(H[:,0,0] * pk[j * 2], i, dim=0, dim_size=numParticles)
            apk[::2]  += scatter_sum(H[:,0,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles)

            apk[1::2] += scatter_sum(H[:,1,0] * pk[j * 2], i, dim=0, dim_size=numParticles)
            apk[1::2] += scatter_sum(H[:,1,1] * pk[j * 2 + 1], i, dim=0, dim_size=numParticles)
            rho = torch.dot(rk, r0)
            alpha = torch.dot(rk, r0) / (torch.dot(apk, r0) + 1e-8)
            sk = rk - alpha * apk
            ask = torch.zeros_like(x0)

            ask[::2]  += scatter_sum(H[:,0,0] * sk[j * 2], i, dim=0, dim_size=numParticles)
            ask[::2]  += scatter_sum(H[:,0,1] * sk[j * 2 + 1], i, dim=0, dim_size=numParticles)

            ask[1::2] += scatter_sum(H[:,1,0] * sk[j * 2], i, dim=0, dim_size=numParticles)
            ask[1::2] += scatter_sum(H[:,1,1] * sk[j * 2 + 1], i, dim=0, dim_size=numParticles)

            omega = torch.dot(ask, sk) / (torch.dot(ask, ask) + 1e-8)
            xk = xk + alpha * pk + omega * sk
            rho_prev = torch.dot(r0, r0) # this is BROKEN
            rk = sk - omega * ask

            # Apply the preconditioner
            zk = M_inv * rk
            beta = torch.dot(rk, r0) / (rho_prev + 1e-8) * (alpha / (omega + 1e-8))
            pk = zk + beta * (pk - omega * apk)
            if torch.abs(alpha) < 1e-8 or torch.abs(omega) < 1e-8 or torch.abs(beta) < 1e-8:
                break

            # print(f'\t[{num_iter:3d}]\tResidual: {torch.linalg.norm(rk)} | rho: {torch.abs(rho)} | alpha: {torch.linalg.norm(alpha)} | omega: {torch.linalg.norm(omega)}')
            # print('###############################################################################')
            # print(f'Iter: {num_iter}, Residual: {torch.linalg.norm(rk)}, Threshold {tol}')
            # print(f'alpha: {alpha}, omega: {omega}, beta: {beta}')
            # print(f'rk: {rk}, pk: {pk}, xk: {xk}')
            # print(f'apk: {apk}, ask: {ask}')
            # print(torch.dot(rk, r0))
            # print(torch.dot(r0, r0))
            # print((alpha / omega))

            residual = torch.zeros_like(x0)
            residual[::2]  += scatter_sum(H[:,0,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
            residual[::2]  += scatter_sum(H[:,0,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)

            residual[1::2] += scatter_sum(H[:,1,0] * xk[j * 2], i, dim=0, dim_size=numParticles)
            residual[1::2] += scatter_sum(H[:,1,1] * xk[j * 2 + 1], i, dim=0, dim_size=numParticles)

            residual = B - residual


            num_iter += 1
            convergence.append(torch.linalg.norm(residual))

        return xk, convergence, num_iter, torch.linalg.norm(residual)
