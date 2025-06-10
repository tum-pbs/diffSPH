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
from sphMath.modules.shifting.deltaPlusShifting import computeDeltaShifting
from sphMath.sparseSolver.bicg import bicgstab_shifting
from sphMath.sparseSolver.bicgOld import BiCGStab_wJacobi


def evalKernel(xij, hij, kernel):
    # rij = torch.linalg.norm(xij, dim = -1)
    # nij = xij / (rij.view(-1,1) + 1e-6)
    
    # W_K = wrappedKernel.kernel(rij / hij, hij, 2)
    # W_J = wrappedKernel.Jacobi(rij / hij, nij, hij, 2)
    # W_H = wrappedKernel.Hessian2D(rij / hij, nij, hij, 2)
    # return W_K, W_J, W_H

    return kernel.eval(xij, hij), kernel.jacobian(xij, hij), kernel.hessian(xij, hij)


def getShiftingMatrices(particleState, domain, wrappedKernel, sparseNeighborhood, config, freeSurface = None, freeSurfaceMask = None):
    with record_function("[Shifting] - Implicit Particle Shifting (IPS)"):
        numParticles = particleState.positions.shape[0]
        if config.get('shifting',{}).get('freeSurface', False):
            if config.get('shifting', {}).get('useExtendedMask', True):
                fs = freeSurfaceMask
            else:
                fs = freeSurface
        
        i, j = sparseNeighborhood.row, sparseNeighborhood.col
        rij, xij = computeDistanceTensor(sparseNeighborhood, False, 'gather')
        hij = particleState.supports[i]

        K, J, H = evalKernel(xij, hij, wrappedKernel)
        if config.get('shifting',{}).get('summationDensity', False):
            particleState.densities = computeDensity(particleState, particleState, domain, wrappedKernel, sparseNeighborhood, 'gather')
            omega =  particleState.masses / particleState.densities
        else:
            omega = particleState.masses / config.get('fluid', {}).get('rho0', 1)
        
        J = scatter_sum(J * omega[j,None], i, dim = 0, dim_size = numParticles)
        H = H * omega[j,None,None]

        h2 = particleState.supports.repeat(2,1).T.flatten()
        dx = config.get('particle', {}).get('dx', torch.pow(particleState.masses / config.get('fluid', {}).get('rho0', 1), 1/particleState.positions.shape[1]).mean().cpu().item())
        h2 = dx
        initializer = config.get('shifting', {}).get('initialization', 'zero')
        
        x0 = torch.rand(numParticles * 2).to(rij.device).type(rij.dtype) * h2 / 4 - h2 / 8
        if initializer == 'deltaPlus':
            x0 = -computeDeltaShifting(particleState, domain, wrappedKernel, sparseNeighborhood, config).flatten() * 0.5
        if initializer == 'deltaMinus':
            x0 = computeDeltaShifting(particleState, domain, wrappedKernel, sparseNeighborhood, config).flatten() * 0.5
        if initializer == 'zero':
            x0 = torch.zeros_like(x0)
        
        dtype = particleState.positions.dtype
        B = torch.zeros(numParticles * 2, dtype = dtype, device=rij.device)

        # iActual = i
        # jActual = j
        activeMask = torch.ones_like(i, dtype = torch.bool)
        if config.get('shifting',{}).get('freeSurface', False):

            J2 = torch.zeros(J.shape[0], 2, dtype = dtype, device=rij.device)
            J2[fs < 0.5, :] = J[fs < 0.5, :]

            B[::2] = J2[:,0]
            B[1::2] = J2[:,1]

            # B[::2] = J[:,0]
            # B[1::2] = J[:,1]

            x0 = x0.view(-1,2)
            x0[fs > 0.5,0] = 0
            x0[fs > 0.5,1] = 0
            x0 = x0.flatten()
            # H[fs[i] > 0.5,:,:] = 0
            H[fs[j] > 0.5,:,:] = 0
            activeMask = fs[i] < 0.5
            # iActual = i[fs[i] < 0.5]
            # jActual = j[fs[i] < 0.5]

            # print(f'Iter: {iters}, Residual: {residual}, fs: xk fs: {update[fs > 0.5]}')
            # if scheme == 'BiCG':
            #     xk = BiCG(H[fs[i] < 0.5], B[fs[i] < 0.5], x0[fs[i] < 0.5], iMasked, jMasked, maxIter = config['shifting']['maxSolveIter'])
            # elif scheme == 'BiCGStab':
            #     xk = BiCGStab(H[fs[i] < 0.5], B[fs[i] < 0.5], x0[fs[i] < 0.5], iMasked, jMasked, maxIter = config['shifting']['maxSolveIter'])
            # elif scheme == 'BiCGStab_wJacobi':
            #     xk = BiCGStab_wJacobi(H[fs[i] < 0.5], B[fs[i] < 0.5], x0[fs[i] < 0.5], iMasked, jMasked, maxIter = config['shifting']['maxSolveIter'])
            # else:
            #     xk = LinearCG(H[fs[i] < 0.5], B[fs[i] < 0.5], x0[fs[i] < 0.5], iMasked, jMasked, maxIter = config['shifting']['maxSolveIter'])
        else:
            B[::2] = J[:,0]
            B[1::2] = J[:,1]

        if torch.any(particleState.kinds != 0):
            B.view(-1,2)[particleState.kinds != 0, 0] = 0
            B.view(-1,2)[particleState.kinds != 0, 1] = 0
            H[particleState.kinds != 0, :, :] = 0
            x0.view(-1,2)[particleState.kinds != 0, 0] = 0
            x0.view(-1,2)[particleState.kinds != 0, 1] = 0
            if config['shifting']['freeSurface']:
                activeMask = torch.logical_and(particleState.kinds == 0, fs < 0.5)
            else:
                activeMask = particleState.kinds == 0

    return H, B, x0, i, j, activeMask


def multiplySparseShifting(H, x, i, j):
    numParticles = x.shape[0] // 2

    apk = torch.zeros_like(x)
    apk[::2]  += scatter_sum(H[:,0,0] * x[j * 2], i, dim=0, dim_size=numParticles)
    apk[::2]  += scatter_sum(H[:,0,1] * x[j * 2 + 1], i, dim=0, dim_size=numParticles)

    apk[1::2] += scatter_sum(H[:,1,0] * x[j * 2], i, dim=0, dim_size=numParticles)
    apk[1::2] += scatter_sum(H[:,1,1] * x[j * 2 + 1], i, dim=0, dim_size=numParticles)

    return apk

def computeShifting(particleState, domain, wrappedKernel, sparseNeighborhood, config, freeSurface = None, freeSurfaceMask = None):
    with record_function("[Shifting] - Implicit Particle Shifting (IPS)"):
        numParticles = particleState.positions.shape[0]


        i, j = sparseNeighborhood.row, sparseNeighborhood.col
        rij, xij = computeDistanceTensor(sparseNeighborhood, False, 'gather')
        hij = particleState.supports[i]
        
        dim = particleState.positions.shape[1]
        
        K, J, H = evalKernel(xij, hij, wrappedKernel)
        
        H, B, x0, i, j, activeMask = getShiftingMatrices(particleState, domain, wrappedKernel, sparseNeighborhood, config, freeSurface, freeSurfaceMask)
        
        # H = H * 32
        # h2 = particleState['supports'].repeat(2,1).T.flatten()
        # h2 = config['particle']['dx']
        # x0 = torch.rand(numParticles * 2).to(rij.device).type(rij.dtype) * h2 / 4 - h2 / 8
        # if config['shifting']['initialization'] == 'deltaPlus':
        #     x0 = -deltaPlusShifting(particleState, config).flatten() * 0.5
        # if config['shifting']['initialization'] == 'deltaMinus':
        #     x0 = deltaPlusShifting(particleState, config).flatten() * 0.5
        # if config['shifting']['initialization'] == 'zero':
        #     x0 = torch.zeros_like(x0)
        

        numParticles = B.shape[0] // 2
        ii = torch.unique(i)
        # Calculate the Jacobi preconditioner
        diag = torch.zeros_like(B).view(-1, 2)
        diag[ii,0] = H[i == j, 0, 0]
        diag[ii,1] = H[i == j, 1, 1]
        diag = diag.flatten()

        M_inv = 1 / diag
        
        # print(diag)
        # M_inv = diag
        M_inv[diag.abs() < 1e-8] = 0
        # M_inv[:] = 1

        M_i = torch.arange(0, numParticles*2, device=H.device)
        M_j = torch.arange(0, numParticles*2, device=H.device)
        M_coo = (M_inv, (M_i, M_j), numParticles*2)
        
        # H = H.cpu().to(torch.float64)
        # x0 = x0.cpu().to(torch.float64)
        # B = B.cpu().to(torch.float64)
        # i = i.cpu()
        # j = j.cpu()
        # M_inv = M_inv.cpu().to(torch.float64)
        # M_i = M_i.cpu()
        # M_j = M_j.cpu()
        M_coo = (M_inv, (M_i, M_j), numParticles*2)

        shiftTolerance = config.get('shifting', {}).get('tol', 1e-4)
        relativeTolerance = config.get('shifting', {}).get('rtol', 1e-4)
        maxIterations = config.get('shifting', {}).get('maxSolveIter', 64)
        preconditioner = config.get('shifting', {}).get('preconditioner', 'Jacobi')
        verbose = config.get('shifting', {}).get('verbose', False)
        dx = config.get('particle', {}).get('dx', torch.pow(particleState.masses / config.get('fluid', {}).get('rho0', 1), 1/particleState.positions.shape[1]).mean().cpu().item())
        threshold = config.get('shifting', {}).get('solverThreshold', dx / 2)
        solver = config.get('shifting', {}).get('solver', 'BiCGStab_wJacobi')
        
        # if scheme == 'BiCG':
        #     xk, convergence, iters, residual = BiCG(H[activeMask], B, x0, i[activeMask], j[activeMask], maxIter = config['shifting']['maxSolveIter'])
        # elif scheme == 'BiCGStab':
        #     xk, convergence, iters, residual = BiCGStab(H[activeMask], B, x0, i[activeMask], j[activeMask], maxIter = config['shifting']['maxSolveIter'])
        # elif scheme == 'BiCGStab_wJacobi':
        #     xk, convergence, iters, residual = BiCGStab_wJacobi(H[activeMask], B, x0, i[activeMask], j[activeMask], maxIter = config['shifting']['maxSolveIter'])
        # else:
        #     xk, convergence, iters, residual = LinearCG(H[activeMask], B, x0, i[activeMask], j[activeMask], maxIter = config['shifting']['maxSolveIter'])

        if solver == 'BiCGold':
            xk, convergence, iters, residual = BiCGStab_wJacobi(H[activeMask], B, x0, i[activeMask], j[activeMask], maxIter = config['shifting']['maxSolveIter'])
        else:
            if preconditioner == 'Jacobi':
                xk, iters, convergence = bicgstab_shifting((H[activeMask], (i[activeMask],j[activeMask]), numParticles * 2), B, x0, tol = shiftTolerance, rtol = relativeTolerance, maxiter = maxIterations, M = M_coo, verbose = verbose, threshold = threshold)
            else:
                xk, iters, convergence = bicgstab_shifting((H[activeMask], (i[activeMask],j[activeMask]), numParticles * 2), B, x0, tol = shiftTolerance, rtol = relativeTolerance, maxiter = maxIterations, M = None, verbose = verbose, threshold = threshold)

        # xk, iters, convergence, *_ = bicgstab_shifting((H, (i,j), numParticles), B, x0, tol = 1e-3, rtol = 1e-3, maxiter = 64, M = None)
        residual = torch.linalg.norm(multiplySparseShifting(H[activeMask], xk, i[activeMask], j[activeMask]) - B)
        


        # xk, iters, convergence = bicgstabfn(lambda x: multiplySparseShifting(H, x, i, j), B.shape[0], B, x0, tol = 1e-3, rtol = 1e-3, maxiter = 64, M = M_coo)
        # residual = torch.linalg.norm(multiplySparseShifting(H, xk, i, j) - B)
        # xk = xk.to(torch.float32).to(B.device)

        # print(convergence)

        # xk = xk.to(torch.float32).to(B.device)
        update =  torch.vstack((-xk[::2],-xk[1::2])).T# , J, H, B
        return update, K, J, H, B, convergence, iters, residual
