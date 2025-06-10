from sphMath.util import ParticleSet, DomainDescription
from sphMath.sampling import buildDomainDescription, sampleRegularParticles
from sphMath.neighborhood import buildNeighborhood, coo_to_csr, SparseCOO
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product
from sphMath.kernels import SPHKernel, getSPHKernelv2

import numpy as np
import torch
from typing import Optional

def sumKernelValues(kernel, nPerh, eta_max : Optional[float] = None):
    deta = 1.0/nPerh
    result = torch.tensor(0.0, dtype = torch.float64)
    etar = deta

    vals = []
    eta_max_ = kernel.kernelScale(1) if eta_max is None else eta_max
    while etar <= eta_max_:
        val = 2.0*torch.linalg.norm(kernel.jacobian(torch.tensor([[etar]], dtype = torch.float64), eta_max_), dim = -1)[0]
        result += val
        vals.append((etar, val.item()))
        etar += deta
    return result

def sumKernelValues2D(kernel, nPerh, eta_max : Optional[float] = None):
    deta = 1.0/nPerh
    result = torch.tensor(0.0, dtype = torch.float64)
    etar = deta

    vals = []
    eta_max_ = kernel.kernelScale(2) if eta_max is None else eta_max
    while etar <= eta_max_:
        val = 2.0*torch.pi*etar/deta*torch.linalg.norm(kernel.jacobian(torch.tensor([[etar,0]], dtype = torch.float64), eta_max_), dim = -1)[0]
        result += val
        vals.append((etar, val.item()))
        etar += deta
    return torch.sqrt(result)

def sumKernelValues3D(kernel, nPerh, eta_max : Optional[float] = None):
    deta = 1.0/nPerh
    result = torch.tensor(0.0, dtype = torch.float64)
    etar = deta

    vals = []
    eta_max_ = float(kernel.kernelScale(3)) if eta_max is None else eta_max
    while etar <= eta_max_:
        val = 4.0*torch.pi*(etar/deta)**2*torch.linalg.norm(kernel.jacobian(torch.tensor([[etar,0,0]], dtype = torch.float64), eta_max_), dim = -1)[0]
        result += val
        vals.append((etar, val.item()))
        etar += deta
    return torch.pow(result, 1.0/3.0)

def createPsiLUT(wrappedKernel_: SPHKernel, dims = [1,2,3], n_min = 1.0, n_max = 20.0, xi = 1.0, nLUT = 255):
    psi = [[] for _ in dims]
    psiH = [[] for _ in dims]
    N_H = [[] for _ in dims]
    n_h = np.linspace(n_min, n_max, nLUT)
    
    Xi = np.arange(-20, 21, 1).astype(dtype = np.float64)
    
    h = 2
    wrappedKernel = getSPHKernelv2(wrappedKernel_)
    
    for dim in dims:
        for n_h in np.linspace(n_min, n_max, nLUT):
            spacing = h/(n_h) #/ packingRatio #* ratio
            xxi = torch.tensor(Xi * spacing, dtype = torch.float64)
            dxx = xxi[1] - xxi[0]

            grid = torch.meshgrid([xxi for i in range(dim)], indexing = 'ij')
            points = torch.stack(grid, dim = -1).reshape(-1, dim)
            v = dxx**dim
            
            rij = torch.linalg.norm(points, dim = -1) / h
            mask = rij <= 1.0
            

            neighbors = points[mask]
            # print(neighbors)
            vH = 2.0 * h if dim == 1 else (np.pi * h**2 if dim == 2 else (4/3) * np.pi * h**3)
            vN = neighbors.shape[0] * v
            ratio = vH / vN

            # kSum = BSplineKernel(neighbors, 1.0, 3, dim_ = dim).sum()
            kSum = wrappedKernel.eval(neighbors, torch.ones_like(neighbors[:,0]) * h).sum()# * ratio
            WHSum = torch.linalg.norm(wrappedKernel.jacobian(neighbors, torch.ones_like(neighbors[:,0]) * h), dim = -1).sum()
            
            # print(neighbors)
            # print(cubicSpline.eval(neighbors, torch.ones_like(neighbors[:,0]) * h))
            # print(torch.linalg.norm(cubicSpline.jacobian(neighbors, torch.ones_like(neighbors[:,0]) * h), dim = -1))
            
            hReferenceFactor_W = 2**(-dim)
            hActualFactor_W = h**(-dim)
            hScaling_W = hReferenceFactor_W / hActualFactor_W
            
            hReferenceFactor_WH = 2**(-dim - 1)
            hActualFactor_WH = h**(-dim - 1)
            hScaling_WH = hReferenceFactor_WH / hActualFactor_WH
            
            
            # print(hScaling)
            
            rho = v  * kSum
            psiH_0 = (hScaling_WH * WHSum)**(1/dim) #* 2# * (2**(-dim - 1))

            psiH_0 = 0.0
            if dim == 1:
                scaling = wrappedKernel.kernelScale(1)
                # scaling = wrappedKernel.xi(1)
                psiH_0 = sumKernelValues(wrappedKernel, n_h , eta_max= 1)
            elif dim == 2:
                scaling = wrappedKernel.kernelScale(2)
                # scaling = wrappedKernel.xi(2)
                scaling = wrappedKernel.kernelScale(2) * (wrappedKernel.packingRatio() - 1)
                psiH_0 = sumKernelValues2D(wrappedKernel, n_h, eta_max= 1)
            elif dim == 3:
                psiH_0 = sumKernelValues3D(wrappedKernel, n_h, eta_max = 1)

            psi_0 = (hScaling_W * kSum)**(1/dim) #* 2# * (2**(-dim - 1))
            if len(dims) > 1:
                psi[dim-1].append(psi_0.item())
                psiH[dim-1].append(psiH_0.item())
                N_H[dim-1].append(vH / v)
            else:
                psi[0].append(psi_0.item())
                psiH[0].append(psiH_0.item())
                N_H[0].append(vH / v)
        
    
    return np.linspace(n_min, n_max, nLUT) * xi, psi, psiH, N_H


def linearInterpolateLUT(LUT, x, xvalues):
    ileft = torch.searchsorted(xvalues, x, right = True).clamp(min = 0, max = len(xvalues) - 2)
    iright = ileft + 1
    alpha = (x - xvalues[ileft]) / (xvalues[iright] - xvalues[ileft])
    alpha = alpha.clamp(min = 0.0, max = 1.0)
    
    return LUT[ileft] * (1 - alpha) + LUT[iright] * alpha

def get_n_h(Ln_h, Lpsi, LpsiH, LN_H, psi = None, psiH = None, n_H = None):
    if psi is not None:
        return linearInterpolateLUT(Ln_h, psi, Lpsi)
    elif psiH is not None:
        return linearInterpolateLUT(Ln_h, psiH, LpsiH)
    elif n_H is not None:
        return linearInterpolateLUT(Ln_h, n_H, LN_H)
    else:
        raise UserWarning("Nothing provided to interpolate")

def interpolateLUT(LUT, dim, which = 'n_h', n_h = None, psi = None, psiH = None, n_H = None):
    args = [n_h, psi, psiH, n_H]
    dtypes = [a.dtype if a is not None else None for a in args]
    devices = [a.device if a is not None else None for a in args]
    
    dtype = next((dtype for dtype in dtypes if dtype is not None), None)
    device = next((device for device in devices if device is not None), None)
    
    if len(LUT[1]) == 3:
        Ln_h = LUT[0].to(dtype = dtype, device = device)
        Lpsi = LUT[1][dim - 1].to(dtype = dtype, device = device)
        LpsiH = LUT[2][dim - 1].to(dtype = dtype, device = device)
        LN_H = LUT[3][dim - 1].to(dtype = dtype, device = device)
    else:
        Ln_h = LUT[0].to(dtype = dtype, device = device)
        Lpsi = LUT[1][0].to(dtype = dtype, device = device)
        LpsiH = LUT[2][0].to(dtype = dtype, device = device)
        LN_H = LUT[3][0].to(dtype = dtype, device = device)
    
    # print(Ln_h)
    
    n_h = n_h if n_h is not None else get_n_h(Ln_h, Lpsi, LpsiH, LN_H, psi, psiH, n_H)
    if which == 'n_h':
        return n_h
    elif which == 'psi':
        return linearInterpolateLUT(Lpsi, n_h, Ln_h)
    elif which == 'psiH':
        return linearInterpolateLUT(LpsiH, n_h, Ln_h)
    elif which == 'n_H':
        return linearInterpolateLUT(LN_H, n_h, Ln_h)
    
    
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Tuple, Union, Dict
from sphMath.neighborhood import buildNeighborhood, filterNeighborhood, filterNeighborhoodByKind, SupportScheme, NeighborhoodInformation, evaluateNeighborhood, SparseNeighborhood, PrecomputedNeighborhood
from sphMath.util import getSetConfig
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
    
def computePsi_0(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}):
    i = neighborhood[0].row
    j = neighborhood[0].col
    dim = particles.positions.shape[1]    
    kernelValues = neighborhood[1]
    
    h = particles.supports[i]
    hReferenceFactor_W = 1**(-dim)
    hActualFactor_W = h**(-dim)
    hScaling_W = hReferenceFactor_W / hActualFactor_W
    
    hReferenceFactor_WH = 1**(-dim - 1)
    hActualFactor_WH = h**(-dim - 1)
    hScaling_WH = hReferenceFactor_WH / hActualFactor_WH
    
    # print(particles.positions.shape, i.shape, j.shape)
    # print(get_i(particles.positions, i).shape, get_j(particles.positions, j).shape)
    # print(domain)
    
    xij = kernelValues.x_ij
    
    kTerm = hScaling_W * kernelValues.W_i
    kTerm_H = hScaling_WH * torch.linalg.norm(kernelValues.gradW_i, dim = -1)

    
    psi_0 = scatter_sum(kTerm, i, dim = 0, dim_size = particles.positions.shape[0])**(1/dim)
    psi_0_H = scatter_sum(kTerm_H, i, dim = 0, dim_size = particles.positions.shape[0])**(1/dim)
    
    return psi_0, psi_0_H


# assumes convention from CRKSPH with $\eta$ = 1
def n_h_to_nH(n_h, dim):
    spacing = 1 / n_h
    v = spacing**dim
    vH = 2.0 if dim == 1 else (np.pi if dim == 2 else (4/3) * np.pi)
    return vH / v

def nH_to_n_h(nH, dim):
    vH = 2.0 if dim == 1 else (np.pi if dim == 2 else (4/3) * np.pi)
    v = vH / nH
    return (1 / v)**(1/dim)

from typing import Optional


class computeOwen:
    def __init__(self, kernel: SPHKernel, dim: int, nLUT =511, nMin = 1.0, nMax = 5.0):
        self.kernel = kernel
        n_h, psi, psiH, N_H = createPsiLUT(kernel, dims = [dim], n_min = nMin, n_max = nMax, xi = 1.0, nLUT = nLUT)
        LUTorch = [torch.tensor(n_h), [torch.tensor(t) for t in psi], [torch.tensor(t) for t in psiH], [torch.tensor(t) for t in N_H]]
        self.LUT = LUTorch
        self.dim = dim
        
    def __call__(self, psiH_):
        return interpolateLUT(self.LUT, self.dim, which = 'n_h', psiH = psiH_)
    
    def fromPsiH(self, psiH_):
        return interpolateLUT(self.LUT, self.dim, which = 'n_h', psiH = psiH_)
    def fromPsi(self, psi_):
        return interpolateLUT(self.LUT, self.dim, which = 'n_h', psi = psi_)

def computeNewSupport(target, n_h, h):        
    n_h_Target = target
    s = n_h_Target / n_h
    a = torch.where(s >= 1.0, 0.4 * (1 + s**-3), 0.4 * (1 + s**2))
    h_i_new = (1 - a + a * s) * h
    # h_i_new = s * h
    return h_i_new



def evaluateOptimalSupportOwen(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel_: SPHKernel,
        neighborhood: NeighborhoodInformation = None,
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {}
        ):
    
    PsiLUT_fn = config['support']['LUT']
    if PsiLUT_fn is None:
        PsiLUT_fn = computeOwen(kernel_, dim = config['domain'].dim, nMin = 2.0, nMax = 6.0, nLUT = 1024)
        config['support']['LUT'] = PsiLUT_fn
    # particles, domain, kernel, targetNeighbors, PsiLUT_fn, nIter = 16, neighborhood = None, verbose = False,eps = 1e-3, neighborhoodAlgorithm = 'compact'):
    hs = [particles.supports]
    # print(particles)
    psis = []
    neighborhood_ = neighborhood
    verletScale = 1.4 if neighborhood is None else neighborhood.verletScale
    supportMode = 'superSymmetric' if neighborhood is None else neighborhood.mode
    if supportMode not in ['gather', 'superSymmetric']:
        supportMode = 'superSymmetric'
    targetNeighbors = config['targetNeighbors']
    
    dim = particles.positions.shape[1]
    nhTarget = nH_to_n_h(targetNeighbors, dim = dim)
    # verletScale = 1.0 if config is None else (config['neighborhood']['verletScale'] if 'verletScale' in config else 1.0)
    hMin = particles.supports.min()
    hMax = particles.supports.max()

    nIter = getSetConfig(config, 'support', 'iterations', 16)
    adaptiveHThreshold = getSetConfig(config, 'support', 'adaptiveHThreshold', 1e-3)
    verbose = False
    
    for i in range(nIter):
        if verbose:
            print('----------------------------------')
            print(f'Iteration {i}, target: {nhTarget}')

        neighborhood, neighbors = evaluateNeighborhood(particles, config['domain'], kernel_, verletScale = 1.0, mode = SupportScheme.SuperSymmetric, priorNeighborhood=neighborhood)
        
        psi_0, psi_0_H = computePsi_0(particles, kernel_, neighbors.get('noghost'), supportScheme, config)
        if verbose:
            print(f'Psi: {psi_0_H.min()} | {psi_0_H.max()} | {psi_0_H.mean()}')
            
        n_h_i = PsiLUT_fn.fromPsiH(psi_0_H)
        if verbose:
            print(f'n_h: {n_h_i.min()} | {n_h_i.max()} | {n_h_i.mean()}')
        
        h = computeNewSupport(nhTarget, n_h_i, particles.supports)
        psis.append(psi_0_H)
        h = h.clamp(min = hMin * 0.25, max = hMax * 4.0)
        hMin = h.min()
        hMax = h.max()
        
        h_ratio = h / particles.supports
        
        if verbose: 
            print(f'Support: {h.min()} | {h.max()} | {h.mean()}')
            print(f'Ratio: {(h_ratio).min()} | {(h_ratio).max()} | {(h_ratio).mean()}')
            
        
        if isinstance(particles, Tuple):
            particles = particles._replace(supports = h)
        else:
            particles.supports = h

        hs.append(h)

        if (h_ratio - 1).abs().max() < adaptiveHThreshold:
            if verbose: 
                print('Stopping Early')
            # print('Stopping Early')
            break
    if neighborhood is not None:
        return psi_0_H, hs[-1], psis, hs, neighborhood_
    return psi_0_H, hs[-1], psis, hs