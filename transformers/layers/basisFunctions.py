import warnings
import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional

# Spacing for basis functions
@torch.jit.script
def getSpacing(n : int, periodic : bool = False):
    if n == 1:
        return 2.
    else:
        return 2. / n if periodic else 2./(n-1)
    
# Function that returns the distance between a given set of points and a set of basis function centers
# Caches the basis function center positions for computational efficiency
centroidCache = {False:{'cuda':{},'cpu':{}},True:{'cuda':{},'cpu':{}}}
def getDistancesRelCached(n, x, periodic = False):
    if n in centroidCache[periodic][x.device.type]:
        centroids = centroidCache[periodic][x.device.type][n]
        if periodic:
            spacing = getSpacing(n, True)
            offset = -1 + spacing / 2.
            ra = torch.unsqueeze(x,axis=0) - centroids
            rb = torch.unsqueeze(x,axis=0) - centroids - 2.
            rc = torch.unsqueeze(x,axis=0) - centroids + 2.
            return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        else:
            spacing = getSpacing(n, False)
            r = torch.unsqueeze(x,axis=0) - centroids
            return r  / spacing


    if periodic:
        spacing = getSpacing(n, True)
        centroids = torch.unsqueeze(torch.linspace(-1.,1.,n+1, device = x.device)[:n],axis=1)
        centroidCache[periodic][x.device.type][n] = centroids

        ra = torch.unsqueeze(x,axis=0) - centroids
        rb = torch.unsqueeze(x,axis=0) - centroids - 2.
        rc = torch.unsqueeze(x,axis=0) - centroids + 2.
        return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        
    spacing = getSpacing(n, False)
    
    centroids = torch.linspace(-1.,1.,n, device = x.device) if n > 1 else torch.tensor([0.], device = x.device)
    centroids = torch.unsqueeze(centroids, axis = 1)
    centroidCache[periodic][x.device.type][n] = centroids
    r = torch.unsqueeze(x,axis=0) - centroids
    return r  / spacing
@torch.jit.script
def getDistancesRel(n : int, x : torch.Tensor, periodic : bool = False):
    if periodic:
        spacing = getSpacing(n, True)
        centroids = torch.unsqueeze(torch.linspace(-1.,1.,n+1, device = x.device)[:n],dim=1)

        ra = torch.unsqueeze(x,dim=0) - centroids
        rb = torch.unsqueeze(x,dim=0) - centroids - 2.
        rc = torch.unsqueeze(x,dim=0) - centroids + 2.
        return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        
    spacing = getSpacing(n, False)
    
    centroids = torch.linspace(-1.,1.,n, device = x.device) if n > 1 else torch.tensor([0.], device = x.device)
    centroids = torch.unsqueeze(centroids, dim = 1)
    r = torch.unsqueeze(x,dim=0) - centroids
    return r  / spacing

@torch.jit.script
def getDistancesRel_offset(n : int, x : torch.Tensor, periodic : bool = False):
    if periodic:
        spacing = getSpacing(n, True)
        centroids = torch.unsqueeze(torch.linspace(-1.,1.,n+1, device = x.device)[:n],dim=1)

        ra = torch.unsqueeze(x,dim=0) - centroids
        rb = torch.unsqueeze(x,dim=0) - centroids - 2.
        rc = torch.unsqueeze(x,dim=0) - centroids + 2.
        return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        
    spacing = getSpacing(n, False)
    
    # centroids = torch.linspace(-1. + spacing / 2,1. - spacing/2,n, device = x.device) if n > 1 else torch.tensor([0.], device = x.device)
    centroids = torch.linspace(-1.,1.,n+1, device = x.device)[:-1] if n > 1 else torch.tensor([0.], device = x.device)
    centroids = torch.unsqueeze(centroids, dim = 1) + spacing / 2
    r = torch.unsqueeze(x,dim=0) - centroids
    return r#  / (centroids[1] - centroids[0])



@torch.jit.script
def cpow(x : torch.Tensor, p : int):
    return torch.maximum(x, torch.zeros_like(x)) ** p

import numpy as np

@torch.jit.script
def evalRBFSeries(n : int, x : torch.Tensor, which : str = 'linear', epsilon : float = 1., periodic : bool = False, adjustSpacing : bool = False, normalized : bool = False):   
    k = int(epsilon)
    if adjustSpacing:
        if which == 'gaussian' or which == 'inverse_quadric' or which == 'inverse_multiquadric' or 'spline' in which  or 'wendland' in which:
            x = x * (1 - 2/n)
        if which == 'bump':
            x = x * (1 - 4/n)
    
    rRel = getDistancesRel(n, x, periodic)
    r = torch.abs(rRel)
    if n == 1:
        return torch.ones_like(r)
    res = torch.zeros_like(r)
    if not adjustSpacing and not normalized:
        if which == 'linear':               res = torch.clamp(1. - r / epsilon,0,1)
        elif which == 'gaussian':             res = torch.exp(-(epsilon * r)**2)
        elif which == 'multiquadric':         res = torch.sqrt(1. + (epsilon * r) **2)
        elif which == 'inverse_quadric':      res = 1. / ( 1 + (epsilon * r) **2)
        elif which == 'inverse_multiquadric': res = 1. / torch.sqrt(1. + (epsilon * r) **2)
        elif which == 'polyharmonic':         res = torch.pow(r, k) if k % 2 == 1 else torch.pow(r,k-1) * torch.log(torch.pow(r,r))
        elif which == 'bump':                 res = torch.where(r < 1./epsilon, torch.exp(-1./(1- (epsilon * r)**2)), torch.zeros_like(r))
        elif which == 'cubic_spline':         res = cpow(1-r/(epsilon * 1.),3) - 4. * cpow(1/2-r/(epsilon * 1.),3)
        elif which == 'quartic_spline':       res = cpow(1-r/(epsilon * 1.),4) - 5 * cpow(3/5-r/(epsilon * 1.),4) + 10 * cpow(1/5-r/(epsilon * 1.),4)
        elif which == 'quintic_spline':       res = cpow(1-r/(epsilon * 1.),5) - 6 * cpow(2/3-r/(epsilon * 1.),5) + 15 * cpow(1/3-r/(epsilon * 1.),5)
        elif which == 'wendland2':            res = cpow(1 - r/(epsilon * 1.), 4) * (1 + 4 * r/(epsilon * 1.))
        elif which == 'wendland4':            res = cpow(1 - r/(epsilon * 1.), 6) * (1 + 6 * r/(epsilon * 1.) + 35/3 * (r/(epsilon * 1.))**2)
        elif which == 'wendland6':            res = cpow(1 - r/(epsilon * 1.), 8) * (1 + 8 * r/(epsilon * 1.) + 25 * (r/(epsilon * 1.)) **2 + 32 * (r * (epsilon * 1.))**3)
        elif which == 'poly6':                res = cpow(1 - (r/epsilon)**2, 3)
        elif which == 'spiky':                res = cpow(1 - r/epsilon, 3)
        elif which == 'square':               
            minDist, minDistIdx = torch.min(rRel**2, 0)
            res = torch.zeros(n, rRel.shape[1], device = r.device)
            res[minDistIdx, torch.arange(rRel.shape[1], device = r.device)] = 1
        elif which == 'square_offset':
            rRel = getDistancesRel_offset(n, x, False)
            minDist, minDistIdx = torch.min(rRel**2, 0)
            res = torch.zeros(n, rRel.shape[1], device = r.device)
            res[minDistIdx, torch.arange(rRel.shape[1], device = r.device)] = 1
        else:
            raise ValueError('Unknown basis function')
    elif adjustSpacing and not normalized:
        if which == 'linear':               res = torch.clamp(1. - r / epsilon,0,1)
        elif which == 'gaussian':             res = torch.exp(-(epsilon * r)**2)
        elif which == 'multiquadric':         res = torch.sqrt(1. + (epsilon * r) **2)
        elif which == 'inverse_quadric':      res = 1. / ( 1 + (epsilon * r) **2)
        elif which == 'inverse_multiquadric': res = 1. / torch.sqrt(1. + (epsilon * r) **2)
        elif which == 'polyharmonic':         res = torch.pow(r, k) if k % 2 == 1 else torch.pow(r,k-1) * torch.log(torch.pow(r,r))
        elif which == 'bump':                 res = torch.where(r < 1./epsilon, torch.exp(-1./(1- (epsilon * r)**2)), torch.zeros_like(r))
        elif which == 'cubic_spline':         res = cpow(1-r/(epsilon * 1.732051),3) - 4. * cpow(1/2-r/(epsilon * 1.732051),3)
        elif which == 'quartic_spline':       res = cpow(1-r/(epsilon * 1.936492),4) - 5 * cpow(3/5-r/(epsilon * 1.936492),4) + 10 * cpow(1/5-r/(epsilon * 1.732051),4)
        elif which == 'quintic_spline':       res = cpow(1-r/(epsilon * 2.121321),5) - 6 * cpow(2/3-r/(epsilon * 2.121321),5) + 15 * cpow(1/3-r/(epsilon * 2.121321),5)
        elif which == 'wendland2':            res = cpow(1 - r/(epsilon * 1.620185), 4) * (1 + 4 * r/(epsilon * 1.620185))
        elif which == 'wendland4':            res = cpow(1 - r/(epsilon * 1.936492), 6) * (1 + 6 * r/(epsilon * 1.936492) + 35/3 * (r/(epsilon * 1.936492))**2)
        elif which == 'wendland6':            res = cpow(1 - r/(epsilon * 2.207940), 8) * (1 + 8 * r/(epsilon * 2.207940) + 25 * (r/(epsilon * 2.207940)) **2 + 32 * (r * (epsilon * 2.207940))**3)
        elif which == 'poly6':                res = cpow(1 - (r/epsilon)**2, 3)
        elif which == 'spiky':                res = cpow(1 - r/epsilon, 3)
        elif which == 'square':               
            minDist, minDistIdx = torch.min(rRel**2, 0)
            res = torch.zeros(n, rRel.shape[1], device = r.device)
            res[minDistIdx, torch.arange(rRel.shape[1], device = r.device)] = 1
        elif which == 'square_offset':
            rRel = getDistancesRel_offset(n, x, False)
            minDist, minDistIdx = torch.min(rRel**2, 0)
            res = torch.zeros(n, rRel.shape[1], device = r.device)
            res[minDistIdx, torch.arange(rRel.shape[1], device = r.device)] = 1
        else:
            raise ValueError('Unknown basis function')
    elif not adjustSpacing and normalized:
        if which == 'linear':               res = torch.clamp(1. - r / 1,0,1)
        elif which == 'gaussian':             res = torch.exp(-(0.9919394235466537 * r)**2)
        elif which == 'multiquadric':         res = torch.sqrt(1. + (1 * r) **2)
        elif which == 'inverse_quadric':      res = 1. / ( 1 + (1.1480214948705423 * r) **2)
        elif which == 'inverse_multiquadric': res = 1. / torch.sqrt(1. + (1.6382510991695163 * r) **2)
        elif which == 'polyharmonic':         res = torch.pow(r, k) if k % 2 == 1 else torch.pow(r,k-1) * torch.log(torch.pow(r,r))
        elif which == 'bump':                 res = torch.where(r < 1./0.38739618954567656, torch.exp(-1./(1- (0.38739618954567656 * r)**2)), torch.zeros_like(r))
        elif which == 'cubic_spline':         res = cpow(1-r/(epsilon * 2.009770395701026),3) - 4. * cpow(1/2-r/(epsilon * 2.009770395701026),3)
        elif which == 'quartic_spline':       res = cpow(1-r/(epsilon * 2.4318514899853443),4) - 5 * cpow(3/5-r/(epsilon * 2.4318514899853443),4) + 10 * cpow(1/5-r/(epsilon * 2.4318514899853443),4)
        elif which == 'quintic_spline':       res = cpow(1-r/(epsilon * 2.8903273082559844),5) - 6 * cpow(2/3-r/(epsilon * 2.8903273082559844),5) + 15 * cpow(1/3-r/(epsilon * 2.8903273082559844),5)
        elif which == 'wendland2':            res = cpow(1 - r/(epsilon * 3.6238397655105032), 4) * (1 + 4 * r/(epsilon * 3.6238397655105032))
        elif which == 'wendland4':            res = cpow(1 - r/(epsilon * 3.7338788470933073), 6) * (1 + 6 * r/(epsilon * 3.7338788470933073) + 35/3 * (r/(epsilon * 3.7338788470933073))**2)
        elif which == 'wendland6':            res = cpow(1 - r/(epsilon * 1.3856863702979971), 8) * (1 + 8 * r/(epsilon * 1.3856863702979971) + 25 * (r/(epsilon * 1.3856863702979971)) **2 + 32 * (r * (epsilon * 1.3856863702979971))**3)
        elif which == 'poly6':                res = cpow(1 - (r/ 2.6936980947728384)**2, 3)
        elif which == 'spiky':                res = cpow(1 - r/3, 3)
        elif which == 'square':               
            minDist, minDistIdx = torch.min(rRel**2, 0)
            res = torch.zeros(n, rRel.shape[1], device = r.device)
            res[minDistIdx, torch.arange(rRel.shape[1], device = r.device)] = 1
        elif which == 'square_offset':
            rRel = getDistancesRel_offset(n, x, False)
            minDist, minDistIdx = torch.min(rRel**2, 0)
            res = torch.zeros(n, rRel.shape[1], device = r.device)
            res[minDistIdx, torch.arange(rRel.shape[1], device = r.device)] = 1
            # print('square_offset')
        else:
            raise ValueError('Unknown basis function')
    else:
        raise ValueError('Normalized and adjusted spacing is not supported')
    
    if normalized:
        res = res / torch.sum(res, dim = 0)
    return res
# Evaluate a chebyshev series of the first kind
@torch.jit.script
def evalChebSeries(n : int,x : torch.Tensor):
    cs = []
    for i in range(n):
        if i == 0:
            cs.append(torch.ones_like(x))
        elif i == 1:
            cs.append(x)
        else:
            cs.append(2. * x * cs[i-1] - cs[i-2])
    return torch.stack(cs)

# Evaluate a chebyshev series of the second kind
@torch.jit.script
def evalChebSeries2(n : int,x : torch.Tensor):
    cs = []
    for i in range(n):
        if i == 0:
            cs.append(torch.ones_like(x))
        elif i == 1:
            cs.append(2 * x)
        else:
            cs.append(2. * x * cs[i-1] - cs[i-2])
    return torch.stack(cs)

# precomputed value for computational efficiency
sqrt_pi_1 = 1. / np.sqrt(np.pi)
sqrt_2pi_1 = 1. / np.sqrt(2. * np.pi)
# Evaluate a fourier series
@torch.jit.script
def fourier(n : int, x : torch.Tensor):
    sqrt_pi_1 = 0.5641895835477563
    sqrt_2pi_1 = 0.3989422804014327

    if n == 0:
        return torch.ones_like(x) * sqrt_2pi_1
    elif n % 2 == 0:
        return torch.cos((n // 2 + 1) * x) * sqrt_pi_1
    return torch.sin((n // 2 + 1) * x) * sqrt_pi_1
@torch.jit.script
def evalFourierSeries(n : int, x : torch.Tensor):
    fs = []
    for i in range(n):
        fs.append(fourier(i, x))
    return torch.stack(fs)

@torch.jit.script
def fourier2(n : int, x : torch.Tensor):
    sqrt_pi_1 = 0.5641895835477563
    sqrt_2pi_1 = 0.3989422804014327
    if n == 0:
        return torch.ones_like(x) * sqrt_2pi_1
    elif n  % 2 == 0:
        return torch.cos(((n - 1) // 2 + 1) * x) * sqrt_pi_1
    return torch.sin(((n-1) // 2 + 1) * x) * sqrt_pi_1
@torch.jit.script
def evalFourierSeries2(n : int, x : torch.Tensor):
    fs = []
    for i in range(n):
        fs.append(fourier2(i, x))
    return torch.stack(fs)

@torch.jit.script
def wrongFourierBasis(n : int, x : torch.Tensor):
    sqrt_pi_1 = 0.5641895835477563
    if n % 2 == 0:
        return (torch.cos((n // 2 + 1) * x) * sqrt_pi_1)
    return (torch.sin((n // 2 + 1) * x) * sqrt_pi_1)
@torch.jit.script
def correctFourierBasis(n : int, x : torch.Tensor):
    sqrt_pi_1 = 0.5641895835477563
    if n % 2 == 0:
        return (torch.cos((n // 2 + 1) * x) * sqrt_pi_1)
    return (torch.sin((n // 2 + 1) * x) * sqrt_pi_1)

@torch.jit.script
def buildFourierSeries(n : int, x : torch.Tensor, kind : str = 'fourier'):
    sqrt_pi_1 = 0.5641895835477563
    sqrt_2pi_1 = 0.3989422804014327
    ndc = True if 'ndc' in kind else False
    fs = []
    for i in range(n):
        if not ndc and i == 0:
            if 'lin' in  kind:
                fs.append(x / 2. * np.pi)
            elif 'sgn' in kind:
                fs.append(torch.sign(x) / 2. * np.pi)
            else:
                fs.append(torch.ones_like(x) * sqrt_2pi_1)
            continue
        if 'odd' in kind:
            fs.append(torch.sin(((i - (0 if ndc else 1)) + 1) * x) * sqrt_pi_1)
        elif 'even' in kind:
            fs.append(torch.cos(((i - (0 if ndc else 1)) + 1) * x) * sqrt_pi_1)
        elif 'ffourier' in kind:
            fs.append(correctFourierBasis(i - (0 if ndc else 1),x))
        else:
            fs.append(wrongFourierBasis(i + (1 if ndc else 0),x))
    return torch.stack(fs)


# Parent function that delegates the call to the corresponding evaluation functions
@torch.jit.script
def evalBasisFunction(n : int, x : torch.Tensor, which : str = 'chebyshev', periodic : bool = False):   
    s = which.split()    
    if s[0] == 'chebyshev':
        return evalChebSeries(n, x)
    if s[0] == 'chebyshev2':
        return evalChebSeries2(n, x)
    if 'fourier' in which:
        return buildFourierSeries(n, x * np.pi, kind = which)
    if s[0] == 'linear':
        return evalRBFSeries(n, x, which = 'linear', epsilon = 1., periodic = periodic)        
    if s[0] == 'dmcf':
        return evalRBFSeries(n, x, which = 'linear', epsilon = 1., periodic = periodic)      #torch.sign(x) * evalRBFSeries(n, torch.abs(x) * 2 - 1, which = 'linear', epsilon = 1., periodic = periodic)     
    if s[0] == 'log':
        if n != 1:
            warnings.warn('Log basis function only supports n = 1, ignoring n')
        return (torch.sign(x) * torch.log1p(torch.abs(x))).unsqueeze(0)

    if s[0] == 'rbf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic)     
    if s[0] == 'abf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic, adjustSpacing = True)     
    if s[0] == 'ubf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic, normalized = True)
    return torch.ones([n,x.shape[0]], device= x.device, dtype = x.dtype) * np.nan


@torch.jit.script
def basisEncoderLayer(edgeLengths, basisTerms : int, basisFunction : str = 'ffourier', mode : str = 'cat'):
    bTerms = []
    for e in edgeLengths.T:
        bTerm = evalBasisFunction(basisTerms, e, basisFunction).mT
        bTerms.append(bTerm)

    if mode == 'cat':
        return torch.cat(bTerms, dim = 1)
    elif mode == 'sum':
        return torch.stack(bTerms, dim = 0).sum(dim = 0)
    elif mode == 'prod':
        return torch.stack(bTerms, dim = 0).prod(dim = 0)
    elif mode == 'outer':
        return torch.einsum('ij,ik->ijk', bTerms[0], bTerms[1]).reshape(-1, basisTerms * basisTerms)
    elif mode == 'i':
        return bTerms[0]
    elif mode == 'j':
        return bTerms[1]
    elif mode == 'k':
        return bTerms[2]
    else:
        raise ValueError(f'Unknown mode: {mode}')
    



# from ml import evalBasisFunction
@torch.jit.script
def basisEncoderLayer(edgeLengths, basisTerms : int, basisFunction : str = 'ffourier', mode : str = 'cat'):
    bTerms = []
    for e in edgeLengths.T:
        bTerm = evalBasisFunction(basisTerms, e, basisFunction).mT
        bTerms.append(bTerm)
    if mode == 'cat':
        return torch.cat(bTerms, dim = 1)
    elif mode == 'sum':
        return torch.stack(bTerms, dim = 0).sum(dim = 0)
    elif mode == 'prod':
        return torch.stack(bTerms, dim = 0).prod(dim = 0)
    elif mode == 'outer':
        return torch.einsum('ij,ik->ijk', bTerms[0], bTerms[1]).reshape(-1, basisTerms * basisTerms)
    elif mode == 'i':
        return bTerms[0]
    elif mode == 'j':
        return bTerms[1]
    elif mode == 'k':
        return bTerms[2]
    else:
        raise ValueError(f'Unknown mode: {mode}')
    