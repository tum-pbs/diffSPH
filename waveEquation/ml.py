import torch
import torch
import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    


import torch
import torch.nn as nn
from collections import OrderedDict

def getActivationFunctions():
    return ['elu', 'relu', 'hardtanh', 'hardswish', 'selu', 'celu', 'leaky_relu', 'prelu', 'rrelu', 'glu', 'gelu', 'logsigmoid', 'hardshrink', 'tanhshrink', 'softsign', 'softplus', 'softmin', 'softmax', 'softshrink', 'log_softmax', 'tanh', 'sigmoid', 'hardsigmoid', 'silu', 'mish']
def getActivationLayer(function: str):
    if function == 'elu':
        return nn.ELU()
    elif function == 'relu':
        return nn.ReLU()
    elif function == 'hardtanh':
        return nn.Hardtanh()
    elif function == 'hardswish':
        return nn.Hardswish()
    elif function == 'selu':
        return nn.SELU()
    elif function == 'celu':
        return nn.CELU()
    elif function == 'leaky_relu':
        return nn.LeakyReLU()
    elif function == 'prelu':
        return nn.PReLU()
    elif function == 'rrelu':
        return nn.RReLU()
    elif function == 'glu':
        return nn.GLU()
    elif function == 'gelu':
        return nn.GELU()
    elif function == 'logsigmoid':
        return nn.LogSigmoid()
    elif function == 'hardshrink':
        return nn.Hardshrink()
    elif function == 'tanhshrink':
        return nn.Tanhshrink()
    elif function == 'softsign':
        return nn.Softsign()
    elif function == 'softplus':
        return nn.Softplus()
    elif function == 'softmin':
        return nn.Softmin()
    elif function == 'softmax':
        return nn.Softmax()
    elif function == 'softshrink':
        return nn.Softshrink()
    elif function == 'log_softmax':
        return nn.LogSoftmax()
    elif function == 'tanh':
        return nn.Tanh()
    elif function == 'sigmoid':
        return nn.Sigmoid()
    elif function == 'hardsigmoid':
        return nn.Hardsigmoid()
    elif function == 'silu':
        return nn.SiLU()
    elif function == 'mish':
        return nn.Mish()
    elif function == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'Unknown activation function: {function}')
    

def getActivationFunctions():
    return ['elu', 'relu', 'hardtanh', 'hardswish', 'selu', 'celu', 'leaky_relu', 'prelu', 'rrelu', 'glu', 'gelu', 'logsigmoid', 'hardshrink', 'tanhshrink', 'softsign', 'softplus', 'softmin', 'softmax', 'softshrink', 'gumbel_softmax', 'log_softmax', 'tanh', 'sigmoid', 'hardsigmoid', 'silu', 'mish']
def getActivationFunction(function : str):
    return getattr(nn.functional, function)

class TransposeLayer(nn.Module):
    def __init__(self, dim1=0, dim2=1):
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, input):
        return torch.transpose(input, self.dim1, self.dim2)

import numpy as np
def buildMLPwActivation(layers, inputFeatures = 1, gain = 1/np.sqrt(34), activation = 'gelu', norm = False, groups = 1, preNorm = False, postNorm = False, noLinear = False, bias = True):
    # print(f'layers: {layers}, inputFeatures: {inputFeatures}, gain: {gain}, activation: {activation}, norm: {norm}, channels: {channels}, preNorm: {preNorm}, postNorm: {postNorm}, noLinear: {noLinear}')
    activationFn = getActivationLayer(activation)
    transposeCounter = 0
    normCounter = 0
    linear = 0
    modules = []
    if preNorm:
        modules.append((f'transposeLayer{transposeCounter}', TransposeLayer(1,2)))
        transposeCounter += 1
        # print(f'groups: {groups[0] if isinstance(groups, list) else groups}, inputFeatures: {inputFeatures}')
        # print(f'PreNorm: {groups} | {inputFeatures}')
        if isinstance(groups,list):
            numGroups = groups[0]
        if numGroups == -1:
            numGroups = inputFeatures
        modules.append((f'norm{normCounter}', nn.GroupNorm(numGroups, inputFeatures)))
        normCounter += 1
        modules.append((f'transposeLayer{transposeCounter}', TransposeLayer(1,2)))
        transposeCounter += 1

    if not noLinear:
        if len(layers) > 1:
            for i in range(len(layers) - 1):
                modules.append((f'linear{linear}', nn.Linear(inputFeatures if i == 0 else layers[i-1],layers[i])))
                linear += 1

    #             torch.nn.init.uniform_(modules[-1].weight,-0.5, 0.5)
                torch.nn.init.xavier_normal_(modules[-1][1].weight,1)
        #         torch.nn.init.zeros_(modules[-1].weight)
                torch.nn.init.zeros_(modules[-1][1].bias)
                # modules.append(nn.BatchNorm1d(layers[i]))
                if norm:
                    modules.append((f'transposeLayer{transposeCounter}',TransposeLayer(1,2)))
                    transposeCounter += 1
                    # print(f'groups: {groups}, layers[i]: {layers[i]}')

                    numGroups = groups[(i + 1) if preNorm else i] if isinstance(groups,list) else groups
                    if numGroups == -1:
                        numGroups = layers[i]
                    modules.append((f'norm{normCounter}', nn.GroupNorm(numGroups, layers[i])))
                    normCounter += 1
                    modules.append((f'transposeLayer{transposeCounter}',TransposeLayer(1,2)))
                    transposeCounter += 1
                modules.append((f'activation{linear-1}', activationFn))
            modules.append((f'linear{linear}', nn.Linear(layers[-2],layers[-1], bias = bias)))
        else:
            modules.append((f'linear{linear}', nn.Linear(inputFeatures,layers[-1], bias = bias))  )
        torch.nn.init.xavier_normal_(modules[-1][1].weight,gain)
        if bias:
            torch.nn.init.zeros_(modules[-1][1].bias)     
    if postNorm:
        modules.append((f'transposeLayer{transposeCounter}', TransposeLayer(1,2)))
        transposeCounter += 1
        # print(f'groups: {channels}, layers[-1]: {layers[-1]}')
        # print(f'groups: {groups[-1] if isinstance(groups,list) else groups}, layers[-1]: {layers[-1]}')
        numGroups = groups[-1] if isinstance(groups,list) else groups
        if numGroups == -1:
            numGroups = layers[-1]
        modules.append((f'norm{normCounter}', nn.GroupNorm(numGroups, layers[-1])))
        normCounter += 1
        modules.append((f'transposeLayer{transposeCounter}', TransposeLayer(1,2)))
        transposeCounter += 1
    moduleDict = OrderedDict()
    for i, module in enumerate(modules):
        moduleDict[module[0]] = module[1]
    return nn.Sequential(moduleDict)

def buildMLPwDict(properties : dict):
    layout = properties['layout'] if 'layout' in properties else []
    output = properties['output']
    inputFeatures = properties['inputFeatures']
    groups = properties['channels'] if 'channels' in properties else 1


    gain = properties['gain'] if 'gain' in properties else 1/np.sqrt(34)
    activation = properties['activation'] if 'activation' in properties else 'celu'
    norm = properties['norm'] if 'norm' in properties else True
    preNorm = properties['preNorm'] if 'preNorm' in properties else False
    postNorm = properties['postNorm'] if 'postNorm' in properties else False
    noLinear = properties['noLinear'] if 'noLinear' in properties else False
    
    numberOfNorms = 0
    if preNorm:
        numberOfNorms += 1
    if postNorm:
        numberOfNorms += 1
    if norm and not noLinear:
        numberOfNorms += len(layout)
    if numberOfNorms >0 and (isinstance(groups,list) and numberOfNorms != len(groups)):
        raise ValueError(f'Number of groups {len(groups)} does not match number of norms {numberOfNorms}')

    mlp = buildMLPwActivation(layout + [output], inputFeatures, gain = gain, activation = activation, norm = norm, groups = groups, preNorm = preNorm, postNorm = postNorm, noLinear = noLinear, bias = properties['bias'] if 'bias' in properties else True)
    return mlp

# from ml import *
from sphMath.sphOperations.shared import scatter_sum

class ScatterSumLayer(nn.Module):
    def __init__(self):
        super(ScatterSumLayer, self).__init__()
    def forward(self, input, index, dim, dim_size):
        return scatter_sum(input, index, dim=dim, dim_size=dim_size)
    
def runMLP_(mlp : torch.nn.Module, features : torch.Tensor, batches : int, verbose : bool = False):  
    if verbose:
        print(f'MLP {features.shape} -> {mlp[-1].out_features} features')
    transposedFeatures = features.view(batches,-1, *features.shape[1:])
    
    processedFeatures = mlp(transposedFeatures)
    processedFeatures = processedFeatures.view(-1, *processedFeatures.shape[2:])
    if verbose:
        print(f'\tFeatures: {processedFeatures.shape} [min: {torch.min(processedFeatures)}, max: {torch.max(processedFeatures)}, mean: {torch.mean(processedFeatures)}]')
    return processedFeatures

def runMLP(mlp : torch.nn.Module, features : torch.Tensor, batches : int, verbose : bool = False, checkpoint : bool = True):      
    if checkpoint:
        return torch.utils.checkpoint.checkpoint(runMLP_, mlp, features, batches, verbose, use_reentrant = False)
    else:
        return runMLP_(mlp, features, batches, verbose)
class SimpleGNN(torch.nn.Module):
    def __init__(self, hyperParameterDict):
        super(SimpleGNN, self).__init__()
            
        nodeFeatures = 3+ hyperParameterDict['featureCount']

        self.edgeMLP1 = buildMLPwDict({
            'layout': [hyperParameterDict['hiddenUnits']] * hyperParameterDict['hiddenLayers'],
            'inputFeatures': nodeFeatures*2 + hyperParameterDict['coordinateFeatures'],
            'output': hyperParameterDict['nodeFeatures'],
            'preNorm': False,
            'norm': False,
            # 'channels': [1],
            'activation': hyperParameterDict['activation'],
            'gain': hyperParameterDict['gain'],
        })
        self.activationLayer = getActivationLayer(hyperParameterDict['activation'])
        self.ScatterSumLayer = ScatterSumLayer()

        numFeatures = nodeFeatures*2 + hyperParameterDict['coordinateFeatures'] + hyperParameterDict['nodeFeatures']*2

        messagePassingVertexLayers = []
        messagePassingEdgeLayers = []
        for layer in range(hyperParameterDict['messagePassingLayers']):
            vertexMLP = buildMLPwDict({
                'layout': [hyperParameterDict['hiddenUnits']] * hyperParameterDict['hiddenLayers'],
                'inputFeatures': hyperParameterDict['nodeFeatures'],
                'output': hyperParameterDict['nodeFeatures'],
                'preNorm': False,
                'norm': False,
                # 'channels': [1],
                'activation': hyperParameterDict['activation'],
                'gain': 1,
            })
            edgeMLP = buildMLPwDict({
                'layout': [hyperParameterDict['hiddenUnits']] * hyperParameterDict['hiddenLayers'],
                'inputFeatures': numFeatures,
                'output': hyperParameterDict['output'] if layer == hyperParameterDict['messagePassingLayers'] - 1 and not hyperParameterDict['finalVertexMLP'] else hyperParameterDict['nodeFeatures'],
                'preNorm': False,
                'norm': False,
                'activation': hyperParameterDict['activation'],
                'gain': hyperParameterDict['gain'],
            })
            messagePassingVertexLayers.append(vertexMLP)
            messagePassingEdgeLayers.append(edgeMLP)

        self.messagePassingVertexLayers = torch.nn.ModuleList(messagePassingVertexLayers)
        self.messagePassingEdgeLayers = torch.nn.ModuleList(messagePassingEdgeLayers)
        self.hyperParameterDict = hyperParameterDict

        if hyperParameterDict['finalVertexMLP'] == True:
            self.finalVertexMLP = buildMLPwDict({
                'layout': [hyperParameterDict['hiddenUnits']] * hyperParameterDict['hiddenLayers'],
                'inputFeatures': hyperParameterDict['nodeFeatures'],
                'output': hyperParameterDict['output'],
                'preNorm': False,
                'norm': False,
                # 'channels': [1],
                'activation': hyperParameterDict['activation'],
                'gain': 1,
                'bias': False
            })
        else:
            self.finalVertexMLP = None

    def forward(self, particles, neighbors, quantity):
        i, j = neighbors.get('noghost')[0].row, neighbors.get('noghost')[0].col
        hij = (particles.supports[i] + particles.supports[j]) / 2
        encodedDistances = basisEncoderLayer(
            neighbors.get('noghost')[1].x_ij/ hij.view(-1, 1),
            self.hyperParameterDict['basisTerms'], 
            self.hyperParameterDict['basis'],
            self.hyperParameterDict['encoderMode']
        )

        node_features = torch.cat([
            particles.supports.view(-1,1),
            particles.masses.view(-1,1),
            particles.densities.view(-1,1),
            quantity,]
        , dim = -1)

        edge_features = torch.cat([
            encodedDistances ,
            node_features[i],
            node_features[j]
        ], dim = -1)

        processedEdges = runMLP(self.edgeMLP1, edge_features,1, verbose = False, checkpoint = True) * (particles.masses[j] / particles.densities[j]).view(-1, 1)

        aggregatedEdges = scatter_sum(processedEdges, i, dim = 0, dim_size = particles.positions.shape[0])
        for layer in range(self.hyperParameterDict['messagePassingLayers']):
            vertexFeatures = runMLP(self.messagePassingVertexLayers[layer], aggregatedEdges, 1, verbose = False, checkpoint = True)
            vertexFeatures = self.activationLayer(vertexFeatures)

            newFeatures = torch.cat([
                vertexFeatures,
                particles.masses.view(-1, 1),
                particles.supports.view(-1, 1),
                particles.densities.view(-1,1),
                quantity,
            ], dim = -1)
            v_j = newFeatures[j]
            v_i = newFeatures[i]

            v_ij = torch.cat([v_i, v_j, encodedDistances / hij.view(-1, 1)], dim = -1)

            processedEdges = runMLP(self.messagePassingEdgeLayers[layer], v_ij, 1, verbose = False, checkpoint = True) * (particles.masses[j] / particles.densities[j]).view(-1, 1)
            aggregatedEdges = scatter_sum(processedEdges, i, dim = 0, dim_size = particles.positions.shape[0])

        if self.finalVertexMLP is not None:
            vertexFeatures = runMLP(self.finalVertexMLP, aggregatedEdges, 1, verbose = False, checkpoint = True)
            # vertexFeatures = self.activationLayer(vertexFeatures)
        else:
            vertexFeatures = aggregatedEdges
            
        return vertexFeatures
        # return scatter_sum(term, i, dim = 0, dim_size = particles.positions.shape[0])[:,0]

