import numpy as np
import torch

@torch.jit.script
def cpow(q, p : int):
    return torch.clamp(q, 0, 1)**p

def get_epsilon(dtype: torch.dtype):
    # return torch.finfo(torch.float64).eps
    if dtype == torch.bfloat16:
        return 0.0078125 * 8
    elif dtype == torch.float16:
        return 0.0009765625 * 8
    elif dtype == torch.float32:
        return 1.1920928955078125e-07 * 8
    elif dtype == torch.float64:
        return 2.220446049250313e-16 * 8
    else:
        raise ValueError("Unsupported dtype {}".format(dtype))

# For Kernel Normalization:
# in 3D: https://www.wolframalpha.com/input?i2d=true&i=Integrate%5BIntegrate%5BIntegrate%5BPower%5B%5C%2840%291-q%5C%2841%29%2C3%5DPower%5Bq%2C2%5DSin%5B%CF%86%5D%2C%7B%CF%86%2C0%2C%CF%80%7D%5D%2C%7B%CE%98%2C0%2C2%CF%80%7D%5D%2C%7Bq%2C0%2C1%7D%5D
# in 2D: https://www.wolframalpha.com/input?i2d=true&i=Integrate%5BIntegrate%5BPower%5B%5C%2840%291-q%5C%2841%29%2C3%5D%2C%7B%CE%98%2C0%2C2%CF%80%7D%5D%2C%7Bq%2C0%2C1%7D%5D
# in 1D: https://www.wolframalpha.com/input?i2d=true&i=Integrate%5BPower%5B%5C%2840%291-q%5C%2841%29%2C3%5D%2C%7Bq%2C0%2C1%7D%5D

import sphMath.kernelFunctions.Wendland2 as Wendland2
import sphMath.kernelFunctions.Wendland4 as Wendland4
import sphMath.kernelFunctions.Wendland6 as Wendland6
import sphMath.kernelFunctions.CubicSpline as CubicSpline
import sphMath.kernelFunctions.QuarticSpline as QuarticSpline
import sphMath.kernelFunctions.QuinticSpline as QuinticSpline
import sphMath.kernelFunctions.B7 as B7

import sphMath.kernelFunctions.Spiky as Spiky
import sphMath.kernelFunctions.ViscosityKernel as ViscosityKernel
import sphMath.kernelFunctions.Poly6 as Poly6
import sphMath.kernelFunctions.CohesionKernel as CohesionKernel
import sphMath.kernelFunctions.AdhesionKernel as AdhesionKernel

def norm_(input):
    return torch.linalg.norm(input, dim = -1, keepdim = False)
def norm_grad(input):
    length = norm_(input)
    float_eps = get_epsilon(input.dtype)
    length = length.clamp(min = float_eps)
    return input / length.view(-1,1)
def norm_hess(input):
    eps = get_epsilon(input.dtype)
    r = torch.linalg.norm(input, dim = -1).view(-1,1,1)# + eps

    outerProd = torch.einsum('ij,ik->ijk', input, input)
    diagTerm = torch.eye(input.shape[1], device = input.device) * (r**2 + eps**3)

    tensor = 1/(r**3 + eps**3) * (diagTerm - outerProd)

    # print(tensor)
    return tensor

class vectorNormalize_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        float_eps = get_epsilon(input.dtype)
        output = norm_grad(input + float_eps*0)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # input = input.to(grad_output.device)
        tensor = norm_hess(input)
        output = torch.einsum('...ij, ...j -> ...i', tensor, grad_output)
        return output

vectorNormalize = vectorNormalize_.apply
class vectorNorm_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = norm_(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # input = input.to(grad_output.device)
        grad_output = grad_output

        float_eps = get_epsilon(input.dtype)

        grad_input = torch.einsum('...i, ... -> ...i', vectorNormalize(input + float_eps*0), grad_output)
        return grad_input

vectorNorm = vectorNorm_.apply

from enum import Enum
from sphMath.enums import KernelType

# @torch.jit.script
def eval_k(kernel : KernelType, q, dim: int = 2):
    if kernel == KernelType.Poly6:
        return Poly6.k(q, dim)
    elif kernel == KernelType.CubicSpline:
        return CubicSpline.k(q, dim)
    elif kernel == KernelType.QuarticSpline:
        return QuarticSpline.k(q, dim)
    elif kernel == KernelType.QuinticSpline:
        return QuinticSpline.k(q, dim)
    elif kernel == KernelType.B7:
        return B7.k(q, dim)
    elif kernel == KernelType.Wendland2:
        return Wendland2.k(q, dim)
    elif kernel == KernelType.Wendland4:
        return Wendland4.k(q, dim)
    elif kernel == KernelType.Wendland6:
        return Wendland6.k(q, dim)
    elif kernel == KernelType.Spiky:
        return Spiky.k(q, dim)
    elif kernel == KernelType.ViscosityKernel:
        return ViscosityKernel.k(q, dim)
    elif kernel == KernelType.CohesionKernel:
        return CohesionKernel.k(q, dim)
    elif kernel == KernelType.AdhesionKernel:
        return AdhesionKernel.k(q, dim)
    else:
        raise ValueError("Unknown kernel type {}".format(kernel))
    
# @torch.jit.script
def eval_dkdq(kernel : KernelType, q, dim: int = 2):
    if kernel == KernelType.Poly6:
        return Poly6.dkdq(q, dim)
    elif kernel == KernelType.CubicSpline:
        return CubicSpline.dkdq(q, dim)
    elif kernel == KernelType.QuarticSpline:
        return QuarticSpline.dkdq(q, dim)
    elif kernel == KernelType.QuinticSpline:
        return QuinticSpline.dkdq(q, dim)
    elif kernel == KernelType.B7:
        return B7.dkdq(q, dim)
    elif kernel == KernelType.Wendland2:
        return Wendland2.dkdq(q, dim)
    elif kernel == KernelType.Wendland4:
        return Wendland4.dkdq(q, dim)
    elif kernel == KernelType.Wendland6:
        return Wendland6.dkdq(q, dim)
    elif kernel == KernelType.Spiky:
        return Spiky.dkdq(q, dim)
    elif kernel == KernelType.ViscosityKernel:
        return ViscosityKernel.dkdq(q, dim)
    elif kernel == KernelType.CohesionKernel:
        return CohesionKernel.dkdq(q, dim)
    elif kernel == KernelType.AdhesionKernel:
        return AdhesionKernel.dkdq(q, dim)
    else:
        raise ValueError("Unknown kernel type {}".format(kernel))
    
# @torch.jit.script
def eval_d2kdq2(kernel : KernelType, q, dim: int = 2):
    if kernel == KernelType.Poly6:
        return Poly6.d2kdq2(q, dim)
    elif kernel == KernelType.CubicSpline:
        return CubicSpline.d2kdq2(q, dim)
    elif kernel == KernelType.QuarticSpline:
        return QuarticSpline.d2kdq2(q, dim)
    elif kernel == KernelType.QuinticSpline:
        return QuinticSpline.d2kdq2(q, dim)
    elif kernel == KernelType.B7:
        return B7.d2kdq2(q, dim)
    elif kernel == KernelType.Wendland2:
        return Wendland2.d2kdq2(q, dim)
    elif kernel == KernelType.Wendland4:
        return Wendland4.d2kdq2(q, dim)
    elif kernel == KernelType.Wendland6:
        return Wendland6.d2kdq2(q, dim)
    elif kernel == KernelType.Spiky:
        return Spiky.d2kdq2(q, dim)
    elif kernel == KernelType.ViscosityKernel:
        return ViscosityKernel.d2kdq2(q, dim)
    elif kernel == KernelType.CohesionKernel:
        return CohesionKernel.d2kdq2(q, dim)
    elif kernel == KernelType.AdhesionKernel:
        return AdhesionKernel.d2kdq2(q, dim)
    else:
        raise ValueError("Unknown kernel type {}".format(kernel))

# @torch.jit.script
def eval_d3kdq3(kernel : KernelType, q, dim: int = 2):
    if kernel == KernelType.Poly6:
        return Poly6.d3kdq3(q, dim)
    elif kernel == KernelType.CubicSpline:
        return CubicSpline.d3kdq3(q, dim)
    elif kernel == KernelType.QuarticSpline:
        return QuarticSpline.d3kdq3(q, dim)
    elif kernel == KernelType.QuinticSpline:
        return QuinticSpline.d3kdq3(q, dim)
    elif kernel == KernelType.B7:
        return B7.d3kdq3(q, dim)
    elif kernel == KernelType.Wendland2:
        return Wendland2.d3kdq3(q, dim)
    elif kernel == KernelType.Wendland4:
        return Wendland4.d3kdq3(q, dim)
    elif kernel == KernelType.Wendland6:
        return Wendland6.d3kdq3(q, dim)
    elif kernel == KernelType.Spiky:
        return Spiky.d3kdq3(q, dim)
    elif kernel == KernelType.ViscosityKernel:
        return ViscosityKernel.d3kdq3(q, dim)
    elif kernel == KernelType.CohesionKernel:
        return CohesionKernel.d3kdq3(q, dim)
    elif kernel == KernelType.AdhesionKernel:
        return AdhesionKernel.d3kdq3(q, dim)
    else:
        raise ValueError("Unknown kernel type {}".format(kernel))

# @torch.jit.script
def eval_C_d(kernel : KernelType, dim : int):
    if kernel == KernelType.Poly6:
        return Poly6.C_d(dim)
    elif kernel == KernelType.CubicSpline:
        return CubicSpline.C_d(dim)
    elif kernel == KernelType.QuarticSpline:
        return QuarticSpline.C_d(dim)
    elif kernel == KernelType.QuinticSpline:
        return QuinticSpline.C_d(dim)
    elif kernel == KernelType.B7:
        return B7.C_d(dim)
    elif kernel == KernelType.Wendland2:
        return Wendland2.C_d(dim)
    elif kernel == KernelType.Wendland4:
        return Wendland4.C_d(dim)
    elif kernel == KernelType.Wendland6:
        return Wendland6.C_d(dim)
    elif kernel == KernelType.Spiky:
        return Spiky.C_d(dim)
    elif kernel == KernelType.ViscosityKernel:
        return ViscosityKernel.C_d(dim)
    elif kernel == KernelType.CohesionKernel:
        return CohesionKernel.C_d(dim)
    elif kernel == KernelType.AdhesionKernel:
        return AdhesionKernel.C_d(dim)
    else:
        raise ValueError("Unknown kernel type {}".format(kernel))
    
# @torch.jit.script
def eval_kernelScale(kernel : KernelType, dim: int = 2):
    # if kernel == KernelType.Poly6:
        # return Poly6.kernelScale(dim)
    if kernel == KernelType.CubicSpline:
        return CubicSpline.kernelScale(dim)
    elif kernel == KernelType.QuarticSpline:
        return QuarticSpline.kernelScale(dim)
    elif kernel == KernelType.QuinticSpline:
        return QuinticSpline.kernelScale(dim)
    elif kernel == KernelType.B7:
        return B7.kernelScale(dim)
    elif kernel == KernelType.Wendland2:
        return Wendland2.kernelScale(dim)
    elif kernel == KernelType.Wendland4:
        return Wendland4.kernelScale(dim)
    elif kernel == KernelType.Wendland6:
        return Wendland6.kernelScale(dim)
    # elif kernel == KernelType.Spiky:
        # return Spiky.kernelScale(dim)
    # elif kernel == KernelType.ViscosityKernel:
    #     return ViscosityKernel.kernelScale(dim)
    # elif kernel == KernelType.CohesionKernel:
    #     return CohesionKernel.kernelScale(dim)
    # elif kernel == KernelType.AdhesionKernel:
    #     return AdhesionKernel.kernelScale(dim)
    else:
        raise ValueError("Unknown kernel type {}".format(kernel))

# @torch.jit.script  
def eval_packingRatio(kernel : KernelType):
    # if kernel == KernelType.Poly6:
        # return Poly6.packingRatio()
    if kernel == KernelType.CubicSpline:
        return CubicSpline.packingRatio()
    elif kernel == KernelType.QuarticSpline:
        return QuarticSpline.packingRatio()
    elif kernel == KernelType.QuinticSpline:
        return QuinticSpline.packingRatio()
    elif kernel == KernelType.B7:
        return B7.packingRatio()
    elif kernel == KernelType.Wendland2:
        return Wendland2.packingRatio()
    elif kernel == KernelType.Wendland4:
        return Wendland4.packingRatio()
    elif kernel == KernelType.Wendland6:
        return Wendland6.packingRatio()
    # elif kernel == KernelType.Spiky:
    #     return Spiky.packingRatio()
    # elif kernel == KernelType.ViscosityKernel:
    #     return ViscosityKernel.packingRatio()
    # elif kernel == KernelType.CohesionKernel:
    #     return CohesionKernel.packingRatio()
    # elif kernel == KernelType.AdhesionKernel:
    #     return AdhesionKernel.packingRatio()
    else:
        raise ValueError("Unknown kernel type {}".format(kernel))

@torch.jit.script
class SPHKernel:
    def __init__(self, module : KernelType):
        self.kernel = module
    def xi(self, dim: int):
        return eval_packingRatio(self.kernel) * eval_kernelScale(self.kernel, dim)
        # return eval_packingRatio(self.kernel) * eval_kernelScale(self.kernel, dim)
    
    # actual kernel function
    # Calls the kernel function of the module and multiplies it with the normalization constants
    # The normalization constants are calculated by integrating the kernel function over the domain
    # of the kernel function. This is done to ensure that the kernel function is normalized.
    #
    # returns $$W(x, h) = \frac{C_d}{h^d} \hat{W}_{\text{kernel}}\left(\frac{\norm{x}}{h}\right)$$
    def eval(self, x, h):
        dim = x.shape[1]
        r = vectorNorm(x)
        q = r / h
        return eval_k(self.kernel, q, dim) * self.C_d(dim) / h**dim
    
    # Spatial gradient functions
    def jacobian(self, x, h):
        dim = x.shape[1]
        r = vectorNorm(x)
        q = r / h
        grad = vectorNormalize(x)
        return grad * ((eval_dkdq(self.kernel, q, dim)) * self.C_d(dim) / h**(dim + 1)).view(-1,1)
    
    def derivative(self, x, h):
        dim = x.shape[1]
        r = vectorNorm(x)
        q = r / h
        grad = vectorNormalize(x)
        return eval_dkdq(self.kernel, q, dim) * self.C_d(dim) / h**(dim + 1)
    
    def hessian(self, x, h):
        dim = x.shape[1]
        r = vectorNorm(x)
        q = r / h
        eps = get_epsilon(x.dtype)

        k1 = eval_dkdq(self.kernel, q, dim)   * eval_C_d(self.kernel, dim) / h**(dim + 1)
        k2 = eval_d2kdq2(self.kernel, q, dim) * eval_C_d(self.kernel, dim) / h**(dim + 2)
        s = (r**2 + eps**2 * h**2)


        factorA = torch.einsum('nu, nv -> nuv', x, x) / s.view(-1,1,1)# + torch.where(q < 1e-5, )
        for i in range(dim):
            factorA[:,i,i] = torch.where(q < 1e-5, 1, factorA[:,i,i])
        # factorA[:,0,0] = torch.where(q < 1e-5, 1, factorA[:,0,0])
        # factorA[:,1,1] = torch.where(q < 1e-5, 1, factorA[:,1,1])

        # factorA

        factorB = -torch.einsum('nu, nv -> nuv', x, x) / (r**3 + eps**3 * h**3).view(-1,1,1)
        factorB += torch.eye(dim, device = x.device, dtype = x.dtype) / (r + eps**2 * h).view(-1,1,1)

        hessian = factorA * k2.view(-1,1,1) + factorB * k1.view(-1,1,1)
        return hessian
    
    def laplacian(self, x, h):
        dim = x.shape[1]
        r = torch.linalg.norm(x, dim = -1)
        q = r / h
        eps = get_epsilon(x.dtype)
        r_eps = r + eps * h

        k1 = eval_dkdq(self.kernel, q, dim)   * eval_C_d(self.kernel, dim) / h**(dim + 1)
        k2 = eval_d2kdq2(self.kernel, q, dim) * eval_C_d(self.kernel, dim) / h**(dim + 2)

        s = (x**2).sum(dim=-1) / (r_eps**2)
        s = torch.where(q < 1e-5, 1, s)
        t = - (x**2).sum(dim=-1) / (r_eps**3)
        t += dim / (r_eps)

        laplacian = (s * k2 + t * k1)
        laplacian[q < 1e-5] = 0
        return laplacian
    
    def laplaciangrad(self, x, h):
        dim = x.shape[1]
        r = torch.linalg.norm(x, dim = -1)
        eps = get_epsilon(x.dtype)
        r_eps = r + eps * h
        q = r / h
        n_ij = torch.nn.functional.normalize(x, dim = -1)
        
        k1 = eval_dkdq(self.kernel, q, dim)   * eval_C_d(self.kernel, dim) / h**(dim + 1)
        k2 = eval_d2kdq2(self.kernel, q, dim) * eval_C_d(self.kernel, dim) / h**(dim + 2)
        k3 = eval_d3kdq3(self.kernel, q, dim) * eval_C_d(self.kernel, dim) / h**(dim + 3)

        k1_prime = n_ij * k2.view(-1,1)
        k2_prime = n_ij * k3.view(-1,1)

        s = 1
        t = (dim - 1) / r_eps
        
        s_prime = 0
        t_prime = x * ((1 - dim) / r_eps**3)[:,None]

        sk_prime = s_prime * k2.view(-1,1) + s * k2_prime
        tk_prime = t_prime * k1.view(-1,1) + t.view(-1,1) * k1_prime

        laplacian_grad = (sk_prime + tk_prime)
        return laplacian_grad
    
    def laplacianApproximation(self, x_ij, h_ij):
        dim = x_ij.shape[1]
        r_ij = torch.linalg.norm(x_ij, dim = -1)
        eps = get_epsilon(x_ij.dtype)
        r_eps = r_ij + eps * h_ij

        gradW_ij = self.jacobian(x_ij, h_ij)
        kernelNorm = torch.linalg.norm(gradW_ij, dim = -1)

        return kernelNorm / r_eps
    
    def laplacianApproximationGrad(self, x_ij, h_ij):
        dim = x_ij.shape[1]
        r_ij = torch.linalg.norm(x_ij, dim = -1)
        n_ij = torch.nn.functional.normalize(x_ij, dim = -1)
        eps = get_epsilon(x_ij.dtype)
        g = r_ij + eps * h_ij

        gradW_ij = self.jacobian(x_ij, h_ij)
        f = torch.linalg.norm(gradW_ij, dim = -1)

        outer = gradW_ij / f.view(-1,1)
        inner = self.hessian(x_ij, h_ij)
        f_prime = torch.einsum('nij, nj -> ni', inner, outer)

        g_prime = n_ij
        return (f_prime * g.view(-1,1) - f.view(-1,1) * g_prime) / (g**2).view(-1,1)
        
        
    def thirdOrderDerivatives(self, x_ij, h_ij):
        r_ij = torch.linalg.norm(x_ij, dim = -1)
        r_eps = r_ij #+ 1e-5 * h_ij
        eps = get_epsilon(x_ij.dtype) * h_ij
        r_eps = r_eps / h_ij + eps

        q = r_ij / h_ij
        dim = x_ij.shape[1]

        vec = torch.tensor([3,5,7])[:dim]
        t = torch.einsum('ij, k -> ijk', torch.einsum('i, j -> ij', vec, vec), vec)

        k1 = eval_dkdq(self.kernel, q, dim) * eval_C_d(self.kernel, dim) / (h_ij**(dim + 1))
        k2 = eval_d2kdq2(self.kernel, q, dim) * eval_C_d(self.kernel, dim) / (h_ij**(dim + 2))
        k3 = eval_d3kdq3(self.kernel, q, dim) * eval_C_d(self.kernel, dim) / (h_ij**(dim + 3))

        # The multiplication here is done to ensure that values dont get _extremely_ small
        # with **5 an r of 1e-7 would otherwise be the limit due to floating point precision
        # for single precision. this is a bit of a hack, but it works.
        termA =  3 * k1 / r_eps**5 / h_ij**2
        termB = -    k1 / r_eps**3
        termC = -3 * k2 / r_eps**4 / h_ij
        termD =  1 * k2 / r_eps**2 * h_ij
        termE =  1 * k3 / r_eps**3

        outer_prod = torch.einsum('nij, nk -> nijk', torch.einsum('ni, nj -> nij', x_ij, x_ij), x_ij)

        baseTerm = (termA + termC + termE).view(-1,1,1,1) * outer_prod 

        x = x_ij[:,0][:,None]
        y = x_ij[:,1][:,None] if dim > 1 else torch.zeros_like(x)
        z = x_ij[:,2][:,None] if dim > 2 else torch.zeros_like(x)

        complexTerm = torch.zeros_like(baseTerm)
        complexTerm[:, t == 3*3*3] = 3 * x
        complexTerm[:, t == 5*5*5] = 3 * y 
        complexTerm[:, t == 7*7*7] = 3 * z 
        # x^2 terms
        complexTerm[:, t == 3*3*5] = y # xxy is missing a y
        complexTerm[:, t == 3*3*7] = z # xxz is missing a z
        # y^2 terms
        complexTerm[:, t == 3*5*5] = x # xyy is missing a x
        complexTerm[:, t == 5*5*7] = z # yyz is missing a z
        # z^2 terms
        complexTerm[:, t == 3*7*7] = x # xzz is missing a x
        complexTerm[:, t == 5*7*7] = y # yzz is missing a y

        finalTerm = baseTerm + complexTerm * (termB + termD).view(-1,1,1,1)
        return finalTerm / (h_ij**3).view(-1,1,1,1)
    
    ## Utility functions
    # def __getattr__(self, name):
        # return getattr(self.module, name)

    def packingRatio(self):
        return eval_packingRatio(self.kernel)
    
    def kernelScale(self, dim: int = 2):
        return eval_kernelScale(self.kernel, dim)
    
    def C_d(self, dim : int):
        return eval_C_d(self.kernel, dim)
    
    def N_H(self, dim : int):
        # if not hasattr(self.module, 'packingRatio'):
            # return 8 if dim == 1 else (27 if dim == 2 else 50)
        packingRatio = eval_packingRatio(self.kernel)
        fac = 2.0 if dim == 1 else (np.pi if dim == 2 else 4 * np.pi / 3)
        N = fac * packingRatio**dim * eval_kernelScale(self.kernel, dim)**dim
        return N

    
    ## Grad-h functions
    def dkdh(self, x, h):
        dim = x.shape[1]
        r = vectorNorm(x)
        q = r / h

        k = eval_k(self.kernel, q, dim)
        dkdq = eval_dkdq(self.kernel, q, dim)

        return -self.C_d(dim)/ h ** (dim + 2) * (dim * h * k + r * dkdq)
    
    def djacobiandh(self, x, h):
        dim = x.shape[1]
        r = vectorNorm(x)
        q = r / h
        grad = vectorNormalize(x)

        dkdq = eval_dkdq(self.kernel, q, dim)
        d2kdq2 = eval_d2kdq2(self.kernel, q, dim)

        factor = self.C_d(dim) / h**(dim + 3)

        return -grad * (factor * ((dim + 1) * h * dkdq + r * d2kdq2)).view(-1,1)
    
    def dlaplaciandh(self, x, h):
        dim = x.shape[1]
        r = torch.linalg.norm(x, dim = -1)
        eps = get_epsilon(x.dtype)
        r_eps = r + eps * h
        q = r / h
        k1 = eval_dkdq(self.kernel, q, dim) * eval_C_d(self.kernel, dim)
        k2 = eval_d2kdq2(self.kernel, q, dim) * eval_C_d(self.kernel, dim) 
        k3 = eval_d3kdq3(self.kernel, q, dim) * eval_C_d(self.kernel, dim) 

        ddh_k1 = - 1 / h**(dim + 3) * ((dim + 1) * h * k1 + r * k2)
        ddh_k2 = - 1 / h**(dim + 4) * ((dim + 2) * h * k2 + r * k3)

        s  =   (x**2).sum(dim=-1) / r_eps**2
        t  = - (x**2).sum(dim=-1) / r_eps**3
        t +=   dim                / r_eps

        laplacian = (s * ddh_k2 + t * ddh_k1)
        laplacian[q < 1e-5] = 0
        return laplacian
    
    def ddhlaplacianApproximation(self, x_ij, h_ij):
        dim = x_ij.shape[1]
        r_ij = torch.linalg.norm(x_ij, dim = -1)
        eps = get_epsilon(x_ij.dtype)
        r_eps = r_ij + eps * h_ij

        gradW_ij = self.jacobian(x_ij, h_ij)
        kernelNorm = torch.linalg.norm(gradW_ij, dim = -1)

        inner = self.djacobiandh(x_ij, h_ij)
        outer = gradW_ij / kernelNorm.view(-1,1)

        # print(inner.shape, outer.shape, r_eps.shape)

        return torch.einsum('nu, nu -> n ', inner, outer) / r_eps

    # old implementation
    # def kernel(self, rij, hij, dim : int = 2):
        # return self.module.kernel(rij, hij, dim)
    
    # def kernelGradient(self, rij, xij, hij, dim : int = 2):
    #     return self.module.kernelGradient(rij, xij, hij, dim)
    
    # def kernelLaplacian(self, rij, hij, dim : int = 2):
    #     return self.module.kernelLaplacian(rij, hij, dim)
    
    # def Jacobi(self, rij, xij, hij, dim : int = 2):
    #     return self.module.kernelGradient(rij, xij, hij, dim)

    # def Hessian2D(self, rij, xij, hij, dim : int = 2):
    #     hessian = torch.zeros(rij.shape[0], 2, 2, device=rij.device, dtype=rij.dtype)
    #     factor = eval_C_d(self.kernel, dim) / hij**(dim)

    #     r_ij = rij * hij
    #     x_ij = xij * r_ij.unsqueeze(-1)
    #     q_ij = rij


    #     k2 = eval_d2kdq2(self.kernel, q_ij, dim = dim)
    #     k1 = eval_dkdq(self.kernel, q_ij, dim = dim)

    #     termA_x = x_ij[:,0]**2 / (hij**2 * r_ij**2    + 1e-5 * hij) * k2
    #     termA_y = x_ij[:,1]**2 / (hij**2 * r_ij**2    + 1e-5 * hij) * k2

    #     termB_x = x_ij[:,1]**2 / (hij * r_ij**3 + 1e-10 * hij) * k1
    #     termB_y = x_ij[:,0]**2 / (hij * r_ij**3 + 1e-10 * hij) * k1
            
    #     d2Wdx2 = factor * (termA_x + termB_x + 1/hij**2 * torch.where(q_ij < 1e-5, eval_d2kdq2(self.kernel, torch.tensor(0.), dim = dim), 0))
    #     d2Wdy2 = factor * (termA_y + termB_y + 1/hij**2 * torch.where(q_ij < 1e-5, eval_d2kdq2(self.kernel, torch.tensor(0.), dim = dim), 0))
    #     # termA_x = x_ij[:,0]**2 / (hij * r_ij    + 1e-5 * hij)**2 * self.module.d2kdq2(q_ij, dim = dim)
    #     # termA_y = x_ij[:,1]**2 / (hij * r_ij    + 1e-5 * hij)**2 * self.module.d2kdq2(q_ij, dim = dim)

    #     # termB_x = torch.where(r_ij / hij > 1e-5, 1 /(hij * r_ij + 1e-6 * hij),0) * self.module.dkdq(q_ij, dim = dim)
    #     # termB_y = torch.where(r_ij / hij > 1e-5, 1 /(hij * r_ij + 1e-6 * hij),0) * self.module.dkdq(q_ij, dim = dim)

    #     # termC_x = - (x_ij[:,0]**2) / (hij * r_ij**3 + 1e-5 * hij) * self.module.dkdq(q_ij, dim = dim)
    #     # termC_y = - (x_ij[:,1]**2) / (hij * r_ij**3 + 1e-5 * hij) * self.module.dkdq(q_ij, dim = dim)
                
    #     # d2Wdx2 = factor * (termA_x + termB_x + termC_x + 1/hij**2 * torch.where(q_ij < 1e-5, self.module.d2kdq2(torch.tensor(0.), dim = dim), 0))
    #     # d2Wdy2 = factor * (termA_y + termB_y + termC_y + 1/hij**2 * torch.where(q_ij < 1e-5, self.module.d2kdq2(torch.tensor(0.), dim = dim), 0))

    #     d2Wdxy = eval_C_d(self.kernel, dim) * hij**(-dim) * torch.where(q_ij > -1e-7, \
    #         ( x_ij[:,0] * x_ij[:,1]) / (hij * r_ij **2 + 1e-5 * hij) * (1 / hij * eval_d2kdq2(self.kernel, q_ij, dim = dim) - 1 / (r_ij + 1e-3 * hij) * eval_dkdq(self.kernel, q_ij, dim = dim)),0)
    #     d2Wdyx = d2Wdxy
            
    #     hessian[:,0,0] = torch.where(q_ij <= 1, d2Wdx2, 0)
    #     hessian[:,1,1] = torch.where(q_ij <= 1, d2Wdy2, 0)
    #     hessian[:,0,1] = torch.where(q_ij <= 1, d2Wdxy, 0)
    #     hessian[:,1,0] = torch.where(q_ij <= 1, d2Wdyx, 0)

    #     return hessian

def evalW(kernel, x, h_i, h_j, scheme):
    if scheme == 'gather':
        return kernel.eval(x, h_i)
    elif scheme == 'scatter':
        return kernel.eval(x, h_j)
    elif scheme == 'mean':
        return kernel.eval(x, (h_i + h_j) / 2)
    elif scheme == 'min':
        return kernel.eval(x, torch.min(h_i, h_j))
    elif scheme == 'max':
        return kernel.eval(x, torch.max(h_i, h_j))
    elif scheme == 'symmetric':
        return (kernel.eval(x, h_i) + kernel.eval(x, h_j)) / 2
    else:
        raise ValueError(f'Unknown support scheme {scheme}')
def evalGradW(kernel, x, h_i, h_j, scheme):
    if scheme == 'gather':
        return kernel.jacobian(x, h_i)
    elif scheme == 'scatter':
        return kernel.jacobian(x, h_j)
    elif scheme == 'mean':
        return kernel.jacobian(x, (h_i + h_j) / 2)
    elif scheme == 'min':
        return kernel.jacobian(x, torch.min(h_i, h_j))
    elif scheme == 'max':
        return kernel.jacobian(x, torch.max(h_i, h_j))
    elif scheme == 'symmetric':
        return (kernel.jacobian(x, h_i) + kernel.jacobian(x, h_j)) / 2
    else:
        raise ValueError(f'Unknown support scheme {scheme}')
def evalDerivativeW(kernel, x, h_i, h_j, scheme):
    if scheme == 'gather':
        return kernel.derivative(x, h_i)
    elif scheme == 'scatter':
        return kernel.derivative(x, h_j)
    elif scheme == 'mean':
        return kernel.derivative(x, (h_i + h_j) / 2)
    elif scheme == 'min':
        return kernel.derivative(x, torch.min(h_i, h_j))
    elif scheme == 'max':
        return kernel.derivative(x, torch.max(h_i, h_j))
    elif scheme == 'symmetric':
        return (kernel.derivative(x, h_i) + kernel.derivative(x, h_j)) / 2
    else:
        raise ValueError(f'Unknown support scheme {scheme}')


# def getSPHKernel(kernel = 'Wendland2'):
#     if kernel == 'Wendland2': return SPHKernel(KernelType.Wendland2)
#     elif kernel == 'Wendland4': return SPHKernel(KernelType.Wendland4)
#     elif kernel == 'Wendland6': return SPHKernel(KernelType.Wendland6)
#     elif kernel == 'CubicSpline': return SPHKernel(KernelType.CubicSpline)
#     elif kernel == 'QuarticSpline': return SPHKernel(KernelType.QuarticSpline)
#     elif kernel == 'QuinticSpline': return SPHKernel(KernelType.QuinticSpline)
#     elif kernel == 'Poly6': return SPHKernel(KernelType.Poly6)
#     elif kernel == 'B7': return SPHKernel(KernelType.B7)
#     elif kernel == 'Spiky': return SPHKernel(KernelType.Spiky)
#     elif kernel == 'ViscosityKernel': return SPHKernel(KernelType.ViscosityKernel)
#     elif kernel == 'CohesionKernel': return SPHKernel(KernelType.CohesionKernel)
#     elif kernel == 'AdhesionKernel': return SPHKernel(KernelType.AdhesionKernel)
#     else:
#         raise ValueError("Unknown kernel type {}".format(kernel))


def getKernelEnum(kernel: str = 'Wendland2'):
    if kernel == 'Wendland2': return KernelType.Wendland2
    elif kernel == 'Wendland4': return KernelType.Wendland4
    elif kernel == 'Wendland6': return KernelType.Wendland6
    elif kernel == 'CubicSpline': return KernelType.CubicSpline
    elif kernel == 'QuarticSpline': return KernelType.QuarticSpline
    elif kernel == 'QuinticSpline': return KernelType.QuinticSpline
    elif kernel == 'Poly6': return KernelType.Poly6
    elif kernel == 'B7': return KernelType.B7
    elif kernel == 'Spiky': return KernelType.Spiky
    elif kernel == 'ViscosityKernel': return KernelType.ViscosityKernel
    elif kernel == 'CohesionKernel': return KernelType.CohesionKernel
    elif kernel == 'AdhesionKernel': return KernelType.AdhesionKernel
    else:
        raise ValueError("Unknown kernel type {}".format(kernel))
    
from typing import Union
def getSPHKernelv2(kernel: Union[str, KernelType] = 'Wendland2'):
    if isinstance(kernel, str):
        kernelEnum = getKernelEnum(kernel)
    elif isinstance(kernel, KernelType):
        kernelEnum = kernel
    else:
        raise ValueError("Unknown kernel type {}".format(kernel))
    return SPHKernel(kernelEnum)


kernels = ['Wendland2', 'Wendland4', 'Wendland6', 'CubicSpline', 'QuarticSpline', 'QuinticSpline', 'Poly6', 'B7']



def Kernel_Scale(kernel: KernelType, dim: int = 2):
    return eval_kernelScale(kernel, dim)
def Kernel_C_d(kernel: KernelType, dim: int = 2):
    return eval_C_d(kernel, dim)
def Kernel_N_H(kernel: KernelType, dim: int = 2):
    packingRatio = eval_packingRatio(kernel)
    fac = 2.0 if dim == 1 else (np.pi if dim == 2 else 4 * np.pi / 3)
    N = fac * packingRatio**dim * eval_kernelScale(kernel, dim)**dim
    return N
def Kernel_xi(kernel: KernelType, dim: int = 2):
    return eval_packingRatio(kernel) * eval_kernelScale(kernel, dim)

from torch.utils.checkpoint import checkpoint
# @torch.jit.script
def Kernel(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
    dim = x.shape[1]
    # r = torch.linalg.norm(x, dim = -1)
    r = vectorNorm(x)
    # r = checkpoint(vectorNorm, x)
    q = r / h
    return eval_k(kernel, q, dim) * eval_C_d(kernel, dim) / h**float(dim)
# @torch.jit.script
def Kernel_Gradient(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
    dim = x.shape[1]
    # r = torch.linalg.norm(x, dim = -1)

    r = vectorNorm(x)
    # r = checkpoint(vectorNorm, x)
    q = r / h
    grad = vectorNormalize(x)
    # grad = checkpoint(vectorNormalize, x)
    # grad = torch.nn.functional.normalize(x, dim = -1)
    normalizationTerm = eval_C_d(kernel, dim) / h**(dim + 1)
    kernelTerm = eval_dkdq(kernel, q, dim)
    normalizedKernelTerm = kernelTerm * normalizationTerm
    return grad * normalizedKernelTerm.view(-1,1)

def Kernel_Derivative(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
    dim = x.shape[1]
    r = vectorNorm(x)
    q = r / h
    # grad = vectorNormalize(x)
    return eval_dkdq(kernel, q, dim) * eval_C_d(kernel, dim) / h**(dim + 1)
def Kernel_Hessian(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
    dim = x.shape[1]
    r = vectorNorm(x)
    q = r / h
    eps = get_epsilon(x.dtype)

    k1 = eval_dkdq(kernel, q, dim)   * eval_C_d(kernel, dim) / h**(dim + 1)
    k2 = eval_d2kdq2(kernel, q, dim) * eval_C_d(kernel, dim) / h**(dim + 2)
    s = (r**2 + eps**2 * h**2)

    factorA = torch.einsum('nu, nv -> nuv', x, x) / s.view(-1,1,1)# + torch.where(q < 1e-5, )
    for i in range(dim):
        factorA[:,i,i] = torch.where(q < 1e-5, 1, factorA[:,i,i])
    # factorA[:,0,0] = torch.where(q < 1e-5, 1, factorA[:,0,0])
    # factorA[:,1,1] = torch.where(q < 1e-5, 1, factorA[:,1,1])

    # factorA

    factorB = -torch.einsum('nu, nv -> nuv', x, x) / (r**3 + eps**3 * h**3).view(-1,1,1)
    factorB += torch.eye(dim, device = x.device, dtype = x.dtype) / (r + eps**2 * h).view(-1,1,1)

    hessian = factorA * k2.view(-1,1,1) + factorB * k1.view(-1,1,1)
    return hessian
def Kernel_Laplacian(kernel: KernelType, x: torch.Tensor, h:torch.Tensor):
    dim = x.shape[1]
    r = torch.linalg.norm(x, dim = -1)
    q = r / h
    eps = get_epsilon(x.dtype)
    r_eps = r + eps * h

    k1 = eval_dkdq(kernel, q, dim)   * eval_C_d(kernel, dim) / h**(dim + 1)
    k2 = eval_d2kdq2(kernel, q, dim) * eval_C_d(kernel, dim) / h**(dim + 2)

    s = (x**2).sum(dim=-1) / (r_eps**2)
    s = torch.where(q < 1e-5, 1, s)
    t = - (x**2).sum(dim=-1) / (r_eps**3)
    t += dim / (r_eps)

    laplacian = (s * k2 + t * k1)
    laplacian[q < 1e-5] = 0
    return laplacian
def Kernel_LaplacianGrad(kernel: KernelType, x: torch.Tensor, h:torch.Tensor):
    dim = x.shape[1]
    r = torch.linalg.norm(x, dim = -1)
    eps = get_epsilon(x.dtype)
    r_eps = r + eps * h
    q = r / h
    n_ij = torch.nn.functional.normalize(x, dim = -1)
    
    k1 = eval_dkdq(kernel, q, dim)   * eval_C_d(kernel, dim) / h**(dim + 1)
    k2 = eval_d2kdq2(kernel, q, dim) * eval_C_d(kernel, dim) / h**(dim + 2)
    k3 = eval_d3kdq3(kernel, q, dim) * eval_C_d(kernel, dim) / h**(dim + 3)

    k1_prime = n_ij * k2.view(-1,1)
    k2_prime = n_ij * k3.view(-1,1)

    s = 1
    t = (dim - 1) / r_eps
    
    s_prime = 0
    t_prime = x * ((1 - dim) / r_eps**3)[:,None]

    sk_prime = s_prime * k2.view(-1,1) + s * k2_prime
    tk_prime = t_prime * k1.view(-1,1) + t.view(-1,1) * k1_prime

    laplacian_grad = (sk_prime + tk_prime)
    return laplacian_grad
def Kernel_LaplacianApproximation(kernel: KernelType, x: torch.Tensor, h:torch.Tensor): 
    dim = x.shape[1]
    r_ij = torch.linalg.norm(x, dim = -1)
    eps = get_epsilon(x.dtype)
    r_eps = r_ij + eps * h

    gradW_ij = Kernel_Gradient(kernel, x, h)
    kernelNorm = torch.linalg.norm(gradW_ij, dim = -1)

    return kernelNorm / r_eps
def Kernel_LaplacianApproximationGrad(kernel: KernelType, x: torch.Tensor, h:torch.Tensor):
    dim = x.shape[1]
    r_ij = torch.linalg.norm(x, dim = -1)
    n_ij = torch.nn.functional.normalize(x, dim = -1)
    eps = get_epsilon(x.dtype)
    g = r_ij + eps * h

    gradW_ij = Kernel_Gradient(kernel, x, h)
    f = torch.linalg.norm(gradW_ij, dim = -1)

    outer = gradW_ij / f.view(-1,1)
    inner = Kernel_Hessian(kernel, x, h)
    f_prime = torch.einsum('nij, nj -> ni', inner, outer)

    g_prime = n_ij
    return (f_prime * g.view(-1,1) - f.view(-1,1) * g_prime) / (g**2).view(-1,1)
def Kernel_ThirdOrderDerivatives(kernel: KernelType, x: torch.Tensor, h:torch.Tensor):
    r_ij = torch.linalg.norm(x, dim = -1)
    r_eps = r_ij #+ 1e-5 * h
    eps = get_epsilon(x.dtype) * h
    r_eps = r_eps / h + eps

    q = r_ij / h
    dim = x.shape[1]

    vec = torch.tensor([3,5,7])[:dim]
    t = torch.einsum('ij, k -> ijk', torch.einsum('i, j -> ij', vec, vec), vec)

    k1 = eval_dkdq(kernel, q, dim) * eval_C_d(kernel, dim) / (h**(dim + 1))
    k2 = eval_d2kdq2(kernel, q, dim) * eval_C_d(kernel, dim) / (h**(dim + 2))
    k3 = eval_d3kdq3(kernel, q, dim) * eval_C_d(kernel, dim) / (h**(dim + 3))

    # The multiplication here is done to ensure that values dont get _extremely_ small
    # with **5 an r of 1e-7 would otherwise be the limit due to floating point precision
    # for single precision. this is a bit of a hack, but it works.
    termA =  3 * k1 / r_eps**5 / h**2
    termB = -    k1 / r_eps**3
    termC = -3 * k2 / r_eps**4 / h
    termD =  1 * k2 / r_eps**2 * h
    termE =  1 * k3 / r_eps**3

    outer_prod = torch.einsum('nij, nk -> nijk', torch.einsum('ni, nj -> nij', x, x), x)

    baseTerm = (termA + termC + termE).view(-1,1,1,1) * outer_prod 

    x = x[:,0][:,None]
    y = x[:,1][:,None] if dim > 1 else torch.zeros_like(x)
    z = x[:,2][:,None] if dim > 2 else torch.zeros_like(x)

    complexTerm = torch.zeros_like(baseTerm)
    complexTerm[:, t == 3*3*3] = 3 * x
    complexTerm[:, t == 5*5*5] = 3 * y 
    complexTerm[:, t == 7*7*7] = 3 * z 
    # x^2 terms
    complexTerm[:, t == 3*3*5] = y # xxy is missing a y
    complexTerm[:, t == 3*3*7] = z # xxz is missing a z
    # y^2 terms
    complexTerm[:, t == 3*5*5] = x # xyy is missing a x
    complexTerm[:, t == 5*5*7] = z # yyz is missing a z
    # z^2 terms
    complexTerm[:, t == 3*7*7] = x # xzz is missing a x
    complexTerm[:, t == 5*7*7] = y # yzz is missing a y

    finalTerm = baseTerm + complexTerm * (termB + termD).view(-1,1,1,1)
    return finalTerm / (h**3).view(-1,1,1,1)

def Kernel_DkDh(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
    dim = x.shape[1]
    r = torch.linalg.norm(x, dim = -1)
    q = r / h

    k = eval_k(kernel, q, dim)
    dkdq = eval_dkdq(kernel, q, dim)

    normConstant = -float(Kernel_C_d(kernel, dim))/ h ** float(dim + 2)

    return normConstant * (float(dim) * h * k + r * dkdq)
def Kernel_Grad_Dh(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
    dim = x.shape[1]
    r = torch.linalg.norm(x, dim = -1)
    q = r / h
    grad = torch.nn.functional.normalize(x, dim = -1)

    dkdq = eval_dkdq(kernel, q, dim)
    d2kdq2 = eval_d2kdq2(kernel, q, dim)

    factor = Kernel_C_d(kernel, dim) / h**(dim + 3)

    return -grad * (factor * (float(dim + 1) * h * dkdq + r * d2kdq2)).view(-1,1)
def Kernel_Laplacian_Dh(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
    dim = x.shape[1]
    r = torch.linalg.norm(x, dim = -1)
    eps = get_epsilon(x.dtype)
    r_eps = r + eps * h
    q = r / h
    k1 = eval_dkdq(kernel, q, dim) * eval_C_d(kernel, dim)
    k2 = eval_d2kdq2(kernel, q, dim) * eval_C_d(kernel, dim) 
    k3 = eval_d3kdq3(kernel, q, dim) * eval_C_d(kernel, dim) 

    ddh_k1 = - 1 / h**(dim + 3) * ((dim + 1) * h * k1 + r * k2)
    ddh_k2 = - 1 / h**(dim + 4) * ((dim + 2) * h * k2 + r * k3)

    s  =   (x**2).sum(dim=-1) / r_eps**2
    t  = - (x**2).sum(dim=-1) / r_eps**3
    t +=   dim                / r_eps

    laplacian = (s * ddh_k2 + t * ddh_k1)
    laplacian[q < 1e-5] = 0
    return laplacian
def Kernel_LaplacianApproximation_DH(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
    dim = x.shape[1]
    r_ij = torch.linalg.norm(x, dim = -1)
    eps = get_epsilon(x.dtype)
    r_eps = r_ij + eps * h

    gradW_ij = Kernel_Gradient(kernel, x, h)
    kernelNorm = torch.linalg.norm(gradW_ij, dim = -1)

    inner = Kernel_Grad_Dh(x, h)
    outer = gradW_ij / kernelNorm.view(-1,1)

    # print(inner.shape, outer.shape, r_eps.shape)

    return torch.einsum('nu, nu -> n ', inner, outer) / r_eps

