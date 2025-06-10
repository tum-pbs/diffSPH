from sphMath.modules.renorm import computeCovarianceMatrices
from sphMath.neighborhood import computeDistanceTensor
from sphMath.sphOperations.shared import scatter_sum
from sphMath.operations import sph_op, SPHOperationCompiled
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.sphOperations.shared import mod
# Maronne surface detection

import numpy as np
import torch


from torch.profiler import record_function
from sphMath.operations import sph_op, SPHOperation
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from typing import Dict, Tuple, Union
from sphMath.kernels import SPHKernel
from enum import Enum
from sphMath.util import getSetConfig

@torch.jit.script
def computeNormalsMaronne(
    particles : WeaklyCompressibleState,
    L: torch.Tensor,
    lambdas: torch.Tensor,
    neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
    supportScheme: SupportScheme = SupportScheme.Scatter):
    
    with record_function("[SPH] - [Surface Detection] - Compute Normals (Maronne)"):
        ones = particles.positions.new_ones(particles.positions.shape[0])
        term = SPHOperationCompiled(
            particles,
            quantity = (ones, ones),
            neighborhood= neighborhood[0],
            kernelValues = neighborhood[1],
            operation= Operation.Gradient,
            gradientMode = GradientMode.Naive,
            supportScheme= supportScheme
        )
        
        nu = torch.bmm(L, term.unsqueeze(-1)).squeeze(-1)
        n = -torch.nn.functional.normalize(nu, dim = -1)
        lMin = torch.min(torch.abs(lambdas), dim = -1).values
    
    return n, lMin


from sphMath.neighborhood import DomainDescription
# See Maronne et al: Fast free-surface detection and level-set function definition in SPH solvers
import math

@torch.jit.script
def detectFreeSurfaceMaronne(
    particles : WeaklyCompressibleState,
    normals: torch.Tensor,
    domain: DomainDescription,
    xi :float, 
    neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
    supportScheme: SupportScheme = SupportScheme.Scatter
    ):
    with record_function("[SPH] - [Surface Detection] - Detect Free Surface (Maronne)"):
        positions = particles.positions
        n = normals
        numParticles = positions.shape[0]
        supports = particles.supports
        periodicity = domain.periodic
        domainMin = domain.min
        domainMax = domain.max
        
        rij = neighborhood[1].r_ij
        i, j = neighborhood[0].row, neighborhood[0].col
        
        T = positions + n * supports.view(-1,1) / xi
        
        hij = supports[j]
        
        tau = torch.vstack((-n[:,1], n[:,0])).mT
        
        xjt = positions[j] - T[i]
        xjt = torch.stack([xjt[:,i_] if not periodicity[i_] else mod(xjt[:,i_], domainMin[i_], domainMax[i_]) for i_ in range(xjt.shape[1])], dim = -1)
        
        condA1 = rij >= math.sqrt(2) * hij / xi
        condA2 = torch.linalg.norm(xjt, dim = -1) <= hij / xi
        condA = (condA1 & condA2) & (i != j)
        cA = scatter_sum(condA, i, dim = 0, dim_size = numParticles)
        
        condB1 = rij < math.sqrt(2) * hij / xi
        condB2 = torch.abs(torch.einsum('ij,ij->i', -n[i], xjt)) + torch.abs(torch.einsum('ij,ij->i', tau[i], xjt)) < hij / xi
        condB = (condB1 & condB2) & (i != j)
        cB = scatter_sum(condB, i, dim = 0, dim_size = numParticles)
        
        fs = torch.where(~cA & ~cB & (torch.linalg.norm(n, dim = -1) > 0.5), 1.,0.)
        return fs, cA, cB
    
@torch.jit.script
def computeColorField(
    particles : WeaklyCompressibleState,
    neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
    supportScheme: SupportScheme = SupportScheme.Scatter
    ):
    with record_function("[SPH] - [Surface Detection] - Compute Color Field"):
        ones = particles.positions.new_ones(particles.positions.shape[0])
        term = SPHOperationCompiled(
            particles,
            quantity = (ones, ones),
            neighborhood= neighborhood[0],
            kernelValues = neighborhood[1],
            operation= Operation.Interpolate,
            supportScheme= supportScheme
        )
        termGrad = SPHOperationCompiled(
            particles,
            quantity = (term, term),
            neighborhood= neighborhood[0],
            kernelValues = neighborhood[1],
            operation= Operation.Gradient,
            gradientMode = GradientMode.Difference,
            supportScheme= supportScheme
        )
        
        return term, termGrad

@torch.jit.script
def detectFreeSurfaceColorField(
    particles : WeaklyCompressibleState,
    colorField : torch.Tensor,
    colorGrad : torch.Tensor,
    detectionThreshold : float,
    targetNeighbors: float,
    neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
    supportScheme: SupportScheme = SupportScheme.Scatter
    ):
    with record_function("[SPH] - [Surface Detection] - Detect Free Surface (Color Field)"):
        
        i, j = neighborhood[0].row, neighborhood[0].col
        nj = particles.numNeighbors
        if nj is None:
            raise ValueError("nj is None. Please check the neighborhood.")
        else:
            colorFieldMean = scatter_sum(colorField[j], i, dim = 0, dim_size = particles.positions.shape[0]) / nj
        
        fs = torch.where((colorField < colorFieldMean) & (nj < targetNeighbors * detectionThreshold), 1., 0.)
        return fs

@torch.jit.script
def detectFreeSurfaceColorFieldGradient(
    particles : WeaklyCompressibleState,
    colorField : torch.Tensor, 
    colorGrad: torch.Tensor,
    xi :float,
    colorFieldGradientThreshold : float,
    neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
    supportScheme: SupportScheme = SupportScheme.Scatter
    ):
    with record_function("[SPH] - [Surface Detection] - Detect Free Surface (Color Field Gradient)"):
        
        fs = torch.linalg.norm(colorGrad, dim = -1) > colorFieldGradientThreshold * particles.supports / xi
        return fs

# Barecasco et al 2013: Simple free-surface detection in two and three-dimensional SPH solver
@torch.jit.script
def detectFreeSurfaceBarecasco(
                               particles : WeaklyCompressibleState,
                               barecascoThreshold : float,
                               neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
                               supportScheme: SupportScheme = SupportScheme.Scatter,):
    with record_function("[SPH] - [Surface Detection] - Detect Free Surface (Barecasco)"):
        xij = neighborhood[1].x_ij
        i, j = neighborhood[0].row, neighborhood[0].col
        n_ij = torch.nn.functional.normalize(xij, dim = 1)
        
        coverVector = scatter_sum(-n_ij, i, dim = 0, dim_size = particles.positions.shape[0])
        normalized = torch.nn.functional.normalize(coverVector)
        angle = torch.arccos(torch.einsum('ij,ij->i', n_ij, normalized[i]))
        threshold = barecascoThreshold
        condition = ((angle <= threshold / 2) & (i != j)) | (torch.linalg.norm(normalized, dim = -1)[i] <= 0.5)
        # condition = (torch.linalg.norm(normalized, dim = -1)[i] <= 0.5)
        fs = ~scatter_sum(condition, i, dim = 0, dim_size = particles.positions.shape[0])
        return fs , normalized

@torch.jit.script
def expandFreeSurfaceMask(fs, expansionIters: int, neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood]):
    with record_function("[SPH] - [Surface Detection] - Expand Free Surface Mask"):
        # expansionIters = solverConfig.get('surfaceDetection', {}).get('expansionIters', 1)
        
        fsm = fs.clone()
        for i in range(expansionIters):
            fsm = scatter_sum(fsm[neighborhood[0].col], neighborhood[0].row, dim = 0, dim_size = fsm.shape[0])
        return fsm

from sphMath.enums import KernelCorrectionScheme
@torch.jit.script
def computeLambdaGrad(
    particles : WeaklyCompressibleState,
    L: torch.Tensor,
    lambdas: torch.Tensor,
    neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
    supportScheme: SupportScheme = SupportScheme.Scatter
):
    with record_function("[SPH] - [Surface Detection] - Compute Lambda Grad"):
        return torch.nn.functional.normalize(SPHOperationCompiled(
            particles,
            quantity = (lambdas, lambdas),
            neighborhood= neighborhood[0],
            kernelValues = neighborhood[1],
            operation= Operation.Gradient,
            gradientMode = GradientMode.Difference,
            supportScheme= supportScheme,
            correctionTerms=[KernelCorrectionScheme.gradientRenorm]
        ))
        # return sph_op(particles, particles, domain, wrappedKernel, sparseNeighborhood, operation = 'gradient', gradientMode = 'difference', supportScheme = supportScheme, correctionTerms=[KernelCorrectionScheme.gradientRenorm], quantity = (lambdas, lambdas))



class SurfaceDetectionSchemes(Enum):
    Maronne = 'Maronne'
    Barecasco = 'Barecasco'
    ColorGradient = 'colorGrad'
    
class NormalSchemes(Enum):
    Lambda = 'lambda'
    Color = 'color'
    
from sphMath.modules.renorm import computeCovarianceMatrices_
from typing import Optional

@torch.jit.script
def surfaceDetection_(
        particles: WeaklyCompressibleState,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        kernelXi: float,
        targetNeighbors: float,
        colorSurfaceThreshold: float, 
        colorGradientSurfaceThreshold: float,
        barecascoThreshold: float,
        fsIter: int,
        domain: DomainDescription,
        supportScheme: SupportScheme = SupportScheme.Scatter,
        surfaceDetectionScheme: SurfaceDetectionSchemes = SurfaceDetectionSchemes.Barecasco,
        surfaceNormalScheme: NormalSchemes = NormalSchemes.Lambda,
        computeNormals: bool = False,):
    # particles, domain, wrappedKernel, actualNeighbors, config, freeSurfaceConfig, computeNormals = True):
    with record_function("[SPH] - [Surface Detection]"):
        n: Optional[torch.Tensor] = None
        lMin: Optional[torch.Tensor]  = None
        with record_function("[SPH] - [Surface Detection] - Compute Surface"):
            if surfaceDetectionScheme == SurfaceDetectionSchemes.Maronne:
                if particles.covarianceMatrices is None or particles.gradCorrectionMatrices is None or particles.eigenValues is None:
                    C, L, eVals = computeCovarianceMatrices_(particles, neighborhood, supportScheme)
                    particles.gradCorrectionMatrices = L
                else:
                    C = particles.covarianceMatrices
                    L = particles.gradCorrectionMatrices
                    eVals = particles.eigenValues
                if L is None or eVals is None:
                    raise ValueError("L or eVals is None. Please check the covariance matrices.")
                    
                n, lMin = computeNormalsMaronne(particles, L, eVals, neighborhood, supportScheme)
                fs, cA, cB = detectFreeSurfaceMaronne(particles, n, domain, kernelXi, neighborhood, supportScheme)
                # if lMin is not None:
                #     n = computeLambdaGrad(particles, L, lMin, neighborhood, supportScheme)
                
            elif surfaceDetectionScheme == SurfaceDetectionSchemes.ColorGradient:
                color, colorGrad = computeColorField(particles, neighborhood, supportScheme)
                fs = detectFreeSurfaceColorFieldGradient(particles, color, colorGrad, kernelXi, colorGradientSurfaceThreshold, neighborhood, supportScheme)
                n = torch.nn.functional.normalize(colorGrad, dim=1)
                
            elif surfaceDetectionScheme == SurfaceDetectionSchemes.Barecasco:
                fs, n = detectFreeSurfaceBarecasco(particles, barecascoThreshold, neighborhood, supportScheme)
            else:
                raise ValueError(f'Unknown surface detection scheme {surfaceDetectionScheme}')
            fsMask = expandFreeSurfaceMask(fs, fsIter, neighborhood)

        if computeNormals:
            with record_function("[SPH] - [Surface Detection] - Compute Normals"):
                if (surfaceNormalScheme == NormalSchemes.Lambda and surfaceDetectionScheme == SurfaceDetectionSchemes.Maronne) or (surfaceNormalScheme==NormalSchemes.Color and surfaceDetectionScheme == SurfaceDetectionSchemes.ColorGradient):
                    pass
                else:
                    if surfaceNormalScheme == NormalSchemes.Lambda:
                        if particles.covarianceMatrices is None or particles.gradCorrectionMatrices is None or particles.eigenValues is None:
                            C, L, eVals = computeCovarianceMatrices_(particles, neighborhood, supportScheme)
                            particles.gradCorrectionMatrices = L
                        else:
                            C = particles.covarianceMatrices
                            L = particles.gradCorrectionMatrices
                            eVals = particles.eigenValues
                        if L is None or eVals is None:
                            raise ValueError("L or eVals is None. Please check the covariance matrices.")
                        n, lMin = computeNormalsMaronne(particles, L, eVals, neighborhood, supportScheme)
                        n = computeLambdaGrad(particles, L, lMin, neighborhood, supportScheme)
                        if lMin is not None:
                            n = torch.nn.functional.normalize(n, dim=1)
                        
                    elif surfaceNormalScheme == NormalSchemes.Color:
                        color, colorGrad = computeColorField(particles, neighborhood, supportScheme)
                        n = torch.nn.functional.normalize(colorGrad, dim=1)
                    else:
                        raise ValueError(f'Unknown normal scheme {surfaceNormalScheme}')
                                
                if lMin is None:
                    if particles.covarianceMatrices is None or particles.gradCorrectionMatrices is None or particles.eigenValues is None:
                        C, L, eVals = computeCovarianceMatrices_(particles, neighborhood, supportScheme)
                        particles.gradCorrectionMatrices = L
                    else: 
                        C = particles.covarianceMatrices
                        L = particles.gradCorrectionMatrices
                        eVals = particles.eigenValues
                    if L is None or eVals is None:
                        raise ValueError("L or eVals is None. Please check the covariance matrices.")
                    
                    n, lMin = computeNormalsMaronne(particles, L, eVals, neighborhood, supportScheme)

        return fs, fsMask, n, lMin
    
    

from sphMath.kernels import KernelType, Kernel_Scale

def surfaceDetection(
        particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: KernelType,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {},
        computeNormals: bool = False):
        freeSurfaceScheme = getSetConfig(config, 'surfaceDetection', 'surfaceDetection', 'Barecasco')
        normalScheme = getSetConfig(config, 'surfaceDetection', 'normalScheme', 'Lambda')
        # computeNormals = getSetConfig(config, 'surfaceDetection', 'computeNormals', False)
        
        colorSurfaceThreshold = getSetConfig(config, 'surfaceDetection', 'colorSurfaceThreshold', 1.5)
        colorGradientSurfaceThreshold = getSetConfig(config, 'surfaceDetection', 'colorGradientSurfaceThreshold', 10.0)
        barecascoThreshold = getSetConfig(config, 'surfaceDetection', 'barecascoThreshold', np.pi/3)
        fsIter = getSetConfig(config, 'surfaceDetection', 'expansionIters', 1)
        targetNeighbors = config['targetNeighbors']
        
        neighborhood = (neighborhood[0](neighborhood[0], kernel)) if neighborhood[1] is None else neighborhood
    
        freeSurfaceSchemeEnum = SurfaceDetectionSchemes[freeSurfaceScheme]
        normalSchemeEnum = NormalSchemes[normalScheme]
        return surfaceDetection_(
            particles,
            neighborhood,
            Kernel_Scale(kernel, particles.positions.shape[1]),
            targetNeighbors,
            colorSurfaceThreshold,
            colorGradientSurfaceThreshold,
            barecascoThreshold,
            fsIter,
            neighborhood[0].domain,
            supportScheme,
            freeSurfaceSchemeEnum,
            normalSchemeEnum,
            computeNormals
        )