from torch.profiler import profile, record_function, ProfilerActivity
from sphMath.neighborhood import filterNeighborhood, coo_to_csr, evaluateNeighborhood, SupportScheme
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
from sphMath.modules.shifting.implicitShifting import computeShifting
import copy
from sphMath.neighborhood import buildNeighborhood, SparseNeighborhood, filterNeighborhoodByKind
from sphMath.modules.surfaceDetection import surfaceDetection
from torch.profiler import record_function
from sphMath.operations import SPHOperation, Operation, GradientMode
from sphMath.modules.mDBC import mDBCDensity

from torch.utils.checkpoint import checkpoint

def solveShifting(SPHSystem, dt, config, verbose = False):
    with record_function("[SPH] - [Shifting]"):
        with record_function("[SPH] - [Shifting] - Init"):
            domain          = config['domain']
            wrappedKernel   = config['kernel']
            particles       = SPHSystem.systemState.clone()
            neighborhood    = SPHSystem.neighborhoodInfo
            
            shiftIters = config.get('shifting', {}).get('maxIterations', 1)
            summationDensity = config.get('shifting', {}).get('summationDensity', False)
            freeSurface = config.get('shifting', {}).get('freeSurface', False)
            freeSurfaceScheme = config.get('shifting', {}).get('surfaceDetection', 'Barecasco')
            shiftingScheme = config.get('shifting', {}).get('scheme', 'delta')
            normalScheme = config.get('shifting', {}).get('normalScheme', 'lambda')
            projectionScheme = config.get('shifting', {}).get('projectionScheme', 'mat')
            surfaceScaling = config.get('shifting', {}).get('surfaceScaling', 0.1)
            shiftingThreshold = config.get('shifting', {}).get('threshold', 0.5)
            rho0 = config.get('fluid', {}).get('rho0', 1)
            spacing = config.get('particle', {}).get('dx', torch.pow(particles.masses / rho0, 1/particles.positions.shape[1]).mean().cpu().item())
                    
            projectQuantities = config.get('shifting', {}).get('projectQuantities', False)

        neighbors0 = None

        overallStates = []
        for i in range(shiftIters):      
            with record_function("[SPH] - [Shifting] - Neighborsearch"):      
                neighborhood, neighbors = evaluateNeighborhood(particles, domain, wrappedKernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=neighborhood, computeHessian=config['neighborhood']['computeHessian'] if shiftingScheme != 'IPS' else True, computeDkDh=config['neighborhood']['computeDkDh'])
                particles.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries
                if i == 0:
                    neighbors0 = neighbors
                particles.covarianceMatrices = None
                particles.gradCorrectionMatrices = None
                particles.eigenValues = None
                
            with record_function("[SPH] - [Shifting] - Density"):
                if summationDensity:
                    particles.densities = computeDensity(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather)
                if torch.any(particles.kinds > 1):
                    with record_function("[deltaSPH] - 03 - mDBC"):
                        particles.densities,_ = mDBCDensity(particles, wrappedKernel, neighbors.get('fluidToGhost'), SupportScheme.Scatter, config, False)

                
                
            with record_function("[SPH] - [Shifting] - Surface Detection"):
                particles.covarianceMatrices = None
                if freeSurface:
                    fs, fsMask, n, lMin = surfaceDetection(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config, True)
                else:
                    fs = fsMask = n = lMin = None

            with record_function("[SPH] - [Shifting] - Compute Shift"):
                if shiftingScheme == 'IPS':
                    update, K, J, H, B, convergence, iters, residual = computeShifting(particles, domain, wrappedKernel, neighbors.get('noghost')[0], config, freeSurface = fs, freeSurfaceMask = fsMask)
                    overallStates.append((convergence, iters, residual))
                else:
                    update = -checkpoint(computeDeltaShifting, particles, domain, wrappedKernel, neighbors.get('noghost')[0], config, use_reentrant=True)
                    # update = -computeDeltaShifting(particles, domain, wrappedKernel, neighbors.get('noghost')[0], config)
            
            with record_function("[SPH] - [Shifting] - Project Shift"):
                if freeSurface:                
                    if projectionScheme == 'dot':
                        result = update - torch.einsum('ij,ij->i', update, n).view(-1,1) * n
                        update[fsMask > 0.5] = result[fsMask > 0.5] * surfaceScaling
                        update[lMin < 0.4] = 0
                    elif projectionScheme == 'mat':
                        nMat = torch.einsum('ij, ik -> ikj', n, n)
                        M = torch.diag_embed(particles.positions.new_ones(particles.positions.shape)) - nMat
                        result = torch.bmm(M, update.unsqueeze(-1)).squeeze(-1)
                        update[fsMask > 0.5] = result[fsMask > 0.5]
                        update[lMin < 0.4] = 0
                        update[fs > 0.5] = update[fs> 0.5] * surfaceScaling
                    else:
                        update[lMin < 0.4] = 0
                        update[fs > 0.5] = 0
                    
            with record_function("[SPH] - [Shifting] - Apply Shift"):
                spacing = torch.pow(particles.masses / rho0, 1/particles.positions.shape[1])
                spacing = config['particle']['dx']
                update = torch.clamp(update, -shiftingThreshold * spacing, shiftingThreshold * spacing)
                update[particles.kinds != 0] = 0
                particles.positions -= update
                        
        dx = particles.positions - SPHSystem.systemState.positions

        if projectQuantities:
            with record_function("[SPH] - [Shifting] - Project Quantities"):
                ftfNeighbors = neighbors0.get('fluid')
                velocityGradient = SPHOperation(particles, SPHSystem.systemState.velocities, config['kernel'], ftfNeighbors[0], ftfNeighbors[1], Operation.Gradient, SupportScheme.Scatter, GradientMode.Difference)

                projectedVelocity = torch.einsum('ijk,ij->ik', velocityGradient, dx)
                velocities = SPHSystem.systemState.velocities - projectedVelocity

                densityGradient = SPHOperation(particles, SPHSystem.systemState.densities, config['kernel'], ftfNeighbors[0], ftfNeighbors[1], Operation.Gradient, SupportScheme.Scatter, GradientMode.Difference)
                projectedDensity = torch.einsum('ij,ij->i', densityGradient, dx)
                densities = SPHSystem.systemState.densities - projectedDensity
        else:
            velocities = SPHSystem.systemState.velocities
            densities = SPHSystem.systemState.densities
        
        return dx, neighborhood, neighbors, overallStates, densities, velocities, fs, n, lMin
            
            
from sphMath.schemes.deltaSPH import DeltaPlusSPHSystem
def shuffleParticles(particleState, config, shiftIters = 256):
    kindMask = particleState.kinds == 0
    particleState.positions[kindMask] += torch.randn_like(particleState.positions)[kindMask] * particleState.supports[kindMask].view(-1,1) * 0.01

    configS = {
        'domain': config['domain'],
        'kernel': config['kernel'],
        'targetNeighbors': config['targetNeighbors'],
        'neighborhood': config['neighborhood'],
        'verletScale': 1.4,
        'particle': {'dx': config['particle']['dx'], 'support': particleState.supports.max().item()},
        'shifting': {'scheme': 'delta',
                    'threshold': 5,
                    'computeMach': False,
                    'summationDensity': True
                    },
        'fluid':config['fluid'],
        'particle': config['particle'],
    }
    particleSystem = DeltaPlusSPHSystem(config['domain'], None, 0., copy.deepcopy(particleState), 'momentum', None, rigidBodies = config['rigidBodies'], regions = config['regions'], config = config)

    for i in range(shiftIters):
        dx, *_ = solveShifting(particleSystem, 0.1, configS)
        particleSystem.systemState.positions[kindMask,:] += dx[kindMask,:]
        print(f'[{i:03d}] dx = {torch.linalg.norm(dx[kindMask,:], dim = 1).mean().item():.4f}')
    return particleSystem.systemState.positions