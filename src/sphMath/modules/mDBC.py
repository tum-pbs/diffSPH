from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.neighborhood import filterNeighborhoodByKind, buildNeighborhood, filterNeighborhood, computeDistanceTensor, coo_to_csr, DomainDescription, SparseCOO
from sphMath.operations import sph_op
import torch
from typing import Union, Tuple, Dict
from torch.profiler import record_function
from sphMath.operations import sph_op, SPHOperation, SPHOperationCompiled, access_optional
from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.gasDynamics import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from sphMath.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from sphMath.boundary import evalGhostQuantity
from sphMath.kernels import SPHKernel
from sphMath.sphOperations.shared import scatter_sum

@torch.jit.script
def mDBCDensity_(particles: WeaklyCompressibleState,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        rho0: float,
        neighCounts: torch.Tensor,
        supportScheme: SupportScheme = SupportScheme.Scatter,
        clampDensity: bool = True):
    if particles.ghostIndices is None:
        return particles.densities, None
    else:
        with record_function("[SPH] - [mDBC]"):
            with record_function("[SPH] - [mDBC] - shepard"):            
                shepardNominator = SPHOperationCompiled(
                    particles,
                    quantity = particles.densities,
                    neighborhood= neighborhood[0],
                    kernelValues = neighborhood[1],
                    operation= Operation.Interpolate,
                    supportScheme= supportScheme
                )
                shepardDenominator = SPHOperationCompiled(
                    particles,
                    quantity = torch.ones_like(particles.densities),
                    neighborhood= neighborhood[0],
                    kernelValues = neighborhood[1],
                    operation= Operation.Interpolate,
                    supportScheme= supportScheme
                )
                shepardDensity = shepardNominator / shepardDenominator
                shepardDensity = torch.where(shepardDenominator > 0, shepardDensity, rho0)
            with record_function("[SPH] - [mDBC] - b"):    
                gradientSum = SPHOperationCompiled(
                    particles,
                    quantity = particles.densities,
                    neighborhood= neighborhood[0],
                    kernelValues = neighborhood[1],
                    operation= Operation.Gradient,
                    gradientMode = GradientMode.Naive,
                    supportScheme= supportScheme
                )            
                b = torch.hstack((shepardNominator[:,None], gradientSum))
            with record_function("[SPH] - [mDBC] - A_g"):
                volumeSum = SPHOperationCompiled(
                    particles,
                    quantity = torch.ones_like(particles.densities),
                    neighborhood= neighborhood[0],
                    kernelValues = neighborhood[1],
                    operation= Operation.Interpolate,
                    supportScheme= supportScheme
                )
                volumeGradSum = SPHOperationCompiled(
                    particles,
                    quantity = torch.ones_like(particles.densities),
                    neighborhood= neighborhood[0],
                    kernelValues = neighborhood[1],
                    operation= Operation.Gradient,
                    gradientMode = GradientMode.Naive,
                    supportScheme= supportScheme
                )

                r_ij, x_ij = neighborhood[1].r_ij, neighborhood[1].x_ij

                positionSum = SPHOperationCompiled(
                    particles,
                    quantity = x_ij,
                    neighborhood= neighborhood[0],
                    kernelValues = neighborhood[1],
                    operation= Operation.Interpolate,
                    supportScheme= supportScheme
                )
                positionMatrix = SPHOperationCompiled(
                    particles,
                    quantity = x_ij,
                    neighborhood= neighborhood[0],
                    kernelValues = neighborhood[1],
                    operation= Operation.Gradient,
                    gradientMode = GradientMode.Naive,
                    supportScheme= supportScheme
                )
                            
                positions = particles.positions
                dtype = positions.dtype
                device = positions.device
                numPtcls = positions.shape[0]
                dim = positions.shape[1]

                A_g = torch.zeros((particles.positions.shape[0], 3, 3), dtype = dtype, device = device)

                A_g[:,0,0] = volumeSum
                A_g[:,1,0] = volumeGradSum[:,0]
                A_g[:,2,0] = volumeGradSum[:,1]

                A_g[:,0,1] = positionSum[:,0]
                A_g[:,0,2] = positionSum[:,1]

                A_g[:,1,1] = positionMatrix[:,0,0]
                A_g[:,1,2] = positionMatrix[:,0,1]
                A_g[:,2,1] = positionMatrix[:,1,0]
                A_g[:,2,2] = positionMatrix[:,1,1]
                
                ghostMask = particles.kinds == 2

                assert torch.all(particles.kinds[~ghostMask] != 2)
                assert torch.all(A_g[~ghostMask,:,:] == 0)
                assert torch.all(b[~ghostMask,:] == 0)
                assert torch.all(particles.kinds[ghostMask] == 2)

                A_g = A_g[ghostMask,:,:]
                b = b[ghostMask,:]
                
                A_g_inv = torch.zeros_like(A_g)
                neighCounts = scatter_sum(torch.ones_like(particles.densities)[neighborhood[0].col], neighborhood[0].row, dim = 0, dim_size = particles.densities.shape[0])[ghostMask]


            with record_function("[SPH] - [mDBC] - A_g_inv"):
                A_g_inv[neighCounts > 4] = torch.linalg.pinv(A_g[neighCounts > 4])

            with record_function("[SPH] - [mDBC] - solve"):
                res = torch.matmul(A_g_inv, b.unsqueeze(2))[:,:,0]
                restDensity = rho0

                bIndices = access_optional(particles.ghostIndices, ghostMask)
                boundaryDensity = torch.ones(numPtcls, dtype = dtype, device = device) * restDensity
                boundaryDensity[bIndices] = torch.where(neighCounts > 0, shepardDensity[ghostMask], restDensity) #/ restDensity
                threshold = 5

                assert torch.all(particles.kinds[bIndices] == 1)
                assert torch.all(shepardDensity[~ghostMask] == rho0)
                assert torch.all(shepardDensity[bIndices] == rho0)
                # assert torch.all(bIndices == )

                # boundaryParticlePositions = perennialState['boundary']['positions']
                # ghostParticlePositions = boundaryGhostState['positions']
                relPos = access_optional(particles.ghostOffsets, ghostMask)
                relDist = torch.linalg.norm(relPos, dim = 1)
                # relDist = torch.clamp(relDist, min = 1e-7, max = config['particle']['support']*3.)
                # relPos = relPos * (relDist / (torch.linalg.norm(relPos, dim = 1) + 1e-7))[:,None]
                
                # print()

                # print(boundaryDensity[bIndices])
                # print(res.shape)
                # print(neighCounts.shape)
                # print(bIndices.shape)

                # boundaryDensity[bIndices] = torch.where(neighCounts > threshold, (shepardDensity[ghostMask] - torch.einsum('nu, nu -> n',(relPos), res[:, 1:] )), boundaryDensity[bIndices])
                boundaryDensity[bIndices] = torch.where(neighCounts > threshold, (res[:,0] - torch.einsum('nu, nu -> n',(relPos), res[:, 1:] )), boundaryDensity[bIndices])
                
                # boundaryDensity[bIndices] = torch.where(neighCounts == 0, restDensity, boundaryDensity[bIndices])
                # if clampDensity:ffmp
                    # boundaryDensity = torch.clamp(boundaryDensity, min = restDensity)
        # self.fluidVolume = self.boundaryVolume / self.boundaryDensity

        # solution, M, b = LiuLiuConsistent(boundaryGhostState, perennialState['fluid'], perennialState['fluid']['densities'])
        # boundaryDensity = 

        
        # print(boundaryDensity[bIndices])
        
        # boundaryDensity = torch.ones(numPtcls, dtype = xij.dtype, device = xij.device) * restDensity
        # boundaryDensity[neighCounts > 0] = shepardDensity[neighCounts > 0] #/ restDensity
        # threshold = 5
                assert torch.all(boundaryDensity[particles.kinds == 0] == rho0)
                assert torch.all(boundaryDensity[ghostMask] == rho0)
        
                mergedDensitities = particles.densities.clone()
                mergedDensitities[bIndices] = boundaryDensity[bIndices]

                # mergedDensitities[bIndices] = rho0
        
                assert torch.all(mergedDensitities[particles.kinds == 0] == particles.densities[particles.kinds == 0])
        return mergedDensitities, boundaryDensity



def mDBCDensity(particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {},
        clampDensity: bool = True):
    ghostMask = particles.kinds == 2
    neighCounts = coo_to_csr(neighborhood[0]).rowEntries[ghostMask]
    rho0 = config['fluid']['rho0']

    with record_function("[SPH] - [mDBC] - density"):
        return mDBCDensity_(
            particles,
            neighborhood,
            rho0,
            neighCounts,
            supportScheme,
            clampDensity
        )