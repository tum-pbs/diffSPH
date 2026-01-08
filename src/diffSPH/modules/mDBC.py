from diffSPH.schemes.states.wcsph import WeaklyCompressibleState
from diffSPH.neighborhood import filterNeighborhoodByKind, buildNeighborhood, filterNeighborhood, computeDistanceTensor, coo_to_csr, DomainDescription, SparseCOO
from diffSPH.operations import sph_op
import torch
from typing import Union, Tuple, Dict
from torch.profiler import record_function
from diffSPH.operations import sph_op, SPHOperation, SPHOperationCompiled, access_optional
from diffSPH.schemes.states.wcsph import WeaklyCompressibleState
from diffSPH.schemes.gasDynamics import CompressibleState
from diffSPH.neighborhood import SparseNeighborhood, PrecomputedNeighborhood, SupportScheme
from diffSPH.operations import DivergenceMode, GradientMode, Operation, LaplacianMode
from diffSPH.boundary import evalGhostQuantity
from diffSPH.kernels import SPHKernel
from diffSPH.sphOperations.shared import scatter_sum

@torch.jit.script
def mDBCDensity_(particles: WeaklyCompressibleState,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        rho0: float,
        c_s: float,
        gravity: torch.Tensor,
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

                # densityGradSum = SPHOperationCompiled(
                #     particles,
                #     quantity = particles.densities,
                #     neighborhood= neighborhood[0],
                #     kernelValues = neighborhood[1],
                #     operation= Operation.Gradient,
                #     gradientMode = GradientMode.Naive,
                #     supportScheme= supportScheme
                # )

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
                # densityGradSum = densityGradSum[ghostMask,:]
                
                A_g_inv = torch.zeros_like(A_g)
                neighCounts = scatter_sum(torch.ones_like(particles.densities)[neighborhood[0].col], neighborhood[0].row, dim = 0, dim_size = particles.densities.shape[0])[ghostMask]


            with record_function("[SPH] - [mDBC] - A_g_inv"):
                A_g_inv[neighCounts > 4] = torch.linalg.pinv(A_g[neighCounts > 4])

            with record_function("[SPH] - [mDBC] - solve"):
                res = torch.matmul(A_g_inv, b.unsqueeze(2))[:,:,0]
                restDensity = rho0

                # print(f'[mDBC] - A_g: {A_g.shape}, b: {b.shape}, A_g_inv: {A_g_inv.shape}, res: {res.shape}')
                # v = torch.hstack((shepardNominator[ghostMask].view(-1,1), densityGradSum))
                # print(f'[mDBC] - v: {v.shape}')
                # res2 = torch.matmul(A_g_inv, v.unsqueeze(2))
                # print(f'[mDBC] - res2: {res2.shape}')

                bIndices = access_optional(particles.ghostIndices, ghostMask)
                boundaryDensity = torch.ones(numPtcls, dtype = dtype, device = device) * restDensity
                boundaryDensity[bIndices] = torch.where(neighCounts > 0, shepardDensity[ghostMask], restDensity) #/ restDensity
                threshold = 9

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

                # boundaryDensity[bIndices] = torch.where(neighCounts > threshold, res[:,0], boundaryDensity[bIndices])

                c_0 = 34.5
                c_0 = c_s
                rho_0 = 1
                rho_0 = rho0

                rho_g = boundaryDensity[bIndices]
                P_g = c_0**2 * (rho_g - rho_0)
                

                g = gravity
                nb = -relPos
                # This normalization is not in the paper https://www.sciencedirect.com/science/article/pii/S0045793025003305?via%3Dihub
                # But it is correct, see the dualsphysics code and thanks Aaron!
                nb = torch.nn.functional.normalize(nb, dim = 1)

                dot = torch.einsum('ni, i -> n', nb, g)
                dot2 = torch.einsum('ni, ni -> n', relPos, nb)

                mask = neighCounts > threshold

                # print(f'[mDBC] - max dot: {torch.max(dot[mask]).item()}, min dot: {torch.min(dot[mask]).item()}')
                # print(f'[mDBC] - max dot2: {torch.max(dot2[mask]).item()}, min dot2: {torch.min(dot2[mask]).item()}')
                # print(f'[mDBC] - dot * dot2 stats: max {torch.max(dot[mask] * dot2[mask]).item()}, min {torch.min(dot[mask] * dot2[mask]).item()}, mean {torch.mean(dot[mask] * dot2[mask]).item()}')

                P_b = P_g + rho_0 * (dot * dot2)
                rho_b = rho_0 + P_b / c_0**2

                # rho_b /= rho_0
                boundaryDensity[bIndices] = rho_b
                
                boundaryDensity[bIndices] = torch.where(neighCounts > threshold, (res[:,0] - torch.einsum('nu, nu -> n',(relPos), res[:, 1:] )), boundaryDensity[bIndices])

                # boundaryDensity[bIndices] = torch.where(neighCounts > threshold, rho_b, boundaryDensity[bIndices])
                
                
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
    c_s = config['fluid']['c_s']

    gravity = torch.tensor(config['gravity']['direction'], dtype = particles.positions.dtype, device = particles.positions.device) * config['gravity']['magnitude']

    with record_function("[SPH] - [mDBC] - density"):
        return mDBCDensity_(
            particles,
            neighborhood,
            rho0,
            c_s,
            gravity,
            neighCounts,
            supportScheme,
            clampDensity
        )
    

def mDBCPenetrationCheck(particles: Union[CompressibleState, WeaklyCompressibleState],
        kernel: SPHKernel,
        neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
        supportScheme: SupportScheme = SupportScheme.Scatter,
        config: Dict = {},
        clampDensity: bool = True        
):
    
    neighbors, kernelValues = neighborhood
    fluidIndices = neighbors.row 
    boundaryIndices = neighbors.col

    boundaryNormal = particles.ghostOffsets[boundaryIndices]
    n_b = torch.linalg.norm(boundaryNormal, dim = 1,)

    # boundaryNormal = boundaryNormal / (n_b[:,None] + 1e-12)

    kind_i = particles.kinds[fluidIndices]
    kind_j = particles.kinds[boundaryIndices]

    if not torch.all(kind_i == 0):
        print(f'[mDBC] - Warning: fluidIndices contains non-fluid particles: {kind_i}')
    if not torch.all(kind_j == 1):
        print(f'[mDBC] - Warning: boundaryIndices contains non-boundary particles: {kind_j}')   

    # print(f'n_b stats: min {torch.min(n_b).item()}, max {torch.max(n_b).item()}, mean {torch.mean(n_b).item()}')

    x_ib = particles.positions[fluidIndices] - particles.positions[boundaryIndices]
    r_ib = torch.linalg.norm(x_ib, dim = 1)
    # print(f'r_ib stats: min {torch.min(r_ib).item()}, max {torch.max(r_ib).item()}, mean {torch.mean(r_ib).item()}')

    dp = config['particle']['dx']

    # r_ib = rrmag

    check_a = r_ib < 1.25 * dp

    norm = torch.linalg.norm(boundaryNormal, dim = 1)
    normalized = boundaryNormal / (norm[:,None] + 1e-12)
    normdist = torch.einsum('ni, ni -> n', x_ib, normalized).abs()

    check_b = torch.logical_and(normdist < 0.75 * norm, norm < 1.75 * dp)
    # check_b = normdist < 0.75 * norm

    # print(f'Dot: {torch.einsum("ni, ni -> n", x_ib, boundaryNormal)}')

    check_c = torch.einsum('ni, ni -> n', particles.velocities[fluidIndices] - particles.velocities[boundaryIndices], boundaryNormal) < 0.0

    # print(f'[mDBC] - Total checks: {check_a.shape[0]}')
    # print(f'[mDBC] - Check A (r_ib > 1.25 * n_b): {torch.sum(check_a).item()} [check_a shape: {check_a.shape}]')
    # print(f'[mDBC] - Check B (dot(x_ib, n_b) > 0.75 * n_b): {torch.sum(check_b).item()} [check_b shape: {check_b.shape}]')
    # print(f'[mDBC] - Check C (dot(v_i - v_b, n_b) > 0): {torch.sum(check_c).item()} [check_c shape: {check_c.shape}]')

    check_ab = torch.logical_and(check_a, check_b)
    check_abc = torch.logical_and(check_ab, check_c)

    # print(f'[mDBC] - Check AB (A and B): {torch.sum(check_ab).item()}')
    # print(f'[mDBC] - Check ABC (A and B and C): {torch.sum(check_abc).item()}')

    # penetrationMask = torch.logical_and(
    #     (check_a),
    #     torch.logical_and(
    #     (check_b), 
    #     (check_c)
    #     )
    # )

    adjustedVelocities = particles.velocities.clone()[fluidIndices]

    # print(f'[mDBC] - Penetration corrections: {torch.sum(penetrationMask).item()} / {penetrationMask.shape[0]}')

    # print(f'Mask shape: {penetrationMask.shape}')
    # print(f'adjusted velocities shape: {adjustedVelocities.shape}')
    # print(f'boundary normal shape: {boundaryNormal.shape}')
    # print(f'r_ib shape: {r_ib.shape}')

    nopenshift = torch.zeros_like(adjustedVelocities)
    nopencount = torch.zeros(adjustedVelocities.shape, dtype = torch.int32, device = adjustedVelocities.device)

    nopen_mask_a = torch.zeros(adjustedVelocities.shape, dtype = torch.int32, device = adjustedVelocities.device)
    nopen_mask_b = torch.zeros(adjustedVelocities.shape, dtype = torch.int32, device = adjustedVelocities.device)
    nopen_mask_c = torch.zeros(adjustedVelocities.shape, dtype = torch.int32, device = adjustedVelocities.device)

    for d in range(particles.positions.shape[1]):
        u_i = particles.velocities[fluidIndices][:,d]
        u_b = particles.velocities[boundaryIndices][:,d]

        # print((r_ib / n_b).shape)

        norm = -boundaryNormal[:,d]
        dr = x_ib[:,d]
        absx = normalized[:,d].abs()

        mask = check_ab

        mask_a = torch.logical_and(dr * -normalized[:,d] < 0.75, absx > 0.001 * dp)

        nopen_mask_a[:,d] = mask_a.int()
        nopen_mask_b[:,d] = mask.int()

        mask = torch.logical_and(mask, mask_a)
        nopen_mask_c[:,d] = mask.int()

        dv = u_i - u_b
        vfc = dv * norm

        mask_b = torch.logical_and(mask, vfc < 0)

        ratio = torch.clamp((dr / norm).abs(), min = 0.25)
        # ratio = torch.ones_like(ratio) *0.25
        factor = - 4 * ratio + 3

        nopenshiftTerm = -factor * dv * norm * norm
        if torch.sum(mask_b) == 0:
            # print(f'[mDBC] - Direction {d}: No penetration corrections applied.')
            continue
        # print(f'[mDBC] - Direction {d}: Applying {torch.sum(mask_b).item()} penetration corrections.')
        # print(f'[mDBC] - Direction {d}: Max correction magnitude: {torch.max(nopenshiftTerm[mask_b].abs()).item():.6f}')
        # print(f'[mDBC] - Direction {d}: Avg correction magnitude: {torch.mean(nopenshiftTerm[mask_b].abs()).item():.6f}')
        # print(f'[mDBC] - dv: max {torch.max(dv[mask_b]).item():.6f}, min {torch.min(dv[mask_b]).item():.6f}, mean {torch.mean(dv[mask_b]).item():.6f}')
        # print(f'[mDBC] - norm: max {torch.max(norm[mask_b]).item():.6f}, min {torch.min(norm[mask_b]).item():.6f}, mean {torch.mean(norm[mask_b]).item():.6f}')
        # print(f'[mDBC] - ratio: max {torch.max(ratio[mask_b]).item():.6f}, min {torch.min(ratio[mask_b]).item():.6f}, mean {torch.mean(ratio[mask_b]).item():.6f}')
        # print(f'[mDBC] - factor: max {torch.max(factor[mask_b]).item():.6f}, min {torch.min(factor[mask_b]).item():.6f}, mean {torch.mean(factor[mask_b]).item():.6f}')

        nopenshift[:,d] += torch.where(
            mask_b,
            nopenshiftTerm,
            torch.zeros_like(nopenshiftTerm)
        )
        nopencount[:,d] += mask_b.int()

        # u_adj_k = - (3 - 4 * torch.clamp(r_ib / n_b, min = 0.25)) * (u_i - u_b) * (boundaryNormal[:,d]**2)

        # print(f'u_adj_k shape: {u_adj_k.shape}')
        # print(f'u_i shape: {u_i.shape}')
        # print(f'u_b shape: {u_b.shape}')

        # adjustedVelocities[:,d] = torch.where(
        #     penetrationMask.view(-1),
        #     u_adj_k,
        #     adjustedVelocities[:,d]
        # )
    
    nopenshift = scatter_sum(nopenshift, fluidIndices, dim = 0, dim_size = particles.positions.shape[0])
    nopencount = scatter_sum(nopencount, fluidIndices, dim = 0, dim_size = particles.positions.shape[0])

    nopencounta = scatter_sum(nopen_mask_a.int(), fluidIndices, dim = 0, dim_size = particles.positions.shape[0])
    nopencountb = scatter_sum(nopen_mask_b.int(), fluidIndices, dim = 0, dim_size = particles.positions.shape[0])
    nopencountc = scatter_sum(nopen_mask_c.int(), fluidIndices, dim = 0, dim_size = particles.positions.shape[0])

    avgShift = nopenshift / (nopencount.float() + 1e-12) * 2
    # print('-' * 40)
    # print(f'[mDBC] - Total penetration corrections applied: {torch.sum(nopencount > 0).item()} [{torch.sum(nopencounta > 0).item()} / {torch.sum(nopencountb > 0).item()} / {torch.sum(nopencountc > 0).item()}] / {particles.positions.shape[0]}')
    # print(f'[mDBC] - Average penetration correction magnitude: {torch.mean(torch.linalg.norm(avgShift[nopencount > 0], dim = 0)).item():.6f}')
    # print(f'[mDBC] - Max penetration correction magnitude: {torch.max(torch.linalg.norm(avgShift[nopencount > 0], dim = 0)).item():.6f}')



    mergedVelocities = particles.velocities.clone()
    # adjustedParticles = fluidIndices[penetrationMask]

    # particles.velocities += avgShift

    checked = scatter_sum(check_ab.int(), fluidIndices, dim = 0, dim_size = particles.positions.shape[0])

    # particles.velocities[checked > 0,:] = 0

    # avgShift[checked>0] = -particles.velocities[checked>0]

    return avgShift