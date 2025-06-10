import torch
from sphMath.kernels import SPHKernel
from typing import Union, Tuple, Optional
from sphMath.sphOperations.shared import get_i, get_j, mod_distance, getSupport, scatter_sum, product

from sphMath.schemes.states.wcsph import WeaklyCompressibleState
from sphMath.schemes.states.compressiblesph import CompressibleState
from sphMath.neighborhood import SparseNeighborhood, DomainDescription, PrecomputedNeighborhood
from sphMath.enums import Operation, SupportScheme, GradientMode, DivergenceMode, LaplacianMode
from sphMath.enums import KernelCorrectionScheme
from typing import List
from sphMath.util import KernelTerms



from sphMath.enums import KernelCorrectionScheme
from typing import List
from sphMath.util import KernelTerms
# from sphMath.neighborhood 
# from sphMath.sphOperations.shared import evalSupport, evalPrecomputed, get_qj, get_qs, correctedKernel_CRK, correctedKernelGradient_CRK
from sphMath.sphOperations.interpolate import interpolate_precomputed, interpolate_precomputed_op
from sphMath.sphOperations.density import density_precomputed, density_precomputed_op
from sphMath.sphOperations.gradient import gradient_precomputed, gradient_precomputed_op
from sphMath.sphOperations.divergence import divergence_precomputed, divergence_precomputed_op
from sphMath.sphOperations.curl import curl_precomputed, curl_precomputed_op
from sphMath.sphOperations.laplacian import laplacian_precomputed, laplacian_precomputed_op


def invokeOperation(
        positions_a : torch.Tensor,
        positions_b : torch.Tensor,

        supports_a : torch.Tensor,
        supports_b : torch.Tensor,

        masses_a : torch.Tensor,
        masses_b : torch.Tensor,

        densities_a : torch.Tensor,
        densities_b : torch.Tensor,

        apparentArea_a: Optional[torch.Tensor],
        apparentArea_b: Optional[torch.Tensor],

        quantity_a : Optional[torch.Tensor],
        quantity_b : Optional[torch.Tensor],
        quantity_ab : Optional[torch.Tensor],
    
        i: torch.Tensor,
        j: torch.Tensor,
        numRows: int,
        numCols: int,

        r_ij: torch.Tensor,
        x_ij: torch.Tensor,

        W_i: torch.Tensor,
        W_j: torch.Tensor,
        gradW_i: torch.Tensor,
        gradW_j: torch.Tensor,
        H_i: Optional[torch.Tensor],
        H_j: Optional[torch.Tensor],
        gradH_i: Optional[torch.Tensor],
        gradH_j: Optional[torch.Tensor],

        operation : Operation = Operation.Interpolate,
        supportScheme : SupportScheme = SupportScheme.Scatter,
        gradientMode : GradientMode = GradientMode.Naive,
        divergenceMode : DivergenceMode = DivergenceMode.div,
        laplacianMode : LaplacianMode = LaplacianMode.naive,
        consistentDivergence : bool = False,
        useApparentArea: bool = False,
        

        correctionTerms: Optional[List[KernelCorrectionScheme]] = None,
        kernelCorrectionValues: Tuple[KernelTerms, KernelTerms] = (KernelTerms(), KernelTerms()),
        positiveDivergence: bool = False
        # gradientRenormalizationMatrix : Optional[torch.Tensor] = None,
        # gradHTerms : Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):

    if operation == Operation.Interpolate:
        return interpolate_precomputed_op(
        positions_a,
        positions_b,

        supports_a,
        supports_b,

        masses_a,
        masses_b,

        densities_a,
        densities_b,
        apparentArea_a,
        apparentArea_b,

        quantity_a,
        quantity_b,
        quantity_ab,
    
        i,
        j,
        numRows,
        numCols,

        r_ij,
        x_ij,

        W_i,
        W_j,
        gradW_i,
        gradW_j,
        H_i,
        H_j,
        gradH_i,
        gradH_j,

        operation,
        supportScheme,
        gradientMode,
        divergenceMode,
        laplacianMode,
        consistentDivergence,
        useApparentArea,
        correctionTerms,
        kernelCorrectionValues[0].A, kernelCorrectionValues[1].A,
        kernelCorrectionValues[0].B, kernelCorrectionValues[1].B,
        kernelCorrectionValues[0].gradA, kernelCorrectionValues[1].gradA,
        kernelCorrectionValues[0].gradB, kernelCorrectionValues[1].gradB,
        kernelCorrectionValues[0].gradCorrectionMatrices, kernelCorrectionValues[1].gradCorrectionMatrices,
        kernelCorrectionValues[0].omega, kernelCorrectionValues[1].omega,
        positiveDivergence
    )
    elif operation == Operation.Density:
        return density_precomputed_op(
        positions_a,
        positions_b,

        supports_a,
        supports_b,

        masses_a,
        masses_b,

        apparentArea_a,
        apparentArea_b,

        quantity_a,
        quantity_b,
        quantity_ab,
    
        i,
        j,
        numRows,
        numCols,

        r_ij,
        x_ij,

        W_i,
        W_j,
        gradW_i,
        gradW_j,
        H_i,
        H_j,
        gradH_i,
        gradH_j,

        operation,
        supportScheme,
        gradientMode,
        divergenceMode,
        laplacianMode,
        consistentDivergence,
        useApparentArea,
        correctionTerms,
        kernelCorrectionValues[0].A, kernelCorrectionValues[1].A,
        kernelCorrectionValues[0].B, kernelCorrectionValues[1].B,
        kernelCorrectionValues[0].gradA, kernelCorrectionValues[1].gradA,
        kernelCorrectionValues[0].gradB, kernelCorrectionValues[1].gradB,
        kernelCorrectionValues[0].gradCorrectionMatrices, kernelCorrectionValues[1].gradCorrectionMatrices,
        kernelCorrectionValues[0].omega, kernelCorrectionValues[1].omega,
        positiveDivergence
    )
    elif operation == Operation.Gradient:
        return gradient_precomputed_op(
        positions_a,
        positions_b,

        supports_a,
        supports_b,

        masses_a,
        masses_b,

        densities_a,
        densities_b,
        apparentArea_a,
        apparentArea_b,

        quantity_a,
        quantity_b,
        quantity_ab,
    
        i,
        j,
        numRows,
        numCols,

        r_ij,
        x_ij,

        W_i,
        W_j,
        gradW_i,
        gradW_j,
        H_i,
        H_j,
        gradH_i,
        gradH_j,

        operation,
        supportScheme,
        gradientMode,
        divergenceMode,
        laplacianMode,
        consistentDivergence,
        useApparentArea,
        correctionTerms,
        kernelCorrectionValues[0].A, kernelCorrectionValues[1].A,
        kernelCorrectionValues[0].B, kernelCorrectionValues[1].B,
        kernelCorrectionValues[0].gradA, kernelCorrectionValues[1].gradA,
        kernelCorrectionValues[0].gradB, kernelCorrectionValues[1].gradB,
        kernelCorrectionValues[0].gradCorrectionMatrices, kernelCorrectionValues[1].gradCorrectionMatrices,
        kernelCorrectionValues[0].omega, kernelCorrectionValues[1].omega,
        positiveDivergence)
    elif operation == Operation.Divergence:
        return divergence_precomputed_op(
        positions_a,
        positions_b,

        supports_a,
        supports_b,

        masses_a,
        masses_b,

        densities_a,
        densities_b,
        apparentArea_a,
        apparentArea_b,

        quantity_a,
        quantity_b,
        quantity_ab,
    
        i,
        j,
        numRows,
        numCols,

        r_ij,
        x_ij,

        W_i,
        W_j,
        gradW_i,
        gradW_j,
        H_i,
        H_j,
        gradH_i,
        gradH_j,

        operation,
        supportScheme,
        gradientMode,
        divergenceMode,
        laplacianMode,
        consistentDivergence,
        useApparentArea,
        correctionTerms,
        kernelCorrectionValues[0].A, kernelCorrectionValues[1].A,
        kernelCorrectionValues[0].B, kernelCorrectionValues[1].B,
        kernelCorrectionValues[0].gradA, kernelCorrectionValues[1].gradA,
        kernelCorrectionValues[0].gradB, kernelCorrectionValues[1].gradB,
        kernelCorrectionValues[0].gradCorrectionMatrices, kernelCorrectionValues[1].gradCorrectionMatrices,
        kernelCorrectionValues[0].omega, kernelCorrectionValues[1].omega,
        positiveDivergence
    )
    elif operation == Operation.Curl:
        return curl_precomputed_op(
        positions_a,
        positions_b,

        supports_a,
        supports_b,

        masses_a,
        masses_b,

        densities_a,
        densities_b,
        apparentArea_a,
        apparentArea_b,

        quantity_a,
        quantity_b,
        quantity_ab,
    
        i,
        j,
        numRows,
        numCols,

        r_ij,
        x_ij,

        W_i,
        W_j,
        gradW_i,
        gradW_j,
        H_i,
        H_j,
        gradH_i,
        gradH_j,

        operation,
        supportScheme,
        gradientMode,
        divergenceMode,
        laplacianMode,
        consistentDivergence,
        useApparentArea,
        correctionTerms,
        kernelCorrectionValues[0].A, kernelCorrectionValues[1].A,
        kernelCorrectionValues[0].B, kernelCorrectionValues[1].B,
        kernelCorrectionValues[0].gradA, kernelCorrectionValues[1].gradA,
        kernelCorrectionValues[0].gradB, kernelCorrectionValues[1].gradB,
        kernelCorrectionValues[0].gradCorrectionMatrices, kernelCorrectionValues[1].gradCorrectionMatrices,
        kernelCorrectionValues[0].omega, kernelCorrectionValues[1].omega,
        positiveDivergence
    )
    elif operation == Operation.Laplacian:
        return laplacian_precomputed_op(
        positions_a,
        positions_b,

        supports_a,
        supports_b,

        masses_a,
        masses_b,

        densities_a,
        densities_b,
        apparentArea_a,
        apparentArea_b,

        quantity_a,
        quantity_b,
        quantity_ab,
    
        i,
        j,
        numRows,
        numCols,

        r_ij,
        x_ij,

        W_i,
        W_j,
        gradW_i,
        gradW_j,
        H_i,
        H_j,
        gradH_i,
        gradH_j,

        operation,
        supportScheme,
        gradientMode,
        divergenceMode,
        laplacianMode,
        consistentDivergence,
        useApparentArea,
        correctionTerms,
        kernelCorrectionValues[0].A, kernelCorrectionValues[1].A,
        kernelCorrectionValues[0].B, kernelCorrectionValues[1].B,
        kernelCorrectionValues[0].gradA, kernelCorrectionValues[1].gradA,
        kernelCorrectionValues[0].gradB, kernelCorrectionValues[1].gradB,
        kernelCorrectionValues[0].gradCorrectionMatrices, kernelCorrectionValues[1].gradCorrectionMatrices,
        kernelCorrectionValues[0].omega, kernelCorrectionValues[1].omega,
        positiveDivergence
    )
    else:
        raise ValueError(f"Unknown operation {operation}")
    