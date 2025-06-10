
from typing import Tuple
from sphMath.sphOperations.shared import mod_distance
from sphMath.neighborhood import SupportScheme
from sphMath.kernels import SPHKernel
from typing import Optional, Tuple
import torch

from typing import Union, Tuple


def evalSupport(h : Tuple[torch.Tensor, torch.Tensor], i: torch.Tensor , j: torch.Tensor, mode : SupportScheme = SupportScheme.Scatter):
    if mode == SupportScheme.Scatter:
        return h[1][j]
    elif mode == SupportScheme.Gather:
        return h[0][i]
    elif mode == SupportScheme.Symmetric:
        return (h[0][i] + h[1][j])/2
    elif mode == SupportScheme.SuperSymmetric:
        return torch.max(h[0][i], h[1][j])
    else:
        raise ValueError(f"Unknown mode {mode}")

def evalPrecomputed(
        values : Tuple[torch.Tensor, torch.Tensor],
        mode : SupportScheme
):
    if mode == SupportScheme.Scatter:
        return values[1]
    elif mode == SupportScheme.Gather:
        return values[0]
    else:
        return (values[0] + values[1])/2


@torch.jit.script
def access_element_or_raise(tensor: Optional[torch.Tensor], i: torch.Tensor) -> torch.Tensor:
    if tensor is not None:
        return tensor[i]
    else:
        raise ValueError("Tensor is None and cannot be accessed.")

def get_qj(quantity: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], i: torch.Tensor, j: torch.Tensor, gradH: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]) -> torch.Tensor:
    if quantity[1] is None:
        raise ValueError("Quantity of the second particle set cannot be none")
    else:
        q = quantity[1]
        qj = access_element_or_raise(q,j)
        if gradH[1] is None:
            return qj
        else:
            return torch.einsum('n..., n -> n...', qj, 1 / access_element_or_raise(gradH[1], j))

def get_qi(quantity: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], i: torch.Tensor, j: torch.Tensor, gradH: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]) -> torch.Tensor:
    if quantity[0] is None:
        raise ValueError("Quantity of the first particle set cannot be none")
    else:
        q = quantity[0]
        qi = access_element_or_raise(q,i)
        if gradH[0] is None:
            return qi
        else:
            return torch.einsum('n..., n -> n...', qi, 1 / access_element_or_raise(gradH[0], i))

def get_qs(quantity: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], i: torch.Tensor, j: torch.Tensor, gradH: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:  
    return (get_qi(quantity, i, j, gradH), get_qj(quantity, i, j, gradH))

def correctedKernel_CRK(i, j, A: Optional[torch.Tensor], B: Optional[torch.Tensor], x_ij, W_ij, xji: bool = False):
    if A is None or B is None:
        raise ValueError("A or B is None, they are required for CRK correction")
    index = j if xji else i
    fac = -1 if xji else 1
    prod = torch.einsum('...i, ...j -> ...', B[index], fac * x_ij)
    return A[index] * (1 + prod) * W_ij


def correctedKernelGradient_CRK(i, j, A: Optional[torch.Tensor], B: Optional[torch.Tensor], gradA: Optional[torch.Tensor], gradB: Optional[torch.Tensor], x_ij, W_ij, gradW_ij, xji: bool = False):
    if A is None or B is None or gradA is None or gradB is None:
        raise ValueError("A, B, gradA or gradB is None, they are required for CRK correction")
    index = j if xji else i
    fac = -1 if xji else 1
    gradW_ij_ = fac * gradW_ij
        
    firstTerm = (A[index] * (1 + torch.einsum('...i, ...j -> ...', B[index], fac * x_ij))).view(-1,1) * gradW_ij_
    secondTerm = gradA[index] * ((1 + torch.einsum('...i, ...j -> ...', B[index], fac * x_ij)) * W_ij).view(-1,1)
    thirdTerm = (torch.einsum('...ca,...a -> ...c', gradB[index], fac * x_ij) + B[index]) * (W_ij * A[index]).view(-1,1)

    return firstTerm + secondTerm + thirdTerm


from sphMath.sphOperations.shared import scatter_sum
from typing import List

def correctedKernel_CRK_ij(i, j, A_i: Optional[torch.Tensor], B_i: Optional[torch.Tensor], x_ij, W_ij, xji: bool = False):
    if A_i is None or B_i is None:
        raise ValueError("A or B is None, they are required for CRK correction")
    # index = i
    fac = 1
    prod = torch.einsum('...i, ...j -> ...', B_i, fac * x_ij)
    return A_i * (1 + prod) * W_ij

def correctedKernel_CRK_ij(i, j, A_j: Optional[torch.Tensor], B_j: Optional[torch.Tensor], x_ij, W_ij, xji: bool = False):
    if A_j is None or B_j is None:
        raise ValueError("A or B is None, they are required for CRK correction")
    # index = j
    fac = -1
    prod = torch.einsum('...i, ...j -> ...', B_j, fac * x_ij)
    return A_j * (1 + prod) * W_ij

def evaluateKernel_(W_i: torch.Tensor, W_j: torch.Tensor, 
            supportScheme: SupportScheme, 
            crkCorrection: bool,             
            i: torch.Tensor, j: torch.Tensor, 
            x_ij: torch.Tensor, 
            correctionTerm_A_i: torch.Tensor, correctionTerm_B_i: torch.Tensor):
    W_ij = evalPrecomputed((W_i, W_j), mode = supportScheme)
    if crkCorrection:
        W_ij = correctedKernel_CRK_ij(i, j, correctionTerm_A_i, correctionTerm_B_i, x_ij, W_ij, False)
    return W_ij

def correctedKernelGradient_CRK_ij(i, j, A_i: Optional[torch.Tensor], B_i: Optional[torch.Tensor], gradA_i: Optional[torch.Tensor], gradB_i: Optional[torch.Tensor], x_ij, W_ij, gradW_ij, xji: bool = False):
    if A_i is None or B_i is None or gradA_i is None or gradB_i is None:
        raise ValueError("A, B, gradA or gradB is None, they are required for CRK correction")
    # index = i
    fac = 1
    gradW_ij_ = fac * gradW_ij
        
    firstTerm = (A_i * (1 + torch.einsum('...i, ...j -> ...', B_i, fac * x_ij))).view(-1,1) * gradW_ij_
    secondTerm = gradA_i * ((1 + torch.einsum('...i, ...j -> ...', B_i, fac * x_ij)) * W_ij).view(-1,1)
    thirdTerm = (torch.einsum('...ca,...a -> ...c', gradB_i, fac * x_ij) + B_i) * (W_ij * A_i).view(-1,1)

    return firstTerm + secondTerm + thirdTerm

def correctedKernelGradient_CRK_ji(i, j, A_j: Optional[torch.Tensor], B_j: Optional[torch.Tensor], gradA_j: Optional[torch.Tensor], gradB_j: Optional[torch.Tensor], x_ij, W_ij, gradW_ij, xji: bool = False):
    if A_j is None or B_j is None or gradA_j is None or gradB_j is None:
        raise ValueError("A, B, gradA or gradB is None, they are required for CRK correction")
    # index = i
    fac = -1
    gradW_ij_ = fac * gradW_ij
        
    firstTerm = (A_j * (1 + torch.einsum('...i, ...j -> ...', B_j, fac * x_ij))).view(-1,1) * gradW_ij_
    secondTerm = gradA_j * ((1 + torch.einsum('...i, ...j -> ...', B_j, fac * x_ij)) * W_ij).view(-1,1)
    thirdTerm = (torch.einsum('...ca,...a -> ...c', gradB_j, fac * x_ij) + B_j) * (W_ij * A_j).view(-1,1)

    return firstTerm + secondTerm + thirdTerm

def evaluateKernelGradient_(
        W_i: torch.Tensor, W_j: torch.Tensor,
        gradW_i: torch.Tensor, gradW_j: torch.Tensor,
            supportScheme: SupportScheme,
            crkCorrection: bool,
            i: torch.Tensor, j: torch.Tensor,
            x_ij: torch.Tensor,
            correctionTerm_A_i: torch.Tensor, correctionTerm_B_i: torch.Tensor,
            correctionTerm_gradA_i: torch.Tensor, correctionTerm_gradB_i: torch.Tensor,
            correctionTerm_A_j: torch.Tensor, correctionTerm_B_j: torch.Tensor,
            correctionTerm_gradA_j: torch.Tensor, correctionTerm_gradB_j: torch.Tensor):
    
    gradW_ij = evalPrecomputed((gradW_i, gradW_j), mode = supportScheme)
    if crkCorrection:
        W_ij = (W_i + W_j) / 2
        gradW_ij = (gradW_i + gradW_j) / 2

        gradW_ij_CRK = correctedKernelGradient_CRK_ij(i, j, correctionTerm_A_i, correctionTerm_B_i, correctionTerm_gradA_i, correctionTerm_gradB_i, x_ij, W_ij, gradW_ij, False)
        gradW_ij_CRK += correctedKernelGradient_CRK_ji(i, j, correctionTerm_A_j, correctionTerm_B_j, correctionTerm_gradA_j, correctionTerm_gradB_j, x_ij, W_ij, gradW_ij, True)
        gradW_ij = (gradW_ij_CRK - gradW_ij_CRK) / 2

    return gradW_ij


def custom_forwards(fnForward, i: torch.Tensor, j: torch.Tensor, numRows: int, numCols: int, iTensors: List[torch.Tensor], jTensors: List[torch.Tensor], ijTensors: List[torch.Tensor], *args):
    inputs_i = [x[i] if x is not None else None for x in iTensors]
    inputs_j = [x[j] if x is not None else None for x in jTensors]
    inputs_ij = [x if x is not None else None for x in ijTensors]
    inputs = inputs_i + inputs_j + inputs_ij
    return fnForward(*args, i, j, numRows, numCols, *inputs)

def custom_backwards(fnForward, grad_output: torch.Tensor, i: torch.Tensor, j: torch.Tensor, numRows: int, numCols: int, iTensors: List[torch.Tensor], jTensors: List[torch.Tensor], ijTensors: List[torch.Tensor], *args):
    inputs_i = [x[i].detach() if x is not None else None for x in iTensors]
    inputs_j = [x[j].detach() if x is not None else None for x in jTensors]
    inputs_ij = [x.detach() if x is not None else None for x in ijTensors]
    inputs = inputs_i + inputs_j + inputs_ij
    # print('Inputs:', inputs)
    # print('Args:', args)

    grad_inputs_i = [x.requires_grad_() for x in inputs_i if x is not None]
    grad_inputs_j = [x.requires_grad_() for x in inputs_j if x is not None]
    grad_inputs_ij = [x.requires_grad_() for x in inputs_ij if x is not None]
    grad_inputs = grad_inputs_i + grad_inputs_j + grad_inputs_ij


    grad_enabled = False
    if not torch.is_grad_enabled():
        _ = torch.set_grad_enabled(True)
    else:
        grad_enabled = True

    fn_ = fnForward(*args, i, j, numRows, numCols, *inputs)
    
    if not grad_enabled:
        torch.set_grad_enabled(False)

    gradients = torch.autograd.grad(
            fn_, grad_inputs,
            grad_outputs=grad_output,
            allow_unused=True,
            retain_graph=False)

    gradient_outputs_i = [None for _ in range(len(inputs_i))]
    gradient_outputs_j = [None for _ in range(len(inputs_j))]
    gradient_outputs_ij = [None for _ in range(len(inputs_ij))]

    index = 0
    for i_, input_i in enumerate(inputs_i):
        if input_i is not None:
            gradient_outputs_i[i_] = scatter_sum(gradients[index], i, dim_size=numRows, dim = 0) if gradients[index] is not None else None
            index += 1
    for j_, input_j in enumerate(inputs_j):
        if input_j is not None:
            gradient_outputs_j[j_] = scatter_sum(gradients[index], j, dim_size=numCols, dim = 0) if gradients[index] is not None else None
            index += 1
    for ij_, input_ij in enumerate(inputs_ij):
        if input_ij is not None:
            gradient_outputs_ij[ij_] = gradients[index]
            index += 1
    
    # print('Gradients:', gradients)
    return gradient_outputs_i + gradient_outputs_j + gradient_outputs_ij



def get_q(q: Optional[torch.Tensor], omega: Optional[torch.Tensor], omegaCorrection: bool):
    if omegaCorrection:
        if omega is None:
            raise ValueError("omega is None")
        if q is None:
            raise ValueError("q is None")
        q = torch.einsum('n..., n -> n...', q, 1 / omega)
    if q is None:
        raise ValueError("q is None")
    else:
        return q
