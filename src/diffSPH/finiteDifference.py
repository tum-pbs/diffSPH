import torch
import numpy as np

def getCentralSchemes():
    return {1 : [2,4,6,8],
            2 : [2,4,6,8],
            3 : [2,4,6],
            4 : [2,4,6]}
def getForwardSchemes():
    return {1 : [1,2,3,4,5,6],
            2 : [1,2,3,4,5,6],
            3 : [1,2,3,4,5,6],
            4 : [1,2,3,4,5]}
def getBackwardSchemes():
    return {1 : [1,2,3,4,5,6],
            2 : [1,2,3,4,5,6],
            3 : [1,2,3,4,5,6],
            4 : [1,2,3,4,5]}

def centralDifferenceStencil(derivative: int, accuracy: int, dtype = torch.float32, device = 'cpu'):
    if derivative == 1:
        if accuracy == 2:
            return torch.tensor([-1/2, 0, 1/2], dtype = dtype, device = device), torch.tensor([-1, 0, 1], dtype = torch.int32, device = device)
        elif accuracy == 4:
            return torch.tensor([1/12, -2/3, 0, 2/3, -1/12], dtype = dtype, device = device), torch.tensor([-2, -1, 0, 1, 2], dtype = torch.int32, device = device)
        elif accuracy == 6:
            return torch.tensor([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60], dtype = dtype, device = device), torch.tensor([-3, -2, -1, 0, 1, 2, 3], dtype = torch.int32, device = device)
        elif accuracy == 8:
            return torch.tensor([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280], dtype = dtype, device = device), torch.tensor([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype = torch.int32, device = device)
        else:
            raise ValueError("Accuracy must be 2, 4, 6, or 8")
    elif derivative == 2:
        if accuracy == 2:
            return torch.tensor([1, -2, 1], dtype = dtype, device = device), torch.tensor([-1, 0, 1], dtype = torch.int32, device = device)
        elif accuracy == 4:
            return torch.tensor([-1/12, 4/3, -5/2, 4/3, -1/12], dtype = dtype, device = device), torch.tensor([-2, -1, 0, 1, 2], dtype = torch.int32, device = device)
        elif accuracy == 6:
            return torch.tensor([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90], dtype = dtype, device = device), torch.tensor([-3, -2, -1, 0, 1, 2, 3], dtype = torch.int32, device = device)
        elif accuracy == 8:
            return torch.tensor([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560],  dtype = dtype, device = device), torch.tensor([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype = torch.int32, device = device)
        else:
            raise ValueError("Accuracy must be 2, 4, 6, or 8")
    elif derivative == 3:
        if accuracy == 2:
            return torch.tensor([-1/2, 1, 0, -1, 1/2], dtype = dtype, device = device), torch.tensor([-1, 0, 1], dtype = torch.int32, device = device)
        elif accuracy == 4:
            return torch.tensor([1/8, -1, 13/8, 0, -13/8, 1, -1/8], dtype = dtype, device = device), torch.tensor([-2, -1, 0, 1, 2], dtype = torch.int32, device = device)
        elif accuracy == 6:
            return torch.tensor([-7/240, 3/10, -169/120, 61/30, -61/30, 169/120, -3/10, 7/240], dtype = dtype, device = device), torch.tensor([-3, -2, -1, 0, 1, 2, 3], dtype = torch.int32, device = device)
        else:
            raise ValueError("Accuracy must be 2, 4, 6")
    elif derivative == 4:
        if accuracy == 2:
            return torch.tensor([1, -4, 6, -4, 1], dtype = dtype, device = device), torch.tensor([-2, -1, 0, 1, 2], dtype = torch.int32, device = device)
        elif accuracy == 4:
            return torch.tensor([-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6], dtype = dtype, device = device), torch.tensor([-3, -2, -1, 0, 1, 2, 3], dtype = torch.int32, device = device)
        elif accuracy == 6:
            return torch.tensor([7/240, -2/5, 169/60, -122/15, 91/8, -122/15, 169/60, -2/5, 7/240], dtype = dtype, device = device), torch.tensor([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype = torch.int32, device = device)
        else:
            raise ValueError("Accuracy must be 2, 4, 6")
    else:
        raise ValueError("Derivative must be 1, 2, 3, or 4")
    
def forwardDifferenceStencil(derivative: int, accuracy: int, dtype = torch.float32, device = 'cpu'):
    if derivative == 1:
        if accuracy == 1:
            return torch.tensor([-1, 1], dtype = dtype, device = device), torch.tensor([0, 1], dtype = torch.int32, device = device)
        elif accuracy == 2:
            return torch.tensor([-3/2, 2, -1/2], dtype = dtype, device = device), torch.tensor([0, 1, 2], dtype = torch.int32, device = device)
        elif accuracy == 3:
            return torch.tensor([-11/6, 3, -3/2, 1/3], dtype = dtype, device = device), torch.tensor([0, 1, 2, 3], dtype = torch.int32, device = device)
        elif accuracy == 4:
            return torch.tensor([-25/12, 4, -3, 4/3, -1/4], dtype = dtype, device = device), torch.tensor([0, 1, 2, 3, 4], dtype = torch.int32, device = device)
        elif accuracy == 5:
            return torch.tensor([-137/60, 5, -5, 10/3, -5/4, 1/5], dtype = dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5], dtype = torch.int32, device = device)
        elif accuracy == 6:
            return torch.tensor([-49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6], dtype = dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype = torch.int32, device = device)
        else:
            raise ValueError("Accuracy must be 1, 2, 3, 4, 5, or 6")
    elif derivative == 2:
        if accuracy == 1:
            return torch.tensor([1, -2, 1], dtype = dtype, device = device), torch.tensor([0, 1, 2], dtype = torch.int32, device = device)
        elif accuracy == 2:
            return torch.tensor([2, -5, 4, -1], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3], dtype = torch.int32, device = device)
        elif accuracy == 3:
            return torch.tensor([35/12, -26/3, 19/2, -14/3, 11/12], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3, 4], dtype = torch.int32, device = device)
        elif accuracy == 4:
            return torch.tensor([15/4, -77/6, 107/6, -13, 61/12, -5/6], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5], dtype = torch.int32, device = device)
        elif accuracy == 5:
            return torch.tensor([203/45, -87/5, 117/4, -254/9, 33/2, -27/5, 137/180], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype = torch.int32, device = device)
        elif accuracy == 6:
            return torch.tensor([469/90, -223/10, 879/20, -949/18, 41, -201/10, 1019/180, -7/10], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype = torch.int32, device = device)
        else:
            raise ValueError("Accuracy must be 1, 2, 3, 4, 5, or 6")
    elif derivative == 3:
        if accuracy == 1:
            return torch.tensor([1, -3, 3, -1], dtype = dtype, device = device), torch.tensor([0, 1, 2, 3], dtype = torch.int32, device = device)
        elif accuracy == 2:
            return torch.tensor([5/2, -9, 12, -7, 3/2], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3, 4], dtype = torch.int32, device = device)
        elif accuracy == 3:
            return torch.tensor([-17/4, 71/4, -59/2, 49/2, -41/4, 7/4], dtype = dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5], dtype = torch.int32, device = device)
        elif accuracy == 4:
            return torch.tensor([-49/18, 29, -461/18, 62, -307/18, 13, -15/8], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype = torch.int32, device = device)
        elif accuracy == 5:
            return torch.tensor([-967/120, 638/15, -3929/40, 389/3, -2545/24, 268/5, -1849/120, 29/15], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype = torch.int32, device = device)
        elif accuracy == 6:
            return torch.tensor([-801/80, 349/6, -18353/120, 2391/10, -1457/6, 4891/30, -561/8, 527/30, -469/240], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype = torch.int32, device = device)
        else:
            raise ValueError("Accuracy must be 1, 2, 3, 4, 5, or 6")
    elif derivative == 4:
        if accuracy == 1:
            return torch.tensor([1, -4, 6, -4, 1], dtype = dtype, device = device), torch.tensor([0, 1, 2, 3, 4], dtype = torch.int32, device = device)
        elif accuracy == 2:
            return torch.tensor([3, -14, 26, -24, 11, -2], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5], dtype = torch.int32, device = device)
        elif accuracy == 3:
            return torch.tensor([35/6, -31, 137/2, -242/3, 107/2, -19, 17/6], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype = torch.int32, device = device)
        elif accuracy == 4:
            return torch.tensor([28/3, -111/2, 142, -1219/6, 176, -185/2, 82/3, -7/2], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype = torch.int32, device = device)
        elif accuracy == 5:
            return torch.tensor([1069/80, -1316/15, 15289/60, -2144/5, 10993/24, -4772/15, 2803/20, -536/15, 967/240], dtype=dtype, device = device), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype = torch.int32, device = device)
        else:
            raise ValueError("Accuracy must be 1, 2, 3, 4, or 5")
    else:
        raise ValueError("Derivative must be 1, 2, 3, or 4")
def backwardDifferenceStencil(derivative : int, accuracy:int):
    stencil, offsets = forwardDifferenceStencil(derivative, accuracy)

    if derivative % 2 == 0:
        return stencil.flip(0), -offsets.flip(0)
    else:
        return -stencil.flip(0), -offsets.flip(0)
    

def centralDifference(f, h, derivative, accuracy):
    stencil = centralDifferenceStencil(derivative, accuracy)
    print(stencil)
    n = len(stencil)
    pad = (n-1)//2
    print(pad)
    print(f)
    f = torch.nn.functional.pad(f, (pad, pad), mode='replicate', value = 0)
    return torch.sum(stencil * f.unfold(-1, n, 1), dim=-1)* (1/h**derivative)
def circular_pad1d(input, pad, dim):
    # Pad the input tensor along dimension `dim` with `pad` circular padding
    idx = tuple(slice(None, None) if i != dim
                else slice(-pad, None)
                for i in range(input.dim()))
    return torch.cat([input[idx], input, input[idx]], dim=dim)
def replicate_pad1d(input, pad, dim):
    # Pad the input tensor along dimension `dim` with `pad` replicate padding
    idx1 = tuple(slice(None, None) if i != dim
                 else slice(0, 1)
                 for i in range(input.dim()))
    idx2 = tuple(slice(None, None) if i != dim
                 else slice(-1, None)
                 for i in range(input.dim()))
    return torch.cat([input[idx1].repeat(1, pad), input, input[idx2].repeat(1, pad)], dim=dim)
def constant_pad1d(input, pad, dim, value):
    # Create a tensor of the same type as `input` filled with `value`
    pad_tensor = torch.full_like(input, fill_value=value)

    # Create slices for padding
    pad_slices = [slice(None, None) if i != dim else slice(0, pad) for i in range(input.dim())]

    # Concatenate the padding and the input tensor along dimension `dim`
    return torch.cat([pad_tensor[pad_slices], input, pad_tensor[pad_slices]], dim=dim)



def applyStencilAlongDimension_2d(fx, stencil, dim, padding = 'circular'):
    n = len(stencil)
    pad = (n-1)//2
    if padding == 'circular':
        fx_padded = circular_pad1d(fx, pad, dim)
    elif padding == 'replicate':
        fx_padded = replicate_pad1d(fx, pad, dim)
    elif padding == 'zeros':
        fx_padded = constant_pad1d(fx, pad, dim, 0)
    # print('fx', fx.shape, fx)
    # print('padded', fx_padded.shape, fx_padded)
    filter = stencil.view(1,1,-1).unsqueeze(3-dim)
    convolved = torch.nn.functional.conv2d(fx_padded.unsqueeze(0).unsqueeze(0), filter, padding = (0,0))
    return convolved



def gradient(fx, extent, order, accuracy):
    stencil = centralDifferenceStencil(order, accuracy).to(torch.float32)
    h = extent / (fx.shape[-1] - 1)
    if isinstance(h, torch.Tensor):
        output = torch.stack([applyStencilAlongDimension_2d(fx, stencil, 0, padding = 'replicate') / h[0]**order, applyStencilAlongDimension_2d(fx, stencil, 1, padding = 'replicate')/ h[1]**order] , dim=-1)
    else:
        output = torch.stack([applyStencilAlongDimension_2d(fx, stencil, 0, padding = 'replicate') / h**order, applyStencilAlongDimension_2d(fx, stencil, 1, padding = 'replicate')/ h**order] , dim=-1)
    return output.flatten(0,2)

def getStencils(order, accuracy):
    centralStencil = centralDifferenceStencil(order, accuracy - accuracy % 2 + 2).to(torch.float32)
    forwardStencil = forwardDifferenceStencil(order, accuracy).to(torch.float32)
    backwardStencil = backwardDifferenceStencil(order, accuracy).to(torch.float32)
    return centralStencil, forwardStencil, backwardStencil

def computeFiniteDifference(fx, dim, extent, order, accuracy):
    ngrid  = fx.shape[dim]
    centralStencil, forwardStencil, backwardStencil = getStencils(order, accuracy)
    h = extent / (ngrid - 1)
    unfolded = fx.transpose(dim,-1).unfold(-1, len(centralStencil),1)
    central = torch.sum(centralStencil.to(fx.device) * unfolded, dim = -1) * (1/h**order)
    forward = torch.vstack([torch.tensordot(forwardStencil.to(fx.device), fx.transpose(dim, 0)[a: a + len(forwardStencil),:], dims=1) * (1/h**order) for a in range(accuracy)]).mT
    backward = torch.vstack([torch.tensordot(backwardStencil.to(fx.device), fx.transpose(dim, 0)[-len(backwardStencil)-a:fx.shape[0] - a,:], dims=1) * (1/h**order) for a in range(accuracy)]).mT
    output = torch.hstack([forward, central, backward]).transpose(dim, -1)
    return output

def computeGradient(fx, extent, dim = 2, accuracy = 1):
    wasScalar = False
    if fx.dim() == dim:
        wasScalar = True
        fx = fx.unsqueeze(-1)
    outputShape = fx.shape + (dim,)
    output = torch.zeros(outputShape, dtype = torch.float32, device = fx.device)
    inputIndices = np.ndindex(fx.shape[dim:])
    for input in inputIndices:
        for i in range(dim):
            inputIndex = tuple(list(list(input)))
            outputIndex = tuple(list(list(input)) + [i])
            dimIndex = tuple([slice(None)] * dim)
            output[(*dimIndex, *outputIndex)] = computeFiniteDifference(fx[(*dimIndex,*inputIndex)], i, extent, 1, accuracy)
    if wasScalar:
        output = output.flatten(dim, dim+1)
    return output


def stencilFunction(fn, p, stencil, stencil_offsets, dx, dim, order):
    # print('p', p.shape)
    # print('stencil', stencil.shape, stencil)
    stencil = stencil.to(p.device)

    offsets = p.new_zeros(stencil.shape[0], p.shape[1])
    offsets[:,dim] = stencil_offsets
    # print('p', p.shape)
    # print('offsets', offsets.shape, offsets)
    
    offset_positions = (p[:, None,:] + offsets[None,:,:] * dx).view(-1, p.shape[1])
    # print('offset_positions', offset_positions.shape, offset_positions)
    functionValues = fn(offset_positions).to(p.device).view(p.shape[0], -1)
    # print('functionValues', functionValues.shape, functionValues)
    result = torch.sum(stencil * functionValues, dim = -1) / dx**order

    return result

def continuousGradient(fn, p, stencil, stencil_offsets, dx, order):
    return torch.stack([stencilFunction(fn, p, stencil, stencil_offsets, dx, d, order) for d in range(p.shape[1])], dim = -1)