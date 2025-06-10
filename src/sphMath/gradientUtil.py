import torch
from torch import autograd

from typing import Tuple, Callable
def derivative(out_fn: Tuple[torch.Tensor, Callable], *args):
    # outputs = fn(*args)
    if isinstance(out_fn, torch.Tensor):
        outputs = out_fn
    else:
        outputs = out_fn(*args)
    # if not isinstance(outputs, torch.Tensor):
    #     gradients = []
    #     for o, out in enumerate(outputs):
    #         gradients.append(derivative(lambda *a: fn(*a)[o], *args))
    #     return gradients

    # print('Output Shape:', outputs.shape)
    # print('------------------------------')

    argGrads = []
    trailingOutputShape = outputs.shape[1:]
    if len(trailingOutputShape) == 0:
        grad_outputs = torch.ones_like(outputs)
        gradients = autograd.grad(outputs=outputs, inputs=args, grad_outputs=grad_outputs, create_graph=True)
        return gradients
    else:
        for iarg, arg in enumerate(args):
            # print(f'\tArgument {iarg} [{arg.shape}]')
            argGrad = arg.new_zeros((arg.shape[0], *outputs.shape[1:], *arg.shape[1:]))
            # print(arg.shape)
        
            # print('\tTrailing Output Shape:', trailingOutputShape)
            # print('\tArg Grad Shape:', argGrad.shape)

            flattenedOutput = outputs.flatten(start_dim = 1)
            expanded = False
            if len(arg.shape) == 1:
                expanded = True
                argGrad = argGrad.view(-1,*trailingOutputShape,1)
                # print('Expanding to:', argGrad.shape)
            flattenedGrad = argGrad.flatten(start_dim = 1, end_dim = -2)
            for i in range(flattenedOutput.shape[1]):
                grad_outputs = torch.zeros_like(flattenedOutput)
                grad_outputs[:,i] = 1
                grad_outputs = grad_outputs.view(*outputs.shape)
                gradients = autograd.grad(outputs=outputs, inputs=arg, grad_outputs=grad_outputs, create_graph=True)
                
                # print(f'flattenedGrad: {flattenedGrad.shape}, gradients: {gradients[0].shape}')
                # print(gradients[0].shape)
                if len(gradients[0].shape) == 1:
                    # gradients[0] = gradients[0].view(-1,1)
                    flattenedGrad[:,i,:] = gradients[0].view(-1,1).flatten(start_dim = 1)
                else:
                    flattenedGrad[:,i,:] = gradients[0].flatten(start_dim = 1)

            if expanded:
                # print('Squeezing')
                argGrad = argGrad.view(arg.shape[0],*trailingOutputShape)
            argGrads.append(argGrad)
            # print('\t', argGrad.shape)
            # print('------------------------------')
        return argGrads
    
    

def eval_fn_grad(out_fn: Tuple[torch.Tensor, Callable], arg, order = 1):
    # print('Evaluating Function')
    if isinstance(out_fn, torch.Tensor):
        gradients = [out_fn]
    else:
        gradients = [out_fn(arg)]
    for o in range(1, order+1):
        # print('Gradient Order:', o)
        gradients.append(derivative(gradients[-1], arg)[0])
    return gradients

def Jacobian(out_fn: Tuple[torch.Tensor, Callable], arg):
    return eval_fn_grad(out_fn, arg, 1)[-1]
def Hessian(out_fn: Tuple[torch.Tensor, Callable], arg):
    return eval_fn_grad(out_fn, arg, 2)[-1]
def Patrician(out_fn: Tuple[torch.Tensor, Callable], arg):
    return eval_fn_grad(out_fn, arg, 3)[-1]
