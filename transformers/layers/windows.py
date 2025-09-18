# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# import torch
# from torch_geometric.loader import DataLoader
# import argparse
# from torch_geometric.nn import radius
# from torch.optim import Adam
# import copy
import torch
# from torch_geometric.loader import DataLoader
# import argparse
# from torch_geometric.nn import radius
# from torch.optim import Adam
# import matplotlib.pyplot as plt
# import portalocker
# import seaborn as sns
import torch
# import torch.nn as nn
# from datautils import *
# from plotting import *

# Use dark theme
# from tqdm.autonotebook import trange, tqdm
# import os

normDict = {}
normDict['zero'] = {}
normDict['zero']['cubicSpline'] = 0.5
normDict['zero']['quarticSpline'] = 0.3680000000000001
normDict['zero']['quinticSpline'] = 0.2716049382716052

normDict['zero']['Wendland2_1D'] = 1.0
normDict['zero']['Wendland4_1D'] = 1.0
normDict['zero']['Wendland6_1D'] = 1.0

normDict['zero']['Wendland2'] = 1.0
normDict['zero']['Wendland4'] = 1.0
normDict['zero']['Wendland6'] = 1.0

normDict['zero']['Hoct4'] = 0.9004611977424557
normDict['zero']['Spiky'] = 1.0
normDict['zero']['Mueller'] = 1.0
normDict['zero']['poly6'] = 1.0
normDict['zero']['Parabola'] = 1.0
normDict['zero']['Linear'] = 1.0

normDict['integral'] = {}
normDict['integral']['cubicSpline'] = 0.3750000004808612
normDict['integral']['quarticSpline'] = 0.24576000000063475
normDict['integral']['quinticSpline'] = 0.16460905349817873

normDict['integral']['Wendland2_1D'] = 0.8000000007695265
normDict['integral']['Wendland4_1D'] = 0.6666666666675429
normDict['integral']['Wendland6_1D'] = 0.5818181818188082

normDict['integral']['Wendland2'] = 0.6666666679481377
normDict['integral']['Wendland4'] = 0.5925925925933454
normDict['integral']['Wendland6'] = 0.5333333333335031

normDict['integral']['Hoct4'] = 0.4724016135230473
normDict['integral']['Spiky'] = 0.5000309999743467
normDict['integral']['Mueller'] = 0.9142857147993859
normDict['integral']['poly6'] = 0.5000309999743467
normDict['integral']['Parabola'] = 1.3333126666253334
normDict['integral']['Linear'] = 0.999999999970667

def getWindowFunction(windowFunction, norm = None):
    windowFn = lambda r: torch.ones_like(r)
    if windowFunction == 'cubicSpline':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 3 - 4 * torch.clamp(1/2 - r, min = 0) ** 3
    elif windowFunction == 'quarticSpline':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 4 - 5 * torch.clamp(3/5 - r, min = 0) ** 4 + 10 * torch.clamp(1/5- r, min = 0) ** 4
    elif windowFunction == 'quinticSpline':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 5 - 6 * torch.clamp(2/3 - r, min = 0) ** 5 + 15 * torch.clamp(1/3 - r, min = 0) ** 5
    elif windowFunction == 'Wendland2_1D':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 3 * (1 + 3 * r)
    elif windowFunction == 'Wendland4_1D':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 5 * (1 + 5 * r + 8 * r**2)
    elif windowFunction == 'Wendland6_1D':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 7 * (1 + 7 * r + 19 * r**2 + 21 * r**3)
    elif windowFunction == 'Wendland2':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 4 * (1 + 4 * r)
    elif windowFunction == 'Wendland4':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 6 * (1 + 6 * r + 35/3 * r**2)
    elif windowFunction == 'Wendland6':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 8 * (1 + 8 * r + 25 * r**2 + 32 * r**3)
    elif windowFunction == 'Hoct4':
        def hoct4(x):
            alpha = 0.0927 # Subject to 0 = (1 − α)** nk−2 + A(γ − α)**nk−2 + B(β − α)**nk−2
            beta = 0.5 # Free parameter
            gamma = 0.75 # Free parameter
            nk = 4 # order of kernel

            A = (1 - beta**2) / (gamma ** (nk - 3) * (gamma ** 2 - beta ** 2))
            B = - (1 + A * gamma ** (nk - 1)) / (beta ** (nk - 1))
            P = -nk * (1 - alpha) ** (nk - 1) - nk * A * (gamma - alpha) ** (nk - 1) - nk * B * (beta - alpha) ** (nk - 1)
            Q = (1 - alpha) ** nk + A * (gamma - alpha) ** nk + B * (beta - alpha) ** nk - P * alpha

            termA = P * x + Q
            termB = (1 - x) ** nk + A * (gamma - x) ** nk + B * (beta - x) ** nk
            termC = (1 - x) ** nk + A * (gamma - x) ** nk
            termD = (1 - x) ** nk
            termE = 0 * x

            termA[x > alpha] = 0
            termB[x <= alpha] = 0
            termB[x > beta] = 0
            termC[x <= beta] = 0
            termC[x > gamma] = 0
            termD[x <= gamma] = 0
            termD[x > 1] = 0
            termE[x < 1] = 0

            return termA + termB + termC + termD + termE

        windowFn = lambda r: hoct4(r)
    elif windowFunction == 'Spiky':
        windowFn = lambda r: torch.clamp(1 - r, min = 0) ** 3
    elif windowFunction == 'Mueller':
        windowFn = lambda r: torch.clamp(1 - r ** 2, min = 0) ** 3
    elif windowFunction == 'poly6':
        windowFn = lambda r: torch.clamp((1 - r)**3, min = 0)
    elif windowFunction == 'Parabola':
        windowFn = lambda r: torch.clamp(1 - r**2, min = 0)
    elif windowFunction == 'Linear':
        windowFn = lambda r: torch.clamp(1 - r, min = 0)
    else:
        raise NotImplementedError(f'Window function {windowFunction} not implemented')
        
    if norm is not None:
        return lambda q: windowFn(q) / normDict[norm][windowFunction]
    return windowFn
# Window Function normalization test
# norm = 'integral'
# print('cubicSpline', getWindowFunction('cubicSpline', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('cubicSpline', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('quarticSpline', getWindowFunction('quarticSpline', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('quarticSpline', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('quinticSpline', getWindowFunction('quinticSpline', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('quinticSpline', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('Wendland2_1D', getWindowFunction('Wendland2_1D', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('Wendland2_1D', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('Wendland4_1D', getWindowFunction('Wendland4_1D', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('Wendland4_1D', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('Wendland6_1D', getWindowFunction('Wendland6_1D', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('Wendland6_1D', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('Wendland2', getWindowFunction('Wendland2', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('Wendland2', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('Wendland4', getWindowFunction('Wendland4', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('Wendland4', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('Wendland6', getWindowFunction('Wendland6', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('Wendland6', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('Hoct4', getWindowFunction('Hoct4', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('Hoct4', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('Spiky', getWindowFunction('Spiky', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('Spiky', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('Mueller', getWindowFunction('Mueller', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('Mueller', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('poly6', getWindowFunction('poly6', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('poly6', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('Parabola', getWindowFunction('Parabola', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('Parabola', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())
# print('Linear', getWindowFunction('Linear', norm = norm)(torch.tensor([0]).type(torch.float64)).numpy().item(), torch.trapezoid(getWindowFunction('Linear', norm = norm)(torch.abs(torch.linspace(-1,1,255).type(torch.float64))),torch.linspace(-1,1,255).type(torch.float64)).numpy().item())       
