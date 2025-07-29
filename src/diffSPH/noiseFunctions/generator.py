from .simplex1d import _noise1
from .simplex2d import _noise2
from .simplex3d import _noise3,_noise3periodic
from .simplex4d import _noise4
from .util import _init
from .perlin import interpolant, perlinNoise1D, perlinNoise2D, perlinNoise3D

import numpy as np
from numba import prange
import torch

def generatePerlin(shape, res, tileable, dim = 2, interpolant  = interpolant , seed =42 , device = 'cpu', dtype = torch.float32):
    if dim == 1:
        return perlinNoise1D(shape, res, tileable, interpolant = interpolant, rng = np.random.default_rng(seed=seed), device = device, dtype = dtype)
    if dim == 2:
        return perlinNoise2D([shape, shape], [res, res], [tileable, tileable], interpolant = interpolant, rng = np.random.default_rng(seed=seed), device = device, dtype = dtype)
    if dim == 3:
        return perlinNoise3D([shape, shape, shape], [res, res, res], [tileable, tileable, tileable], interpolant = interpolant, rng = np.random.default_rng(seed=seed), device = device, dtype = dtype)
     

def generateSimplex(shape, freq, dim = 2, seed = 42, device = 'cpu', dtype = torch.float32, tileable = False):    
    dx = 2/shape
    x = torch.linspace(-1 + dx / 2, 1 - dx/2, shape, device = device, dtype = dtype)
    y = torch.linspace(-1 + dx / 2, 1 - dx/2, shape, device = device, dtype = dtype)
    z = torch.linspace(-1 + dx / 2, 1 - dx/2, shape, device = device, dtype = dtype)
    perm, perm_grad = _init(seed)
    # print('generateSimplex', shape, freq, dim, seed, device, dtype, tileable)
    if not tileable:
        if dim == 1:
            xx = x
            noise = []        
            for point in xx.numpy():
                noise.append(_noise1(point * freq, perm))
            return x, torch.tensor(noise, device = device, dtype = dtype).reshape(xx.shape)
        if dim == 2:
            # print('generateSimplexPP', shape, freq, dim, seed, device, dtype, tileable)
            xx,yy = torch.meshgrid(x,y, indexing = 'xy')
            p = torch.stack((xx,yy), axis = -1).flatten().reshape((-1,2)).numpy()
            noise = []        
            for point in p:
                noise.append(_noise2(point[0] * freq, point[1] * freq, perm))
            return xx, yy, torch.tensor(noise, device = device, dtype = dtype).reshape(xx.shape)
        if dim == 3:
            xx,yy,zz = torch.meshgrid(x,y,z, indexing = 'xy')
            p = torch.stack((xx,yy,zz), axis = -1).flatten().reshape((-1,3)).numpy()
            noise = []        
            for point in p:
                noise.append(_noise3(point[0] * freq, point[1] * freq,point[2] * freq, perm, perm_grad))
            return xx, yy,zz, torch.tensor(noise, device = device, dtype = dtype).reshape(xx.shape)
    else:
        if dim == 1:
            xx = np.sin(x.cpu().numpy() * np.pi)
            yy = np.cos(x.cpu().numpy() * np.pi)
            p = np.stack((xx,yy), axis = -1).flatten().reshape((-1,2))
            noise = []                
            for point in p:
#                 print(point.shape)
                noise.append(_noise2(point[0] * freq / np.pi, point[1] * freq / np.pi, perm))
            return x, torch.tensor(noise, device = device, dtype = dtype).reshape(xx.shape)
        if dim == 2:
            # print('generateSimplexP', shape, freq, dim, seed, device, dtype, tileable)
            noise = np.zeros((x.shape[0], y.shape[0]))
            frequency = 1
            amplitude = 1
            xx,yy = torch.meshgrid(x,y, indexing = 'xy')
            x = x.numpy()
            y = y.numpy()
            for ix in prange(x.shape[0]):
                for iy in prange(y.shape[0]):
                    nx = np.cos(x[ix] * np.pi + np.pi) 
                    ny = np.cos(y[iy] * np.pi + np.pi)
                    nz = np.sin(x[ix] * np.pi + np.pi)
                    nw = np.sin(y[iy] * np.pi + np.pi)
                    noise[ix,iy] += _noise4(freq * nx / np.pi, freq * ny / np.pi, freq * nz / np.pi, freq * nw / np.pi, perm)
            return xx, yy, torch.tensor(noise, device = device, dtype = dtype)
        if dim == 3:
            raise Exception('Not implemented yet (noise is not periodic)')
            xx,yy,zz = torch.meshgrid(x,y,z, indexing = 'xy')
            p = torch.stack((xx,yy,zz), axis = -1).flatten().reshape((-1,3)).numpy()
            noise = []        
            ifreq = int(freq)
            for point in p:                
                noise.append(_noise3periodic(point[0] * ifreq, point[1] * ifreq,point[2] * ifreq, perm, perm_grad, w6 = ifreq, d6 = ifreq, h6 = ifreq))
            return xx, yy,zz, torch.tensor(noise, device = device, dtype = dtype).reshape(xx.shape)
    

def generateOctaveNoise(n, dim = 2, octaves = 4, lacunarity = 2, persistence = 0.5, baseFrequency = 1, tileable = True, kind = 'perlin', device = 'cpu', dtype = torch.float32, seed = 12345, normalized = True):
    freq = baseFrequency
    octave = 1
    amplitude = 1
    noise = torch.zeros([n] * dim, device = device, dtype = dtype)
    for i in range(octaves):
        result = generatePerlin(n, freq, dim = dim, tileable = tileable, device = device, dtype = dtype, seed = seed) if kind == 'perlin' else generateSimplex(n, freq = freq, dim = dim, tileable = tileable, device = device, dtype = dtype, seed = seed)
        noise += amplitude * result[-1]
        freq *= lacunarity
        amplitude *= persistence
    if normalized:
        noise = (noise  - torch.min(noise)) / (torch.max(noise) - torch.min(noise)) * 2 - 1
    return *result[:-1], noise