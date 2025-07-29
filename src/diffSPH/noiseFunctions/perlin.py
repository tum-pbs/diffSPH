import torch
import numpy as np

def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)

def perlinNoise3D(
        shape, res, tileable=(False, False, False),
        interpolant=interpolant, rng = np.random.default_rng(seed=42), device = 'cpu', dtype = torch.float32):
    dx = 2/shape[0]
    x = torch.linspace(-1 + dx / 2, 1 - dx/2, shape[0], device = device, dtype = dtype)
    y = torch.linspace(-1 + dx / 2, 1 - dx/2, shape[1], device = device, dtype = dtype)
    z = torch.linspace(-1 + dx / 2, 1 - dx/2, shape[2], device = device, dtype = dtype)
    xx,yy,zz = torch.meshgrid(x,y,z, indexing = 'xy')
    
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = torch.tensor(grid.transpose(1, 2, 3, 0) % 1).type(dtype).to(device)
    # Gradients
    theta = 2*np.pi*rng.random((res[0] + 1, res[1] + 1, res[2] + 1))
    phi = 2*np.pi*rng.random((res[0] + 1, res[1] + 1, res[2] + 1))
    gradients = np.stack(
        (np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)),
        axis=3
    )
    if tileable[0]:
        gradients[-1,:,:] = gradients[0,:,:]
    if tileable[1]:
        gradients[:,-1,:] = gradients[:,0,:]
    if tileable[2]:
        gradients[:,:,-1] = gradients[:,:,0]
    gradients = torch.tensor(gradients.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)).type(dtype).to(device)
    g000 = gradients[    :-d[0],    :-d[1],    :-d[2]]
    g100 = gradients[d[0]:     ,    :-d[1],    :-d[2]]
    g010 = gradients[    :-d[0],d[1]:     ,    :-d[2]]
    g110 = gradients[d[0]:     ,d[1]:     ,    :-d[2]]
    g001 = gradients[    :-d[0],    :-d[1],d[2]:     ]
    g101 = gradients[d[0]:     ,    :-d[1],d[2]:     ]
    g011 = gradients[    :-d[0],d[1]:     ,d[2]:     ]
    g111 = gradients[d[0]:     ,d[1]:     ,d[2]:     ]
    # Ramps
    n000 = torch.sum(torch.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g000, 3)
    n100 = torch.sum(torch.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g100, 3)
    n010 = torch.sum(torch.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g010, 3)
    n110 = torch.sum(torch.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g110, 3)
    n001 = torch.sum(torch.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g001, 3)
    n101 = torch.sum(torch.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g101, 3)
    n011 = torch.sum(torch.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011, 3)
    n111 = torch.sum(torch.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111, 3)
    # Interpolation
    t = interpolant(grid)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
    n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11
    return xx,yy,zz,((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1)

def perlinNoise2D(shape, res, tileable=(False, False), interpolant=interpolant, rng = np.random.default_rng(seed=42), device = 'cpu', dtype = torch.float32):
    dx = 2/shape[0]
    x = torch.linspace(-1 + dx / 2, 1 - dx/2, shape[0], device = device, dtype = dtype)
    y = torch.linspace(-1 + dx / 2, 1 - dx/2, shape[1], device = device, dtype = dtype)
    xx,yy = torch.meshgrid(x,y, indexing = 'xy')
    
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0],delta[0]), torch.arange(0,res[1], delta[1]), indexing = 'xy'),axis=-1).transpose(0,1) % 1
    # Gradients
    # print('perlinNoise2D', shape, res, tileable, interpolant, rng, device, dtype)
    angles = 2*np.pi*rng.random((res[0]+1, res[1]+1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    gradients = torch.tensor(gradients).type(dtype).to(device)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = torch.sum(torch.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = torch.sum(torch.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = torch.sum(torch.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = torch.sum(torch.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return xx, yy, np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def perlinNoise1D(shape, res, tileable = False, interpolant = interpolant , rng = np.random.default_rng(seed=42), device = 'cpu', dtype = torch.float32):
    xx, yy, noise = perlinNoise2D([shape, shape], [res, res], [tileable, tileable], interpolant = interpolant, rng = rng, device = device, dtype = dtype)
    return xx[0,:], noise[0,:]