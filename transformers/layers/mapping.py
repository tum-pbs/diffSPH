import torch 
import numpy as np

def mapToSpherical(positions):
    x = positions[:,0]
    y = positions[:,1]
    z = positions[:,2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(y, x)
    phi = torch.acos(z / (r + 1e-7))
    
    return torch.vstack((r,theta,phi)).mT


def ballToCylinder(positions):
    r = torch.linalg.norm(positions, dim = 1)
    xy = torch.linalg.norm(positions[:,:2], dim = 1)
    absz = torch.abs(positions[:,2])

#     debugPrint(r)
#     debugPrint(xy)
#     debugPrint(absz)

    x = positions[:,0]
    y = positions[:,1]
    z = positions[:,2]

    termA = torch.zeros_like(positions)

    eps = 1e-7

    xB = x * r / (xy + eps)
    yB = y * r / (xy + eps)
    zB = 3 / 2 * z
    termB = torch.vstack((xB, yB, zB)).mT

    xC = x * torch.sqrt(3 * r / (r + absz + eps))
    yC = y * torch.sqrt(3 * r / (r + absz + eps))
    zC = torch.sign(z) * r
    termC = torch.vstack((xC, yC, zC)).mT

    mapped = torch.zeros_like(positions)

    maskA = r < eps
    maskB = torch.logical_and(torch.logical_not(maskA), 5/4 * z**2 <= x**2 + y**2)
    maskC = torch.logical_and(torch.logical_not(maskA), torch.logical_not(maskB))

    mapped[maskB] = termB[maskB]
    mapped[maskC] = termC[maskC]

#     debugPrint(mapped)
    return mapped
# debugPrint(cylinderPositions)

def cylinderToCube(positions):
    x = positions[:,0]
    y = positions[:,1]
    z = positions[:,2]
    xy = torch.linalg.norm(positions[:,:2], dim = 1)
    eps = 1e-7

    termA = torch.vstack((torch.zeros_like(x), torch.zeros_like(y), z)).mT
    # debugPrint(termA)

    xB = torch.sign(x) * xy
    yB = 4. / np.pi * torch.sign(x) * xy * torch.atan(y/(x+eps))
    zB = z
    termB = torch.vstack((xB, yB, zB)).mT

    xC = 4. / np.pi * torch.sign(y) * xy * torch.atan(x / (y + eps))
    yC = torch.sign(y) * xy
    zC = z
    termC = torch.vstack((xC, yC, zC)).mT

    maskA = torch.logical_and(torch.abs(x) < eps, torch.abs(y) < eps)
    maskB = torch.logical_and(torch.logical_not(maskA), torch.abs(y) <= torch.abs(x))
    maskC = torch.logical_and(torch.logical_not(maskA), torch.logical_not(maskB))

    # debugPrint(torch.sum(maskA))
    # debugPrint(torch.sum(maskB))
    # debugPrint(torch.sum(maskC))


    mapped = torch.zeros_like(positions)
    mapped[maskA] = termA[maskA]
    mapped[maskB] = termB[maskB]
    mapped[maskC] = termC[maskC]
    
    return mapped

def mapToSpherePreserving(positions):
    cylinderPositions = ballToCylinder(positions)
    cubePositions = cylinderToCube(cylinderPositions)
    return cubePositions



def map_positions(edge_attr, coordinateMapping):
    if edge_attr.shape[1] > 1:
        expanded = torch.hstack((edge_attr, torch.zeros_like(edge_attr[:,0])[:,None])) if edge_attr.shape[1] == 2 else edge_attr
        if coordinateMapping == 'polar':
            spherical = mapToSpherical(expanded)
            if expanded.shape[1] == 2:
                mapped = torch.vstack((spherical[:,0] * 2. - 1.,spherical[:,1] / np.pi)).mT
            else:
                mapped = torch.vstack((spherical[:,0] * 2. - 1.,spherical[:,1] / np.pi,spherical[:,2] / np.pi)).mT
        if coordinateMapping == 'cartesian':
            mapped = expanded
        if coordinateMapping == 'preserving':
            cubePositions = mapToSpherePreserving(expanded)
            mapped = cubePositions
            
        return mapped
    else:
        return edge_attr