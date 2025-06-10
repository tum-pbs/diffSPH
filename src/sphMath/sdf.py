import torch
from sphMath.sdfFunctionality.operators import *
from sphMath.sdfFunctionality.implicitFunctions import *

def getSDF(function):
    if function == 'circle':
        return {'function': torch.vmap(sdCircle, in_dims=(0, *([None] * 1)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'r' : 'Radius of sphere, float'}, 'sample': [1.0]}
    elif function == 'box':
        return {'function': torch.vmap(sdBox, in_dims=(0, *([None] * 1)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'b' : 'Size of box, torch.Tensor'}, 'sample': [torch.tensor([1.0, 1.0])]}
    elif function == 'roundedBox':
        return {'function': sdRoundedBox, 'arguments': {'p' : 'batched Input Position', 'b' : 'Size of box, torch.Tensor', 'r' : 'Radius of corners, torch.Tensor'}, 'sample': [torch.tensor([1.0, 1.0]), torch.tensor([0.1, 0.2, 0.3, 0.4])]}
    elif function == 'orientedBox':
        return {'function': torch.vmap(sdOrientedBox, in_dims=(0, *([None] * 3)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'a' : 'Start of box, torch.Tensor', 'b' : 'End of box, torch.Tensor', 'th' : 'Thickness of box, float'}, 'sample': [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]), 0.1]}
    elif function == 'segment':
        return {'function': torch.vmap(sdSegment, in_dims=(0, *([None] * 2)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'a' : 'Start of segment, torch.Tensor', 'b' : 'End of segment, torch.Tensor'}, 'sample': [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])]}
    elif function == 'rhombus':
        return {'function': torch.vmap(sdRhombus, in_dims=(0, *([None] * 1)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'b' : 'Size of rhombus, torch.Tensor'}, 'sample': [torch.tensor([1.0, 1.0])]}
    elif function == 'trapezoid':
        return {'function': torch.vmap(sdTrapezoid, in_dims=(0, *([None] * 3)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'r1' : 'Top radius, float', 'r2' : 'Bottom radius, float', 'he' : 'Height, float'}, 'sample': [1.0, 0.5, 0.5]}
    elif function == 'parallelogram':
        return {'function': torch.vmap(sdParallelogram, in_dims=(0, *([None] * 3)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'wi' : 'Width, float', 'he' : 'Height, float', 'sk' : 'Skew, float'}, 'sample': [1.0, 0.5, 0.5]}
    elif function == 'equilateralTriangle':
        return {'function': sdEquilateralTriangle, 'arguments': {'p' : 'batched Input Position', 'r' : 'Radius of triangle, float'}, 'sample': [1.0]}
    elif function == 'triangleIsosceles':
        return {'function': torch.vmap(sdTriangleIsosceles, in_dims=(0, *([None] * 1)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'q' : 'Base of triangle, torch.Tensor'}, 'sample': [torch.tensor([0.3, 1.1])]}
    elif function == 'triangle':
        return {'function': torch.vmap(sdTriangle, in_dims=(0, *([None] * 3)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'p0' : 'First point of triangle, torch.Tensor', 'p1' : 'Second point of triangle, torch.Tensor', 'p2' : 'Third point of triangle, torch.Tensor'}, 'sample': [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0])]}
    elif function == 'unevenCapsule':
        return {'function': torch.vmap(sdUnevenCapsule, in_dims=(0, *([None] * 3)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'r1' : 'Top radius, float', 'r2' : 'Bottom radius, float', 'h' : 'Height, float'}, 'sample': [1.0, 0.5, 0.5]}
    elif function == 'pentagon':
        return {'function': torch.vmap(sdPentagon, in_dims=(0, *([None] * 1)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'r' : 'Radius of pentagon, float'}, 'sample': [1.0]}
    elif function == 'hexagon':
        return {'function': torch.vmap(sdHexagon, in_dims=(0, *([None] * 1)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'r' : 'Radius of hexagon, float'}, 'sample': [1.0]}
    elif function == 'octogon':
        return {'function': torch.vmap(sdOctogon, in_dims=(0, *([None] * 1)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'r' : 'Radius of octogon, float'}, 'sample': [1.0]}
    elif function == 'hexagram':
        return {'function': torch.vmap(sdHexagram, in_dims=(0, *([None] * 1)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'r' : 'Radius of hexagram, float'}, 'sample': [1.0]}
    elif function == 'star5':
        return {'function': torch.vmap(sdStar5, in_dims=(0, *([None] * 2)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'r' : 'Radius of star, float', 'rf' : 'Radius of inner star, float'}, 'sample': [1.0, 0.75]}
    elif function == 'star':
        return {'function': torch.vmap(sdStar, in_dims=(0, *([None] * 3)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'r' : 'Radius of star, float', 'n' : 'Number of points, int', 'm' : 'Number of inner points, float'}, 'sample': [1.0, 9, 3]}
    elif function == 'pie':
        return {'function': torch.vmap(sdPie, in_dims=(0, *([None] * 2)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'c' : 'sin/cos of aperture, torch.Tensor', 'r' : 'Radius of pie, float'}, 'sample': [torch.tensor([0.0, 1.0]), 1.0]}
    elif function == 'cutDisk':
        return {'function': torch.vmap(sdCutDisk, in_dims=(0, *([None] * 2)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'r' : 'Radius of disk, float', 'h' : 'Height of disk, float'}, 'sample': [1.0, 0.5]}
    elif function == 'arc':
        return {'function': torch.vmap(sdArc, in_dims=(0, *([None] * 3)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'sc' : 'Start of arc, torch.Tensor', 'ra' : 'Radius of arc, float', 'rb' : 'Radius of arc, float'}, 'sample': [torch.tensor([0.0, 0.0]), 1.0, 0.5]}
    elif function == 'ring':
        return {'function': torch.vmap(sdRing, in_dims=(0, *([None] * 3)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'n' : 'Normal of ring, torch.Tensor', 'r' : 'Radius of ring, float', 'th' : 'Thickness of ring, float'}, 'sample': [torch.tensor([0.0, 1.0]), 1.0, 0.5]}
    elif function == 'horseshoe':
        return {'function': torch.vmap(sdHorseshoe, in_dims=(0, *([None] * 3)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'c' : 'sin/cos of aperture, torch.Tensor', 'r' : 'Radius of horseshoe, float', 'w' : 'Width of horseshoe, float'}, 'sample': [torch.tensor([0.0, 1.0]), 1.0, 0.5]}
    elif function == 'vesica':
        return {'function': sdVesica, 'arguments': {'p' : 'batched Input Position', 'r' : 'Radius of vesica, float', 'd' : 'Distance of vesica, float'}, 'sample': [1.0, 0.5]}
    elif function == 'moon':
        return {'function': torch.vmap(sdMoon, in_dims=(0, *([None] * 3)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'd' : 'Distance of moon, float', 'ra' : 'Radius of moon, float', 'rb' : 'Radius of moon, float'}, 'sample': [1.0, 0.5, 0.5]}
    elif function == 'egg':
        return {'function': sdEgg, 'arguments': {'p' : 'batched Input Position', 'ra' : 'Radius of egg, float', 'rb' : 'Radius of egg, float'}, 'sample': [1.0, 0.5]}
    elif function == 'polygon':
        return {'function': torch.vmap(sdPolygon, in_dims=(0, *([None] * 1)), out_dims=0), 'arguments': {'p' : 'batched Input Position', 'v' : 'Vertices of polygon, torch.Tensor'}, 'sample': [torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5 * np.sqrt(3.0)]]).to(torch.float32)]}
    
sdfFunctions = ['circle', 'box', 'roundedBox', 'orientedBox', 'segment', 'rhombus', 'trapezoid', 'parallelogram', 'equilateralTriangle', 'triangleIsosceles', 'triangle', 'unevenCapsule', 'pentagon', 'hexagon', 'octogon', 'hexagram', 'star5', 'star', 'pie', 'cutDisk', 'arc', 'ring', 'horseshoe', 'vesica', 'moon', 'egg', 'polygon']



def sampleSDF(x, sdf, invert = False):
    x_ = x.clone()
    x_.requires_grad = True
    d = sdf(x_)
    grad = torch.autograd.grad(outputs = d, inputs = x_, grad_outputs = torch.ones_like(d), create_graph = True, retain_graph = True)

    grad = grad[0].detach() if x.requires_grad == False else grad[0]
    grad = torch.nn.functional.normalize(grad, dim = -1)

    if invert:
        return -d.detach() if x.requires_grad == False else -d, -grad if x.requires_grad == False else -grad
    else:
        return d.detach() if x.requires_grad == False else d, grad if x.requires_grad == False else grad

