import torch
import numpy as np

def translate_sdf(sdf, translation):
    return lambda p: sdf(p - translation)
def rotate_sdf(sdf, angle):
    return lambda p: sdf(torch.stack([p[:,0] * np.cos(angle) - p[:,1] * np.sin(angle), p[:,0] * np.sin(angle) + p[:,1] * np.cos(angle)], dim=1))
def scale_sdf(sdf, scale):
    return lambda p: sdf(p / scale) *scale
def op_union(a, b):
    return lambda p: torch.min(a(p), b(p))
def op_intersection(a, b):
    return lambda p: torch.max(a(p), b(p))
def op_difference(a, b):
    return lambda p: torch.max(a(p), -b(p))
def op_smooth_union(a, b, k):
    def smooth_union(p):
        d1 = a(p)
        d2 = b(p)
        h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
        return torch.lerp(d2, d1, h) - k * h * (1.0 - h)
    return smooth_union
def op_smooth_intersection(a, b, k):
    def smooth_intersection(p):
        d1 = a(p)
        d2 = b(p)
        h = torch.clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
        return torch.lerp(d2, d1, h) + k * h * (1.0 - h)
    return smooth_intersection
def op_smooth_difference(a, b, k):
    def smooth_difference(p):
        d1 = a(p)
        d2 = b(p)
        h = torch.clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0)
        return torch.lerp(d2, -d1, h) + k * h * (1.0 - h)
    return smooth_difference
def op_twist(a, k):
    def twist(p):
        c = np.cos(k * p[:,1])
        s = np.sin(k * p[:,1])
        q = torch.stack([c * p[:,0] - s * p[:,1], s * p[:,0] + c * p[:,1]], dim=1)
        return a(q)
    return twist
def op_bend(a, k):
    def bend(p):
        c = np.cos(k * p[:,0])
        s = np.sin(k * p[:,0])
        q = torch.stack([c * p[:,0] - s * p[:,1], s * p[:,0] + c * p[:,1]], dim=1)
        return a(q)
    return bend
def op_taper(a, k):
    def taper(p):
        return a(torch.stack([p[:,0] * k, p[:,1]], dim=1))
    return taper
def op_shear(a, k):
    def shear(p):
        return a(torch.stack([p[:,0] + k * p[:,1], p[:,1]], dim=1))
    return shear
def op_scale(a, k):
    def scale(p):
        return a(p / k) * k
    return scale
def op_translate(a, t):
    def translate(p):
        return a(p - t)
    return translate
def op_rotate(a, angle):
    def rotate(p):
        c = np.cos(angle)
        s = np.sin(angle)
        q = torch.stack([c * p[:,0] - s * p[:,1], s * p[:,0] + c * p[:,1]], dim=1)
        return a(q)
    return rotate
def op_mirror(a, axis):
    def mirror(p):
        return a(torch.stack([p[:,0], axis - p[:,1]], dim=1))
    return mirror
def op_flip(a, axis):
    def flip(p):
        return a(torch.stack([axis - p[:,0], p[:,1]], dim=1))
    return flip
def op_invert(a):
    def invert(p):
        return -a(p)
    return invert
def op_shell(a, thickness):
    def shell(p):
        return torch.abs(a(p)) - thickness
    return shell
def op_round(a, radius):
    def round(p):
        return a(p) - radius
    return round
def op_blend(a,b,l):
    def blend(p):
        return torch.lerp(a(p), b(p), l)
    return blend
def op_select(a, b, k):
    def select(p):
        return torch.where(p[:,0] < k, a(p), b(p))
    return select

operatorDict ={
    'union': op_union,
    'intersection': op_intersection,
    'difference': op_difference,
    'smooth_union': op_smooth_union,
    'smooth_intersection': op_smooth_intersection,
    'smooth_difference': op_smooth_difference,
    'twist': op_twist,
    'bend': op_bend,
    'taper': op_taper,
    'shear': op_shear,
    'scale': op_scale,
    'translate': op_translate,
    'rotate': op_rotate,
    'mirror': op_mirror,
    'flip': op_flip,
    'invert': op_invert,
    'shell': op_shell,
    'round': op_round,
    'blend': op_blend,
    'select': op_select
}