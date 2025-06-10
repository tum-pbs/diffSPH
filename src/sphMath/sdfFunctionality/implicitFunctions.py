import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore")
def batchedSDF(fn, p, *args):
    return torch.vmap(fn, in_dims=(0, *([None] * len(args))), out_dims=0)(p, *args)

def sdEquilateralTriangle(p, r):
    k = torch.sqrt(torch.tensor(3.0, dtype = p.dtype, device = p.device))
    p[:, 0] = torch.abs(p[:, 0]) - r
    p[:, 1] = p[:, 1] + r / k
    mask = p[:, 0] + k * p[:, 1] > 0.0
    p[mask] = torch.stack([p[mask, 0] - k * p[mask, 1], -k * p[mask, 0] - p[mask, 1]], dim=1) / 2.0
    p[:, 0] -= torch.clamp(p[:, 0], -2.0 * r, 0.0)
    return -torch.norm(p, dim=1) * torch.sign(p[:, 1])
def sdTriangleIsosceles(p, q):
    p = torch.stack((torch.abs(p[0]), p[1]))
    a = p - q * torch.clamp(torch.dot(p, q) / torch.dot(q, q), 0.0, 1.0)
    b = p - q * torch.stack([torch.clamp(p[0] / q[0], 0.0, 1.0), torch.tensor(1.0, dtype = p.dtype, device = p.device)])
    s = -torch.sign(q[1])
    d = torch.min(torch.stack([torch.dot(a, a), s * (p[0] * q[1] - p[1] * q[0])]),
                  torch.stack([torch.dot(b, b), s * (p[1] - q[1])]))
    return -torch.sqrt(d[0]) * torch.sign(d[1])
def sdTriangle(p, p0, p1, p2):
    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p0 - p2
    v0 = p - p0
    v1 = p - p1
    v2 = p - p2
    pq0 = v0 - e0 * torch.clamp(torch.dot(v0, e0) / torch.dot(e0, e0), 0.0, 1.0)
    pq1 = v1 - e1 * torch.clamp(torch.dot(v1, e1) / torch.dot(e1, e1), 0.0, 1.0)
    pq2 = v2 - e2 * torch.clamp(torch.dot(v2, e2) / torch.dot(e2, e2), 0.0, 1.0)
    s = torch.sign(e0[0] * e2[1] - e0[1] * e2[0])
    d = torch.min(
        torch.min(torch.stack([torch.dot(pq0, pq0), s * (v0[0] * e0[1] - v0[1] * e0[0])]), torch.stack([torch.dot(pq1, pq1), s * (v1[0] * e1[1] - v1[1] * e1[0])])),
                  torch.stack([torch.dot(pq2, pq2), s * (v2[0] * e2[1] - v2[1] * e2[0])]))
    return -torch.sqrt(d[0]) * torch.sign(d[1])
def sdUnevenCapsule(p, r1, r2, h):
    p = torch.stack((torch.abs(p[0]), p[1]))
    b = (r1 - r2) / h
    a = torch.sqrt(torch.tensor(1.0, dtype = p.dtype, device = p.device) - b * b)
    k = torch.dot(p, torch.tensor([-b, a], dtype = p.dtype, device = p.device))
    mask1 = k < 0.0
    mask2 = k > a * h
    dist1 = torch.linalg.norm(p, dim=-1) - r1
    dist2 = torch.linalg.norm(p - torch.tensor([0.0, h], dtype = p.dtype, device = p.device), dim=-1) - r2
    dist3 = torch.dot(p, torch.tensor([a, b], dtype = p.dtype, device = p.device)) - r1
    return torch.where(mask1, dist1, torch.where(mask2, dist2, dist3))
def sdPentagon(p, r):
    k = torch.tensor([0.809016994, 0.587785252, 0.726542528], dtype = p.dtype, device = p.device)
    p = torch.stack((torch.abs(p[0]), p[1]))
    p -= 2.0 * torch.min(torch.dot(torch.stack([-k[0], k[1]]), p), torch.tensor(0.0, dtype = p.dtype, device = p.device)) * torch.stack([-k[0], k[1]])
    p -= 2.0 * torch.min(torch.dot(torch.stack([k[0], k[1]]), p), torch.tensor(0.0, dtype = p.dtype, device = p.device)) * torch.stack([k[0], k[1]])
    p -= torch.stack([torch.clamp(p[0], -r * k[2], r * k[2]), torch.tensor(r, dtype = p.dtype, device = p.device)])
    return torch.norm(p) * torch.sign(p[1])
def sdHexagon(p, r):
    k = torch.tensor([-0.866025404, 0.5, 0.577350269], dtype = p.dtype, device = p.device)
    p = torch.abs(p)
    p = p - 2.0 * torch.min(torch.dot(k[:2], p), torch.tensor(0.0, dtype = p.dtype, device = p.device)) * k[:2]
    p = p - torch.stack([torch.clamp(p[0], -k[2] * r, k[2] * r), torch.tensor(r, dtype = p.dtype, device = p.device)])
    return torch.norm(p) * torch.sign(p[1])
def sdOctogon(p, r):
    k = torch.tensor([-0.9238795325, 0.3826834323, 0.4142135623], dtype = p.dtype, device = p.device)
    p = torch.abs(p)
    p -= 2.0 * torch.min(torch.dot(torch.tensor([k[0], k[1]], dtype = p.dtype, device = p.device), p), torch.tensor(0.0, dtype = p.dtype, device = p.device)) * torch.stack([k[0], k[1]])
    p -= 2.0 * torch.min(torch.dot(torch.tensor([-k[0], k[1]], dtype = p.dtype, device = p.device), p), torch.tensor(0.0, dtype = p.dtype, device = p.device)) * torch.stack([-k[0], k[1]])
    p -= torch.stack([torch.clamp(p[0], -k[2] * r, k[2] * r), torch.tensor(r, dtype = p.dtype, device = p.device)])
    return torch.norm(p) * torch.sign(p[1])

def sdHexagram(p, r):
    k = torch.tensor([-0.5, 0.8660254038, 0.5773502692, 1.7320508076], dtype = p.dtype, device = p.device)
    p = torch.abs(p)
    p -= 2.0 * torch.min(torch.dot(torch.stack([k[0], k[1]]), p), torch.tensor(0.0, dtype = p.dtype, device = p.device)) * torch.stack([k[0], k[1]])
    p -= 2.0 * torch.min(torch.dot(torch.stack([k[1], k[0]]), p), torch.tensor(0.0, dtype = p.dtype, device = p.device)) * torch.stack([k[1], k[0]])
    p -= torch.stack([torch.clamp(p[0], r * k[2], r * k[3]), torch.tensor(r, dtype = p.dtype, device = p.device)])
    return torch.norm(p) * torch.sign(p[1])
def sdStar5(p, r, rf):
    k1 = torch.tensor([0.809016994375, -0.587785252292], dtype = p.dtype, device = p.device)
    k2 = torch.tensor([-k1[0], k1[1]], dtype = p.dtype, device = p.device)
    p = torch.stack((torch.abs(p[0]), p[1]))
    p -= 2.0 * torch.max(torch.dot(k1, p), torch.tensor(0.0, dtype = p.dtype, device = p.device)) * k1
    p -= 2.0 * torch.max(torch.dot(k2, p), torch.tensor(0.0, dtype = p.dtype, device = p.device)) * k2
    p = torch.stack((torch.abs(p[0]), p[1]))
    p[1] -= r
    ba = rf * torch.stack([-k1[1], k1[0]]) - torch.tensor([0, 1], dtype = p.dtype, device = p.device)
    h = torch.clamp(torch.dot(p, ba) / torch.dot(ba, ba), 0.0, r)
    return torch.norm(p - ba * h) * torch.sign(p[1] * ba[0] - p[0] * ba[1])
def sdStar(p, r, n, m):
    an = 3.141593 / float(n)
    en = 3.141593 / m
    acs = torch.tensor([np.cos(an, dtype = p.dtype, device = p.device), np.sin(an)]).to(p.dtype)
    ecs = torch.tensor([np.cos(en, dtype = p.dtype, device = p.device), np.sin(en)]).to(p.dtype)

    bn = torch.remainder(torch.atan2(p[0], p[1]), 2.0 * an) - an
    p = torch.norm(p) * torch.stack([torch.cos(bn), torch.abs(torch.sin(bn))])
    p -= r * acs
    p += ecs * torch.clamp(-torch.dot(p, ecs), 0.0, r * acs[1] / ecs[1])
    return torch.norm(p) * torch.sign(p[0])
def sdPie(p, c, r):
    p = torch.stack((torch.abs(p[0]), p[1]))
    l = torch.norm(p) - r
    m = torch.norm(p - c * torch.clamp(torch.dot(p, c), 0.0, r))  # c=sin/cos of aperture
    return torch.max(l, m * torch.sign(c[1] * p[0] - c[0] * p[1]))
def sdCutDisk(p, r, h):
    w = float(np.sqrt(r*r - h*h))
    p = torch.stack((torch.abs(p[0]), p[1]))
    s = torch.max((h - r) * p[0] * p[0] + w * w * (h + r - 2.0 * p[1]), h * p[0] - w * p[1])
    return torch.where(s < 0.0, torch.norm(p) - r, torch.where(p[0] < w, h - p[1], torch.norm(p - torch.tensor([w, h], dtype = p.dtype, device = p.device))))
def sdArc(p, sc, ra, rb):
    p = torch.stack((torch.abs(p[0]), p[1]))
    return torch.where(sc[1] * p[0] > sc[0] * p[1], torch.linalg.norm(p - sc * ra), torch.abs(torch.linalg.norm(p) - ra)) - rb
def sdRing(p, n, r, th):
    p = torch.stack((torch.abs(p[0]), p[1]))
    M = torch.tensor(np.array([[n[0], n[1]], [-n[1], n[0]]], dtype = p.dtype, device = p.device)).to(torch.float32)
    p = torch.matmul(M, p)
    return torch.max(torch.abs(torch.linalg.norm(p) - r) - th * 0.5,
                     torch.linalg.norm(torch.stack([p[0], torch.max(torch.tensor(0.0, dtype = p.dtype, device = p.device), torch.abs(r - p[1]) - th * 0.5)])) * torch.sign(p[0]))
def sdHorseshoe(p, c, r, w):
    # p = torch.stack((torch.abs(p[0]), p[1]))
    p = torch.stack((torch.abs(p[0]), p[1]))
    l = torch.linalg.norm(p)
    p = torch.matmul(torch.tensor([[-c[0], c[1]], [c[1], c[0]]], dtype = p.dtype, device = p.device), p.T).T
    p = torch.stack([torch.where((p[1] > 0.0) | (p[0] > 0.0), p[0], l * torch.sign(-c[0])),
                     torch.where(p[0] > 0.0, p[1], l)])
    p = torch.stack([p[0], torch.abs(p[1] - r)]) - w
    return torch.linalg.norm(torch.max(p, torch.tensor(0.0, dtype = p.dtype, device = p.device))) + torch.min(torch.tensor(0.0, dtype = p.dtype, device = p.device), torch.max(p[0], p[1]))
def sdVesica(p, r, d):
    p = torch.abs(p)
    b = float(np.sqrt(r*r - d*d))
    return torch.where((p[:, 1] - b) * d > p[:, 0] * b, torch.linalg.norm(p - torch.tensor([0.0, b], dtype = p.dtype, device = p.device), dim = -1), torch.linalg.norm(p - torch.tensor([-d, 0.0], dtype = p.dtype, device = p.device), dim = -1) - r)
def sdMoon(p, d, ra, rb):
      p[1] = torch.abs(p[1])
      a = (ra * ra - rb * rb + d * d) / (2.0 * d)
      b = torch.sqrt(torch.max(torch.tensor(ra * ra - a * a, dtype = p.dtype, device = p.device), torch.tensor(0.0, dtype = p.dtype, device = p.device)))
      condition = d * (p[0] * b - p[1] * a) > d * d * torch.max(b - p[1], torch.tensor(0.0, dtype = p.dtype, device = p.device))
      return torch.where(condition, torch.linalg.norm(p - torch.tensor([a, b], dtype = p.dtype, device = p.device)), torch.max(torch.linalg.norm(p) - ra, -(torch.linalg.norm(p - torch.tensor([d, 0.0], dtype = p.dtype, device = p.device)) - rb)))
def sdRoundedCross(p, h):
  k = 0.5 * (h + 1.0 / h)
  p = torch.abs(p)
  condition = (p[:, 0] < 1.0) & (p[:, 1] < p[:, 0] * (k - h) + h)
  return torch.where(condition, k - torch.sqrt(torch.sum((p - torch.tensor([1.0, k], dtype = p.dtype, device = p.device))**2, dim=1)), torch.sqrt(torch.min(torch.sum((p - torch.tensor([0.0, h], dtype = p.dtype, device = p.device))**2, dim=1), torch.sum((p - torch.tensor([1.0, 0.0], dtype = p.dtype, device = p.device))**2, dim=1))))
def sdEgg(p, ra, rb):
    k = torch.sqrt(torch.tensor(3.0, dtype = p.dtype, device = p.device))
    p[:, 0] = torch.abs(p[:, 0])
    r = ra - rb
    return torch.where(p[:, 1] < 0.0, torch.linalg.norm(p, dim=-1) - r,                        
                       torch.where(k * (p[:, 0] + r) < p[:, 1], torch.linalg.norm(torch.stack([p[:, 0], p[:, 1] - k * r], dim = 1), dim=-1),
                                  torch.linalg.norm(torch.stack([p[:, 0] + r, p[:, 1]], dim = 1), dim=-1) - 2.0 * r)) - rb

def sdPolygon(p, v):
    d = torch.dot(p - v[0], p - v[0])
    s = torch.tensor(-1.0, dtype = p.dtype, device = p.device)
    for i in range(len(v)):
        j = i - 1 if i > 0 else len(v) - 1
        e = v[j] - v[i]
        w = p - v[i]
        b = w - e * torch.clamp(torch.dot(w, e) / torch.dot(e, e), 0.0, 1.0)
        d = torch.min(d, torch.dot(b, b))
        c = torch.stack([p[1] >= v[i][1], p[1] < v[j][1], e[0] * w[1] > e[1] * w[0]])
        s = torch.where(torch.all(c) | torch.all(~c), s, -s)

        # if torch.all(c) or torch.all(~c):
            # s *= -1.0
    return s * torch.sqrt(d)

def sdCircle(p : torch.Tensor, r : float):
    return torch.linalg.norm(p, dim=-1) - r
def sdBox(p : torch.Tensor, b : torch.Tensor):
    q = torch.abs(p) - b
    return torch.linalg.norm(torch.clamp(q, min = 0.0), dim=-1) + torch.clamp(torch.max(q, dim=-1)[0], max = 0.0)
def sdRoundedBox(p : torch.Tensor, b : torch.Tensor, r : torch.Tensor):
    r = r.repeat(p.shape[0], 1)
    print('r', r.shape)
    r[:,:2] = torch.where((p[:,0] > 0.0).repeat(2,1).mT, r[:,:2], r[:,2:])
    r[:,0] = torch.where(p[:,1] > 0.0, r[:,0], r[:,1])

    q = torch.abs(p) - b + r[:,0].view(-1,1)
    a1 = torch.clamp(torch.max(q, dim = -1)[0], max = 0.0)
    a2 = torch.linalg.norm(torch.clamp(q, min = 0.0), dim=-1) - r[:,0]
    return a1 + a2
def sdOrientedBox(p, a, b, th):
    l = torch.linalg.norm(b - a)
    d = (b - a) / l
    q = (p - (a + b) * 0.5)
    q = torch.matmul(torch.tensor([[d[0], -d[1]], [d[1], d[0]]], dtype = p.dtype, device = p.device), q)
    print(q.shape)
    q = torch.abs(q) - torch.tensor([l, th], dtype = p.dtype, device = p.device) * 0.5
    return torch.linalg.norm(torch.max(q, torch.tensor(0.0, dtype = p.dtype, device = p.device))) + torch.min(torch.max(q[0], q[1]), torch.tensor(0.0, dtype = p.dtype, device = p.device))
def sdSegment(p, a, b):
    pa = p - a
    ba = b - a
    h = torch.clamp(torch.dot(pa, ba) / torch.dot(ba, ba), 0.0, 1.0)
    return torch.linalg.norm(pa - ba * h)


def ndot(a, b):
    return a[0] * b[0] - a[1] * b[1]

def sdRhombus(p, b):
    p = torch.abs(p)
    h = torch.clamp(ndot(b - 2.0 * p, b) / torch.dot(b, b), -1.0, 1.0)
    d = torch.linalg.norm(p - 0.5 * b * torch.stack([1.0 - h, 1.0 + h]))
    return d * torch.sign(p[0] * b[1] + p[ 1] * b[ 0] - b[ 0] * b[ 1])
def sdTrapezoid(p, r1, r2, he):
    k1 = torch.tensor([r2, he], dtype = p.dtype, device = p.device)
    k2 = torch.tensor([r2 - r1, 2.0 * he], dtype = p.dtype, device = p.device)
    p = torch.stack((torch.abs(p[0]), p[1]))
    ca = torch.stack([p[0] - torch.min(p[0], torch.where(p[1] < 0.0, r1, r2)), torch.abs(p[1]) - he])
    cb = p - k1 + k2 * torch.clamp(torch.dot(k1 - p, k2) / torch.dot(k2, k2), 0.0, 1.0)
    s = torch.where((cb[0] < 0.0) & (ca[1] < 0.0), -1.0, 1.0)
    return s * torch.sqrt(torch.min(torch.dot(ca, ca), torch.dot(cb, cb)))
def sdParallelogram(p, wi, he, sk):
    e = torch.tensor([sk, he], dtype = p.dtype, device = p.device)
    p = torch.where(p[1] < 0.0, -p, p)
    w = p - e
    w[0] -= torch.clamp(w[0], -wi, wi)
    d = torch.stack((torch.dot(w,w), -w[1]))
    s = p[0] * e[1] - p[1] * e[0]
    p = torch.where(s < 0.0, -p, p)
    v = p - torch.tensor([wi, 0.0], dtype = p.dtype, device = p.device)
    v -= e * torch.clamp(torch.dot(v, e) / torch.dot(e, e), -1.0, 1.0)
    d = torch.min(d, torch.stack((torch.dot(v,v), wi * he - torch.abs(s))))
    return torch.sqrt(d[ 0]) * torch.sign(-d[1])

functionDict = {
    'circle': sdCircle,
    'box': sdBox,
    'roundedBox': sdRoundedBox,
    'orientedBox': sdOrientedBox,
    'segment': sdSegment,
    'rhombus': sdRhombus,
    'trapezoid': sdTrapezoid,
    'parallelogram': sdParallelogram,
    'equilateralTriangle': sdEquilateralTriangle,
    'triangleIsosceles': sdTriangleIsosceles,
    'triangle': sdTriangle,
    'unevenCapsule': sdUnevenCapsule,
    'pentagon': sdPentagon,
    'hexagon': sdHexagon,
    'octogon': sdOctogon,
    'hexagram': sdHexagram,
    'star5': sdStar5,
    'star': sdStar,
    'pie': sdPie,
    'cutDisk': sdCutDisk,
    'arc': sdArc,
    'ring': sdRing,
    'horseshoe': sdHorseshoe,
    'vesica': sdVesica,
    'moon': sdMoon,
    'egg': sdEgg,
    'polygon': sdPolygon
}