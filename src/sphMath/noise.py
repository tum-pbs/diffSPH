from sphMath.noiseFunctions.generator import generateOctaveNoise
import torch

def generateNoise(n, dim = 2, octaves = 4, lacunarity = 2, persistence = 0.5, baseFrequency = 1, tileable = True, kind = 'perlin', device = 'cpu', dtype = torch.float32, seed = 12345, normalized = True):
    return generateOctaveNoise(n, dim, octaves, lacunarity, persistence, baseFrequency, tileable, kind, device, dtype, seed, normalized)