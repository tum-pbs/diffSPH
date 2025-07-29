# It all started with this gist, once upon a time:
# https://gist.github.com/KdotJPG/b1270127455a94ac5d19

from .constants import *
from math import floor
from ctypes import c_int64

try:
    from numba import njit, prange
except ImportError:
    prange = range

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper


def _overflow(x):
    # Since normal python ints and longs can be quite humongous we have to use
    # self hack to make them be able to _overflow.
    # Using a np.int64 won't work either, as it will still complain with:
    # "_overflowError: int too big to convert"
    return c_int64(x).value


def _init(seed):
    # Have to zero fill so we can properly loop over it later
    perm = np.zeros(256, dtype=np.int64)
    perm_grad_index3 = np.zeros(256, dtype=np.int64)
    source = np.arange(256)
    # Generates a proper permutation (i.e. doesn't merely perform N
    # successive pair swaps on a base array)
    seed = _overflow(seed * 6364136223846793005 + 1442695040888963407)
    seed = _overflow(seed * 6364136223846793005 + 1442695040888963407)
    seed = _overflow(seed * 6364136223846793005 + 1442695040888963407)
    for i in range(255, -1, -1):
        seed = _overflow(seed * 6364136223846793005 + 1442695040888963407)
        r = int((seed + 31) % (i + 1))
        if r < 0:
            r += i + 1
        perm[i] = source[r]
        perm_grad_index3[i] = int((perm[i] % (len(GRADIENTS3) / 3)) * 3)
        source[r] = source[i]
    return perm, perm_grad_index3


@njit(cache=True)
def _extrapolate2(perm, xsb, ysb, dx, dy):
    index = perm[(perm[xsb & 0xFF] + ysb) & 0xFF] & 0x0E
    g1, g2 = GRADIENTS2[index : index + 2]
    return g1 * dx + g2 * dy

@njit(cache=True)
def _extrapolate3periodic(perm, perm_grad_index3, xsb, ysb, zsb, dx, dy, dz, w6, h6, d6, wrap):
    bSum = xsb + ysb + zsb
    xc = (wrap / 2 * xsb + bSum) / (wrap * wrap / 2) // w6
    yc = (wrap / 2 * ysb + bSum) / (wrap * wrap / 2) // h6
    zc = (wrap / 2 * zsb + bSum) / (wrap * wrap / 2) // d6
    
    xsbm = int((-(wrap - 1) * w6 * xc) + (h6 * yc) + (d6 * zc) + xsb)
    ysbm = int((w6 * xc) + (-(wrap - 1) * h6 * yc) + (d6 * zc) + ysb)
    zsbm = int((w6 * xc) + (h6 * yc) + (-(wrap - 1) * d6 * zc) + zsb)
		
    index = perm_grad_index3[(perm[(perm[xsbm & 0xFF] + ysbm) & 0xFF] + zsbm) & 0xFF]
    g1, g2, g3 = GRADIENTS3[index : index + 3]
    return g1 * dx + g2 * dy + g3 * dz


@njit(cache=True)
def _extrapolate3(perm, perm_grad_index3, xsb, ysb, zsb, dx, dy, dz):
    index = perm_grad_index3[(perm[(perm[xsb & 0xFF] + ysb) & 0xFF] + zsb) & 0xFF]
    g1, g2, g3 = GRADIENTS3[index : index + 3]
    return g1 * dx + g2 * dy + g3 * dz


@njit(cache=True)
def _extrapolate4(perm, xsb, ysb, zsb, wsb, dx, dy, dz, dw):
    index = perm[(perm[(perm[(perm[xsb & 0xFF] + ysb) & 0xFF] + zsb) & 0xFF] + wsb) & 0xFF] & 0xFC
    g1, g2, g3, g4 = GRADIENTS4[index : index + 4]
    return g1 * dx + g2 * dy + g3 * dz + g4 * dw

# from .simplex2d import _noise2
# from .simplex3d import _noise3
# from .simplex4d import _noise4
# @njit(cache=True, parallel=True)
# def _noise2a(x, y, perm):
#     noise = np.empty((y.size, x.size), dtype=np.double)
#     for y_i in prange(y.size):
#         for x_i in prange(x.size):
#             noise[y_i, x_i] = _noise2(x[x_i], y[y_i], perm)
#     return noise


# @njit(cache=True, parallel=True)
# def _noise3a(x, y, z, perm, perm_grad_index3):
#     noise = np.empty((z.size, y.size, x.size), dtype=np.double)
#     for z_i in prange(z.size):
#         for y_i in prange(y.size):
#             for x_i in prange(x.size):
#                 noise[z_i, y_i, x_i] = _noise3(x[x_i], y[y_i], z[z_i], perm, perm_grad_index3)
#     return noise


# @njit(cache=True, parallel=True)
# def _noise4a(x, y, z, w, perm):
#     noise = np.empty((w.size, z.size, y.size, x.size), dtype=np.double)
#     for w_i in prange(w.size):
#         for z_i in prange(z.size):
#             for y_i in prange(y.size):
#                 for x_i in prange(x.size):
#                     noise[w_i, z_i, y_i, x_i] = _noise4(x[x_i], y[y_i], z[z_i], w[w_i], perm)
#     return noise


# ################################################################################
# # There be dragons in the depths below..




