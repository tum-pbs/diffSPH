# from .util import _extrapolate1
# from .constants import STRETCH_CONSTANT1, SQUISH_CONSTANT1, NORM_CONSTANT1
from math import floor
from ctypes import c_int64
from numba import njit, prange

from .simplex2d import _noise2

@njit(cache=True)
def _noise1(x, perm):
    return _noise2(x,0, perm) 