import numba
import numpy as np

@numba.jit(nopython=True)
def rescale_and_lookup(data, scale, offset, lut):
    # data should be floating point and 2d
    # lut is 1d
    vmin, vmax = 0, lut.shape[0] - 1
    out = np.empty_like(data, dtype=lut.dtype)
    for (x, y) in np.nditer((data, out)):
        val = (x - offset) * scale
        val = min(max(val, vmin), vmax)
        y[...] = lut[int(val)]
    return out

@numba.jit(nopython=True)
def rescale_and_clip(data, scale, offset, vmin, vmax):
    # vmin and vmax <= 255
    out = np.empty_like(data, dtype=np.uint8)
    for (x, y) in np.nditer((data, out)):
        val = (x - offset) * scale
        val = min(max(val, vmin), vmax)
        y[...] = val
    return out

@numba.jit(nopython=True)
def numba_take(lut, data):
    # numba supports only the 1st two arguments of np.take
    return np.take(lut, data)
