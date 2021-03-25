import numpy as np
import numba

@numba.guvectorize(
    '(n),(),(),(),()->(n)', nopython=True)
def rescale_clip(xx, scale, offset, vmin, vmax, yy):
    for i in range(xx.size):
        val = (xx[i] - offset) * scale
        yy[i] = min(max(val, vmin), vmax)

@numba.guvectorize(
    '(n),(),()->(n)', nopython=True)
def rescale_noclip(xx, scale, offset, yy):
    for i in range(xx.size):
        yy[i] = (xx[i] - offset) * scale


def rescaleData(data, scale, offset, dtype, clip):
    data_out = np.empty_like(data, dtype=dtype)
    if clip is None:
        return rescale_noclip(data, scale, offset, out=data_out)
    else:
        return rescale_clip(data, scale, offset, clip[0], clip[1], out=data_out)