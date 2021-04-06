import numpy as np
import numba

@numba.guvectorize(
    '(n),(),(),(),()->(n)', nopython=True)
def rescale_clip_uint8(xx, scale, offset, vmin, vmax, yy):
    for i in range(xx.size):
        val = (xx[i] - offset) * scale
        yy[i] = min(max(val, vmin), vmax)

@numba.guvectorize(
    '(n),(),()->(n)', nopython=True)
def rescale_noclip_uint8(xx, scale, offset, yy):
    for i in range(xx.size):
        yy[i] = (xx[i] - offset) * scale

@numba.guvectorize(
    '(n),(),(),(),()->(n)', nopython=True)
def rescale_clip_uint16(xx, scale, offset, vmin, vmax, yy):
    for i in range(xx.size):
        val = (xx[i] - offset) * scale
        yy[i] = min(max(val, vmin), vmax)

@numba.guvectorize(
    '(n),(),()->(n)', nopython=True)
def rescale_noclip_uint16(xx, scale, offset, yy):
    for i in range(xx.size):
        yy[i] = (xx[i] - offset) * scale


def rescaleData(data, scale, offset, dtype, clip):
    data_out = np.empty_like(data, dtype=dtype)
    if dtype == np.uint8:
        if clip is None:
            rescale_noclip_uint8(data, scale, offset, out=data_out)
        else:
            rescale_clip_uint8(data, scale, offset, clip[0], clip[1], out=data_out)
    else:
        if clip is None:
            rescale_noclip_uint16(data, scale, offset, out=data_out)
        else:
            rescale_clip_uint16(data, scale, offset, clip[0], clip[1], out=data_out)
    return data_out