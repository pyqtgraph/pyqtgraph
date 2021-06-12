import numpy as np
import numba

rescale_functions = {}

def rescale_clip_source(xx, scale, offset, vmin, vmax, yy):
    for i in range(xx.size):
        val = (xx[i] - offset) * scale
        yy[i] = min(max(val, vmin), vmax)

def rescaleData(data, scale, offset, dtype, clip):
    data_out = np.empty_like(data, dtype=dtype)
    key = (data.dtype.name, data_out.dtype.name)
    func = rescale_functions.get(key)
    if func is None:
        func = numba.guvectorize(
            [f'{key[0]}[:],f8,f8,f8,f8,{key[1]}[:]'],
            '(n),(),(),(),()->(n)',
            nopython=True)(rescale_clip_source)
        rescale_functions[key] = func
    func(data, scale, offset, clip[0], clip[1], out=data_out)
    return data_out
