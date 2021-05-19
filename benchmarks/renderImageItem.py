# -*- coding: utf-8 -*-
import numpy as np

import pyqtgraph as pg

try:
    import cupy as cp

    pg.setConfigOption("useCupy", True)
except ImportError:
    cp = None

try:
    import numba
except ImportError:
    numba = None


def renderQImage(*args, **kwargs):
    imgitem = pg.ImageItem(axisOrder='row-major')
    if 'autoLevels' not in kwargs:
        kwargs['autoLevels'] = False
    imgitem.setImage(*args, **kwargs)
    imgitem.render()


def prime_numba():
    shape = (64, 64)
    lut_small = np.random.randint(256, size=(256, 3), dtype=np.uint8)
    lut_big = np.random.randint(256, size=(512, 3), dtype=np.uint8)
    for lut in [lut_small, lut_big]:
        renderQImage(np.zeros(shape, dtype=np.uint8), levels=(20, 220), lut=lut)
        renderQImage(np.zeros(shape, dtype=np.uint16), levels=(250, 3000), lut=lut)
        renderQImage(np.zeros(shape, dtype=np.float32), levels=(-4.0, 4.0), lut=lut)


class _TimeSuite(object):
    def __init__(self):
        super(_TimeSuite, self).__init__()
        self.float_data = None
        self.uint8_data = None
        self.uint8_lut = None
        self.uint16_data = None
        self.uint16_lut = None

    def setup(self):
        size = (self.size, self.size)
        self.float_data, self.uint16_data, self.uint8_data, self.uint16_lut, self.uint8_lut = self._create_data(
            size, np
        )
        if numba is not None:
            # ensure JIT compilation
            pg.setConfigOption("useNumba", True)
            prime_numba()
            pg.setConfigOption("useNumba", False)
        if cp:
            _d1, _d2, _d3, self.cupy_uint16_lut, self.cupy_uint8_lut = self._create_data(size, cp)
            renderQImage(cp.asarray(self.uint16_data["data"]))  # prime the gpu

    @property
    def numba_uint16_lut(self):
        return self.uint16_lut

    @property
    def numba_uint8_lut(self):
        return self.uint8_lut

    @property
    def numpy_uint16_lut(self):
        return self.uint16_lut

    @property
    def numpy_uint8_lut(self):
        return self.uint8_lut

    @staticmethod
    def _create_data(size, xp):
        float_data = {
            "data": xp.random.normal(size=size).astype("float32"),
            "levels": [-4.0, 4.0],
        }
        uint16_data = {
            "data": xp.random.randint(100, 4500, size=size).astype("uint16"),
            "levels": [250, 3000],
        }
        uint8_data = {
            "data": xp.random.randint(0, 255, size=size).astype("ubyte"),
            "levels": [20, 220],
        }
        c_map = xp.array([[-500.0, 255.0], [-255.0, 255.0], [0.0, 500.0]])
        uint8_lut = xp.zeros((256, 4), dtype="ubyte")
        for i in range(3):
            uint8_lut[:, i] = xp.clip(xp.linspace(c_map[i][0], c_map[i][1], 256), 0, 255)
        uint8_lut[:, 3] = 255
        uint16_lut = xp.zeros((2 ** 16, 4), dtype="ubyte")
        for i in range(3):
            uint16_lut[:, i] = xp.clip(xp.linspace(c_map[i][0], c_map[i][1], 2 ** 16), 0, 255)
        uint16_lut[:, 3] = 255
        return float_data, uint16_data, uint8_data, uint16_lut, uint8_lut


def make_test(dtype, kind, use_levels, lut_name, func_name):
    def time_test(self):
        data = getattr(self, dtype + "_data")
        levels = data["levels"] if use_levels else None
        lut = getattr(self, f"{kind}_{lut_name}_lut", None) if lut_name is not None else None
        pg.setConfigOption("useNumba", kind == "numba")
        img_data = data["data"]
        if kind == "cupy":
            img_data = cp.asarray(img_data)
        renderQImage(img_data, lut=lut, levels=levels)

    time_test.__name__ = func_name
    return time_test


for kind in ["cupy", "numba", "numpy"]:
    if kind == "cupy" and cp is None:
        continue
    if kind == "numba" and numba is None:
        continue
    for dtype in ["float", "uint16", "uint8"]:
        for levels in [True, False]:
            if dtype == "float" and not levels:
                continue
            for lutname in [None, "uint8", "uint16"]:
                name = (
                    f'time_1x_renderImageItem_{kind}_{dtype}_{"" if levels else "no"}levels_{lutname or "no"}lut'
                )
                setattr(_TimeSuite, name, make_test(dtype, kind, levels, lutname, name))


class Time4096Suite(_TimeSuite):
    def __init__(self):
        self.size = 4096
        super(Time4096Suite, self).__init__()
