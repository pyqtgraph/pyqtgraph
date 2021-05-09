import numpy as np

import pyqtgraph as pg

try:
    import cupy as cp

    pg.setConfigOption("useCupy", True)
except ImportError:
    cp = None


def renderQImage(*args, **kwargs):
    imgitem = pg.ImageItem(axisOrder='row-major')
    if 'autoLevels' not in kwargs:
        kwargs['autoLevels'] = False
    imgitem.setImage(*args, **kwargs)
    imgitem.render()


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
        renderQImage(self.uint16_data["data"])  # prime the cpu
        if cp:
            renderQimage(cp.asarray(self.uint16_data["data"]))  # prime the gpu


    @staticmethod
    def _create_data(size, xp):
        float_data = {
            "data": xp.random.normal(size=size),
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


def make_test(dtype, use_cupy, use_levels, lut_name, func_name):
    def time_test(self):
        data = getattr(self, dtype + "_data")
        levels = data["levels"] if use_levels else None
        lut = getattr(self, lut_name + "_lut", None) if lut_name is not None else None
        for _ in range(10):
            img_data = data["data"]
            if use_cupy:
                img_data = cp.asarray(img_data)
            renderQImage(img_data, lut=lut, levels=levels)

    time_test.__name__ = func_name
    return time_test


for cupy in [True, False]:
    if cupy and cp is None:
        continue
    for dtype in ["float", "uint16", "uint8"]:
        for levels in [True, False]:
            if dtype == "float" and not levels:
                continue
            for lutname in [None, "uint8", "uint16"]:
                name = (
                    f'time_10x_renderImageItem_{"cupy" if cupy else ""}{dtype}_{"" if levels else "no"}levels_{lutname or "no"}lut'
                )
                setattr(_TimeSuite, name, make_test(dtype, cupy, levels, lutname, name))


class Time0256Suite(_TimeSuite):
    def __init__(self):
        self.size = 256
        super(Time0256Suite, self).__init__()


class Time0512Suite(_TimeSuite):
    def __init__(self):
        self.size = 512
        super(Time0512Suite, self).__init__()


class Time1024Suite(_TimeSuite):
    def __init__(self):
        self.size = 1024
        super(Time1024Suite, self).__init__()


class Time2048Suite(_TimeSuite):
    def __init__(self):
        self.size = 2048
        super(Time2048Suite, self).__init__()


class Time3072Suite(_TimeSuite):
    def __init__(self):
        self.size = 3072
        super(Time3072Suite, self).__init__()


class Time4096Suite(_TimeSuite):
    def __init__(self):
        self.size = 4096
        super(Time4096Suite, self).__init__()
