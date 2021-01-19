import numpy as np

import pyqtgraph as pg


class TimeSuite(object):
    def __init__(self):
        self.c_map = None
        self.float_data = None
        self.uint8_data = None
        self.uint8_lut = None
        self.uint16_data = None
        self.uint16_lut = None

    def setup(self):
        size = (500, 500)

        self.float_data = {
            'data': np.random.normal(size=size),
            'levels': [-4., 4.],
        }

        self.uint16_data = {
            'data': np.random.randint(100, 4500, size=size).astype('uint16'),
            'levels': [250, 3000],
        }

        self.uint8_data = {
            'data': np.random.randint(0, 255, size=size).astype('ubyte'),
            'levels': [20, 220],
        }

        self.c_map = np.array([
            [-500., 255.],
            [-255., 255.],
            [0., 500.],
        ])

        self.uint8_lut = np.zeros((256, 4), dtype='ubyte')
        for i in range(3):
            self.uint8_lut[:, i] = np.clip(np.linspace(self.c_map[i][0], self.c_map[i][1], 256), 0, 255)
        self.uint8_lut[:, 3] = 255

        self.uint16_lut = np.zeros((2 ** 16, 4), dtype='ubyte')
        for i in range(3):
            self.uint16_lut[:, i] = np.clip(np.linspace(self.c_map[i][0], self.c_map[i][1], 2 ** 16), 0, 255)
        self.uint16_lut[:, 3] = 255


def make_test(dtype, use_levels, lut_name, func_name):
    def time_test(self):
        data = getattr(self, dtype + '_data')
        pg.makeARGB(
            data['data'],
            lut=getattr(self, lut_name + '_lut', None),
            levels=use_levels and data['levels'],
        )

    time_test.__name__ = func_name
    return time_test


for dt in ['float', 'uint16', 'uint8']:
    for levels in [True, False]:
        for ln in [None, 'uint8', 'uint16']:
            name = f'time_makeARGB_{dt}_{"" if levels else "no"}levels_{ln or "no"}lut'
            setattr(TimeSuite, name, make_test(dt, levels, ln, name))
