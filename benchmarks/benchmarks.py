import pyqtgraph as pg
import numpy as np


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
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

        self.cmap = np.array([
            [-500., 255.],
            [-255., 255.],
            [0., 500.],
        ])

        self.uint8_lut = np.zeros((256, 4), dtype='ubyte')
        for i in range(3):
            self.uint8_lut[:, i] = np.clip(np.linspace(self.cmap[i][0], self.cmap[i][1], 256), 0, 255)
        self.uint8_lut[:,3] = 255

        self.uint16_lut = np.zeros((2**16, 4), dtype='ubyte')
        for i in range(3):
            self.uint16_lut[:, i] = np.clip(np.linspace(self.cmap[i][0], self.cmap[i][1], 2**16), 0, 255)
        self.uint16_lut[:,3] = 255


def make_test(*args):
    def time_test(self):
        data, levels, lut, name = args
        data = getattr(self, data+'_data')
        if lut is not None:
            lut = getattr(self, lut+'_lut')
        if levels is not None:
            levels = data['levels']
        data = data['data']
        pg.makeARGB(data, levels=levels, lut=lut)
    time_test.__name__ = name
    return time_test


for data in ['float', 'uint16', 'uint8']:
    for levels in [True, False]:
        for lut in [None, 'uint8', 'uint16']:
            name = 'time_makeARGB_%s_%s_%slut' % (data, 'levels' if levels else 'nolevels', lut or 'no')
            setattr(TimeSuite, name, make_test(data, levels, lut, name))

