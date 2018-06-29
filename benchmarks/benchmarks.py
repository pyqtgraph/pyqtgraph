import pyqtgraph as pg
import numpy as np


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self):
        self.data = np.random.normal((1000, 1000))
        self.levels = [-4., 4.]
        self.cmap = np.array([
            [-500., 255.],
            [-255., 255.],
            [0., 500.],
        ])
        self.lut = np.zeros((256, 4), dtype='ubyte')
        for i in range(3):
            self.lut[:, i] = np.clip(np.linspace(self.cmap[i][0], self.cmap[i][1], 256), 0, 255)
        self.lut[:,3] = 255

    def time_makeARGB(self):
        pg.makeARGB(self.data, levels=self.levels, lut=self.lut)

    def time_makeRGBA(self):
        pg.makeRGBA(self.data, levels=self.levels, lut=self.lut)
