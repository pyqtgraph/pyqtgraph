import pyqtgraph as pg
import numpy as np
app = pg.mkQApp()
plot = pg.plot()
app.processEvents()

# set view range equal to its bounding rect. 
# This causes plots to look the same regardless of pxMode.
plot.setRange(rect=plot.boundingRect())


def test_modes():
    for i, pxMode in enumerate([True, False]):
        for j, useCache in enumerate([True, False]):
            s = pg.ScatterPlotItem()
            s.opts['useCache'] = useCache
            plot.addItem(s)
            s.setData(x=np.array([10,40,20,30])+i*100, y=np.array([40,60,10,30])+j*100, pxMode=pxMode)
            s.addPoints(x=np.array([60, 70])+i*100, y=np.array([60, 70])+j*100, size=[20, 30])


if __name__ == '__main__':
    test_modes()
