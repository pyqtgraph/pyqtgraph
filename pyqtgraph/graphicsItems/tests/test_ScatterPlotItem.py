import pyqtgraph as pg
import numpy as np
app = pg.mkQApp()
plot = pg.plot()
app.processEvents()

# set view range equal to its bounding rect. 
# This causes plots to look the same regardless of pxMode.
plot.setRange(rect=plot.boundingRect())


def test_scatterplotitem():
    for i, pxMode in enumerate([True, False]):
        for j, useCache in enumerate([True, False]):
            s = pg.ScatterPlotItem()
            s.opts['useCache'] = useCache
            plot.addItem(s)
            s.setData(x=np.array([10,40,20,30])+i*100, y=np.array([40,60,10,30])+j*100, pxMode=pxMode)
            s.addPoints(x=np.array([60, 70])+i*100, y=np.array([60, 70])+j*100, size=[20, 30])
            
            # Test uniform spot updates
            s.setSize(10)
            s.setBrush('r')
            s.setPen('g')
            s.setSymbol('+')
            app.processEvents()
            
            # Test list spot updates
            s.setSize([10] * 6)
            s.setBrush([pg.mkBrush('r')] * 6)
            s.setPen([pg.mkPen('g')] * 6)
            s.setSymbol(['+'] * 6)
            s.setPointData([s] * 6)
            app.processEvents()

            # Test array spot updates
            s.setSize(np.array([10] * 6))
            s.setBrush(np.array([pg.mkBrush('r')] * 6))
            s.setPen(np.array([pg.mkPen('g')] * 6))
            s.setSymbol(np.array(['+'] * 6))
            s.setPointData(np.array([s] * 6))
            app.processEvents()

            # Test per-spot updates
            spot = s.points()[0]
            spot.setSize(20)
            spot.setBrush('b')
            spot.setPen('g')
            spot.setSymbol('o')
            spot.setData(None)
            

if __name__ == '__main__':
    test_scatterplotitem()
