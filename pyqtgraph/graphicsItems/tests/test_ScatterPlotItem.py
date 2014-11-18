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
            app.processEvents()

    plot.clear()


def test_init_spots():
    spots = [
        {'x': 0, 'y': 1},
        {'pos': (1, 2), 'pen': None, 'brush': None, 'data': 'zzz'},
    ]
    s = pg.ScatterPlotItem(spots=spots)
    
    # Check we can display without errors
    plot.addItem(s)
    app.processEvents()
    plot.clear()
    
    # check data is correct
    spots = s.points()
    
    defPen = pg.mkPen(pg.getConfigOption('foreground'))

    assert spots[0].pos().x() == 0
    assert spots[0].pos().y() == 1
    assert spots[0].pen() == defPen
    assert spots[0].data() is None
    
    assert spots[1].pos().x() == 1
    assert spots[1].pos().y() == 2
    assert spots[1].pen() == pg.mkPen(None)
    assert spots[1].brush() == pg.mkBrush(None)
    assert spots[1].data() == 'zzz'
    

if __name__ == '__main__':
    test_scatterplotitem()
