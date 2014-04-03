"""
Test for unwanted reference cycles

"""
import pyqtgraph as pg
import numpy as np
import gc
app = pg.mkQApp()

def processEvents():
    for i in range(3):
        gc.collect()
        app.processEvents()
        # processEvents ignored DeferredDelete events; we must process these
        # manually.
        app.sendPostedEvents(None, pg.QtCore.QEvent.DeferredDelete)

def test_PlotItem():
    for i in range(10):
        plt = pg.PlotItem()
        plt.plot(np.random.normal(size=10000))
    processEvents()
    
    ot = pg.debug.ObjTracker()
    
    plots = []
    for i in range(10):
        plt = pg.PlotItem()
        plt.plot(np.random.normal(size=10000))
        plots.append(plt)
    processEvents()

    ot.diff()
    
    del plots
    processEvents()
    
    ot.diff()
    
    
    return ot

    
if __name__ == '__main__':
    ot = test_PlotItem()
