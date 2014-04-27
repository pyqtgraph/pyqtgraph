"""
Test for unwanted reference cycles

"""
import pyqtgraph as pg
import numpy as np
import gc, weakref
app = pg.mkQApp()

def assert_alldead(refs):
    for ref in refs:
        assert ref() is None

def mkrefs(*objs):
    return map(weakref.ref, objs)

def test_PlotWidget():
    def mkobjs(*args, **kwds):
        w = pg.PlotWidget(*args, **kwds)
        data = pg.np.array([1,5,2,4,3])
        c = w.plot(data, name='stuff')
        w.addLegend()
        
        # test that connections do not keep objects alive
        w.plotItem.vb.sigRangeChanged.connect(mkrefs)
        app.focusChanged.connect(w.plotItem.vb.invertY)
        
        # return weakrefs to a bunch of objects that should die when the scope exits.
        return mkrefs(w, c, data, w.plotItem, w.plotItem.vb, w.plotItem.getMenu(), w.plotItem.getAxis('left'))
    
    for i in range(5):
        assert_alldead(mkobjs())
    
def test_ImageView():
    def mkobjs():
        iv = pg.ImageView()
        data = np.zeros((10,10,5))
        iv.setImage(data)
        
        return mkrefs(iv, iv.imageItem, iv.view, iv.ui.histogram, data)
    
    for i in range(5):
        assert_alldead(mkobjs())


    
if __name__ == '__main__':
    ot = test_PlotItem()
