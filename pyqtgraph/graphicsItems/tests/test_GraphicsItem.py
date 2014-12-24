import gc
import weakref
try:
    import faulthandler
    faulthandler.enable()
except ImportError:
    pass

import pyqtgraph as pg
pg.mkQApp()

def test_getViewWidget():
    view = pg.PlotWidget()
    vref = weakref.ref(view)
    item = pg.InfiniteLine()
    view.addItem(item)
    assert item.getViewWidget() is view
    del view
    gc.collect()
    assert vref() is None
    assert item.getViewWidget() is None

def test_getViewWidget_deleted():
    view = pg.PlotWidget()
    item = pg.InfiniteLine()
    view.addItem(item)
    assert item.getViewWidget() is view
    
    # Arrange to have Qt automatically delete the view widget
    obj = pg.QtGui.QWidget()
    view.setParent(obj)
    del obj
    gc.collect()

    assert not pg.Qt.isQObjectAlive(view)
    assert item.getViewWidget() is None


#if __name__ == '__main__':
    #view = pg.PlotItem()
    #vref = weakref.ref(view)
    #item = pg.InfiniteLine()
    #view.addItem(item)
    #del view
    #gc.collect()
    
    