# -*- coding: utf-8 -*-
"""
Test for unwanted reference cycles

"""
import pyqtgraph as pg
import numpy as np
import gc, weakref
import six
import pytest
app = pg.mkQApp()

skipreason = ('This test is failing on pyside and pyside2 for an unknown reason.')
                 
def assert_alldead(refs):
    for ref in refs:
        assert ref() is None

def qObjectTree(root):
    """Return root and its entire tree of qobject children"""
    childs = [root]
    for ch in pg.QtCore.QObject.children(root):
        childs += qObjectTree(ch)
    return childs

def mkrefs(*objs):
    """Return a list of weakrefs to each object in *objs.
    QObject instances are expanded to include all child objects.
    """
    allObjs = {}
    for obj in objs:
        if isinstance(obj, pg.QtCore.QObject):
            obj = qObjectTree(obj)
        else:
            obj = [obj]
        for o in obj:
            allObjs[id(o)] = o
    return [weakref.ref(obj) for obj in allObjs.values()]


@pytest.mark.skipif(pg.Qt.QT_LIB in {'PySide'}, reason=skipreason)
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
    
@pytest.mark.skipif(pg.Qt.QT_LIB in {'PySide', 'PySide2', 'PySide6'}, reason=skipreason)
def test_ImageView():
    def mkobjs():
        iv = pg.ImageView()
        data = np.zeros((10,10,5))
        iv.setImage(data)
        
        return mkrefs(iv, iv.imageItem, iv.view, iv.ui.histogram, data)
    for i in range(5):
        gc.collect()
        assert_alldead(mkobjs())


@pytest.mark.skipif(pg.Qt.QT_LIB in {'PySide'}, reason=skipreason)
def test_GraphicsWindow():
    def mkobjs():
        w = pg.GraphicsWindow()
        p1 = w.addPlot()
        v1 = w.addViewBox()
        return mkrefs(w, p1, v1)
    
    for i in range(5):
        assert_alldead(mkobjs())

    
    
if __name__ == '__main__':
    ot = test_PlotItem()
