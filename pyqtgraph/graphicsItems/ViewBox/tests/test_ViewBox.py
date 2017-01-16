#import PySide
import pyqtgraph as pg
import pytest

app = pg.mkQApp()
qtest = pg.Qt.QtTest.QTest
QRectF = pg.QtCore.QRectF

def assertMapping(vb, r1, r2):
    assert vb.mapFromView(r1.topLeft()) == r2.topLeft()
    assert vb.mapFromView(r1.bottomLeft()) == r2.bottomLeft()
    assert vb.mapFromView(r1.topRight()) == r2.topRight()
    assert vb.mapFromView(r1.bottomRight()) == r2.bottomRight()

def init_viewbox():
    """Helper function to init the ViewBox
    """
    global win, vb
    
    win = pg.GraphicsWindow()
    win.ci.layout.setContentsMargins(0,0,0,0)
    win.resize(200, 200)
    win.show()
    vb = win.addViewBox()
    
    # set range before viewbox is shown
    vb.setRange(xRange=[0, 10], yRange=[0, 10], padding=0)
    
    # required to make mapFromView work properly.
    qtest.qWaitForWindowShown(win)
    
    g = pg.GridItem()
    vb.addItem(g)
    
    app.processEvents()
    
def test_ViewBox():
    init_viewbox()
    
    w = vb.geometry().width()
    h = vb.geometry().height()
    view1 = QRectF(0, 0, 10, 10)
    size1 = QRectF(0, h, w, -h)
    assertMapping(vb, view1, size1)
    
    # test resize
    win.resize(400, 400)
    app.processEvents()
    w = vb.geometry().width()
    h = vb.geometry().height()
    size1 = QRectF(0, h, w, -h)
    assertMapping(vb, view1, size1)
    
    # now lock aspect
    vb.setAspectLocked()
    
    # test wide resize
    win.resize(800, 400)
    app.processEvents()
    w = vb.geometry().width()
    h = vb.geometry().height()
    view1 = QRectF(-5, 0, 20, 10)
    size1 = QRectF(0, h, w, -h)
    assertMapping(vb, view1, size1)
    
    # test tall resize
    win.resize(400, 800)
    app.processEvents()
    w = vb.geometry().width()
    h = vb.geometry().height()
    view1 = QRectF(0, -5, 10, 20)
    size1 = QRectF(0, h, w, -h)
    assertMapping(vb, view1, size1)


skipreason = "Skipping this test until someone has time to fix it."
@pytest.mark.skipif(True, reason=skipreason)
def test_limits_and_resize():
    init_viewbox()

    # now lock aspect
    vb.setAspectLocked()
    # test limits + resize  (aspect ratio constraint has priority over limits
    win.resize(400, 400)
    app.processEvents()
    vb.setLimits(xMin=0, xMax=10, yMin=0, yMax=10)
    win.resize(800, 400)
    app.processEvents()
    w = vb.geometry().width()
    h = vb.geometry().height()
    view1 = QRectF(-5, 0, 20, 10)
    size1 = QRectF(0, h, w, -h)
    assertMapping(vb, view1, size1)
