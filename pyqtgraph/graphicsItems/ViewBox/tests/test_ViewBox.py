#import PySide
import pyqtgraph as pg
import pytest
import numpy as np

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
    
    win = pg.GraphicsLayoutWidget()
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
    win.resize(200, 400)
    app.processEvents()
    w = vb.geometry().width()
    h = vb.geometry().height()
    view1 = QRectF(0, -5, 10, 20)
    size1 = QRectF(0, h, w, -h)
    assertMapping(vb, view1, size1)

    win.close()


def test_ViewBox_setMenuEnabled():
    init_viewbox()
    vb.setMenuEnabled(True)
    assert vb.menu is not None
    vb.setMenuEnabled(False)
    assert vb.menu is None


def test_enableAutoRange():
    init_viewbox()

    x = np.linspace(-3, 3, 1001)
    y = np.abs(np.sinc(x))

    plotItem = win.addPlot()
    plotItem.plot(x, y)
    inds = [np.argmax(y)]

    for ind in inds:
        xp = x[ind]
        text = pg.TextItem('{:.2f}'.format(xp), anchor=(0.5,1.9))
        plotItem.addItem(text) 

    vb = plotItem.vb

    xRange, yRange = vb.viewRange()
    vb.scaleBy([1e-10, 5e-9])
    zoomedXRange, zoomedYRange = vb.viewRange()
    assert xRange[0] < zoomedXRange[0]
    assert xRange[1] > zoomedXRange[1]
    assert yRange[0] < zoomedYRange[0]
    assert yRange[1] > zoomedYRange[1]

    vb.enableAutoRange()
    app.processEvents()
    autoXRange, autoYRange = vb.viewRange()

    assert xRange[0] == pytest.approx(autoXRange[0], rel=1e-1)
    assert xRange[1] == pytest.approx(autoXRange[1], rel=1e-1)
    assert yRange[0] == pytest.approx(autoYRange[0], rel=1e-1)
    assert yRange[1] == pytest.approx(autoYRange[1], rel=1e-1)

    vb.scaleBy([5_000_000, 1_000_000])
    zoomedXRange, zoomedYRange = vb.viewRange()
    assert xRange[0] > zoomedXRange[0]
    assert xRange[1] < zoomedXRange[1]
    assert yRange[0] > zoomedYRange[0]
    assert yRange[1] < zoomedYRange[1]
    vb.enableAutoRange()
    app.processEvents()
    autoXRange, autoYRange = vb.viewRange()



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
