#import PySide
import pyqtgraph as pg
import pytest

QRectF = None
app = None
win = None
vb = None

def setup_module():
    global app, win, vb, QRectF
    app = pg.mkQApp()
    QRectF = pg.QtCore.QRectF
    qtest = pg.Qt.QtTest.QTest
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


def teardown_module():
    global app, win, vb
    app.exit()
    app = None
    win = None
    vb = None


def test_initial_shape():
    w = vb.geometry().width()
    h = vb.geometry().height()
    
    view1 = QRectF(0, 0, 10, 10)
    size1 = QRectF(0, h, w, -h)
    _assert_mapping(vb, view1, size1)

def test_resize():
    # test resize
    win.resize(400, 400)
    app.processEvents()
    
    w = vb.geometry().width()
    h = vb.geometry().height()
    view1 = QRectF(0, 0, 10, 10)
    size1 = QRectF(0, h, w, -h)
    size1 = QRectF(0, h, w, -h)
    _assert_mapping(vb, view1, size1)


skipreason = ('unclear why these tests are failing. skipping until someone '
              'has time to fix it.')
@pytest.mark.skipif(True, reason=skipreason)
def test_wide_resize():
    # test wide resize
    win.resize(800, 400)
    app.processEvents()
    w = vb.geometry().width()
    h = vb.geometry().height()
    view1 = QRectF(-5, 0, 20, 10)
    size1 = QRectF(0, h, w, -h)
    _assert_mapping(vb, view1, size1)


skipreason = ('unclear why these tests are failing. skipping until someone '
              'has time to fix it.')
@pytest.mark.skipif(True, reason=skipreason)
def test_tall_resize():
    # test tall resize
    win.resize(400, 800)
    app.processEvents()
    w = vb.geometry().width()
    h = vb.geometry().height()
    view1 = QRectF(0, -5, 10, 20)
    size1 = QRectF(0, h, w, -h)
    _assert_mapping(vb, view1, size1)


skipreason = ('unclear why these tests are failing. skipping until someone '
              'has time to fix it.')
@pytest.mark.skipif(True, reason=skipreason)
def test_aspect_radio_constraint():
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
    _assert_mapping(vb, view1, size1)


def _assert_mapping(vb, r1, r2):
    assert vb.mapFromView(r1.topLeft()) == r2.topLeft()
    assert vb.mapFromView(r1.bottomLeft()) == r2.bottomLeft()
    assert vb.mapFromView(r1.topRight()) == r2.topRight()
    assert vb.mapFromView(r1.bottomRight()) == r2.bottomRight()


if __name__ == '__main__':
    import user,sys
    test_ViewBox()
