import time
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtTest
import numpy as np
import pyqtgraph as pg
from pyqtgraph.tests import assertImageApproved

app = pg.mkQApp()


def test_ImageItem():
    
    view = pg.plot()
    view.resize(200, 200)
    img = pg.ImageItem(border=0.5)
    view.addItem(img)
    
    
    # test mono float
    np.random.seed(0)
    data = np.random.normal(size=(20, 20))
    dmax = data.max()
    data[:10, 1] = dmax + 10
    data[1, :10] = dmax + 12
    data[3, :10] = dmax + 13
    img.setImage(data)
    
    QtTest.QTest.qWaitForWindowShown(view)
    time.sleep(0.1)
    app.processEvents()
    assertImageApproved(view, 'imageitem/init', 'Init image item. View is auto-scaled, image axis 0 marked by 1 line, axis 1 is marked by 2 lines. Origin in bottom-left.')
    
    # ..with colormap
    cmap = pg.ColorMap([0, 0.25, 0.75, 1], [[0, 0, 0, 255], [255, 0, 0, 255], [255, 255, 0, 255], [255, 255, 255, 255]])
    img.setLookupTable(cmap.getLookupTable())
    assertImageApproved(view, 'imageitem/lut', 'Set image LUT.')
    
    # ..and different levels
    img.setLevels([dmax+9, dmax+13])
    assertImageApproved(view, 'imageitem/levels1', 'Levels show only axis lines.')

    img.setLookupTable(None)

    # test mono int
    data = np.fromfunction(lambda x,y: x+y*10, (129, 128)).astype(np.int16)
    img.setImage(data)
    assertImageApproved(view, 'imageitem/gradient_mono_int', 'Mono int gradient.')
    
    img.setLevels([640, 641])
    assertImageApproved(view, 'imageitem/gradient_mono_int_levels', 'Mono int gradient w/ levels to isolate diagonal.')

    # test mono byte
    data = np.fromfunction(lambda x,y: x+y, (129, 128)).astype(np.ubyte)
    img.setImage(data)
    assertImageApproved(view, 'imageitem/gradient_mono_byte', 'Mono byte gradient.')
    
    img.setLevels([127, 128])
    assertImageApproved(view, 'imageitem/gradient_mono_byte_levels', 'Mono byte gradient w/ levels to isolate diagonal.')

    # test RGBA byte
    data = np.zeros((100, 100, 4), dtype='ubyte')
    data[..., 0] = np.linspace(0, 255, 100).reshape(100, 1)
    data[..., 1] = np.linspace(0, 255, 100).reshape(1, 100)
    data[..., 3] = 255
    img.setImage(data)
    assertImageApproved(view, 'imageitem/gradient_rgba_byte', 'RGBA byte gradient.')
    
    img.setLevels([[128, 129], [128, 255], [0, 1], [0, 255]])
    assertImageApproved(view, 'imageitem/gradient_rgba_byte_levels', 'RGBA byte gradient. Levels set to show x=128 and y>128.')
    
    # test RGBA float
    data = data.astype(float)
    img.setImage(data / 1e9)
    assertImageApproved(view, 'imageitem/gradient_rgba_float', 'RGBA float gradient.')

    # checkerboard to test alpha
    img2 = pg.ImageItem()
    img2.setImage(np.fromfunction(lambda x,y: (x+y)%2, (10, 10)), levels=[-1,2])
    view.addItem(img2)
    img2.scale(10, 10)
    img2.setZValue(-10)
    
    data[..., 0] *= 1e-9
    data[..., 1] *= 1e9
    data[..., 3] = np.fromfunction(lambda x,y: np.sin(0.1 * (x+y)), (100, 100))
    img.setImage(data, levels=[[0, 128e-9],[0, 128e9],[0, 1],[-1, 1]])
    assertImageApproved(view, 'imageitem/gradient_rgba_float_alpha', 'RGBA float gradient with alpha.')    

    # test composition mode
    img.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
    assertImageApproved(view, 'imageitem/gradient_rgba_float_additive', 'RGBA float gradient with alpha and additive composition mode.')    
    
    img2.hide()
    img.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
    
    # test downsampling
    data = np.fromfunction(lambda x,y: np.cos(0.002 * x**2), (800, 100))
    img.setImage(data, levels=[-1, 1])
    assertImageApproved(view, 'imageitem/resolution_without_downsampling', 'Resolution test without downsampling.')
    
    img.setAutoDownsample(True)
    assertImageApproved(view, 'imageitem/resolution_with_downsampling_x', 'Resolution test with downsampling axross x axis.')
    
    img.setImage(data.T, levels=[-1, 1])
    assertImageApproved(view, 'imageitem/resolution_with_downsampling_y', 'Resolution test with downsampling across y axis.')

@pytest.mark.skipif(pg.Qt.USE_PYSIDE, reason="pyside does not have qWait")
def test_dividebyzero():
    import pyqtgraph as pg
    im = pg.image(pg.np.random.normal(size=(100,100)))
    im.imageItem.setAutoDownsample(True)
    im.view.setRange(xRange=[-5+25, 5e+25],yRange=[-5e+25, 5e+25])
    app.processEvents()
    QtTest.QTest.qWait(1000)
    # must manually call im.imageItem.render here or the exception
    # will only exist on the Qt event loop
    im.imageItem.render()
