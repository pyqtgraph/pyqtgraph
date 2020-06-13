import numpy as np
import pyqtgraph as pg

pg.mkQApp()


def test_fft():
    f = 20.
    x = np.linspace(0, 1, 1000)
    y = np.sin(2 * np.pi * f * x)
    pd = pg.PlotDataItem(x, y)
    pd.setFftMode(True)
    x, y = pd.getData()
    assert abs(x[np.argmax(y)] - f) < 0.03

    x = np.linspace(0, 1, 1001)
    y = np.sin(2 * np.pi * f * x)
    pd.setData(x, y)
    x, y = pd.getData()
    assert abs(x[np.argmax(y)]- f) < 0.03

    pd.setLogMode(True, False)
    x, y = pd.getData()
    assert abs(x[np.argmax(y)] - np.log10(f)) < 0.01

def test_setData():
    pdi = pg.PlotDataItem()

    #test empty data
    pdi.setData([])

    #test y data
    y = list(np.random.normal(size=100))
    pdi.setData(y)
    assert len(pdi.xData) == 100
    assert len(pdi.yData) == 100

    #test x, y data
    y += list(np.random.normal(size=50))
    x = np.linspace(5, 10, 150)

    pdi.setData(x, y)
    assert len(pdi.xData) == 150
    assert len(pdi.yData) == 150

    #test dict of x, y list
    y += list(np.random.normal(size=50))
    x = list(np.linspace(5, 10, 200))
    pdi.setData({'x': x, 'y': y})
    assert len(pdi.xData) == 200
    assert len(pdi.yData) == 200

def test_clear():
    y = list(np.random.normal(size=100))
    x = np.linspace(5, 10, 100)
    pdi = pg.PlotDataItem(x, y)
    pdi.clear()

    assert pdi.xData == None
    assert pdi.yData == None

def test_clear_in_step_mode():
    w = pg.PlotWidget()
    c = pg.PlotDataItem([1,4,2,3], [5,7,6], stepMode=True)
    w.addItem(c)
    c.clear()

def test_clipping():
    y = np.random.normal(size=150)
    x = np.exp2(np.linspace(5, 10, 150))  # non-uniform spacing

    w = pg.PlotWidget(autoRange=True, downsample=5)
    c = pg.PlotDataItem(x, y)
    w.addItem(c)
    w.show()

    c.setClipToView(True)

    w.setXRange(200, 600)

    for x_min in range(100, 2**10 - 100, 100):
        w.setXRange(x_min, x_min + 100)

        xDisp, _ = c.getData()
        vr = c.viewRect()

        assert xDisp[0] <= vr.left()
        assert xDisp[-1] >= vr.right()

    w.close()
