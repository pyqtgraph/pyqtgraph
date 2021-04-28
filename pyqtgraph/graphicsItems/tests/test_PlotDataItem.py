# -*- coding: utf-8 -*-
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

pg.mkQApp()


def test_bool():
    truths = np.random.randint(0, 2, size=(100,)).astype(np.bool)
    pdi = pg.PlotDataItem(truths)
    bounds = pdi.dataBounds(1)
    assert isinstance(bounds[0], np.integer)
    assert isinstance(bounds[1], np.integer)
    xdata, ydata = pdi.getData()
    assert ydata.dtype == np.integer


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
    
    #test clear by empty call
    pdi.setData()
    assert pdi.xData is None
    assert pdi.yData is None

    #test dict of x, y list
    y += list(np.random.normal(size=50))
    x = list(np.linspace(5, 10, 200))
    pdi.setData({'x': x, 'y': y})
    assert len(pdi.xData) == 200
    assert len(pdi.yData) == 200

    #test clear by zero length arrays call
    pdi.setData([],[])
    assert pdi.xData is None
    assert pdi.yData is None

def test_opts():
    # test that curve and scatter plot properties get updated from PlotDataItem methods
    y = list(np.random.normal(size=100))
    x = np.linspace(5, 10, 100)
    pdi = pg.PlotDataItem(x, y)
    pen = QtGui.QPen( QtGui.QColor('#FF0000') )
    pen2 = QtGui.QPen( QtGui.QColor('#FFFF00') )
    brush = QtGui.QBrush( QtGui.QColor('#00FF00'))
    brush2 = QtGui.QBrush( QtGui.QColor('#00FFFF'))
    float_value = 1.0 + 20*np.random.random()
    pen2.setWidth( int(float_value) )
    pdi.setPen(pen)
    assert pdi.curve.opts['pen'] == pen
    pdi.setShadowPen(pen2)
    assert pdi.curve.opts['shadowPen'] == pen2
    pdi.setFillLevel( float_value )
    assert pdi.curve.opts['fillLevel'] == float_value
    pdi.setFillBrush(brush2)
    assert pdi.curve.opts['brush'] == brush2

    pdi.setSymbol('t')
    assert pdi.scatter.opts['symbol'] == 't'
    pdi.setSymbolPen(pen)
    assert pdi.scatter.opts['pen'] == pen
    pdi.setSymbolBrush(brush)
    assert pdi.scatter.opts['brush'] == brush
    pdi.setSymbolSize( float_value )
    assert pdi.scatter.opts['size'] == float_value

def test_clear():
    y = list(np.random.normal(size=100))
    x = np.linspace(5, 10, 100)
    pdi = pg.PlotDataItem(x, y)
    pdi.clear()

    assert pdi.xData is None
    assert pdi.yData is None

def test_clear_in_step_mode():
    w = pg.PlotWidget()
    c = pg.PlotDataItem([1,4,2,3], [5,7,6], stepMode="center")
    w.addItem(c)
    c.clear()

def test_clipping():
    y = np.random.normal(size=150)
    x = np.exp2(np.linspace(5, 10, 150))  # non-uniform spacing

    w = pg.PlotWidget(autoRange=True, downsample=5)
    c = pg.PlotDataItem(x, y)
    w.addItem(c)

    c.setClipToView(True)
    w.setXRange(200, 600)
    for x_min in range(100, 2**10 - 100, 100):
        w.setXRange(x_min, x_min + 100)

        xDisp, _ = c.getData()
        vr = c.viewRect()

        assert xDisp[0] <= vr.left()
        assert xDisp[-1] >= vr.right()

    w.close()
