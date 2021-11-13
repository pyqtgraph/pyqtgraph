import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

pos = [0.0, 0.5, 1.0]
qcols = [
    QtGui.QColor('#FF0000'),
    QtGui.QColor('#00FF00'),
    QtGui.QColor('#0000FF')
]
float_tuples = [
    (1.0, 0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 1.0)
]
int_tuples = [
    (255,  0,  0,255),
    (  0,255,  0,255),
    (  0,  0,255,255)
] 

@pytest.mark.parametrize("color_list", (qcols, int_tuples))
def test_ColorMap_getStops(color_list):
    cm = pg.ColorMap(pos, color_list, name='test')
    # default is byte format:
    stops, colors = cm.getStops()
    assert (stops == pos).all()
    assert (colors == int_tuples).all()

    # manual byte format:
    stops, colors = cm.getStops(pg.ColorMap.BYTE)
    assert (stops == pos).all()
    assert (colors == int_tuples).all()

    stops, colors = cm.getStops('bYTe')
    assert (stops == pos).all()
    assert (colors == int_tuples).all()

    # manual float format:
    stops, colors = cm.getStops(pg.ColorMap.FLOAT)
    assert (stops == pos).all()
    assert (colors == float_tuples).all()

    stops, colors = cm.getStops('floaT')
    assert (stops == pos).all()
    assert (colors == float_tuples).all()

    # manual QColor format:
    stops, colors = cm.getStops(pg.ColorMap.QCOLOR)
    assert (stops == pos).all()
    for actual, good in zip(colors, qcols):
        assert actual.getRgbF() == good.getRgbF()

    stops, colors = cm.getStops('qColor')
    assert (stops == pos).all()
    for actual, good in zip(colors, qcols):
        assert actual.getRgbF() == good.getRgbF()


@pytest.mark.parametrize("color_list", (qcols, int_tuples))
def test_ColorMap_getColors(color_list):
    cm = pg.ColorMap(pos, color_list, name='from QColors')

    colors = cm.getColors()
    assert (colors == int_tuples).all()

    colors = cm.getColors('byte')
    assert (colors == int_tuples).all()

    colors = cm.getColors('float')
    assert (colors == float_tuples).all()

    colors = cm.getColors('qcolor')
    for actual, good in zip(colors, qcols):
        assert actual.getRgbF() == good.getRgbF()
