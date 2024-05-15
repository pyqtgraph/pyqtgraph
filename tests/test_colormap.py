import pytest

import numpy as np
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

def test_ColorMap_getByIndex():
    cm = pg.ColorMap([0.0, 1.0], [(0,0,0), (255,0,0)])
    assert cm.getByIndex(0) == QtGui.QColor.fromRgbF(0.0, 0.0, 0.0, 1.0)
    assert cm.getByIndex(1) == QtGui.QColor.fromRgbF(1.0, 0.0, 0.0, 1.0)

def test_round_trip():
    # test that colormap survives a round trip.
    # note that while both input and output are in BYTE,
    # internally the colors are stored as float; thus
    # there is a conversion BYTE -> float -> BYTE
    nPts = 256
    zebra = np.zeros((nPts, 3), dtype=np.uint8)
    zebra[1::2, :] = 255
    cmap = pg.ColorMap(None, zebra)
    lut = cmap.getLookupTable(nPts=nPts)
    assert np.all(lut == zebra)
