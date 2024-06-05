import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


def test_qpainterpathprivate_read():
    x0, y0 = 100, 200
    size = 100

    qpath = QtGui.QPainterPath()
    qpath.moveTo(x0, y0)
    for idx in range(1, size):
        qpath.lineTo(x0 + idx, y0 + idx)

    memory = pg.Qt.internals.get_qpainterpath_element_array(qpath)
    assert len(memory) == size
    assert np.all(memory['x'] == np.arange(x0, x0 + size))
    assert np.all(memory['y'] == np.arange(y0, y0 + size))
    assert memory['c'][0] == 0
    assert np.all(memory['c'][1:] == 1)

def test_qpainterpathprivate_write():
    x0, y0 = 100, 200
    size = 100

    qpath0 = QtGui.QPainterPath()
    qpath0.moveTo(x0, y0)
    for idx in range(1, size):
        qpath0.lineTo(x0 + idx, y0 + idx)

    qpath1 = QtGui.QPainterPath()
    memory = pg.Qt.internals.get_qpainterpath_element_array(qpath1, size)
    assert len(memory) == size

    memory['x'] = np.arange(x0, x0 + size)
    memory['y'] = np.arange(y0, y0 + size)
    memory['c'][:1] = 0
    memory['c'][1:] = 1
    assert qpath0 == qpath1
