import numpy as np

import pyqtgraph as pg
import pyqtgraph.functions as fn
from pyqtgraph.exporters import ImageExporter
from pyqtgraph.Qt import QtGui

app = pg.mkQApp()


def test_ImageExporter_filename_dialog():
    """Tests ImageExporter code path that opens a file dialog. Regression test
    for pull request 1133."""
    p = pg.plot()
    exp = ImageExporter(p.getPlotItem())
    exp.export()


def test_ImageExporter_toBytes():
    p = pg.plot()
    p.hideAxis('bottom')
    p.hideAxis('left')
    exp = ImageExporter(p.getPlotItem())
    qimg = exp.export(toBytes=True)
    qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    data = fn.ndarray_from_qimage(qimg)
    black = (0, 0, 0, 255)
    assert np.all(data == black), "Exported image should be entirely black."
