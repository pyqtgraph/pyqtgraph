# -*- coding: utf-8 -*-
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.exporters import ImageExporter
import pyqtgraph.functions as fn

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
    data = fn.qimage_to_ndarray(qimg)
    black = (0, 0, 0, 255)
    assert np.all(data == black), "Exported image should be entirely black."
