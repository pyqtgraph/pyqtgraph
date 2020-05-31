# -*- coding: utf-8 -*-
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

app = pg.mkQApp()


def test_ImageExporter_filename_dialog():
    """Tests ImageExporter code path that opens a file dialog. Regression test
    for pull request 1133."""
    p = pg.plot()
    exp = ImageExporter(p.getPlotItem())
    exp.export()
