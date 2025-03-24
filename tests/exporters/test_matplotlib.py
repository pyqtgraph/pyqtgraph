from importlib.metadata import version

import pytest

import pyqtgraph as pg
from pyqtgraph.exporters import MatplotlibExporter

pytest.importorskip("matplotlib")
from packaging.version import Version, parse


app = pg.mkQApp()

# see https://github.com/matplotlib/matplotlib/pull/24172
if (
    pg.Qt.QT_LIB == "PySide6"
    and parse(pg.Qt.PySide6.__version__) > Version('6.4')
    and parse(version('matplotlib')) < Version('3.6.2')
):
    pytest.skip(
        "matplotlib + PySide6 6.4 bug",
        allow_module_level=True
    )


def test_MatplotlibExporter():
    plt = pg.PlotWidget()
    plt.show()

    # curve item
    plt.plot([0, 1, 2], [0, 1, 2])
    # scatter item
    plt.plot([0, 1, 2], [1, 2, 3], pen=None, symbolBrush='r')
    # curve + scatter
    plt.plot([0, 1, 2], [2, 3, 4], pen='k', symbolBrush='r')

    exp = MatplotlibExporter(plt.getPlotItem())
    exp.export()

def test_MatplotlibExporter_nonplotitem():
    # attempting to export something other than a PlotItem raises an exception
    plt = pg.PlotWidget()
    plt.show()
    plt.plot([0, 1, 2], [2, 3, 4])
    exp = MatplotlibExporter(plt.getPlotItem().getViewBox())
    with pytest.raises(Exception):
        exp.export()

@pytest.mark.parametrize('scale', [1e10, 1e-9])
def test_MatplotlibExporter_siscale(scale):
    # coarse test to verify that plot data is scaled before export when
    # autoSIPrefix is in effect (so mpl doesn't add its own multiplier label)
    plt = pg.PlotWidget()
    plt.show()
    plt.plot([0, 1, 2], [(i+1)*scale for i in range(3)])
    # set the label so autoSIPrefix works
    plt.setLabel('left', 'magnitude')
    exp = MatplotlibExporter(plt.getPlotItem())
    exp.export()

    mpw = MatplotlibExporter.windows[-1]
    fig = mpw.getFigure()
    ymin, ymax = fig.axes[0].get_ylim()

    if scale < 1:
        assert ymax > scale
    else:
        assert ymax < scale
