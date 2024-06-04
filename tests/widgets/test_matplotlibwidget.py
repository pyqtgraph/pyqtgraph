"""
MatplotlibWidget test:

Tests the creation of a MatplotlibWidget.
"""

from importlib.metadata import version

import numpy as np
import pytest
from packaging.version import parse, Version

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

pytest.importorskip("matplotlib")

# see https://github.com/matplotlib/matplotlib/pull/24172
if (
    pg.Qt.QT_LIB == "PySide6"
    and parse(pg.Qt.PySide6.__version__) > Version('6.4')
    and parse(version("matplotlib")) < Version('3.6.2')
):
    pytest.skip(
        "matplotlib + PySide6 6.4 bug",
        allow_module_level=True
    )

from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget

pg.mkQApp()

default_parent = None
default_figsize = MatplotlibWidget.figsize_default
default_dpi = MatplotlibWidget.dpi_default


def assert_widget_fields(mplw, parent, figsize, dpi):
    assert mplw.parent() == parent
    assert np.allclose(mplw.getFigure().get_size_inches(), figsize)
    assert mplw.getFigure().dpi == dpi


def test_init_with_qwidget_arguments():
    """
    Ensures providing only the parent argument to the constructor properly
    initializes the widget to match the QWidget constructor prototype.
    """
    win = QtWidgets.QMainWindow()

    mplw = MatplotlibWidget(win)

    assert_widget_fields(mplw, win, default_figsize, default_dpi)


def test_init_with_matplotlib_arguments():
    """
    Tests the constructor that sets variables associated with Matplotlib and
    abstracts away any details about the underlying QWidget parent class.
    """
    figsize = (1.0, 3.0)
    dpi = 256
    mplw = MatplotlibWidget(figsize, dpi)

    assert_widget_fields(mplw, default_parent, figsize, dpi)


def test_init_with_no_arguments():
    mplw = MatplotlibWidget()

    assert_widget_fields(mplw, default_parent, default_figsize, default_dpi)


def test_init_sanity():
    """
    Tests to ensure the constructor behaves as expected.
    """
    parent = QtWidgets.QMainWindow()
    figsize = (1.0, 4.0)
    dpi = 256

    # These tests will not work if these two assertions do not hold.
    assert figsize != default_figsize
    assert dpi != default_dpi

    mplw = MatplotlibWidget(parent, figsize=figsize)
    assert_widget_fields(mplw, parent, figsize, default_dpi)

    mplw = MatplotlibWidget(parent, dpi=dpi)
    assert_widget_fields(mplw, parent, default_figsize, dpi)

    mplw = MatplotlibWidget(parent, figsize, dpi)
    assert_widget_fields(mplw, parent, figsize, dpi)

    mplw = MatplotlibWidget(figsize, dpi)
    assert_widget_fields(mplw, default_parent, figsize, dpi)

    mplw = MatplotlibWidget(figsize, dpi, parent)
    assert_widget_fields(mplw, parent, figsize, dpi)

    mplw = MatplotlibWidget(dpi=dpi, parent=parent)
    assert_widget_fields(mplw, parent, default_figsize, dpi)
