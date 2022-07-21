"""
MatplotlibWidget test:

Tests the creation of a MatplotlibWidget.
"""

import pytest
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import numpy as np

pytest.importorskip("matplotlib")

pg.mkQApp()

default_parent = None
default_figsize = pg.MatplotlibWidget.figsize_default
default_dpi = pg.MatplotlibWidget.dpi_default


def assert_widget_fields(mplw, parent, figsize, dpi):
    assert mplw.parent() == parent
    assert np.allclose(mplw.getFigure().get_size_inches(), figsize)
    assert mplw.getFigure().dpi == dpi


def test_init_with_qwidget_arguments():
    """
    Ensures providing only the parent argument to the constructor properly
    intializes the widget to match the QWidget constructor prototype.
    """
    win = QtWidgets.QMainWindow()

    mplw = pg.MatplotlibWidget(win)

    assert_widget_fields(mplw, win, default_figsize, default_dpi)


def test_init_with_matplotlib_arguments():
    """
    Tests the contructor that sets variables associated with Matplotlib and
    abstracts away any details about the underlying QWidget parent class.
    """
    figsize = (1.0, 3.0)
    dpi = 256
    mplw = pg.MatplotlibWidget(figsize, dpi)

    assert_widget_fields(mplw, default_parent, figsize, dpi)


def test_init_with_no_arguments():
    mplw = pg.MatplotlibWidget()

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

    mplw = pg.MatplotlibWidget(parent, figsize=figsize)
    assert_widget_fields(mplw, parent, figsize, default_dpi)

    mplw = pg.MatplotlibWidget(parent, dpi=dpi)
    assert_widget_fields(mplw, parent, default_figsize, dpi)

    mplw = pg.MatplotlibWidget(parent, figsize, dpi)
    assert_widget_fields(mplw, parent, figsize, dpi)

    mplw = pg.MatplotlibWidget(figsize, dpi)
    assert_widget_fields(mplw, default_parent, figsize, dpi)

    mplw = pg.MatplotlibWidget(figsize, dpi, parent)
    assert_widget_fields(mplw, parent, figsize, dpi)

    mplw = pg.MatplotlibWidget(dpi=dpi, parent=parent)
    assert_widget_fields(mplw, parent, default_figsize, dpi)
