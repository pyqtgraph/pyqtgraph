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

def test_init_with_qwidget_arguments():
    """
    Ensures providing only the parent argument to the constructor properly 
    intializes the widget to match the QWidget constructor prototype.
    """
    win = QtWidgets.QMainWindow()
    
    mplw = pg.MatplotlibWidget(win)

    assert mplw.parent() == win

def test_init_with_matplotlib_arguments():
    """
    Tests the contructor that sets variables associated with Matplotlib and 
    abstracts away any details about the underlying QWidget parent class.
    """
    figsize = (1.0, 3.0)
    dpi = 256
    mplw = pg.MatplotlibWidget(figsize, dpi)

    assert np.allclose(mplw.getFigure().get_size_inches(), figsize)
    assert mplw.getFigure().dpi == dpi
