import platform
import pytest
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

@pytest.mark.skipif(
    platform.python_implementation() == "PyPy",
    reason="PyPy has no getrefcount"
)
def test_qmenu_leak_workaround():
    # refer to https://github.com/pyqtgraph/pyqtgraph/pull/2518
    pg.mkQApp()
    topmenu = QtWidgets.QMenu()
    submenu = QtWidgets.QMenu()

    refcnt1 = sys.getrefcount(submenu)

    # check that after the workaround,
    # submenu has no change in refcnt
    topmenu.addMenu(submenu)
    submenu.setParent(None) # this is the workaround for PySide{2,6},
                            # and should have no effect on bindings
                            # where it is not needed.
    refcnt2 = sys.getrefcount(submenu)
    assert refcnt2 == refcnt1
    
    # check that topmenu is not a C++ parent of submenu.
    # i.e. deleting topmenu leaves submenu alive
    del topmenu
    assert pg.Qt.isQObjectAlive(submenu)