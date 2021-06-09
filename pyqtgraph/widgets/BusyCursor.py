# -*- coding: utf-8 -*-
from ..Qt import QtGui, QtCore, QT_LIB

__all__ = ["BusyCursor"]


class BusyCursor(object):
    """Class for displaying a busy mouse cursor during long operations.
    Usage::

        with pyqtgraph.BusyCursor():
            doLongOperation()

    May be nested. If called from a non-gui thread, then the cursor will not be affected.
    """

    active = []
    nesting_count = 0

    def __enter__(self):
        app = QtCore.QCoreApplication.instance()
        self._active = (app is not None) and (QtCore.QThread.currentThread() == app.thread())
        if self._active:
            QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))

    def __exit__(self, *args):
        if self._active:
            QtGui.QApplication.restoreOverrideCursor()
