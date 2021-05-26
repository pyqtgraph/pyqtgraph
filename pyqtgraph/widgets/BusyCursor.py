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
        isGuiThread = (app is not None) and (QtCore.QThread.currentThread() == app.thread())
        if isGuiThread and QtGui.QApplication.instance() is not None:
            if QT_LIB == "PySide":
                # pass CursorShape rather than QCursor for PySide
                # see https://bugreports.qt.io/browse/PYSIDE-243
                QtGui.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            else:
                QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
            BusyCursor.active.append(self)
            BusyCursor.nesting_count += 1
            self._active = True
        else:
            self._active = False

    def __exit__(self, *args):
        if self._active:
            BusyCursor.active.pop(-1)
            if len(BusyCursor.active) == 0:
                for _ in range(BusyCursor.nesting_count):
                    QtGui.QApplication.restoreOverrideCursor()
                BusyCursor.nesting_count = 0
