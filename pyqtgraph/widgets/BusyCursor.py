# -*- coding: utf-8 -*-
from contextlib import contextmanager

from ..Qt import QtGui, QtCore

__all__ = ["BusyCursor"]


@contextmanager
def BusyCursor():
    app = QtCore.QCoreApplication.instance()
    active = (app is not None) and (QtCore.QThread.currentThread() == app.thread())
    try:
        if active:
            QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        yield
    finally:
        if active:
            QtGui.QApplication.restoreOverrideCursor()
