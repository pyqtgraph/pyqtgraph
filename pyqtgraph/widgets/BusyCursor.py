from ..Qt import QtGui, QtCore

__all__ = ['BusyCursor']

class BusyCursor(object):
    """Class for displaying a busy mouse cursor during long operations.
    Usage::

        with pyqtgraph.BusyCursor():
            doLongOperation()

    May be nested.
    """
    active = []

    def __enter__(self):
        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        BusyCursor.active.append(self)

    def __exit__(self, *args):
        if self._active:
            BusyCursor.active.pop(-1)
            QtGui.QApplication.restoreOverrideCursor()
        