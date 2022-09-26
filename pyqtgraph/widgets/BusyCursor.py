from contextlib import contextmanager

from ..Qt import QtCore, QtGui, QtWidgets

__all__ = ["BusyCursor"]


@contextmanager
def BusyCursor():
    """
    Display a busy mouse cursor during long operations.
    Usage::

        with BusyCursor():
            doLongOperation()

    May be nested. If called from a non-gui thread, then the cursor will not be affected.
    """
    app = QtCore.QCoreApplication.instance()
    in_gui_thread = (app is not None) and (QtCore.QThread.currentThread() == app.thread())
    try:
        if in_gui_thread:
            guard = QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
            # on PySide6 6.3.0, setOverrideCursor() returns a QOverrideCursorGuard object
            # that, on its destruction, calls restoreOverrideCursor() if the user had not
            # already done so.
            # if the user wants to call it manually, they must do it via the returned object,
            # and not via the QtWidgets.QApplication static method; otherwise the restore
            # would get called twice.
        yield
    finally:
        if in_gui_thread:
            if hasattr(guard, 'restoreOverrideCursor'):
                guard.restoreOverrideCursor()
            else:
                QtWidgets.QApplication.restoreOverrideCursor()
