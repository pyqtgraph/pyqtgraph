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
    need_cleanup = in_gui_thread
    try:
        if in_gui_thread:
            guard = QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
            if hasattr(QtGui, 'QOverrideCursorGuard') and isinstance(guard, QtGui.QOverrideCursorGuard):
                # on PySide6 6.3.0, setOverrideCursor() returns a QOverrideCursorGuard context manager
                # object that calls restoreOverrideCursor() for us
                need_cleanup = False
        yield
    finally:
        if need_cleanup:
            QtWidgets.QApplication.restoreOverrideCursor()
