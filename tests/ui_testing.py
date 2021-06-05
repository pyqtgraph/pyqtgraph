import time
from pyqtgraph.Qt import QtCore, QtGui, QtTest


def resizeWindow(win, w, h, timeout=2.0):
    """Resize a window and wait until it has the correct size.
    
    This is required for unit testing on some platforms that do not guarantee
    immediate response from the windowing system.
    """
    QtGui.QApplication.processEvents()
    # Sometimes the window size will switch multiple times before settling
    # on its final size. Adding qWaitForWindowExposed seems to help with this.
    QtTest.QTest.qWaitForWindowExposed(win)
    win.resize(w, h)
    start = time.time()
    while True:
        w1, h1 = win.width(), win.height()
        if (w,h) == (w1,h1):
            return
        QtTest.QTest.qWait(10)
        if time.time()-start > timeout:
            raise TimeoutError("Window resize failed (requested %dx%d, got %dx%d)" % (w, h, w1, h1))
    

# Functions for generating user input events. 
# We would like to use QTest for this purpose, but it seems to be broken.
# See: http://stackoverflow.com/questions/16299779/qt-qgraphicsview-unit-testing-how-to-keep-the-mouse-in-a-pressed-state

def mousePress(widget, pos, button, modifier=None):
    if isinstance(widget, QtGui.QGraphicsView):
        widget = widget.viewport()
    if modifier is None:
        modifier = QtCore.Qt.NoModifier
    event = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, pos, button, QtCore.Qt.NoButton, modifier)
    QtGui.QApplication.sendEvent(widget, event)


def mouseRelease(widget, pos, button, modifier=None):
    if isinstance(widget, QtGui.QGraphicsView):
        widget = widget.viewport()
    if modifier is None:
        modifier = QtCore.Qt.NoModifier
    event = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease, pos, button, QtCore.Qt.NoButton, modifier)
    QtGui.QApplication.sendEvent(widget, event)


def mouseMove(widget, pos, buttons=None, modifier=None):
    if isinstance(widget, QtGui.QGraphicsView):
        widget = widget.viewport()
    if modifier is None:
        modifier = QtCore.Qt.NoModifier
    if buttons is None:
        buttons = QtCore.Qt.NoButton
    event = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, pos, QtCore.Qt.NoButton, buttons, modifier)
    QtGui.QApplication.sendEvent(widget, event)


def mouseDrag(widget, pos1, pos2, button, modifier=None):
    mouseMove(widget, pos1)
    mousePress(widget, pos1, button, modifier)
    mouseMove(widget, pos2, button, modifier)
    mouseRelease(widget, pos2, button, modifier)

    
def mouseClick(widget, pos, button, modifier=None):
    mouseMove(widget, pos)
    mousePress(widget, pos, button, modifier)
    mouseRelease(widget, pos, button, modifier)
