
# Functions for generating user input events. 
# We would like to use QTest for this purpose, but it seems to be broken.
# See: http://stackoverflow.com/questions/16299779/qt-qgraphicsview-unit-testing-how-to-keep-the-mouse-in-a-pressed-state

from ..Qt import QtCore, QtGui, QT_LIB


def mousePress(widget, pos, button, modifier=None):
    if isinstance(widget, QtGui.QGraphicsView):
        widget = widget.viewport()
    if modifier is None:
        modifier = QtCore.Qt.NoModifier
    if QT_LIB != 'PyQt5' and isinstance(pos, QtCore.QPointF):
        pos = pos.toPoint()
    event = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, pos, button, QtCore.Qt.NoButton, modifier)
    QtGui.QApplication.sendEvent(widget, event)


def mouseRelease(widget, pos, button, modifier=None):
    if isinstance(widget, QtGui.QGraphicsView):
        widget = widget.viewport()
    if modifier is None:
        modifier = QtCore.Qt.NoModifier
    if QT_LIB != 'PyQt5' and isinstance(pos, QtCore.QPointF):
        pos = pos.toPoint()
    event = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease, pos, button, QtCore.Qt.NoButton, modifier)
    QtGui.QApplication.sendEvent(widget, event)


def mouseMove(widget, pos, buttons=None, modifier=None):
    if isinstance(widget, QtGui.QGraphicsView):
        widget = widget.viewport()
    if modifier is None:
        modifier = QtCore.Qt.NoModifier
    if buttons is None:
        buttons = QtCore.Qt.NoButton
    if QT_LIB != 'PyQt5' and isinstance(pos, QtCore.QPointF):
        pos = pos.toPoint()
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
    
