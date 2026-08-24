import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


def test_mouse_move_before_press_keeps_neutral_state():
    pg.mkQApp()
    button = pg.JoystickButton()
    pos = QtCore.QPointF(10, 10)
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        pos,
        pos,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )

    button.mouseMoveEvent(event)

    assert button.getState() == [0, 0]
