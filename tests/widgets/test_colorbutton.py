import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

pg.mkQApp()


def test_color_dialog_inherits_button_palette():
    button = pg.ColorButton()
    palette = button.palette()
    window_color = QtGui.QColor("#123456")
    palette.setColor(QtGui.QPalette.ColorRole.Window, window_color)
    button.setPalette(palette)

    assert button.colorDialog.parent() is button
    assert (
        button.colorDialog.palette().color(QtGui.QPalette.ColorRole.Window)
        == window_color
    )
