import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


def test_findWidget() -> None:
    group = QtWidgets.QGroupBox()
    combo = QtWidgets.QComboBox()
    spin = QtWidgets.QSpinBox()
    check = QtWidgets.QCheckBox()
    line = QtWidgets.QLineEdit()
    radio = QtWidgets.QRadioButton()

    widget_group = pg.WidgetGroup([
        (group, "group", None),
        (combo, "combo", None),
        (spin, "spin", 3),
        (check, "check", None),
        (line, "line_and_radio", None),
        (radio, "line_and_radio", None),
    ])

    assert widget_group.findWidget("combo") is combo
    assert widget_group.findWidget("check") is check
    assert widget_group.findWidget("spin") is spin
    assert widget_group.findWidget("line_and_radio") is line
    assert widget_group.findWidget("nonexistent") is None
