import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.widgets.VerticalLabel import VerticalLabel


pg.mkQApp()


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_size_hint_includes_stylesheet_contents_margins(orientation):
    label = VerticalLabel("Dock", orientation=orientation)
    label.setStyleSheet("""
        QLabel {
            padding-left: 4px;
            padding-right: 6px;
            padding-top: 3px;
            padding-bottom: 5px;
        }
    """)
    label.resize(label.sizeHint())
    image = QtGui.QImage(label.size(), QtGui.QImage.Format.Format_ARGB32)
    image.fill(0)
    label.render(image)

    margins = label.contentsMargins()
    textSize = QtCore.QSize(
        label.fontMetrics().horizontalAdvance(label.text()),
        label.fontMetrics().height()
    )

    if orientation == "vertical":
        expected = QtCore.QSize(
            textSize.height() + margins.left() + margins.right(),
            textSize.width() + margins.top() + margins.bottom()
        )
        assert label.maximumWidth() == expected.width()
        assert label.minimumHeight() == expected.height()
    else:
        expected = QtCore.QSize(
            textSize.width() + margins.left() + margins.right(),
            textSize.height() + margins.top() + margins.bottom()
        )
        assert label.minimumWidth() == expected.width()
        assert label.maximumHeight() == expected.height()

    assert label.sizeHint() == expected
