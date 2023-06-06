import pytest
import sys

import pyqtgraph as pg

pg.mkQApp()

@pytest.mark.skipif(
    sys.platform.startswith("linux")
    and pg.Qt.QT_LIB == "PySide6"
    and (6, 0) < pg.Qt.PySide6.__version_info__ < (6, 4, 3),
    reason="taking gui thread causes segfault"
)
def test_nested_busy_cursors_clear_after_all_exit():
    with pg.BusyCursor():
        wait_cursor = pg.Qt.QtCore.Qt.CursorShape.WaitCursor
        with pg.BusyCursor():
            assert pg.Qt.QtWidgets.QApplication.overrideCursor().shape() == wait_cursor, "Cursor should be waiting"
        assert pg.Qt.QtWidgets.QApplication.overrideCursor().shape() == wait_cursor, "Cursor should be waiting"
    assert pg.Qt.QtWidgets.QApplication.overrideCursor() is None, "No override cursor should be set"
