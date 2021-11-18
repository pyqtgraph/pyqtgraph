import pyqtgraph as pg

pg.mkQApp()


def test_nested_busy_cursors_clear_after_all_exit():
    with pg.BusyCursor():
        wait_cursor = pg.Qt.QtCore.Qt.CursorShape.WaitCursor
        with pg.BusyCursor():
            assert pg.Qt.QtWidgets.QApplication.overrideCursor().shape() == wait_cursor, "Cursor should be waiting"
        assert pg.Qt.QtWidgets.QApplication.overrideCursor().shape() == wait_cursor, "Cursor should be waiting"
    assert pg.Qt.QtWidgets.QApplication.overrideCursor() is None, "No override cursor should be set"
