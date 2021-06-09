# -*- coding: utf-8 -*-
import pyqtgraph as pg

pg.mkQApp()


def test_nested_busy_cursors_clear_after_all_exit():
    with pg.BusyCursor():
        with pg.BusyCursor():
            assert pg.Qt.QtGui.QApplication.overrideCursor().shape() == pg.Qt.QtCore.Qt.CursorShape.WaitCursor, "Cursor should be waiting"
        assert pg.Qt.QtGui.QApplication.overrideCursor().shape() == pg.Qt.QtCore.Qt.CursorShape.WaitCursor, "Cursor should be waiting"
    assert pg.Qt.QtGui.QApplication.overrideCursor() is None, "No override cursor should be set"
