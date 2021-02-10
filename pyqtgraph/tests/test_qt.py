# -*- coding: utf-8 -*-
import pyqtgraph as pg
import os
import pytest


app = pg.mkQApp()

def test_isQObjectAlive():
    o1 = pg.QtCore.QObject()
    o2 = pg.QtCore.QObject()
    o2.setParent(o1)
    del o1
    assert not pg.Qt.isQObjectAlive(o2)

@pytest.mark.skipif(
    pg.Qt.QT_LIB == "PySide2"
    and tuple(map(int, pg.Qt.PySide2.__version__.split("."))) >= (5, 14) 
    and tuple(map(int, pg.Qt.PySide2.__version__.split("."))) < (5, 14, 2, 2), 
    reason="new PySide2 doesn't have loadUi functionality"
)
def test_loadUiType():
    directory = os.path.dirname(__file__)

    uiFile = os.path.realpath(os.path.join(directory, "uictest.ui"))

    formClass, baseClass = pg.Qt.loadUiType(uiFile)
    w = baseClass()
    ui = formClass()
    ui.setupUi(w)
    w.show()
    app.processEvents()
