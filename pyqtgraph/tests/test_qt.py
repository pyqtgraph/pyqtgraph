# -*- coding: utf-8 -*-
import pyqtgraph as pg
import gc, os
import pytest


app = pg.mkQApp()

def test_isQObjectAlive():
    o1 = pg.QtCore.QObject()
    o2 = pg.QtCore.QObject()
    o2.setParent(o1)
    del o1
    assert not pg.Qt.isQObjectAlive(o2)

@pytest.mark.skipif(pg.Qt.QT_LIB == 'PySide', reason='pysideuic does not appear to be '
                                                     'packaged with conda')
@pytest.mark.skipif(
    pg.Qt.QT_LIB == "PySide2"
    and tuple(map(int, pg.Qt.PySide2.__version__.split("."))) >= (5, 14) 
    and tuple(map(int, pg.Qt.PySide2.__version__.split("."))) < (5, 14, 2, 2), 
    reason="new PySide2 doesn't have loadUi functionality"
)
def test_loadUiType():
    path = os.path.dirname(__file__)
    formClass, baseClass = pg.Qt.loadUiType(os.path.join(path, 'uictest.ui'))
    w = baseClass()
    ui = formClass()
    ui.setupUi(w)
    w.show()
    app.processEvents()
