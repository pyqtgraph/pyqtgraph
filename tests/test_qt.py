import os
import warnings

import pytest

import pyqtgraph as pg

app = pg.mkQApp()

def test_isQObjectAlive():
    o1 = pg.QtCore.QObject()
    o2 = pg.QtCore.QObject()
    o2.setParent(o1)
    del o1
    assert not pg.Qt.isQObjectAlive(o2)

def test_loadUiType():
    path = os.path.dirname(__file__)
    # path_search = True is _highly_ not recommended, toggling this to pass tests
    # future versions of pyqtgraph will remove the Runtime importing of ui files
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        formClass, baseClass = pg.Qt.loadUiType(os.path.join(path, 'uictest.ui'), path_search=True)
    w = baseClass()
    ui = formClass()
    ui.setupUi(w)
    w.show()
    app.processEvents()
