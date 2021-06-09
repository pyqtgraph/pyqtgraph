import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtTest

from examples.ExampleApp import ExampleLoader
"""
This file is used by test_examples.py for ensuring the Example App works.
It is not named test_ExampleApp.py as that way the Example application is
not run twice.
"""

pg.mkQApp()

def test_ExampleLoader():
    loader = ExampleLoader()
    QtTest.QTest.qWaitForWindowExposed(loader)
    QtTest.QTest.qWait(200)
    loader.close()

if __name__ == "__main__":
    test_ExampleLoader()
    pg.exec()
