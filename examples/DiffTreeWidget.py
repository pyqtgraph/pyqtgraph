# -*- coding: utf-8 -*-

"""
Simple use of DiffTreeWidget to display differences between structures of 
nested dicts, lists, and arrays.
"""

import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np


app = pg.mkQApp("DiffTreeWidget Example")
A = {
    'a list': [1,2,2,4,5,6, {'nested1': 'aaaa', 'nested2': 'bbbbb'}, "seven"],
    'a dict': {
        'x': 1,
        'y': 2,
        'z': 'three'
    },
    'an array': np.random.randint(10, size=(40,10)),
    #'a traceback': some_func1(),
    #'a function': some_func1,
    #'a class': pg.DataTreeWidget,
}

B = {
    'a list': [1,2,3,4,5,5, {'nested1': 'aaaaa', 'nested2': 'bbbbb'}, "seven"],
    'a dict': {
        'x': 2,
        'y': 2,
        'z': 'three',
        'w': 5
    },
    'another dict': {1:2, 2:3, 3:4},
    'an array': np.random.randint(10, size=(40,10)),
}

tree = pg.DiffTreeWidget()
tree.setData(A, B)
tree.show()
tree.setWindowTitle('pyqtgraph example: DiffTreeWidget')
tree.resize(1000, 800)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()