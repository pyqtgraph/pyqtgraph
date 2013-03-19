# -*- coding: utf-8 -*-
import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

pg.mkQApp()

spw = pg.ScatterPlotWidget()
spw.show()

data = np.array([
    (1, 1, 3, 4, 'x'),
    (2, 3, 3, 7, 'y'),
    (3, 2, 5, 2, 'z'),
    (4, 4, 6, 9, 'z'),
    (5, 3, 6, 7, 'x'),
    (6, 5, 2, 6, 'y'),
    (7, 5, 7, 2, 'z'),
    ], dtype=[('col1', float), ('col2', float), ('col3', int), ('col4', int), ('col5', 'S10')])

spw.setFields([
    ('col1', {'units': 'm'}),
    ('col2', {'units': 'm'}),
    ('col3', {}),
    ('col4', {}),
    ('col5', {'mode': 'enum', 'values': ['x', 'y', 'z']}),
    ])
    
spw.setData(data)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
