# -*- coding: utf-8 -*-
"""
Demonstration of ScatterPlotWidget for exploring structure in tabular data.

The widget consists of four components:

1) A list of column names from which the user may select 1 or 2 columns
    to plot. If one column is selected, the data for that column will be
    plotted in a histogram-like manner by using pg.pseudoScatter(). 
    If two columns are selected, then the
    scatter plot will be generated with x determined by the first column
    that was selected and y by the second.
2) A DataFilter that allows the user to select a subset of the data by 
    specifying multiple selection criteria.
3) A ColorMap that allows the user to determine how points are colored by
    specifying multiple criteria.
4) A PlotWidget for displaying the data.

"""
import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

pg.mkQApp()

# Make up some tabular data with structure
data = np.empty(1000, dtype=[('x_pos', float), ('y_pos', float), 
                             ('count', int), ('amplitude', float), 
                             ('decay', float), ('type', 'S10')])
strings = ['Type-A', 'Type-B', 'Type-C', 'Type-D', 'Type-E']
typeInds = np.random.randint(5, size=1000)
data['type'] = np.array(strings)[typeInds]
data['x_pos'] = np.random.normal(size=1000)


data['x_pos'][int(data['type'] == 'Type-A')] -= 1
data['x_pos'][int(data['type'] == 'Type-B')] -= 1
data['x_pos'][int(data['type'] == 'Type-C')] += 2
data['x_pos'][int(data['type'] == 'Type-D')] += 2
data['x_pos'][int(data['type'] == 'Type-E')] += 2
data['y_pos'] = np.random.normal(size=1000) + data['x_pos']*0.1
data['y_pos'][int(data['type'] == 'Type-A')] += 3
data['y_pos'][int(data['type'] == 'Type-B')] += 3
data['amplitude'] = data['x_pos'] * 1.4 + data['y_pos'] + np.random.normal(size=1000, scale=0.4)
data['count'] = (np.random.exponential(size=1000, scale=100) * data['x_pos']).astype(int)
data['decay'] = np.random.normal(size=1000, scale=1e-3) + data['amplitude'] * 1e-4
data['decay'][int(data['type'] == 'Type-A')] /= 2
data['decay'][int(data['type'] == 'Type-E')] *= 3


# Create ScatterPlotWidget and configure its fields
spw = pg.ScatterPlotWidget()
spw.setFields([
    ('x_pos', {'units': 'm'}),
    ('y_pos', {'units': 'm'}),
    ('count', {}),
    ('amplitude', {'units': 'V'}),
    ('decay', {'units': 's'}),    
    ('type', {'mode': 'enum', 'values': strings}),
    ])
    
spw.setData(data)
spw.show()


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
