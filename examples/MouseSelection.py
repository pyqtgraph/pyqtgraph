# -*- coding: utf-8 -*-
"""
Demonstrates selecting plot curves by mouse click
"""
import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

win = pg.plot()
win.setWindowTitle('pyqtgraph example: Plot data selection')

curves = [
    pg.PlotCurveItem(y=np.sin(np.linspace(0, 20, 1000)), pen='r', clickable=True),
    pg.PlotCurveItem(y=np.sin(np.linspace(1, 21, 1000)), pen='g', clickable=True),
    pg.PlotCurveItem(y=np.sin(np.linspace(2, 22, 1000)), pen='b', clickable=True),
    ]
              
def plotClicked(curve):
    global curves
    for i,c in enumerate(curves):
        if c is curve:
            c.setPen('rgb'[i], width=3)
        else:
            c.setPen('rgb'[i], width=1)
            
    
for c in curves:
    win.addItem(c)
    c.sigClicked.connect(plotClicked)

if __name__ == '__main__':
    pg.mkQApp().exec_()
