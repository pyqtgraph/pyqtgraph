# -*- coding: utf-8 -*-
import sys, os
## Add path to library (just for examples; you do not need this)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])

mw = QtGui.QMainWindow()
mw.resize(800,800)
cw = QtGui.QWidget()
layout = QtGui.QGridLayout()
cw.setLayout(layout)
mw.setCentralWidget(cw)

w1 = pg.PlotWidget()
layout.addWidget(w1, 0,0)

w2 = pg.PlotWidget()
layout.addWidget(w2, 1,0)

w3 = pg.GraphicsView()
w3.enableMouse()
w3.aspectLocked = True
layout.addWidget(w3, 0,1)

w4 = pg.PlotWidget()
#vb = pg.ViewBox()
#w4.setCentralItem(vb)
layout.addWidget(w4, 1,1)

mw.show()


n = 3000
s1 = pg.ScatterPlotItem(size=10, pen=QtGui.QPen(QtCore.Qt.NoPen), brush=QtGui.QBrush(QtGui.QColor(255, 255, 255, 20)))
pos = np.random.normal(size=(2,n), scale=1e-5)
spots = [{'pos': pos[:,i], 'data': 1} for i in range(n)] + [{'pos': [0,0], 'data': 1}]
s1.addPoints(spots)
w1.addDataItem(s1)

def clicked(plot, points):
    print "clicked points", points
    
s1.sigClicked.connect(clicked)


s2 = pg.ScatterPlotItem(pxMode=False)
spots2 = []
for i in range(10):
    for j in range(10):
        spots2.append({'pos': (1e-6*i, 1e-6*j), 'size': 1e-6, 'brush':pg.intColor(i*10+j, 100)})
s2.addPoints(spots2)
w2.addDataItem(s2)

s2.sigClicked.connect(clicked)


s3 = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
pos = np.random.normal(size=(2,3000), scale=1e-5)
spots = [{'pos': pos[:,i], 'data': 1, 'brush':pg.intColor(i, 3000)} for i in range(3000)]
s3.addPoints(spots)
w3.addItem(s3)
w3.setRange(s3.boundingRect())
s3.sigClicked.connect(clicked)


s4 = pg.ScatterPlotItem(identical=True, size=10, pen=QtGui.QPen(QtCore.Qt.NoPen), brush=QtGui.QBrush(QtGui.QColor(255, 255, 255, 20)))
#pos = np.random.normal(size=(2,n), scale=1e-5)
#spots = [{'pos': pos[:,i], 'data': 1} for i in range(n)] + [{'pos': [0,0], 'data': 1}]
s4.addPoints(spots)
w4.addDataItem(s4)


## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()

