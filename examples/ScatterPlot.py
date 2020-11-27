# -*- coding: utf-8 -*-
"""
Example demonstrating a variety of scatter plot features.
"""



## Add path to library (just for examples; you do not need this)
import initExample

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from collections import namedtuple
from itertools import chain

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
mw.resize(800,800)
view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
mw.setCentralWidget(view)
mw.show()
mw.setWindowTitle('pyqtgraph example: ScatterPlot')

## create four areas to add plots
w1 = view.addPlot()
w2 = view.addViewBox()
w2.setAspectLocked(True)
view.nextRow()
w3 = view.addPlot()
w4 = view.addPlot()
print("Generating data, this takes a few seconds...")

## There are a few different ways we can draw scatter plots; each is optimized for different types of data:


## 1) All spots identical and transform-invariant (top-left plot).
## In this case we can get a huge performance boost by pre-rendering the spot
## image and just drawing that image repeatedly.

n = 300
s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
pos = np.random.normal(size=(2,n), scale=1e-5)
spots = [{'pos': pos[:,i], 'data': 1} for i in range(n)] + [{'pos': [0,0], 'data': 1}]
s1.addPoints(spots)
w1.addItem(s1)

## Make all plots clickable
lastClicked = []
def clicked(plot, points):
    global lastClicked
    for p in lastClicked:
        p.resetPen()
    print("clicked points", points)
    for p in points:
        p.setPen('b', width=2)
    lastClicked = points
s1.sigClicked.connect(clicked)



## 2) Spots are transform-invariant, but not identical (top-right plot).
## In this case, drawing is almsot as fast as 1), but there is more startup
## overhead and memory usage since each spot generates its own pre-rendered
## image.

TextSymbol = namedtuple("TextSymbol", "label symbol scale")

def createLabel(label, angle):
    symbol = QtGui.QPainterPath()
    #symbol.addText(0, 0, QFont("San Serif", 10), label)
    f = QtGui.QFont()
    f.setPointSize(10)
    symbol.addText(0, 0, f, label)
    br = symbol.boundingRect()
    scale = min(1. / br.width(), 1. / br.height())
    tr = QtGui.QTransform()
    tr.scale(scale, scale)
    tr.rotate(angle)
    tr.translate(-br.x() - br.width()/2., -br.y() - br.height()/2.)
    return TextSymbol(label, tr.map(symbol), 0.1 / scale)

random_str = lambda : (''.join([chr(np.random.randint(ord('A'),ord('z'))) for i in range(np.random.randint(1,5))]), np.random.randint(0, 360))

s2 = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
pos = np.random.normal(size=(2,n), scale=1e-5)
spots = [{'pos': pos[:,i], 'data': 1, 'brush':pg.intColor(i, n), 'symbol': i%10, 'size': 5+i/10.} for i in range(n)]
s2.addPoints(spots)
spots = [{'pos': pos[:,i], 'data': 1, 'brush':pg.intColor(i, n), 'symbol': label[1], 'size': label[2]*(5+i/10.)} for (i, label) in [(i, createLabel(*random_str())) for i in range(n)]]
s2.addPoints(spots)
w2.addItem(s2)
s2.sigClicked.connect(clicked)


## 3) Spots are not transform-invariant, not identical (bottom-left).
## This is the slowest case, since all spots must be completely re-drawn
## every time because their apparent transformation may have changed.

s3 = pg.ScatterPlotItem(pxMode=False)   ## Set pxMode=False to allow spots to transform with the view
spots3 = []
for i in range(10):
    for j in range(10):
        spots3.append({'pos': (1e-6*i, 1e-6*j), 'size': 1e-6, 'pen': {'color': 'w', 'width': 2}, 'brush':pg.intColor(i*10+j, 100)})
s3.addPoints(spots3)
w3.addItem(s3)
s3.sigClicked.connect(clicked)


## Test performance of large scatterplots

s4 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 20))
pos = np.random.normal(size=(2,10000), scale=1e-9)
s4.addPoints(x=pos[0], y=pos[1])
w4.addItem(s4)
s4.sigClicked.connect(clicked)

# Enable hovering and show a tool tip hovered points
s4.setAcceptHoverEvents(True)
hoverPoints = []
hoverPen = pg.mkPen('g')
hoverRect = None

def hoverEvent(ev):
    global hoverPoints, hoverRect

    if hoverRect is None and len(hoverPoints) > 0:
        hoverRect = hoverPoints[0]._data['sourceRect']  # avoid gc to prevent removal of glyph from cache

    newPoints = [] if ev.exit else s4.pointsAt(ev.pos())

    for pt in chain(hoverPoints, newPoints):
        if pt in newPoints and pt not in hoverPoints:
            pt.setPen(hoverPen)

        elif pt in hoverPoints and pt not in newPoints:
            pt.resetPen()

    cutoff = 3
    tip = '\n\n'.join('index: {}\nx: {:.3g}\ny: {:.3g}'.format(pt.index(), pt.pos().x(), pt.pos().y())
                      for pt in newPoints[:cutoff])
    if len(newPoints) > cutoff:
        tip += '\n\n({} more...)'.format(len(newPoints) - cutoff)

    s4.getViewBox().setToolTip(tip)

    hoverPoints = newPoints

s4.hoverEvent = hoverEvent


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
