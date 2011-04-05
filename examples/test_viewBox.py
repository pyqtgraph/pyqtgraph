#!/usr/bin/python
# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

## This example uses a ViewBox to create a PlotWidget-like interface

#from scipy import random
import numpy as np
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
cw = QtGui.QWidget()
vl = QtGui.QVBoxLayout()
cw.setLayout(vl)
mw.setCentralWidget(cw)
mw.show()
mw.resize(800, 600)


gv = pg.GraphicsView(cw)
gv.enableMouse(False)    ## Mouse interaction will be handled by the ViewBox
l = QtGui.QGraphicsGridLayout()
l.setHorizontalSpacing(0)
l.setVerticalSpacing(0)


vb = pg.ViewBox()
p1 = pg.PlotCurveItem()
vb.addItem(p1)
vl.addWidget(gv)

class movableRect(QtGui.QGraphicsRectItem):
    def __init__(self, *args):
        QtGui.QGraphicsRectItem.__init__(self, *args)
        self.setAcceptHoverEvents(True)
    def hoverEnterEvent(self, ev):
        self.savedPen = self.pen()
        self.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        ev.ignore()
    def hoverLeaveEvent(self, ev):
        self.setPen(self.savedPen)
        ev.ignore()
    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            self.pressDelta = self.mapToParent(ev.pos()) - self.pos()
        else:
            ev.ignore()     
    def mouseMoveEvent(self, ev):
        self.setPos(self.mapToParent(ev.pos()) - self.pressDelta)
        

#rect = QtGui.QGraphicsRectItem(QtCore.QRectF(0, 0, 1, 1))
rect = movableRect(QtCore.QRectF(0, 0, 1, 1))
rect.setPen(QtGui.QPen(QtGui.QColor(100, 200, 100)))
vb.addItem(rect)

l.addItem(vb, 0, 1)
gv.centralWidget.setLayout(l)


xScale = pg.ScaleItem(orientation='bottom', linkView=vb)
l.addItem(xScale, 1, 1)
yScale = pg.ScaleItem(orientation='left', linkView=vb)
l.addItem(yScale, 0, 0)

xScale.setLabel(text=u"<span style='color: #ff0000; font-weight: bold'>X</span> <i>Axis</i>", units="s")
yScale.setLabel('Y Axis', units='V')

def rand(n):
    data = np.random.random(n)
    data[int(n*0.1):int(n*0.13)] += .5
    data[int(n*0.18)] += 2
    data[int(n*0.1):int(n*0.13)] *= 5
    data[int(n*0.18)] *= 20
    return data, np.arange(n, n+len(data)) / float(n)
    

def updateData():
    yd, xd = rand(10000)
    p1.updateData(yd, x=xd)

yd, xd = rand(10000)
updateData()
vb.autoRange()

t = QtCore.QTimer()
t.timeout.connect(updateData)
t.start(50)

## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()
