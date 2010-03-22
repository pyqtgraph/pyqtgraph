#!/usr/bin/python
# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scipy import random
from PyQt4 import QtGui, QtCore
from pyqtgraph.GraphicsView import *
from pyqtgraph.graphicsItems import *

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
cw = QtGui.QWidget()
vl = QtGui.QVBoxLayout()
cw.setLayout(vl)
mw.setCentralWidget(cw)
mw.show()
mw.resize(800, 600)


gv = GraphicsView(cw)
gv.enableMouse(False)
l = QtGui.QGraphicsGridLayout()
l.setHorizontalSpacing(0)
l.setVerticalSpacing(0)


vb = ViewBox()
p1 = PlotCurveItem()
vb.addItem(p1)
vl.addWidget(gv)
rect = QtGui.QGraphicsRectItem(QtCore.QRectF(0, 0, 1, 1))
rect.setPen(QtGui.QPen(QtGui.QColor(100, 200, 100)))
vb.addItem(rect)

l.addItem(vb, 0, 1)
gv.centralWidget.setLayout(l)


xScale = ScaleItem(orientation='bottom', linkView=vb)
l.addItem(xScale, 1, 1)
yScale = ScaleItem(orientation='left', linkView=vb)
l.addItem(yScale, 0, 0)

xScale.setLabel(text=u"<span style='color: #ff0000; font-weight: bold'>X</span> <i>Axis</i>", units="s")
yScale.setLabel('Y Axis', units='V')

def rand(n):
    data = random.random(n)
    data[int(n*0.1):int(n*0.13)] += .5
    data[int(n*0.18)] += 2
    data[int(n*0.1):int(n*0.13)] *= 5
    data[int(n*0.18)] *= 20
    return data, arange(n, n+len(data)) / float(n)
    

def updateData():
    yd, xd = rand(10000)
    p1.updateData(yd, x=xd)

yd, xd = rand(10000)
updateData()
vb.autoRange()

t = QtCore.QTimer()
QtCore.QObject.connect(t, QtCore.SIGNAL('timeout()'), updateData)
t.start(50)

app.exec_()