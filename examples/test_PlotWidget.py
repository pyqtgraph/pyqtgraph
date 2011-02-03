#!/usr/bin/python
# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scipy import random
from numpy import array, arange
from PyQt4 import QtGui, QtCore
from pyqtgraph.PlotWidget import *
from pyqtgraph.graphicsItems import *


app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
cw = QtGui.QWidget()
mw.setCentralWidget(cw)
l = QtGui.QVBoxLayout()
cw.setLayout(l)

pw = PlotWidget()
l.addWidget(pw)
pw2 = PlotWidget()
l.addWidget(pw2)
pw3 = PlotWidget()
l.addWidget(pw3)

pw.registerPlot('Plot1')
pw2.registerPlot('Plot2')

#p1 = PlotCurveItem()
#pw.addItem(p1)
p1 = pw.plot()
rect = QtGui.QGraphicsRectItem(QtCore.QRectF(0, 0, 1, 1))
rect.setPen(QtGui.QPen(QtGui.QColor(100, 200, 100)))
pw.addItem(rect)

#pen = QtGui.QPen(QtGui.QBrush(QtGui.QColor(255, 255, 255, 50)), 5)
#pen.setCosmetic(True)
#pen.setJoinStyle(QtCore.Qt.MiterJoin)
#p1.setShadowPen(pen)
p1.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))

#l1 = QtGui.QGraphicsLineItem(0, 2, 2, 3)
#l1.setPen(QtGui.QPen(QtGui.QColor(255,0,0)))

#l2 = InfiniteLine(pw2, 1.5, 90, movable=True)
#
#lr1 = LinearRegionItem(pw2, 'vertical', [1.1, 1.3])
#pw2.addItem(lr1)
#lr2 = LinearRegionItem(pw2, 'horizontal', [50, 100])
#pw2.addItem(lr2)


#l3 = InfiniteLine(pw, [1.5, 1.5], 45)
#pw.addItem(l1)
#pw2.addItem(l2)
#pw.addItem(l3)

pw3.plot(array([100000]*100))


mw.show()


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
pw.autoRange()

t = QtCore.QTimer()

QtCore.QObject.connect(t, QtCore.SIGNAL('timeout()'), updateData)
t.start(50)


for i in range(0, 5):
    for j in range(0, 3):
        yd, xd = rand(10000)
        pw2.plot(yd*(j+1), xd, params={'iter': i, 'val': j})
    
app.exec_()
