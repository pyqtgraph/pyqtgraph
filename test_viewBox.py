#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy import random
from PyQt4 import QtGui, QtCore
from GraphicsView import *
from graphicsItems import *


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
#w1 = QtGui.QGraphicsWidget()
l = QtGui.QGraphicsGridLayout()
l.setHorizontalSpacing(0)
l.setVerticalSpacing(0)


vb = ViewBox()
p1 = PlotCurveItem()
#gv.scene().addItem(vb)
vb.addItem(p1)
vl.addWidget(gv)
rect = QtGui.QGraphicsRectItem(QtCore.QRectF(0, 0, 1, 1))
rect.setPen(QtGui.QPen(QtGui.QColor(100, 200, 100)))
vb.addItem(rect)

l.addItem(vb, 0, 2)
gv.centralWidget.setLayout(l)


xScale = ScaleItem(orientation='bottom', linkView=vb)
l.addItem(xScale, 1, 2)
yScale = ScaleItem(orientation='left', linkView=vb)
l.addItem(yScale, 0, 1)

xLabel = LabelItem(u"<span style='color: #ff0000; font-weight: bold'>X</span> <i>Axis</i> <span style='font-size: 6pt'>(μV)</span>", html=True, color='CCC')
l.setRowFixedHeight(2, 20)
l.setRowFixedHeight(1, 40)
l.addItem(xLabel, 2, 2)
yLabel = LabelItem("Y Axis", color=QtGui.QColor(200, 200, 200))
yLabel.setAngle(90)
l.setColumnFixedWidth(0, 20)
l.setColumnFixedWidth(1, 60)
l.addItem(yLabel, 0, 0)


#grid = GridItem(gv)
#vb.addItem(grid)

#gv.scene().addItem(w1)
#w1.setGeometry(0, 0, 1, 1)


#c1 = Qwt.QwtPlotCurve()
#c1.setData(range(len(data)), data)
#c1.attach(p1)
#c2 = PlotCurve()
#c2.setData([1,2,3,4,5,6,7,8], [1,2,10,4,3,2,4,1])
#c2.attach(p2)

def rand(n):
    data = random.random(n)
    data[int(n*0.1):int(n*0.13)] += .5
    data[int(n*0.18)] += 2
    data[int(n*0.1):int(n*0.13)] *= 5
    data[int(n*0.18)] *= 20
    #c1.setData(range(len(data)), data)
    return data, arange(n, n+len(data)) / float(n)
    

def updateData():
    yd, xd = rand(10000)
    p1.updateData(yd, x=xd)
    
    #vb.setRange(p1.boundingRect())
    #p1.plot(yd, x=xd, clear=True)

yd, xd = rand(10000)
#p2.plot(yd * 1000, x=xd)
#for i in [1,2]:
    #for j in range(3):
        #yd, xd = rand(1000)
        #p3.plot(yd * 100000 * i, x=xd, params={'repetitions': j, 'scale': i})
updateData()
vb.autoRange()

t = QtCore.QTimer()
QtCore.QObject.connect(t, QtCore.SIGNAL('timeout()'), updateData)
t.start(50)

