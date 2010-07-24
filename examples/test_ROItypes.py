#!/usr/bin/python -i
# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scipy import zeros
from pyqtgraph.graphicsWindows import *
from pyqtgraph.graphicsItems import *
from pyqtgraph.widgets import *
from pyqtgraph.PlotWidget import *
from PyQt4 import QtCore, QtGui

app = QtGui.QApplication([])

#i = PlotWindow(array([0,1,2,1,2]), parent=None, title='')

class Win(QtGui.QMainWindow):
  pass

w = Win()
v = GraphicsView(useOpenGL=False)
v.invertY(True)
v.setAspectLocked(True)
v.enableMouse(True)
v.autoPixelScale = False

w.setCentralWidget(v)
#s = QtGui.QGraphicsScene()
#v.setScene(s)
s = v.scene()

#p = Plot(array([0,2,1,3,4]), copy=False)
#s.addItem(p)

arr = ones((100, 100), dtype=float)
arr[45:55, 45:55] = 0
arr[25, :] = 5
arr[:, 25] = 5
arr[75, :] = 5
arr[:, 75] = 5
arr[50, :] = 10
arr[:, 50] = 10

im1 = ImageItem(arr)
im2 = ImageItem(arr)
s.addItem(im1)
s.addItem(im2)
im2.moveBy(110, 20)
im3 = ImageItem()
s.addItem(im3)
im3.moveBy(0, 130)
im3.setZValue(10)
im4 = ImageItem()
s.addItem(im4)
im4.moveBy(110, 130)
im4.setZValue(10)


#g = Grid(view=v, bounds=QtCore.QRectF(0.1, 0.1, 0.8, 0.8))
#g = Grid(view=v)
#s.addItem(g)

#wid = RectROI([0,  0], [2, 2], maxBounds=QtCore.QRectF(-1, -1, 5, 5))
roi = TestROI([0,  0], [20, 20], maxBounds=QtCore.QRectF(-10, -10, 230, 140))
s.addItem(roi)
roi2 = LineROI([0,  0], [20, 20], width=5)
s.addItem(roi2)
mlroi = MultiLineROI([[0, 50], [50, 60], [60, 30]], width=5)
s.addItem(mlroi)
elroi = EllipseROI([110, 10], [30, 20])
s.addItem(elroi)
croi = CircleROI([110, 50], [20, 20])
s.addItem(croi)
troi = PolygonROI([[0,0], [1,0], [0,1]])
s.addItem(troi)


def updateImg(roi):
  global im1, im2, im3, im4, arr
  arr1 = roi.getArrayRegion(arr, img=im1)
  im3.updateImage(arr1, autoRange=True)
  arr2 = roi.getArrayRegion(arr, img=im2)
  im4.updateImage(arr2, autoRange=True)

roi.connect(roi, QtCore.SIGNAL('regionChanged'), lambda: updateImg(roi))
roi2.connect(roi2, QtCore.SIGNAL('regionChanged'), lambda: updateImg(roi2))
croi.connect(croi, QtCore.SIGNAL('regionChanged'), lambda: updateImg(croi))
elroi.connect(elroi, QtCore.SIGNAL('regionChanged'), lambda: updateImg(elroi))
mlroi.connect(mlroi, QtCore.SIGNAL('regionChanged'), lambda: updateImg(mlroi))


v.setRange(QtCore.QRect(-2, -2, 220, 220))

w.show()
app.exec_()
