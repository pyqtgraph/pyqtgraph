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
from pyqtgraph.functions import mkPen
from PyQt4 import QtCore, QtGui

app = QtGui.QApplication([])

class Win(QtGui.QMainWindow):
  pass

w = Win()
v = GraphicsView(useOpenGL=False)
v.invertY(True)
v.setAspectLocked(True)
v.enableMouse(True)
v.autoPixelScale = False

w.setCentralWidget(v)
s = v.scene()
v.setRange(QtCore.QRect(-2, -2, 220, 220))

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

pi1 = PlotItem()
s.addItem(pi1)
pi1.scale(0.5, 0.5)
pi1.setGeometry(0, 170, 300, 100)

lastRoi = None

def updateRoi(roi):
    global im1, im2, im3, im4, arr, lastRoi
    if roi is None:
        return
    lastRoi = roi
    arr1 = roi.getArrayRegion(im1.image, img=im1)
    im3.updateImage(arr1, autoRange=True)
    arr2 = roi.getArrayRegion(im2.image, img=im2)
    im4.updateImage(arr2, autoRange=True)
    updateRoiPlot(roi, arr1)
    
def updateRoiPlot(roi, data=None):
    if data is None:
        data = roi.getArrayRegion(im1.image, img=im1)
    if data is not None:
        roi.curve.updateData(data.mean(axis=1))

#def updatePlot(roi)

rois = []
rois.append(TestROI([0,  0], [20, 20], maxBounds=QtCore.QRectF(-10, -10, 230, 140), pen=mkPen(0)))
rois.append(LineROI([0,  0], [20, 20], width=5, pen=mkPen(1)))
rois.append(MultiLineROI([[0, 50], [50, 60], [60, 30]], width=5, pen=mkPen(2)))
rois.append(EllipseROI([110, 10], [30, 20], pen=mkPen(3)))
rois.append(CircleROI([110, 50], [20, 20], pen=mkPen(4)))
rois.append(PolygonROI([[2,0], [2.1,0], [2,.1]], pen=mkPen(5)))
for r in rois:
    s.addItem(r)
    c = pi1.plot(pen=r.pen)
    r.curve = c
    r.connect(r, QtCore.SIGNAL('regionChanged'), updateRoi)

def updateImage():
    global im1, arr, lastRoi
    r = abs(random.normal(loc=0, scale=(arr.max()-arr.min())*0.1, size=arr.shape))
    im1.updateImage(arr + r)
    updateRoi(lastRoi)
    for r in rois:
        updateRoiPlot(r)
    
    
t = QtCore.QTimer()
t.connect(t, QtCore.SIGNAL('timeout()'), updateImage)
t.start(50)

w.show()
app.exec_()
