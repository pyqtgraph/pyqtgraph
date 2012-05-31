#!/usr/bin/python -i
# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

## Create image to display
arr = np.ones((100, 100), dtype=float)
arr[45:55, 45:55] = 0
arr[25, :] = 5
arr[:, 25] = 5
arr[75, :] = 5
arr[:, 75] = 5
arr[50, :] = 10
arr[:, 50] = 10
arr += np.sin(np.linspace(0, 20, 100)).reshape(1, 100)
arr += np.random.normal(size=(100,100))


## create GUI
app = QtGui.QApplication([])
w = pg.GraphicsWindow(size=(800,800), border=True)

text = """Data Selection From Image.<br>\n
Drag an ROI or its handles to update the selected image.<br>
Hold CTRL while dragging to snap to pixel boundaries<br>
and 15-degree rotation angles.
"""
w1 = w.addLayout(row=0, col=0)
label1 = w1.addLabel(text, row=0, col=0)
v1a = w1.addViewBox(row=1, col=0, lockAspect=True)
v1b = w1.addViewBox(row=2, col=0, lockAspect=True)
img1a = pg.ImageItem(arr)
v1a.addItem(img1a)
img1b = pg.ImageItem()
v1b.addItem(img1b)

rois = []
rois.append(pg.RectROI([20, 20], [20, 20], pen=(0,9)))
rois[-1].addRotateHandle([1,0], [0.5, 0.5])
rois.append(pg.LineROI([0, 60], [20, 80], width=5, pen=(1,9)))
rois.append(pg.MultiLineROI([[20, 90], [50, 60], [60, 90]], width=5, pen=(2,9)))
rois.append(pg.EllipseROI([60, 10], [30, 20], pen=(3,9)))
rois.append(pg.CircleROI([80, 50], [20, 20], pen=(4,9)))
#rois.append(pg.LineSegmentROI([[110, 50], [20, 20]], pen=(5,9)))
#rois.append(pg.PolyLineROI([[110, 60], [20, 30], [50, 10]], pen=(6,9)))

def update(roi):
    img1b.setImage(roi.getArrayRegion(arr, img1a), levels=(0, arr.max()))
    v1b.autoRange()
    
for roi in rois:
    roi.sigRegionChanged.connect(update)
    v1a.addItem(roi)

update(rois[-1])
    


text = """User-Modifiable ROIs<br>
Click on a line segment to add a new handle.
Right click on a handle to remove.
"""
w2 = w.addLayout(row=0, col=1)
label2 = w2.addLabel(text, row=0, col=0)
v2a = w2.addViewBox(row=1, col=0, lockAspect=True)
r2a = pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True)
v2a.addItem(r2a)
r2b = pg.PolyLineROI([[0,-20], [10,-10], [10,-30]], closed=False)
v2a.addItem(r2b)

text = """Building custom ROI types<Br>
ROIs can be built with a variety of different handle types<br>
that scale and rotate the roi around an arbitrary center location
"""
w3 = w.addLayout(row=1, col=0)
label3 = w3.addLabel(text, row=0, col=0)
v3 = w3.addViewBox(row=1, col=0, lockAspect=True)

r3a = pg.ROI([0,0], [10,10])
v3.addItem(r3a)
## handles scaling horizontally around center
r3a.addScaleHandle([1, 0.5], [0.5, 0.5])
r3a.addScaleHandle([0, 0.5], [0.5, 0.5])

## handles scaling vertically from opposite edge
r3a.addScaleHandle([0.5, 0], [0.5, 1])
r3a.addScaleHandle([0.5, 1], [0.5, 0])

## handles scaling both vertically and horizontally
r3a.addScaleHandle([1, 1], [0, 0])
r3a.addScaleHandle([0, 0], [1, 1])

r3b = pg.ROI([20,0], [10,10])
v3.addItem(r3b)
## handles rotating around center
r3b.addRotateHandle([1, 1], [0.5, 0.5])
r3b.addRotateHandle([0, 0], [0.5, 0.5])

## handles rotating around opposite corner
r3b.addRotateHandle([1, 0], [0, 1])
r3b.addRotateHandle([0, 1], [1, 0])

## handles rotating/scaling around center
r3b.addScaleRotateHandle([0, 0.5], [0.5, 0.5])
r3b.addScaleRotateHandle([1, 0.5], [0.5, 0.5])


text = """Transforming objects with ROI"""
w4 = w.addLayout(row=1, col=1)
label4 = w4.addLabel(text, row=0, col=0)
v4 = w4.addViewBox(row=1, col=0, lockAspect=True)
g = pg.GridItem()
v4.addItem(g)
r4 = pg.ROI([0,0], [100,100])
r4.addRotateHandle([1,0], [0.5, 0.5])
r4.addRotateHandle([0,1], [0.5, 0.5])
img4 = pg.ImageItem(arr)
v4.addItem(r4)
img4.setParentItem(r4)





#v = w.addViewBox(colspan=2)

#v.invertY(True)  ## Images usually have their Y-axis pointing downward
#v.setAspectLocked(True)



### Create image items, add to scene and set position 
#im1 = pg.ImageItem(arr)
#im2 = pg.ImageItem(arr)
#v.addItem(im1)
#v.addItem(im2)
#im2.moveBy(110, 20)
#v.setRange(QtCore.QRectF(0, 0, 200, 120))

#im3 = pg.ImageItem()
#v2 = w.addViewBox(1,0)
#v2.addItem(im3)
#v2.setRange(QtCore.QRectF(0, 0, 60, 60))
#v2.invertY(True)
#v2.setAspectLocked(True)
##im3.moveBy(0, 130)
#im3.setZValue(10)

#im4 = pg.ImageItem()
#v3 = w.addViewBox(1,1)
#v3.addItem(im4)
#v3.setRange(QtCore.QRectF(0, 0, 60, 60))
#v3.invertY(True)
#v3.setAspectLocked(True)
##im4.moveBy(110, 130)
#im4.setZValue(10)

### create the plot
#pi1 = w.addPlot(2,0, colspan=2)
##pi1 = pg.PlotItem()
##s.addItem(pi1)
##pi1.scale(0.5, 0.5)
##pi1.setGeometry(0, 170, 300, 100)

#lastRoi = None

#def updateRoi(roi):
    #global im1, im2, im3, im4, arr, lastRoi
    #if roi is None:
        #return
    #lastRoi = roi
    #arr1 = roi.getArrayRegion(im1.image, img=im1)
    #im3.setImage(arr1)
    #arr2 = roi.getArrayRegion(im2.image, img=im2)
    #im4.setImage(arr2)
    #updateRoiPlot(roi, arr1)
    
#def updateRoiPlot(roi, data=None):
    #if data is None:
        #data = roi.getArrayRegion(im1.image, img=im1)
    #if data is not None:
        #roi.curve.setData(data.mean(axis=1))


### Create a variety of different ROI types
#rois = []
#rois.append(pg.TestROI([0,  0], [20, 20], maxBounds=QtCore.QRectF(-10, -10, 230, 140), pen=(0,9)))
#rois.append(pg.LineROI([0,  0], [20, 20], width=5, pen=(1,9)))
#rois.append(pg.MultiLineROI([[0, 50], [50, 60], [60, 30]], width=5, pen=(2,9)))
#rois.append(pg.EllipseROI([110, 10], [30, 20], pen=(3,9)))
#rois.append(pg.CircleROI([110, 50], [20, 20], pen=(4,9)))
#rois.append(pg.LineSegmentROI([[110, 50], [20, 20]], pen=(5,9)))
#rois.append(pg.PolyLineROI([[110, 60], [20, 30], [50, 10]], pen=(6,9)))
##rois.append(pg.PolygonROI([[2,0], [2.1,0], [2,.1]], pen=(5,9)))
##rois.append(SpiralROI([20,30], [1,1], pen=mkPen(0)))

### Add each ROI to the scene and link its data to a plot curve with the same color
#for r in rois:
    #v.addItem(r)
    #c = pi1.plot(pen=r.pen)
    #r.curve = c
    #r.sigRegionChanged.connect(updateRoi)

#def updateImage():
    #global im1, arr, lastRoi
    #r = abs(np.random.normal(loc=0, scale=(arr.max()-arr.min())*0.1, size=arr.shape))
    #im1.updateImage(arr + r)
    #updateRoi(lastRoi)
    #for r in rois:
        #updateRoiPlot(r)
    
### Rapidly update one of the images with random noise    
#t = QtCore.QTimer()
#t.timeout.connect(updateImage)
#t.start(50)



## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()
