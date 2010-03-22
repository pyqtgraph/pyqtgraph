# -*- coding: utf-8 -*-
"""
ImageView.py -  Widget for basic image dispay and analysis
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.

Widget used for displaying 2D or 3D data. Features:
  - float or int (including 16-bit int) image display via ImageItem
  - zoom/pan via GraphicsView
  - black/white level controls
  - time slider for 3D data sets
  - ROI plotting
  - Image normalization through a variety of methods
"""

from ImageViewTemplate import *
from graphicsItems import *
from widgets import ROI
from PyQt4 import QtCore, QtGui

class PlotROI(ROI):
    def __init__(self, size):
        ROI.__init__(self, pos=[0,0], size=size, scaleSnap=True, translateSnap=True)
        self.addScaleHandle([1, 1], [0, 0])


class ImageView(QtGui.QWidget):
    def __init__(self, parent=None, name="ImageView", *args):
        QtGui.QWidget.__init__(self, parent, *args)
        self.levelMax = 4096
        self.levelMin = 0
        self.name = name
        self.image = None
        self.imageDisp = None
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.scene = self.ui.graphicsView.sceneObj
        self.ui.graphicsView.enableMouse(True)
        self.ui.graphicsView.autoPixelRange = False
        self.ui.graphicsView.setAspectLocked(True)
        self.ui.graphicsView.invertY()
        self.ui.graphicsView.enableMouse()
        
        self.imageItem = ImageItem()
        self.scene.addItem(self.imageItem)
        self.currentIndex = 0
        
        self.ui.normGroup.hide()

        self.roi = PlotROI(10)
        self.roi.setZValue(20)
        self.scene.addItem(self.roi)
        self.roi.hide()
        self.ui.roiPlot.hide()
        self.roiCurve = self.ui.roiPlot.plot()
        self.roiTimeLine = InfiniteLine(self.ui.roiPlot, 0)
        self.roiTimeLine.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0, 200)))
        self.ui.roiPlot.addItem(self.roiTimeLine)
        
        self.normLines = []
        for i in [0,1]:
            l = InfiniteLine(self.ui.roiPlot, 0)
            l.setPen(QtGui.QPen(QtGui.QColor(0, 100, 200, 200)))
            self.ui.roiPlot.addItem(l)
            self.normLines.append(l)
            l.hide()
            
        for fn in ['addItem']:
            setattr(self, fn, getattr(self.ui.graphicsView, fn))

        QtCore.QObject.connect(self.ui.timeSlider, QtCore.SIGNAL('valueChanged(int)'), self.timeChanged)
        QtCore.QObject.connect(self.ui.whiteSlider, QtCore.SIGNAL('valueChanged(int)'), self.updateImage)
        QtCore.QObject.connect(self.ui.blackSlider, QtCore.SIGNAL('valueChanged(int)'), self.updateImage)
        QtCore.QObject.connect(self.ui.roiBtn, QtCore.SIGNAL('clicked()'), self.roiClicked)
        self.roi.connect(QtCore.SIGNAL('regionChanged'), self.roiChanged)
        QtCore.QObject.connect(self.ui.normBtn, QtCore.SIGNAL('toggled(bool)'), self.normToggled)
        QtCore.QObject.connect(self.ui.normDivideRadio, QtCore.SIGNAL('clicked()'), self.updateNorm)
        QtCore.QObject.connect(self.ui.normSubtractRadio, QtCore.SIGNAL('clicked()'), self.updateNorm)
        QtCore.QObject.connect(self.ui.normOffRadio, QtCore.SIGNAL('clicked()'), self.updateNorm)
        QtCore.QObject.connect(self.ui.normROICheck, QtCore.SIGNAL('clicked()'), self.updateNorm)
        QtCore.QObject.connect(self.ui.normFrameCheck, QtCore.SIGNAL('clicked()'), self.updateNorm)
        QtCore.QObject.connect(self.ui.normTimeRangeCheck, QtCore.SIGNAL('clicked()'), self.updateNorm)
        QtCore.QObject.connect(self.ui.normStartSlider, QtCore.SIGNAL('valueChanged(int)'), self.updateNorm)
        QtCore.QObject.connect(self.ui.normStopSlider, QtCore.SIGNAL('valueChanged(int)'), self.updateNorm)
        
        self.ui.roiPlot.registerPlot(self.name + '_ROI')

    def updateNorm(self):
        for l, sl in zip(self.normLines, [self.ui.normStartSlider, self.ui.normStopSlider]):
            if self.ui.normTimeRangeCheck.isChecked():
                l.show()
            else:
                l.hide()
            
            i, t = self.timeIndex(sl)
            l.setPos(t)
        
        
        self.imageDisp = None
        self.updateImage()
        self.roiChanged()

    def normToggled(self, b):
        self.ui.normGroup.setVisible(b)

    def roiClicked(self):
        if self.ui.roiBtn.isChecked():
            self.roi.show()
            self.ui.roiPlot.show()
            self.roiChanged()
        else:
            self.roi.hide()
            self.ui.roiPlot.hide()

    def roiChanged(self):
        if self.image is None:
            return
            
        image = self.getProcessedImage()
        if image.ndim == 2:
            axes = (0, 1)
        elif image.ndim == 3:
            axes = (1, 2)
        else:
            return
        data = self.roi.getArrayRegion(image.view(ndarray), self.imageItem, axes)
        if data is not None:
            while data.ndim > 1:
                data = data.mean(axis=1)
            self.roiCurve.setData(y=data, x=self.tVals)
            #self.ui.roiPlot.replot()

    def setImage(self, img, autoRange=True, autoLevels=True, levels=None):
        self.image = img
        if hasattr(img, 'xvals'):
            self.tVals = img.xvals(0)
        else:
            self.tVals = arange(img.shape[0])
        self.ui.timeSlider.setValue(0)
        #self.ui.normStartSlider.setValue(0)
        #self.ui.timeSlider.setMaximum(img.shape[0]-1)
            
        if img.ndim == 2:
            self.axes = {'t': None, 'x': 0, 'y': 1, 'c': None}
        elif img.ndim == 3:
            if img.shape[2] <= 3:
                self.axes = {'t': None, 'x': 0, 'y': 1, 'c': 2}
            else:
                self.axes = {'t': 0, 'x': 1, 'y': 2, 'c': None}
        elif img.ndim == 4:
            self.axes = {'t': 0, 'x': 1, 'y': 2, 'c': 3}

            
        self.imageDisp = None
        if autoRange:
            self.autoRange()
        if autoLevels:
            self.autoLevels()
        if levels is not None:
            self.levelMax = levels[1]
            self.levelMin = levels[0]
        self.updateImage()
        if self.ui.roiBtn.isChecked():
            self.roiChanged()
            
    def autoLevels(self):
        image = self.getProcessedImage()
        
        self.ui.whiteSlider.setValue(self.ui.whiteSlider.maximum())
        self.ui.blackSlider.setValue(0)
        self.imageItem.setLevels(white=self.whiteLevel(), black=self.blackLevel())
            
    def autoRange(self):
        image = self.getProcessedImage()
        
        self.ui.graphicsView.setRange(QtCore.QRectF(0, 0, image.shape[self.axes['x']], image.shape[self.axes['y']]), padding=0., lockAspect=True)        
        
    def getProcessedImage(self):
        if self.imageDisp is None:
            image = self.normalize(self.image)
            self.imageDisp = image
            self.levelMax = float(image.max())
            self.levelMin = float(image.min())
        return self.imageDisp
        
    def normalize(self, image):
        
        if self.ui.normOffRadio.isChecked():
            return image
            
        div = self.ui.normDivideRadio.isChecked()
        norm = image.copy()
        #if div:
            #norm = ones(image.shape)
        #else:
            #norm = zeros(image.shape)
        if div:
            norm = norm.astype(float32)
            
        if self.ui.normTimeRangeCheck.isChecked() and image.ndim == 3:
            (sind, start) = self.timeIndex(self.ui.normStartSlider)
            (eind, end) = self.timeIndex(self.ui.normStopSlider)
            #print start, end, sind, eind
            n = image[sind:eind+1].mean(axis=0)
            n.shape = (1,) + n.shape
            if div:
                norm /= n
            else:
                norm -= n
                
        if self.ui.normFrameCheck.isChecked() and image.ndim == 3:
            n = image.mean(axis=1).mean(axis=1)
            n.shape = n.shape + (1, 1)
            if div:
                norm /= n
            else:
                norm -= n
            
        return norm
        
        
        
    def timeChanged(self):
        (ind, time) = self.timeIndex(self.ui.timeSlider)
        if ind != self.currentIndex:
            self.currentIndex = ind
            self.updateImage()
        self.roiTimeLine.setPos(time)
        #self.ui.roiPlot.replot()
        self.emit(QtCore.SIGNAL('timeChanged'), ind, time)

    def updateImage(self):
        ## Redraw image on screen
        if self.image is None:
            return
            
        image = self.getProcessedImage()
        #print "update:", image.ndim, image.max(), image.min(), self.blackLevel(), self.whiteLevel()
        if self.axes['t'] is None:
            self.ui.timeSlider.hide()
            self.imageItem.updateImage(image, white=self.whiteLevel(), black=self.blackLevel())
        else:
            self.ui.timeSlider.show()
            self.imageItem.updateImage(image[self.currentIndex], white=self.whiteLevel(), black=self.blackLevel())
            
    def timeIndex(self, slider):
        """Return the time and frame index indicated by a slider"""
        if self.image is None:
            return (0,0)
        v = slider.value()
        vmax = slider.maximum()
        f = float(v) / vmax
        t = 0.0
        #xv = self.image.xvals('Time') 
        xv = self.tVals
        if xv is None:
            ind = int(f * self.image.shape[0])
        else:
            if len(xv) < 2:
                return (0,0)
            totTime = xv[-1] + (xv[-1]-xv[-2])
            t = f * totTime
            inds = argwhere(xv < t)
            if len(inds) < 1:
                return (0,t)
            ind = inds[-1,0]
        #print ind
        return ind, t

    def whiteLevel(self):
        return self.levelMin + (self.levelMax-self.levelMin) * self.ui.whiteSlider.value() / self.ui.whiteSlider.maximum() 
    
    def blackLevel(self):
        return self.levelMin + ((self.levelMax-self.levelMin) / self.ui.blackSlider.maximum()) * self.ui.blackSlider.value()
        