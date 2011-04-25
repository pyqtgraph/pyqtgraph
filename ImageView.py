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
import sys
#from numpy import ndarray
import ptime
import numpy as np

from SignalProxy import proxyConnect

class PlotROI(ROI):
    def __init__(self, size):
        ROI.__init__(self, pos=[0,0], size=size, scaleSnap=True, translateSnap=True)
        self.addScaleHandle([1, 1], [0, 0])


class ImageView(QtGui.QWidget):
    
    sigTimeChanged = QtCore.Signal(object, object)
    
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
        
        self.ignoreTimeLine = False
        
        if 'linux' in sys.platform.lower():   ## Stupid GL bug in linux.
            self.ui.graphicsView.setViewport(QtGui.QWidget())
        
        self.ui.graphicsView.enableMouse(True)
        self.ui.graphicsView.autoPixelRange = False
        self.ui.graphicsView.setAspectLocked(True)
        self.ui.graphicsView.invertY()
        self.ui.graphicsView.enableMouse()
        
        self.ticks = [t[0] for t in self.ui.gradientWidget.listTicks()]
        self.ticks[0].colorChangeAllowed = False
        self.ticks[1].colorChangeAllowed = False
        self.ui.gradientWidget.allowAdd = False
        self.ui.gradientWidget.setTickColor(self.ticks[1], QtGui.QColor(255,255,255))
        self.ui.gradientWidget.setOrientation('right')
        
        self.imageItem = ImageItem()
        self.scene.addItem(self.imageItem)
        self.currentIndex = 0
        
        self.ui.normGroup.hide()

        self.roi = PlotROI(10)
        self.roi.setZValue(20)
        self.scene.addItem(self.roi)
        self.roi.hide()
        self.normRoi = PlotROI(10)
        self.normRoi.setPen(QtGui.QPen(QtGui.QColor(255,255,0)))
        self.normRoi.setZValue(20)
        self.scene.addItem(self.normRoi)
        self.normRoi.hide()
        #self.ui.roiPlot.hide()
        self.roiCurve = self.ui.roiPlot.plot()
        self.timeLine = InfiniteLine(self.ui.roiPlot, 0, movable=True)
        self.timeLine.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0, 200)))
        self.timeLine.setZValue(1)
        self.ui.roiPlot.addItem(self.timeLine)
        self.ui.splitter.setSizes([self.height()-35, 35])
        self.ui.roiPlot.showScale('left', False)
        
        self.keysPressed = {}
        self.playTimer = QtCore.QTimer()
        self.playRate = 0
        self.lastPlayTime = 0
        
        #self.normLines = []
        #for i in [0,1]:
            #l = InfiniteLine(self.ui.roiPlot, 0)
            #l.setPen(QtGui.QPen(QtGui.QColor(0, 100, 200, 200)))
            #self.ui.roiPlot.addItem(l)
            #self.normLines.append(l)
            #l.hide()
        self.normRgn = LinearRegionItem(self.ui.roiPlot, 'vertical')
        self.normRgn.setZValue(0)
        self.ui.roiPlot.addItem(self.normRgn)
        self.normRgn.hide()
            
        ## wrap functions from graphics view
        for fn in ['addItem', 'removeItem']:
            setattr(self, fn, getattr(self.ui.graphicsView, fn))

        #QtCore.QObject.connect(self.ui.timeSlider, QtCore.SIGNAL('valueChanged(int)'), self.timeChanged)
        #self.timeLine.connect(self.timeLine, QtCore.SIGNAL('positionChanged'), self.timeLineChanged)
        self.timeLine.sigPositionChanged.connect(self.timeLineChanged)
        #QtCore.QObject.connect(self.ui.whiteSlider, QtCore.SIGNAL('valueChanged(int)'), self.updateImage)
        #QtCore.QObject.connect(self.ui.blackSlider, QtCore.SIGNAL('valueChanged(int)'), self.updateImage)
        #QtCore.QObject.connect(self.ui.gradientWidget, QtCore.SIGNAL('gradientChanged'), self.updateImage)
        self.ui.gradientWidget.sigGradientChanged.connect(self.updateImage)
        #QtCore.QObject.connect(self.ui.roiBtn, QtCore.SIGNAL('clicked()'), self.roiClicked)
        self.ui.roiBtn.clicked.connect(self.roiClicked)
        #self.roi.connect(self.roi, QtCore.SIGNAL('regionChanged'), self.roiChanged)
        self.roi.sigRegionChanged.connect(self.roiChanged)
        #QtCore.QObject.connect(self.ui.normBtn, QtCore.SIGNAL('toggled(bool)'), self.normToggled)
        self.ui.normBtn.toggled.connect(self.normToggled)
        #QtCore.QObject.connect(self.ui.normDivideRadio, QtCore.SIGNAL('clicked()'), self.updateNorm)
        self.ui.normDivideRadio.clicked.connect(self.updateNorm)
        #QtCore.QObject.connect(self.ui.normSubtractRadio, QtCore.SIGNAL('clicked()'), self.updateNorm)
        self.ui.normSubtractRadio.clicked.connect(self.updateNorm)
        #QtCore.QObject.connect(self.ui.normOffRadio, QtCore.SIGNAL('clicked()'), self.updateNorm)
        self.ui.normOffRadio.clicked.connect(self.updateNorm)
        #QtCore.QObject.connect(self.ui.normROICheck, QtCore.SIGNAL('clicked()'), self.updateNorm)
        self.ui.normROICheck.clicked.connect(self.updateNorm)
        #QtCore.QObject.connect(self.ui.normFrameCheck, QtCore.SIGNAL('clicked()'), self.updateNorm)
        self.ui.normFrameCheck.clicked.connect(self.updateNorm)
        #QtCore.QObject.connect(self.ui.normTimeRangeCheck, QtCore.SIGNAL('clicked()'), self.updateNorm)
        self.ui.normTimeRangeCheck.clicked.connect(self.updateNorm)
        #QtCore.QObject.connect(self.playTimer, QtCore.SIGNAL('timeout()'), self.timeout)
        self.playTimer.timeout.connect(self.timeout)
        
        ##QtCore.QObject.connect(self.ui.normStartSlider, QtCore.SIGNAL('valueChanged(int)'), self.updateNorm)
        #QtCore.QObject.connect(self.ui.normStopSlider, QtCore.SIGNAL('valueChanged(int)'), self.updateNorm)
        self.normProxy = proxyConnect(None, self.normRgn.sigRegionChanged, self.updateNorm)
        #self.normRoi.connect(self.normRoi, QtCore.SIGNAL('regionChangeFinished'), self.updateNorm)
        self.normRoi.sigRegionChangeFinished.connect(self.updateNorm)
        
        self.ui.roiPlot.registerPlot(self.name + '_ROI')
        
        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown]

    #def __dtor__(self):
        ##print "Called ImageView sip destructor"
        #self.quit()
        #QtGui.QWidget.__dtor__(self)
        
    def close(self):
        self.ui.roiPlot.close()
        self.ui.graphicsView.close()
        self.ui.gradientWidget.sigGradientChanged.disconnect(self.updateImage)
        self.scene.clear()
        del self.image
        del self.imageDisp
        #self.image = None
        #self.imageDisp = None
        self.setParent(None)
        
    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Space:
            if self.playRate == 0:
                fps = (self.getProcessedImage().shape[0]-1) / (self.tVals[-1] - self.tVals[0])
                self.play(fps)
                #print fps
            else:
                self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_Home:
            self.setCurrentIndex(0)
            self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_End:
            self.setCurrentIndex(self.getProcessedImage().shape[0]-1)
            self.play(0)
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
        else:
            QtGui.QWidget.keyPressEvent(self, ev)

    def keyReleaseEvent(self, ev):
        if ev.key() in [QtCore.Qt.Key_Space, QtCore.Qt.Key_Home, QtCore.Qt.Key_End]:
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except:
                self.keysPressed = {}
            self.evalKeyState()
        else:
            QtGui.QWidget.keyReleaseEvent(self, ev)
        
        
    def evalKeyState(self):
        if len(self.keysPressed) == 1:
            key = self.keysPressed.keys()[0]
            if key == QtCore.Qt.Key_Right:
                self.play(20)
                self.lastPlayTime = ptime.time() + 0.2  ## 2ms wait before start
                self.jumpFrames(1)
            elif key == QtCore.Qt.Key_Left:
                self.play(-20)
                self.lastPlayTime = ptime.time() + 0.2
                self.jumpFrames(-1)
            elif key == QtCore.Qt.Key_Up:
                self.play(-100)
            elif key == QtCore.Qt.Key_Down:
                self.play(100)
            elif key == QtCore.Qt.Key_PageUp:
                self.play(-1000)
            elif key == QtCore.Qt.Key_PageDown:
                self.play(1000)
        else:
            self.play(0)
        
    def play(self, rate):
        #print "play:", rate
        self.playRate = rate
        if rate == 0:
            self.playTimer.stop()
            return
            
        self.lastPlayTime = ptime.time()
        if not self.playTimer.isActive():
            self.playTimer.start(16)
            
        
    def timeout(self):
        now = ptime.time()
        dt = now - self.lastPlayTime
        if dt < 0:
            return
        n = int(self.playRate * dt)
        #print n, dt
        if n != 0:
            #print n, dt, self.lastPlayTime
            self.lastPlayTime += (float(n)/self.playRate)
            if self.currentIndex+n > self.image.shape[0]:
                self.play(0)
            self.jumpFrames(n)
        
    def setCurrentIndex(self, ind):
        self.currentIndex = clip(ind, 0, self.getProcessedImage().shape[0]-1)
        self.updateImage()
        self.ignoreTimeLine = True
        self.timeLine.setValue(self.tVals[self.currentIndex])
        self.ignoreTimeLine = False

    def jumpFrames(self, n):
        """If this is a video, move ahead n frames"""
        if self.axes['t'] is not None:
            self.setCurrentIndex(self.currentIndex + n)

    def updateNorm(self):
        #for l, sl in zip(self.normLines, [self.ui.normStartSlider, self.ui.normStopSlider]):
            #if self.ui.normTimeRangeCheck.isChecked():
                #l.show()
            #else:
                #l.hide()
            
            #i, t = self.timeIndex(sl)
            #l.setPos(t)
        
        if self.ui.normTimeRangeCheck.isChecked():
            #print "show!"
            self.normRgn.show()
        else:
            self.normRgn.hide()
        
        if self.ui.normROICheck.isChecked():
            #print "show!"
            self.normRoi.show()
        else:
            self.normRoi.hide()
        
        self.imageDisp = None
        self.updateImage()
        self.roiChanged()

    def normToggled(self, b):
        self.ui.normGroup.setVisible(b)
        self.normRoi.setVisible(b and self.ui.normROICheck.isChecked())
        self.normRgn.setVisible(b and self.ui.normTimeRangeCheck.isChecked())

    def roiClicked(self):
        if self.ui.roiBtn.isChecked():
            self.roi.show()
            #self.ui.roiPlot.show()
            self.ui.roiPlot.setMouseEnabled(True, True)
            self.ui.splitter.setSizes([self.height()*0.6, self.height()*0.4])
            self.roiCurve.show()
            self.roiChanged()
            self.ui.roiPlot.showScale('left', True)
        else:
            self.roi.hide()
            self.ui.roiPlot.setMouseEnabled(False, False)
            self.ui.roiPlot.setXRange(self.tVals.min(), self.tVals.max())
            self.ui.splitter.setSizes([self.height()-35, 35])
            self.roiCurve.hide()
            self.ui.roiPlot.showScale('left', False)

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
        data = self.roi.getArrayRegion(image.view(np.ndarray), self.imageItem, axes)
        if data is not None:
            while data.ndim > 1:
                data = data.mean(axis=1)
            self.roiCurve.setData(y=data, x=self.tVals)
            #self.ui.roiPlot.replot()

    def setImage(self, img, autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None):
        """Set the image to be displayed in the widget.
        Options are:
          img:         ndarray; the image to be displayed.
          autoRange:   bool; whether to scale/pan the view to fit the image.
          autoLevels:  bool; whether to update the white/black levels to fit the image.
          levels:      (min, max); the white and black level values to use.
          axes:        {'t':0, 'x':1, 'y':2, 'c':3}; Dictionary indicating the interpretation for each axis.
                       This is only needed to override the default guess.
        """
        if not isinstance(img, np.ndarray):
            raise Exception("Image must be specified as ndarray.")
        self.image = img
        
        if xvals is not None:
            self.tVals = xvals
        elif hasattr(img, 'xvals'):
            try:
                self.tVals = img.xvals(0)
            except:
                self.tVals = np.arange(img.shape[0])
        else:
            self.tVals = np.arange(img.shape[0])
        #self.ui.timeSlider.setValue(0)
        #self.ui.normStartSlider.setValue(0)
        #self.ui.timeSlider.setMaximum(img.shape[0]-1)
        
        if axes is None:
            if img.ndim == 2:
                self.axes = {'t': None, 'x': 0, 'y': 1, 'c': None}
            elif img.ndim == 3:
                if img.shape[2] <= 4:
                    self.axes = {'t': None, 'x': 0, 'y': 1, 'c': 2}
                else:
                    self.axes = {'t': 0, 'x': 1, 'y': 2, 'c': None}
            elif img.ndim == 4:
                self.axes = {'t': 0, 'x': 1, 'y': 2, 'c': 3}
            else:
                raise Exception("Can not interpret image with dimensions %s" % (str(img.shape)))
        elif isinstance(axes, dict):
            self.axes = axes.copy()
        elif isinstance(axes, list) or isinstance(axes, tuple):
            self.axes = {}
            for i in range(len(axes)):
                self.axes[axes[i]] = i
        else:
            raise Exception("Can not interpret axis specification %s. Must be like {'t': 2, 'x': 0, 'y': 1} or ('t', 'x', 'y', 'c')" % (str(axes)))
            
        for x in ['t', 'x', 'y', 'c']:
            self.axes[x] = self.axes.get(x, None)
            
        self.imageDisp = None
        if autoLevels:
            self.autoLevels()
        if levels is not None:
            self.levelMax = levels[1]
            self.levelMin = levels[0]
            
        self.currentIndex = 0
        self.updateImage()
        if self.ui.roiBtn.isChecked():
            self.roiChanged()
            
            
        if self.axes['t'] is not None:
            #self.ui.roiPlot.show()
            self.ui.roiPlot.setXRange(self.tVals.min(), self.tVals.max())
            self.timeLine.setValue(0)
            #self.ui.roiPlot.setMouseEnabled(False, False)
            if len(self.tVals) > 1:
                start = self.tVals.min()
                stop = self.tVals.max() + abs(self.tVals[-1] - self.tVals[0]) * 0.02
            elif len(self.tVals) == 1:
                start = self.tVals[0] - 0.5
                stop = self.tVals[0] + 0.5
            else:
                start = 0
                stop = 1
            for s in [self.timeLine, self.normRgn]:
                s.setBounds([start, stop])
        #else:
            #self.ui.roiPlot.hide()
            
        self.imageItem.resetTransform()
        if scale is not None:
            self.imageItem.scale(*scale)
        if scale is not None:
            self.imageItem.setPos(*pos)
            
        if autoRange:
            self.autoRange()
        self.roiClicked()
            
            
    def autoLevels(self):
        image = self.getProcessedImage()
        
        #self.ui.whiteSlider.setValue(self.ui.whiteSlider.maximum())
        #self.ui.blackSlider.setValue(0)
        
        self.ui.gradientWidget.setTickValue(self.ticks[0], 0.0)
        self.ui.gradientWidget.setTickValue(self.ticks[1], 1.0)
        self.imageItem.setLevels(white=self.whiteLevel(), black=self.blackLevel())
            

    def autoRange(self):
        image = self.getProcessedImage()
        
        #self.ui.graphicsView.setRange(QtCore.QRectF(0, 0, image.shape[self.axes['x']], image.shape[self.axes['y']]), padding=0., lockAspect=True)        
        self.ui.graphicsView.setRange(self.imageItem.sceneBoundingRect(), padding=0., lockAspect=True)
        
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
        norm = image.view(np.ndarray).copy()
        #if div:
            #norm = ones(image.shape)
        #else:
            #norm = zeros(image.shape)
        if div:
            norm = norm.astype(np.float32)
            
        if self.ui.normTimeRangeCheck.isChecked() and image.ndim == 3:
            (sind, start) = self.timeIndex(self.normRgn.lines[0])
            (eind, end) = self.timeIndex(self.normRgn.lines[1])
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
            
        if self.ui.normROICheck.isChecked() and image.ndim == 3:
            n = self.normRoi.getArrayRegion(norm, self.imageItem, (1, 2)).mean(axis=1).mean(axis=1)
            n = n[:,np.newaxis,np.newaxis]
            #print start, end, sind, eind
            if div:
                norm /= n
            else:
                norm -= n
                
        return norm
        
    def timeLineChanged(self):
        #(ind, time) = self.timeIndex(self.ui.timeSlider)
        if self.ignoreTimeLine:
            return
        self.play(0)
        (ind, time) = self.timeIndex(self.timeLine)
        if ind != self.currentIndex:
            self.currentIndex = ind
            self.updateImage()
        #self.timeLine.setPos(time)
        #self.emit(QtCore.SIGNAL('timeChanged'), ind, time)
        self.sigTimeChanged.emit(ind, time)

    def updateImage(self):
        ## Redraw image on screen
        if self.image is None:
            return
            
        image = self.getProcessedImage()
        #print "update:", image.ndim, image.max(), image.min(), self.blackLevel(), self.whiteLevel()
        if self.axes['t'] is None:
            #self.ui.timeSlider.hide()
            self.imageItem.updateImage(image, white=self.whiteLevel(), black=self.blackLevel())
            self.ui.roiPlot.hide()
            self.ui.roiBtn.hide()
        else:
            self.ui.roiBtn.show()
            self.ui.roiPlot.show()
            #self.ui.timeSlider.show()
            self.imageItem.updateImage(image[self.currentIndex], white=self.whiteLevel(), black=self.blackLevel())
            
            
    def timeIndex(self, slider):
        """Return the time and frame index indicated by a slider"""
        if self.image is None:
            return (0,0)
        #v = slider.value()
        #vmax = slider.maximum()
        #f = float(v) / vmax
        
        t = slider.value()
        
        #t = 0.0
        #xv = self.image.xvals('Time') 
        xv = self.tVals
        if xv is None:
            ind = int(t)
            #ind = int(f * self.image.shape[0])
        else:
            if len(xv) < 2:
                return (0,0)
            totTime = xv[-1] + (xv[-1]-xv[-2])
            #t = f * totTime
            inds = np.argwhere(xv < t)
            if len(inds) < 1:
                return (0,t)
            ind = inds[-1,0]
        #print ind
        return ind, t

    def whiteLevel(self):
        return self.levelMin + (self.levelMax-self.levelMin) * self.ui.gradientWidget.tickValue(self.ticks[1])
        #return self.levelMin + (self.levelMax-self.levelMin) * self.ui.whiteSlider.value() / self.ui.whiteSlider.maximum() 
    
    def blackLevel(self):
        return self.levelMin + (self.levelMax-self.levelMin) * self.ui.gradientWidget.tickValue(self.ticks[0])
        #return self.levelMin + ((self.levelMax-self.levelMin) / self.ui.blackSlider.maximum()) * self.ui.blackSlider.value()
        