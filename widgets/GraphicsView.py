# -*- coding: utf-8 -*-
"""
GraphicsView.py -   Extension of QGraphicsView
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""

from pyqtgraph.Qt import QtCore, QtGui, QtOpenGL, QtSvg
#from numpy import vstack
#import time
from pyqtgraph.Point import Point
#from vector import *
import sys, os
#import debug    
from FileDialog import FileDialog
from pyqtgraph.GraphicsScene import GraphicsScene
import numpy as np
import pyqtgraph.functions as fn

__all__ = ['GraphicsView']

class GraphicsView(QtGui.QGraphicsView):
    
    sigRangeChanged = QtCore.Signal(object, object)
    sigMouseReleased = QtCore.Signal(object)
    sigSceneMouseMoved = QtCore.Signal(object)
    #sigRegionChanged = QtCore.Signal(object)
    sigScaleChanged = QtCore.Signal(object)
    lastFileDir = None
    
    def __init__(self, parent=None, useOpenGL=None, background='k'):
        """Re-implementation of QGraphicsView that removes scrollbars and allows unambiguous control of the 
        viewed coordinate range. Also automatically creates a QGraphicsScene and a central QGraphicsWidget
        that is automatically scaled to the full view geometry.
        
        By default, the view coordinate system matches the widget's pixel coordinates and 
        automatically updates when the view is resized. This can be overridden by setting 
        autoPixelRange=False. The exact visible range can be set with setRange().
        
        The view can be panned using the middle mouse button and scaled using the right mouse button if
        enabled via enableMouse()."""
        self.closed = False
        
        QtGui.QGraphicsView.__init__(self, parent)
        
        ## in general openGL is poorly supported in Qt. 
        ## we only enable it where the performance benefit is critical.
        if useOpenGL is None:
            if 'linux' in sys.platform:  ## linux has numerous bugs in opengl implementation
                useOpenGL = False
            elif 'darwin' in sys.platform: ## openGL greatly speeds up display on mac
                useOpenGL = True
            else:
                useOpenGL = False
        self.useOpenGL(useOpenGL)
        
        self.setCacheMode(self.CacheBackground)
        
        if background is not None:
            brush = fn.mkBrush(background)
            self.setBackgroundBrush(brush)
        
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFrameShape(QtGui.QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QtGui.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtGui.QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QtGui.QGraphicsView.MinimalViewportUpdate)
        
        
        #self.setSceneRect(QtCore.QRectF(-1e10, -1e10, 2e10, 2e10))
        
        self.lockedViewports = []
        self.lastMousePos = None
        self.setMouseTracking(True)
        self.aspectLocked = False
        #self.yInverted = True
        self.range = QtCore.QRectF(0, 0, 1, 1)
        self.autoPixelRange = True
        self.currentItem = None
        self.clearMouse()
        self.updateMatrix()
        self.sceneObj = GraphicsScene()
        self.setScene(self.sceneObj)
        
        ## by default we set up a central widget with a grid layout.
        ## this can be replaced if needed.
        self.centralWidget = None
        self.setCentralItem(QtGui.QGraphicsWidget())
        self.centralLayout = QtGui.QGraphicsGridLayout()
        self.centralWidget.setLayout(self.centralLayout)
        
        self.mouseEnabled = False
        self.scaleCenter = False  ## should scaling center around view center (True) or mouse click (False)
        self.clickAccepted = False
        
    #def paintEvent(self, *args):
        #prof = debug.Profiler('GraphicsView.paintEvent '+str(id(self)), disabled=False)
        #QtGui.QGraphicsView.paintEvent(self, *args)
        #prof.finish()
        
    def close(self):
        self.centralWidget = None
        self.scene().clear()
        #print "  ", self.scene().itemCount()
        self.currentItem = None
        self.sceneObj = None
        self.closed = True
        self.setViewport(None)
        
    def useOpenGL(self, b=True):
        if b:
            v = QtOpenGL.QGLWidget()
        else:
            v = QtGui.QWidget()
            
        #v.setStyleSheet("background-color: #000000;")
        self.setViewport(v)
            
    def keyPressEvent(self, ev):
        #QtGui.QGraphicsView.keyPressEvent(self, ev)
        self.scene().keyPressEvent(ev)  ## bypass view, hand event directly to scene
                                        ## (view likes to eat arrow key events)
        
        
    def setCentralItem(self, item):
        return self.setCentralWidget(item)
        
    def setCentralWidget(self, item):
        """Sets a QGraphicsWidget to automatically fill the entire view."""
        if self.centralWidget is not None:
            self.scene().removeItem(self.centralWidget)
        self.centralWidget = item
        self.sceneObj.addItem(item)
        self.resizeEvent(None)
        
    def addItem(self, *args):
        return self.scene().addItem(*args)
        
    def removeItem(self, *args):
        return self.scene().removeItem(*args)
        
    def enableMouse(self, b=True):
        self.mouseEnabled = b
        self.autoPixelRange = (not b)
        
    def clearMouse(self):
        self.mouseTrail = []
        self.lastButtonReleased = None
    
    def resizeEvent(self, ev):
        if self.closed:
            return
        if self.autoPixelRange:
            self.range = QtCore.QRectF(0, 0, self.size().width(), self.size().height())
        self.setRange(self.range, padding=0, disableAutoPixel=False)
        self.updateMatrix()
    
    def updateMatrix(self, propagate=True):
        self.setSceneRect(self.range)
        if self.aspectLocked:
            self.fitInView(self.range, QtCore.Qt.KeepAspectRatio)
        else:
            self.fitInView(self.range, QtCore.Qt.IgnoreAspectRatio)
            
        self.sigRangeChanged.emit(self, self.range)
        
        if propagate:
            for v in self.lockedViewports:
                v.setXRange(self.range, padding=0)
        
    def viewRect(self):
        """Return the boundaries of the view in scene coordinates"""
        ## easier to just return self.range ?
        r = QtCore.QRectF(self.rect())
        return self.viewportTransform().inverted()[0].mapRect(r)

    def visibleRange(self):
        ## for backward compatibility
        return self.viewRect()

    def translate(self, dx, dy):
        self.range.adjust(dx, dy, dx, dy)
        self.updateMatrix()
    
    def scale(self, sx, sy, center=None):
        scale = [sx, sy]
        if self.aspectLocked:
            scale[0] = scale[1]
        #adj = (self.range.width()*0.5*(1.0-(1.0/scale[0])), self.range.height()*0.5*(1.0-(1.0/scale[1])))
        #print "======\n", scale, adj
        #print self.range
        #self.range.adjust(adj[0], adj[1], -adj[0], -adj[1])
        #print self.range
        
        if self.scaleCenter:
            center = None
        if center is None:
            center = self.range.center()
            
        w = self.range.width()  / scale[0]
        h = self.range.height() / scale[1]
        self.range = QtCore.QRectF(center.x() - (center.x()-self.range.left()) / scale[0], center.y() - (center.y()-self.range.top())  /scale[1], w, h)
        
        
        self.updateMatrix()
        self.sigScaleChanged.emit(self)

    def setRange(self, newRect=None, padding=0.05, lockAspect=None, propagate=True, disableAutoPixel=True):
        if disableAutoPixel:
            self.autoPixelRange=False
        if newRect is None:
            newRect = self.visibleRange()
            padding = 0
        
        padding = Point(padding)
        newRect = QtCore.QRectF(newRect)
        pw = newRect.width() * padding[0]
        ph = newRect.height() * padding[1]
        newRect = newRect.adjusted(-pw, -ph, pw, ph)
        scaleChanged = False
        if self.range.width() != newRect.width() or self.range.height() != newRect.height():
            scaleChanged = True
        self.range = newRect
        #print "New Range:", self.range
        self.centralWidget.setGeometry(self.range)
        self.updateMatrix(propagate)
        if scaleChanged:
            self.sigScaleChanged.emit(self)

    def scaleToImage(self, image):
        """Scales such that pixels in image are the same size as screen pixels. This may result in a significant performance increase."""
        pxSize = image.pixelSize()
        image.setPxMode(True)
        try:
            self.sigScaleChanged.disconnect(image.setScaledMode)
        except TypeError:
            pass
        tl = image.sceneBoundingRect().topLeft()
        w = self.size().width() * pxSize[0]
        h = self.size().height() * pxSize[1]
        range = QtCore.QRectF(tl.x(), tl.y(), w, h)
        self.setRange(range, padding=0)
        self.sigScaleChanged.connect(image.setScaledMode)
        
        
        
    def lockXRange(self, v1):
        if not v1 in self.lockedViewports:
            self.lockedViewports.append(v1)
        
    def setXRange(self, r, padding=0.05):
        r1 = QtCore.QRectF(self.range)
        r1.setLeft(r.left())
        r1.setRight(r.right())
        self.setRange(r1, padding=[padding, 0], propagate=False)
        
    def setYRange(self, r, padding=0.05):
        r1 = QtCore.QRectF(self.range)
        r1.setTop(r.top())
        r1.setBottom(r.bottom())
        self.setRange(r1, padding=[0, padding], propagate=False)
        
    #def invertY(self, invert=True):
        ##if self.yInverted != invert:
            ##self.scale[1] *= -1.
        #self.yInverted = invert
        #self.updateMatrix()
    
    
    def wheelEvent(self, ev):
        QtGui.QGraphicsView.wheelEvent(self, ev)
        if not self.mouseEnabled:
            return
        sc = 1.001 ** ev.delta()
        #self.scale *= sc
        #self.updateMatrix()
        self.scale(sc, sc)
        
    def setAspectLocked(self, s):
        self.aspectLocked = s
        
    #def mouseDoubleClickEvent(self, ev):
        #QtGui.QGraphicsView.mouseDoubleClickEvent(self, ev)
        #pass
        
    ### This function is here because interactive mode is disabled due to bugs.
    #def graphicsSceneEvent(self, ev, pev=None, fev=None):
        #ev1 = GraphicsSceneMouseEvent()
        #ev1.setPos(QtCore.QPointF(ev.pos().x(), ev.pos().y()))
        #ev1.setButtons(ev.buttons())
        #ev1.setButton(ev.button())
        #ev1.setModifiers(ev.modifiers())
        #ev1.setScenePos(self.mapToScene(QtCore.QPoint(ev.pos())))
        #if pev is not None:
            #ev1.setLastPos(pev.pos())
            #ev1.setLastScenePos(pev.scenePos())
            #ev1.setLastScreenPos(pev.screenPos())
        #if fev is not None:
            #ev1.setButtonDownPos(fev.pos())
            #ev1.setButtonDownScenePos(fev.scenePos())
            #ev1.setButtonDownScreenPos(fev.screenPos())
        #return ev1
        
    def leaveEvent(self, ev):
        self.scene().leaveEvent(ev)  ## inform scene when mouse leaves
        
    def mousePressEvent(self, ev):
        QtGui.QGraphicsView.mousePressEvent(self, ev)

        #print "Press over:"
        #for i in self.items(ev.pos()):
        #    print i.zValue(), int(i.acceptedMouseButtons()), i, i.scenePos()
        #print "Event accepted:", ev.isAccepted()
        #print "Grabber:", self.scene().mouseGrabberItem()
        

        if not self.mouseEnabled:
            return
        self.lastMousePos = Point(ev.pos())
        self.mousePressPos = ev.pos()
        self.clickAccepted = ev.isAccepted()
        if not self.clickAccepted:
            self.scene().clearSelection()
        return   ## Everything below disabled for now..
        
        #self.currentItem = None
        #maxZ = None
        #for i in self.items(ev.pos()):
            #if maxZ is None or maxZ < i.zValue():
                #self.currentItem = i
                #maxZ = i.zValue()
        #print "make event"
        #self.pev = self.graphicsSceneEvent(ev)
        #self.fev = self.pev
        #if self.currentItem is not None:
            #self.currentItem.mousePressEvent(self.pev)
        ##self.clearMouse()
        ##self.mouseTrail.append(Point(self.mapToScene(ev.pos())))
        #self.emit(QtCore.SIGNAL("mousePressed(PyQt_PyObject)"), self.mouseTrail)
                
    def mouseReleaseEvent(self, ev):
        QtGui.QGraphicsView.mouseReleaseEvent(self, ev)
        if not self.mouseEnabled:
            return 
        #self.mouseTrail.append(Point(self.mapToScene(ev.pos())))
        #self.emit(QtCore.SIGNAL("mouseReleased"), ev)
        self.sigMouseReleased.emit(ev)
        self.lastButtonReleased = ev.button()
        return   ## Everything below disabled for now..
        
        ##self.mouseTrail.append(Point(self.mapToScene(ev.pos())))
        #self.emit(QtCore.SIGNAL("mouseReleased(PyQt_PyObject)"), self.mouseTrail)
        #if self.currentItem is not None:
            #pev = self.graphicsSceneEvent(ev, self.pev, self.fev)
            #self.pev = pev
            #self.currentItem.mouseReleaseEvent(pev)
            #self.currentItem = None

    def mouseMoveEvent(self, ev):
        if self.lastMousePos is None:
            self.lastMousePos = Point(ev.pos())
        delta = Point(ev.pos() - self.lastMousePos)
        self.lastMousePos = Point(ev.pos())

        QtGui.QGraphicsView.mouseMoveEvent(self, ev)
        if not self.mouseEnabled:
            return
        #self.emit(QtCore.SIGNAL("sceneMouseMoved(PyQt_PyObject)"), self.mapToScene(ev.pos()))
        self.sigSceneMouseMoved.emit(self.mapToScene(ev.pos()))
        #print "moved. Grabber:", self.scene().mouseGrabberItem()
        
            
        if self.clickAccepted:  ## Ignore event if an item in the scene has already claimed it.
            return
        
        if ev.buttons() == QtCore.Qt.RightButton:
            delta = Point(np.clip(delta[0], -50, 50), np.clip(-delta[1], -50, 50))
            scale = 1.01 ** delta
            #if self.yInverted:
                #scale[0] = 1. / scale[0]
            self.scale(scale[0], scale[1], center=self.mapToScene(self.mousePressPos))
            #self.emit(QtCore.SIGNAL('regionChanged(QRectF)'), self.range)
            self.sigRangeChanged.emit(self, self.range)

        elif ev.buttons() in [QtCore.Qt.MidButton, QtCore.Qt.LeftButton]:  ## Allow panning by left or mid button.
            px = self.pixelSize()
            tr = -delta * px
            
            self.translate(tr[0], tr[1])
            #self.emit(QtCore.SIGNAL('regionChanged(QRectF)'), self.range)
            self.sigRangeChanged.emit(self, self.range)
        
        #return   ## Everything below disabled for now..
        
        ##self.mouseTrail.append(Point(self.mapToScene(ev.pos())))
        #if self.currentItem is not None:
            #pev = self.graphicsSceneEvent(ev, self.pev, self.fev)
            #self.pev = pev
            #self.currentItem.mouseMoveEvent(pev)
        
        
    def pixelSize(self):
        """Return vector with the length and width of one view pixel in scene coordinates"""
        p0 = Point(0,0)
        p1 = Point(1,1)
        tr = self.transform().inverted()[0]
        p01 = tr.map(p0)
        p11 = tr.map(p1)
        return Point(p11 - p01)
        
        
    def writeSvg(self, fileName=None):
        if fileName is None:
            self.fileDialog = FileDialog()
            self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
            self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
            if GraphicsView.lastFileDir is not None:
                self.fileDialog.setDirectory(GraphicsView.lastFileDir)
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.writeSvg)
            return
        fileName = str(fileName)
        GraphicsView.lastFileDir = os.path.split(fileName)[0]
        self.svg = QtSvg.QSvgGenerator()
        self.svg.setFileName(fileName)
        self.svg.setSize(self.size())
        self.svg.setResolution(600)
        painter = QtGui.QPainter(self.svg)
        self.render(painter)
        
    def writeImage(self, fileName=None):
        if fileName is None:
            self.fileDialog = FileDialog()
            self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
            self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave) ## this is the line that makes the fileDialog not show on mac
            if GraphicsView.lastFileDir is not None:
                self.fileDialog.setDirectory(GraphicsView.lastFileDir)
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.writeImage)
            return
        fileName = str(fileName)
        GraphicsView.lastFileDir = os.path.split(fileName)[0]
        self.png = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(self.png)
        rh = self.renderHints()
        self.setRenderHints(QtGui.QPainter.Antialiasing)
        self.render(painter)
        self.setRenderHints(rh)
        self.png.save(fileName)
        
    def writePs(self, fileName=None):
        if fileName is None:
            self.fileDialog = FileDialog()
            self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
            self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave) 
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.writePs)
            return
        #if fileName is None:
        #    fileName = str(QtGui.QFileDialog.getSaveFileName())
        printer = QtGui.QPrinter(QtGui.QPrinter.HighResolution)
        printer.setOutputFileName(fileName)
        painter = QtGui.QPainter(printer)
        self.render(painter)
        painter.end()
        
    def dragEnterEvent(self, ev):
        ev.ignore()  ## not sure why, but for some reason this class likes to consume drag events
        
        
        
    #def getFreehandLine(self):
        
        ## Wait for click
        #self.clearMouse()
        #while self.lastButtonReleased != QtCore.Qt.LeftButton:
            #QtGui.qApp.sendPostedEvents()
            #QtGui.qApp.processEvents()
            #time.sleep(0.01)
        #fl = vstack(self.mouseTrail)
        #return fl
    
    #def getClick(self):
        #fl = self.getFreehandLine()
        #return fl[-1]
    

