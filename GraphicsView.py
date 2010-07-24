# -*- coding: utf-8 -*-
"""
GraphicsView.py -   Extension of QGraphicsView
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""

from PyQt4 import QtCore, QtGui, QtOpenGL, QtSvg
#from numpy import vstack
#import time
from Point import *
#from vector import *
import sys
            
        
class GraphicsView(QtGui.QGraphicsView):
    def __init__(self, parent=None, useOpenGL=True):
        """Re-implementation of QGraphicsView that removes scrollbars and allows unambiguous control of the 
        viewed coordinate range. Also automatically creates a QGraphicsScene and a central QGraphicsWidget
        that is automatically scaled to the full view geometry.
        
        By default, the view coordinate system matches the widget's pixel coordinates and 
        automatically updates when the view is resized. This can be overridden by setting 
        autoPixelRange=False. The exact visible range can be set with setRange().
        
        The view can be panned using the middle mouse button and scaled using the right mouse button if
        enabled via enableMouse()."""
        
        QtGui.QGraphicsView.__init__(self, parent)
        self.useOpenGL(useOpenGL)
        
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0,0,0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active,QtGui.QPalette.Base,brush)
        brush = QtGui.QBrush(QtGui.QColor(0,0,0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive,QtGui.QPalette.Base,brush)
        brush = QtGui.QBrush(QtGui.QColor(244,244,244))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled,QtGui.QPalette.Base,brush)
        self.setPalette(palette)
        self.setProperty("cursor",QtCore.QVariant(QtCore.Qt.ArrowCursor))
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFrameShape(QtGui.QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QtGui.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtGui.QGraphicsView.AnchorViewCenter)
        #self.setResizeAnchor(QtGui.QGraphicsView.NoAnchor)
        self.setViewportUpdateMode(QtGui.QGraphicsView.SmartViewportUpdate)
        self.setSceneRect(QtCore.QRectF(-1e10, -1e10, 2e10, 2e10))
        #self.setSceneRect(1, 1, 0, 0) ## Set an empty (but non-zero) scene rect so that the view doesn't try to automatically update for us.
        #self.setInteractive(False)
        self.lockedViewports = []
        self.lastMousePos = None
        #self.setMouseTracking(False)
        self.aspectLocked = False
        self.yInverted = True
        self.range = QtCore.QRectF(0, 0, 1, 1)
        self.autoPixelRange = True
        self.currentItem = None
        self.clearMouse()
        self.updateMatrix()
        self.sceneObj = QtGui.QGraphicsScene()
        self.setScene(self.sceneObj)
        self.centralWidget = None
        self.setCentralItem(QtGui.QGraphicsWidget())
        self.mouseEnabled = False
        self.scaleCenter = False  ## should scaling center around view center (True) or mouse click (False)
        self.clickAccepted = False
        
    def useOpenGL(self, b=True):
        if b:
            v = QtOpenGL.QGLWidget()
        else:
            v = QtGui.QWidget()
            
        #v.setStyleSheet("background-color: #000000;")
        self.setViewport(v)
            
    def keyPressEvent(self, ev):
        ev.ignore()
        
    def setCentralItem(self, item):
        if self.centralWidget is not None:
            self.scene().removeItem(self.centralWidget)
        self.centralWidget = item
        self.sceneObj.addItem(item)
        
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
        if self.autoPixelRange:
            self.range = QtCore.QRectF(0, 0, self.size().width(), self.size().height())
        self.setRange(self.range, padding=0, disableAutoPixel=False)
        self.updateMatrix()
    
    def updateMatrix(self, propagate=True):
        #print "udpateMatrix:"
        translate = Point(self.range.center())
        if self.range.width() == 0 or self.range.height() == 0:
            return
        scale = Point(self.size().width()/self.range.width(), self.size().height()/self.range.height())
        
        m = QtGui.QMatrix()
        
        ## First center the viewport at 0
        self.resetMatrix()
        center = self.viewportTransform().inverted()[0].map(Point(self.width()/2., self.height()/2.))
        if self.yInverted:
            m.translate(center.x(), center.y())
            #print "  inverted; translate", center.x(), center.y()
        else:
            m.translate(center.x(), -center.y())
            #print "  not inverted; translate", center.x(), -center.y()
            
        ## Now scale and translate properly
        if self.aspectLocked:
            scale = Point(scale.min())
        if not self.yInverted:
            scale = scale * Point(1, -1)
        m.scale(scale[0], scale[1])
        #print "  scale:", scale
        st = translate
        m.translate(-st[0], -st[1])
        #print "  translate:", st
        self.setMatrix(m)
        self.currentScale = scale
        
        if propagate:
            for v in self.lockedViewports:
                v.setXRange(self.range, padding=0)
        
    def visibleRange(self):
        """Return the boundaries of the view in scene coordinates"""
        ## easier to just return self.range ?
        r = QtCore.QRectF(self.rect())
        return self.viewportTransform().inverted()[0].mapRect(r)

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
        self.range = newRect.adjusted(-pw, -ph, pw, ph)
        #print "New Range:", self.range
        self.centralWidget.setGeometry(self.range)
        self.updateMatrix(propagate)
        self.emit(QtCore.SIGNAL('viewChanged'), self.range)
        
        
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
        
    def invertY(self, invert=True):
        #if self.yInverted != invert:
            #self.scale[1] *= -1.
        self.yInverted = invert
        self.updateMatrix()
    
    
    def wheelEvent(self, ev):
        if not self.mouseEnabled:
            return
        QtGui.QGraphicsView.wheelEvent(self, ev)
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
        self.emit(QtCore.SIGNAL("mouseReleased"), ev)
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
        QtGui.QGraphicsView.mouseMoveEvent(self, ev)
        if not self.mouseEnabled:
            return
        self.emit(QtCore.SIGNAL("sceneMouseMoved(PyQt_PyObject)"), self.mapToScene(ev.pos()))
        #print "moved. Grabber:", self.scene().mouseGrabberItem()
        
        if self.lastMousePos is None:
            self.lastMousePos = Point(ev.pos())
            
        if self.clickAccepted:  ## Ignore event if an item in the scene has already claimed it.
            return
            
        delta = Point(ev.pos()) - self.lastMousePos
        
        self.lastMousePos = Point(ev.pos())
        
        if ev.buttons() == QtCore.Qt.RightButton:
            delta = Point(clip(delta[0], -50, 50), clip(-delta[1], -50, 50))
            scale = 1.01 ** delta
            #if self.yInverted:
                #scale[0] = 1. / scale[0]
            self.scale(scale[0], scale[1], center=self.mapToScene(self.mousePressPos))
            self.emit(QtCore.SIGNAL('regionChanged(QRectF)'), self.range)


        elif ev.buttons() in [QtCore.Qt.MidButton, QtCore.Qt.LeftButton]:  ## Allow panning by left or mid button.
            tr = -delta / self.currentScale
            
            self.translate(tr[0], tr[1])
            self.emit(QtCore.SIGNAL('regionChanged(QRectF)'), self.range)
        
        #return   ## Everything below disabled for now..
        
        ##self.mouseTrail.append(Point(self.mapToScene(ev.pos())))
        #if self.currentItem is not None:
            #pev = self.graphicsSceneEvent(ev, self.pev, self.fev)
            #self.pev = pev
            #self.currentItem.mouseMoveEvent(pev)
        
        
    
        
    def writeSvg(self, fileName=None):
        if fileName is None:
            fileName = str(QtGui.QFileDialog.getSaveFileName())
        self.svg = QtSvg.QSvgGenerator()
        self.svg.setFileName(fileName)
        self.svg.setSize(self.size())
        self.svg.setResolution(600)
        painter = QtGui.QPainter(self.svg)
        self.render(painter)
        
    def writeImage(self, fileName=None):
        if fileName is None:
            fileName = str(QtGui.QFileDialog.getSaveFileName())
        self.png = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(self.png)
        rh = self.renderHints()
        self.setRenderHints(QtGui.QPainter.Antialiasing)
        self.render(painter)
        self.setRenderHints(rh)
        self.png.save(fileName)
        
    def writePs(self, fileName=None):
        if fileName is None:
            fileName = str(QtGui.QFileDialog.getSaveFileName())
        printer = QtGui.QPrinter(QtGui.QPrinter.HighResolution)
        printer.setOutputFileName(fileName)
        painter = QtGui.QPainter(printer)
        self.render(painter)
        painter.end()
        
        
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
    

#class GraphicsSceneMouseEvent(QtGui.QGraphicsSceneMouseEvent):
    #"""Stand-in class for QGraphicsSceneMouseEvent"""
    #def __init__(self):
        #QtGui.QGraphicsSceneMouseEvent.__init__(self)
            
    #def setPos(self, p):
        #self.vpos = p
    #def setButtons(self, p):
        #self.vbuttons = p
    #def setButton(self, p):
        #self.vbutton = p
    #def setModifiers(self, p):
        #self.vmodifiers = p
    #def setScenePos(self, p):
        #self.vscenePos = p
    #def setLastPos(self, p):
        #self.vlastPos = p
    #def setLastScenePos(self, p):
        #self.vlastScenePos = p
    #def setLastScreenPos(self, p):
        #self.vlastScreenPos = p
    #def setButtonDownPos(self, p):
        #self.vbuttonDownPos = p
    #def setButtonDownScenePos(self, p):
        #self.vbuttonDownScenePos = p
    #def setButtonDownScreenPos(self, p):
        #self.vbuttonDownScreenPos = p
    
    #def pos(self):
        #return self.vpos
    #def buttons(self):
        #return self.vbuttons
    #def button(self):
        #return self.vbutton
    #def modifiers(self):
        #return self.vmodifiers
    #def scenePos(self):
        #return self.vscenePos
    #def lastPos(self):
        #return self.vlastPos
    #def lastScenePos(self):
        #return self.vlastScenePos
    #def lastScreenPos(self):
        #return self.vlastScreenPos
    #def buttonDownPos(self):
        #return self.vbuttonDownPos
    #def buttonDownScenePos(self):
        #return self.vbuttonDownScenePos
    #def buttonDownScreenPos(self):
        #return self.vbuttonDownScreenPos
    
