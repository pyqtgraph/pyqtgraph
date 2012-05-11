# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtGui, QtCore
from .GraphicsView import GraphicsView
from pyqtgraph.graphicsItems.GradientEditorItem import GradientEditorItem
import weakref
import numpy as np
import collections

__all__ = ['TickSlider', 'GradientWidget', 'BlackWhiteSlider']


class GradientWidget(GraphicsView):
    
    sigGradientChanged = QtCore.Signal(object)
    
    def __init__(self, parent=None, orientation='bottom',  *args, **kargs):
        GraphicsView.__init__(self, parent, useOpenGL=False, background=None)
        self.maxDim = 27
        kargs['tickPen'] = 'k'
        self.item = GradientEditorItem(*args, **kargs)
        self.item.sigGradientChanged.connect(self.sigGradientChanged)
        self.setCentralItem(self.item)
        self.setOrientation(orientation)
        self.setCacheMode(self.CacheNone)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setFrameStyle(QtGui.QFrame.NoFrame | QtGui.QFrame.Plain)
        self.setBackgroundRole(QtGui.QPalette.NoRole)
        #self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        #self.setAutoFillBackground(False)
        #self.setAttribute(QtCore.Qt.WA_PaintOnScreen, False)
        #self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def setOrientation(self, ort):
        self.item.setOrientation(ort)
        self.orientation = ort
        self.setMaxDim()
        
    def setMaxDim(self, mx=None):
        if mx is None:
            mx = self.maxDim
        else:
            self.maxDim = mx
            
        if self.orientation in ['bottom', 'top']:
            self.setFixedHeight(mx)
            self.setMaximumWidth(16777215)
        else:
            self.setFixedWidth(mx)
            self.setMaximumHeight(16777215)
        
    def __getattr__(self, attr):
        ### wrap methods from GradientEditorItem
        return getattr(self.item, attr)



#Gradients = collections.OrderedDict([
    #('thermal', {'ticks': [(0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)), (1, (255, 255, 255, 255)), (0, (0, 0, 0, 255))], 'mode': 'rgb'}),
    #('flame', {'ticks': [(0.2, (7, 0, 220, 255)), (0.5, (236, 0, 134, 255)), (0.8, (246, 246, 0, 255)), (1.0, (255, 255, 255, 255)), (0.0, (0, 0, 0, 255))], 'mode': 'rgb'}),
    #('yellowy', {'ticks': [(0.0, (0, 0, 0, 255)), (0.2328863796753704, (32, 0, 129, 255)), (0.8362738179251941, (255, 255, 0, 255)), (0.5257586450247, (115, 15, 255, 255)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'} ),
    #('bipolar', {'ticks': [(0.0, (0, 255, 255, 255)), (1.0, (255, 255, 0, 255)), (0.5, (0, 0, 0, 255)), (0.25, (0, 0, 255, 255)), (0.75, (255, 0, 0, 255))], 'mode': 'rgb'}),
    #('spectrum', {'ticks': [(1.0, (255, 0, 255, 255)), (0.0, (255, 0, 0, 255))], 'mode': 'hsv'}),
    #('cyclic', {'ticks': [(0.0, (255, 0, 4, 255)), (1.0, (255, 0, 0, 255))], 'mode': 'hsv'}),
    #('greyclip', {'ticks': [(0.0, (0, 0, 0, 255)), (0.99, (255, 255, 255, 255)), (1.0, (255, 0, 0, 255))], 'mode': 'rgb'}),
#])


#class TickSlider(GraphicsView):
    #def __init__(self, parent=None, orientation='bottom', allowAdd=True, **kargs):
        #self.orientation = orientation
        #self.length = 100
        #self.tickSize = 15
        #self.ticks = {}
        #self.maxDim = 20
        #GraphicsView.__init__(self, parent, useOpenGL=False)
        #self.allowAdd = allowAdd
        ##self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        ##self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        ##self.setTransformationAnchor(QtGui.QGraphicsView.NoAnchor)
        ##self.setResizeAnchor(QtGui.QGraphicsView.AnchorViewCenter)
        #self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        #self.orientations = {
            #'left': (270, 1, -1), 
            #'right': (270, 1, 1), 
            #'top': (0, 1, -1), 
            #'bottom': (0, 1, 1)
        #}
            
        ##self.scene = QtGui.QGraphicsScene()
        ##self.setScene(self.scene)
        
        #self.setOrientation(orientation)
        #self.setFrameStyle(QtGui.QFrame.NoFrame | QtGui.QFrame.Plain)
        #self.setBackgroundRole(QtGui.QPalette.NoRole)
        #self.setMouseTracking(True)
        
        
    #def keyPressEvent(self, ev):
        #ev.ignore()
     
    #def setMaxDim(self, mx=None):
        #if mx is None:
            #mx = self.maxDim
        #else:
            #self.maxDim = mx
            
        #if self.orientation in ['bottom', 'top']:
            #self.setFixedHeight(mx)
            #self.setMaximumWidth(16777215)
        #else:
            #self.setFixedWidth(mx)
            #self.setMaximumHeight(16777215)
        
    #def setOrientation(self, ort):
        #self.orientation = ort
        #self.resetTransform()
        #self.rotate(self.orientations[ort][0])
        #self.scale(*self.orientations[ort][1:])
        #self.setMaxDim()
        
    #def addTick(self, x, color=None, movable=True):
        #if color is None:
            #color = QtGui.QColor(255,255,255)
        #tick = Tick(self, [x*self.length, 0], color, movable, self.tickSize)
        #self.ticks[tick] = x
        #self.scene.addItem(tick)
        #return tick
                
    #def removeTick(self, tick):
        #del self.ticks[tick]
        #self.scene.removeItem(tick)
                
    #def tickMoved(self, tick, pos):
        ##print "tick changed"
        ### Correct position of tick if it has left bounds.
        #newX = min(max(0, pos.x()), self.length)
        #pos.setX(newX)
        #tick.setPos(pos)
        #self.ticks[tick] = float(newX) / self.length
    
    #def tickClicked(self, tick, ev):
        #if ev.button() == QtCore.Qt.RightButton:
            #self.removeTick(tick)
    
    #def widgetLength(self):
        #if self.orientation in ['bottom', 'top']:
            #return self.width()
        #else:
            #return self.height()
    
    #def resizeEvent(self, ev):
        #wlen = max(40, self.widgetLength())
        #self.setLength(wlen-self.tickSize)
        #bounds = self.scene().itemsBoundingRect()
        #bounds.setLeft(min(-self.tickSize*0.5, bounds.left()))
        #bounds.setRight(max(self.length + self.tickSize, bounds.right()))
        ##bounds.setTop(min(bounds.top(), self.tickSize))
        ##bounds.setBottom(max(0, bounds.bottom()))
        #self.setSceneRect(bounds)
        #self.fitInView(bounds, QtCore.Qt.KeepAspectRatio)
        
    #def setLength(self, newLen):
        #for t, x in self.ticks.items():
            #t.setPos(x * newLen, t.pos().y())
        #self.length = float(newLen)
        
    #def mousePressEvent(self, ev):
        #QtGui.QGraphicsView.mousePressEvent(self, ev)
        #self.ignoreRelease = False
        #for i in self.items(ev.pos()):
            #if isinstance(i, Tick):
                #self.ignoreRelease = True
                #break
        ##if len(self.items(ev.pos())) > 0:  ## Let items handle their own clicks
            ##self.ignoreRelease = True
        
    #def mouseReleaseEvent(self, ev):
        #QtGui.QGraphicsView.mouseReleaseEvent(self, ev)
        #if self.ignoreRelease:
            #return
            
        #pos = self.mapToScene(ev.pos())
            
        #if ev.button() == QtCore.Qt.LeftButton and self.allowAdd:
            #if pos.x() < 0 or pos.x() > self.length:
                #return
            #if pos.y() < 0 or pos.y() > self.tickSize:
                #return
            #pos.setX(min(max(pos.x(), 0), self.length))
            #self.addTick(pos.x()/self.length)
        #elif ev.button() == QtCore.Qt.RightButton:
            #self.showMenu(ev)
            
        
    #def showMenu(self, ev):
        #pass

    #def setTickColor(self, tick, color):
        #tick = self.getTick(tick)
        #tick.color = color
        #tick.setBrush(QtGui.QBrush(QtGui.QColor(tick.color)))

    #def setTickValue(self, tick, val):
        #tick = self.getTick(tick)
        #val = min(max(0.0, val), 1.0)
        #x = val * self.length
        #pos = tick.pos()
        #pos.setX(x)
        #tick.setPos(pos)
        #self.ticks[tick] = val
        
    #def tickValue(self, tick):
        #tick = self.getTick(tick)
        #return self.ticks[tick]
        
    #def getTick(self, tick):
        #if type(tick) is int:
            #tick = self.listTicks()[tick][0]
        #return tick

    #def mouseMoveEvent(self, ev):
        #QtGui.QGraphicsView.mouseMoveEvent(self, ev)
        ##print ev.pos(), ev.buttons()

    #def listTicks(self):
        #ticks = self.ticks.items()
        #ticks.sort(lambda a,b: cmp(a[1], b[1]))
        #return ticks


#class GradientWidget(TickSlider):
    
    #sigGradientChanged = QtCore.Signal(object)
    
    #def __init__(self, *args, **kargs):
        #self.currentTick = None
        #self.currentTickColor = None
        #self.rectSize = 15
        #self.gradRect = QtGui.QGraphicsRectItem(QtCore.QRectF(0, -self.rectSize, 100, self.rectSize))
        #self.backgroundRect = QtGui.QGraphicsRectItem(QtCore.QRectF(0, -self.rectSize, 100, self.rectSize))
        #self.backgroundRect.setBrush(QtGui.QBrush(QtCore.Qt.DiagCrossPattern))
        #self.colorMode = 'rgb'
        #TickSlider.__init__(self, *args, **kargs)
        #self.colorDialog = QtGui.QColorDialog()
        #self.colorDialog.setOption(QtGui.QColorDialog.ShowAlphaChannel, True)
        #self.colorDialog.setOption(QtGui.QColorDialog.DontUseNativeDialog, True)
        
        #self.colorDialog.currentColorChanged.connect(self.currentColorChanged)
        #self.colorDialog.rejected.connect(self.currentColorRejected)
        
        ##self.gradient = QtGui.QLinearGradient(QtCore.QPointF(0,0), QtCore.QPointF(100,0))
        #self.scene.addItem(self.backgroundRect)
        #self.scene.addItem(self.gradRect)
        
        #self.setMaxDim(self.rectSize + self.tickSize)
            
        ##self.btn = QtGui.QPushButton('RGB')
        ##self.btnProxy = self.scene.addWidget(self.btn)
        ##self.btnProxy.setFlag(self.btnProxy.ItemIgnoresTransformations)
        ##self.btnProxy.scale(0.7, 0.7)
        ##self.btnProxy.translate(-self.btnProxy.sceneBoundingRect().width()+self.tickSize/2., 0)
        ##if self.orientation == 'bottom':
            ##self.btnProxy.translate(0, -self.rectSize)
        #self.rgbAction = QtGui.QAction('RGB', self)
        #self.rgbAction.setCheckable(True)
        #self.rgbAction.triggered.connect(lambda: self.setColorMode('rgb'))
        #self.hsvAction = QtGui.QAction('HSV', self)
        #self.hsvAction.setCheckable(True)
        #self.hsvAction.triggered.connect(lambda: self.setColorMode('hsv'))
            
        #self.menu = QtGui.QMenu()
        
        ### build context menu of gradients
        #global Gradients
        #for g in Gradients:
            #px = QtGui.QPixmap(100, 15)
            #p = QtGui.QPainter(px)
            #self.restoreState(Gradients[g])
            #grad = self.getGradient()
            #brush = QtGui.QBrush(grad)
            #p.fillRect(QtCore.QRect(0, 0, 100, 15), brush)
            #p.end()
            #label = QtGui.QLabel()
            #label.setPixmap(px)
            #label.setContentsMargins(1, 1, 1, 1)
            #act = QtGui.QWidgetAction(self)
            #act.setDefaultWidget(label)
            #act.triggered.connect(self.contextMenuClicked)
            #act.name = g
            #self.menu.addAction(act)
            
        #self.menu.addSeparator()
        #self.menu.addAction(self.rgbAction)
        #self.menu.addAction(self.hsvAction)
        
        
        #for t in self.ticks.keys():
            #self.removeTick(t)
        #self.addTick(0, QtGui.QColor(0,0,0), True)
        #self.addTick(1, QtGui.QColor(255,0,0), True)
        #self.setColorMode('rgb')
        #self.updateGradient()
    
    #def showMenu(self, ev):
        #self.menu.popup(ev.globalPos())
    
    #def contextMenuClicked(self, b):
        #global Gradients
        #act = self.sender()
        #self.restoreState(Gradients[act.name])
    
    #def setColorMode(self, cm):
        #if cm not in ['rgb', 'hsv']:
            #raise Exception("Unknown color mode %s. Options are 'rgb' and 'hsv'." % str(cm))
        
        #try:
            #self.rgbAction.blockSignals(True)
            #self.hsvAction.blockSignals(True)
            #self.rgbAction.setChecked(cm == 'rgb')
            #self.hsvAction.setChecked(cm == 'hsv')
        #finally:
            #self.rgbAction.blockSignals(False)
            #self.hsvAction.blockSignals(False)
        #self.colorMode = cm
        #self.updateGradient()
        
    #def updateGradient(self):
        #self.gradient = self.getGradient()
        #self.gradRect.setBrush(QtGui.QBrush(self.gradient))
        #self.sigGradientChanged.emit(self)
        
    #def setLength(self, newLen):
        #TickSlider.setLength(self, newLen)
        #self.backgroundRect.setRect(0, -self.rectSize, newLen, self.rectSize)
        #self.gradRect.setRect(0, -self.rectSize, newLen, self.rectSize)
        #self.updateGradient()
        
    #def currentColorChanged(self, color):
        #if color.isValid() and self.currentTick is not None:
            #self.setTickColor(self.currentTick, color)
            #self.updateGradient()
            
    #def currentColorRejected(self):
        #self.setTickColor(self.currentTick, self.currentTickColor)
        #self.updateGradient()
        
    #def tickClicked(self, tick, ev):
        #if ev.button() == QtCore.Qt.LeftButton:
            #if not tick.colorChangeAllowed:
                #return
            #self.currentTick = tick
            #self.currentTickColor = tick.color
            #self.colorDialog.setCurrentColor(tick.color)
            #self.colorDialog.open()
            ##color = QtGui.QColorDialog.getColor(tick.color, self, "Select Color", QtGui.QColorDialog.ShowAlphaChannel)
            ##if color.isValid():
                ##self.setTickColor(tick, color)
                ##self.updateGradient()
        #elif ev.button() == QtCore.Qt.RightButton:
            #if not tick.removeAllowed:
                #return
            #if len(self.ticks) > 2:
                #self.removeTick(tick)
                #self.updateGradient()
                
    #def tickMoved(self, tick, pos):
        #TickSlider.tickMoved(self, tick, pos)
        #self.updateGradient()


    #def getGradient(self):
        #g = QtGui.QLinearGradient(QtCore.QPointF(0,0), QtCore.QPointF(self.length,0))
        #if self.colorMode == 'rgb':
            #ticks = self.listTicks()
            #g.setStops([(x, QtGui.QColor(t.color)) for t,x in ticks])
        #elif self.colorMode == 'hsv':  ## HSV mode is approximated for display by interpolating 10 points between each stop
            #ticks = self.listTicks()
            #stops = []
            #stops.append((ticks[0][1], ticks[0][0].color))
            #for i in range(1,len(ticks)):
                #x1 = ticks[i-1][1]
                #x2 = ticks[i][1]
                #dx = (x2-x1) / 10.
                #for j in range(1,10):
                    #x = x1 + dx*j
                    #stops.append((x, self.getColor(x)))
                #stops.append((x2, self.getColor(x2)))
            #g.setStops(stops)
        #return g
        
    #def getColor(self, x, toQColor=True):
        #ticks = self.listTicks()
        #if x <= ticks[0][1]:
            #c = ticks[0][0].color
            #if toQColor:
                #return QtGui.QColor(c)  # always copy colors before handing them out
            #else:
                #return (c.red(), c.green(), c.blue(), c.alpha())
        #if x >= ticks[-1][1]:
            #c = ticks[-1][0].color
            #if toQColor:
                #return QtGui.QColor(c)  # always copy colors before handing them out
            #else:
                #return (c.red(), c.green(), c.blue(), c.alpha())
            
        #x2 = ticks[0][1]
        #for i in range(1,len(ticks)):
            #x1 = x2
            #x2 = ticks[i][1]
            #if x1 <= x and x2 >= x:
                #break
                
        #dx = (x2-x1)
        #if dx == 0:
            #f = 0.
        #else:
            #f = (x-x1) / dx
        #c1 = ticks[i-1][0].color
        #c2 = ticks[i][0].color
        #if self.colorMode == 'rgb':
            #r = c1.red() * (1.-f) + c2.red() * f
            #g = c1.green() * (1.-f) + c2.green() * f
            #b = c1.blue() * (1.-f) + c2.blue() * f
            #a = c1.alpha() * (1.-f) + c2.alpha() * f
            #if toQColor:
                #return QtGui.QColor(r, g, b,a)
            #else:
                #return (r,g,b,a)
        #elif self.colorMode == 'hsv':
            #h1,s1,v1,_ = c1.getHsv()
            #h2,s2,v2,_ = c2.getHsv()
            #h = h1 * (1.-f) + h2 * f
            #s = s1 * (1.-f) + s2 * f
            #v = v1 * (1.-f) + v2 * f
            #c = QtGui.QColor()
            #c.setHsv(h,s,v)
            #if toQColor:
                #return c
            #else:
                #return (c.red(), c.green(), c.blue(), c.alpha())
                    
    #def getLookupTable(self, nPts, alpha=True):
        #"""Return an RGB/A lookup table."""
        #if alpha:
            #table = np.empty((nPts,4), dtype=np.ubyte)
        #else:
            #table = np.empty((nPts,3), dtype=np.ubyte)
            
        #for i in range(nPts):
            #x = float(i)/(nPts-1)
            #color = self.getColor(x, toQColor=False)
            #table[i] = color[:table.shape[1]]
            
        #return table
            
            

    #def mouseReleaseEvent(self, ev):
        #TickSlider.mouseReleaseEvent(self, ev)
        #self.updateGradient()
        
    #def addTick(self, x, color=None, movable=True):
        #if color is None:
            #color = self.getColor(x)
        #t = TickSlider.addTick(self, x, color=color, movable=movable)
        #t.colorChangeAllowed = True
        #t.removeAllowed = True
        #return t
        
    #def saveState(self):
        #ticks = []
        #for t in self.ticks:
            #c = t.color
            #ticks.append((self.ticks[t], (c.red(), c.green(), c.blue(), c.alpha())))
        #state = {'mode': self.colorMode, 'ticks': ticks}
        #return state
        
    #def restoreState(self, state):
        #self.setColorMode(state['mode'])
        #for t in self.ticks.keys():
            #self.removeTick(t)
        #for t in state['ticks']:
            #c = QtGui.QColor(*t[1])
            #self.addTick(t[0], c)
        #self.updateGradient()



#class BlackWhiteSlider(GradientWidget):
    #def __init__(self, parent):
        #GradientWidget.__init__(self, parent)
        #self.getTick(0).colorChangeAllowed = False
        #self.getTick(1).colorChangeAllowed = False
        #self.allowAdd = False
        #self.setTickColor(self.getTick(1), QtGui.QColor(255,255,255))
        #self.setOrientation('right')

    #def getLevels(self):
        #return (self.tickValue(0), self.tickValue(1))

    #def setLevels(self, black, white):
        #self.setTickValue(0, black)
        #self.setTickValue(1, white)




#class GammaWidget(TickSlider):
    #pass
    
        
#class Tick(QtGui.QGraphicsPolygonItem):
    #def __init__(self, view, pos, color, movable=True, scale=10):
        ##QObjectWorkaround.__init__(self)
        #self.movable = movable
        #self.view = weakref.ref(view)
        #self.scale = scale
        #self.color = color
        ##self.endTick = endTick
        #self.pg = QtGui.QPolygonF([QtCore.QPointF(0,0), QtCore.QPointF(-scale/3**0.5,scale), QtCore.QPointF(scale/3**0.5,scale)])
        #QtGui.QGraphicsPolygonItem.__init__(self, self.pg)
        #self.setPos(pos[0], pos[1])
        #self.setFlags(QtGui.QGraphicsItem.ItemIsMovable | QtGui.QGraphicsItem.ItemIsSelectable)
        #self.setBrush(QtGui.QBrush(QtGui.QColor(self.color)))
        #if self.movable:
            #self.setZValue(1)
        #else:
            #self.setZValue(0)

    ##def x(self):
        ##return self.pos().x()/100.

    #def mouseMoveEvent(self, ev):
        ##print self, "move", ev.scenePos()
        #if not self.movable:
            #return
        #if not ev.buttons() & QtCore.Qt.LeftButton:
            #return
            
            
        #newPos = ev.scenePos() + self.mouseOffset
        #newPos.setY(self.pos().y())
        ##newPos.setX(min(max(newPos.x(), 0), 100))
        #self.setPos(newPos)
        #self.view().tickMoved(self, newPos)
        #self.movedSincePress = True
        ##self.emit(QtCore.SIGNAL('tickChanged'), self)
        #ev.accept()

    #def mousePressEvent(self, ev):
        #self.movedSincePress = False
        #if ev.button() == QtCore.Qt.LeftButton:
            #ev.accept()
            #self.mouseOffset = self.pos() - ev.scenePos()
            #self.pressPos = ev.scenePos()
        #elif ev.button() == QtCore.Qt.RightButton:
            #ev.accept()
            ##if self.endTick:
                ##return
            ##self.view.tickChanged(self, delete=True)
            
    #def mouseReleaseEvent(self, ev):
        ##print self, "release", ev.scenePos()
        #if not self.movedSincePress:
            #self.view().tickClicked(self, ev)
        
        ##if ev.button() == QtCore.Qt.LeftButton and ev.scenePos() == self.pressPos:
            ##color = QtGui.QColorDialog.getColor(self.color, None, "Select Color", QtGui.QColorDialog.ShowAlphaChannel)
            ##if color.isValid():
                ##self.color = color
                ##self.setBrush(QtGui.QBrush(QtGui.QColor(self.color)))
                ###self.emit(QtCore.SIGNAL('tickChanged'), self)
                ##self.view.tickChanged(self)
        
        
    
    
