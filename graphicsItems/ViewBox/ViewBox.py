from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from pyqtgraph.Point import Point
import pyqtgraph.functions as fn
from .. ItemGroup import ItemGroup
from .. GraphicsWidget import GraphicsWidget
from pyqtgraph.GraphicsScene import GraphicsScene
import pyqtgraph
import weakref
from copy import deepcopy
import collections

__all__ = ['ViewBox']


class ChildGroup(ItemGroup):
    
    sigItemsChanged = QtCore.Signal()
    
    def itemChange(self, change, value):
        ret = ItemGroup.itemChange(self, change, value)
        if change == self.ItemChildAddedChange or change == self.ItemChildRemovedChange:
            self.sigItemsChanged.emit()
        
        return ret


class ViewBox(GraphicsWidget):
    """
    Box that allows internal scaling/panning of children by mouse drag. 
    Not really compatible with GraphicsView having the same functionality.
    """
    
    sigYRangeChanged = QtCore.Signal(object, object)
    sigXRangeChanged = QtCore.Signal(object, object)
    sigRangeChangedManually = QtCore.Signal(object)
    sigRangeChanged = QtCore.Signal(object, object)
    #sigActionPositionChanged = QtCore.Signal(object)
    sigStateChanged = QtCore.Signal(object)
    
    ## mouse modes
    PanMode = 3
    RectMode = 1
    
    ## axes
    XAxis = 0
    YAxis = 1
    XYAxes = 2
    
    ## for linking views together
    NamedViews = weakref.WeakValueDictionary()   # name: ViewBox
    AllViews = weakref.WeakKeyDictionary()       # ViewBox: None
    
    
    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True, invertY=False, name=None):
        GraphicsWidget.__init__(self, parent)
        self.name = None
        self.linksBlocked = False
        self.addedItems = []
        #self.gView = view
        #self.showGrid = showGrid
        
        self.state = {
            
            ## separating targetRange and viewRange allows the view to be resized
            ## while keeping all previously viewed contents visible
            'targetRange': [[0,1], [0,1]],   ## child coord. range visible [[xmin, xmax], [ymin, ymax]]
            'viewRange': [[0,1], [0,1]],     ## actual range viewed
        
            'yInverted': invertY,
            'aspectLocked': False,    ## False if aspect is unlocked, otherwise float specifies the locked ratio.
            'autoRange': [True, True],  ## False if auto range is disabled, 
                                          ## otherwise float gives the fraction of data that is visible
            'linkedViews': [None, None],
            
            'mouseEnabled': [enableMouse, enableMouse],
            'mouseMode': ViewBox.PanMode if pyqtgraph.getConfigOption('leftButtonPan') else ViewBox.RectMode,  
            'wheelScaleFactor': -1.0 / 8.0,
        }
        
        
        self.exportMethods = collections.OrderedDict([
            ('SVG', self.saveSvg),
            ('Image', self.saveImage),
            ('Print', self.savePrint),
        ])
        
        self.setFlag(self.ItemClipsChildrenToShape)
        self.setFlag(self.ItemIsFocusable, True)  ## so we can receive key presses
        
        ## childGroup is required so that ViewBox has local coordinates similar to device coordinates.
        ## this is a workaround for a Qt + OpenGL but that causes improper clipping
        ## https://bugreports.qt.nokia.com/browse/QTBUG-23723
        self.childGroup = ChildGroup(self)
        self.childGroup.sigItemsChanged.connect(self.itemsChanged)
        
        #self.useLeftButtonPan = pyqtgraph.getConfigOption('leftButtonPan') # normally use left button to pan
        # this also enables capture of keyPressEvents.
        
        ## Make scale box that is shown when dragging on the view
        self.rbScaleBox = QtGui.QGraphicsRectItem(0, 0, 1, 1)
        self.rbScaleBox.setPen(fn.mkPen((255,0,0), width=1))
        self.rbScaleBox.setBrush(fn.mkBrush(255,255,0,100))
        self.addItem(self.rbScaleBox)
        self.rbScaleBox.hide()
        
        self.axHistory = [] # maintain a history of zoom locations
        self.axHistoryPointer = -1 # pointer into the history. Allows forward/backward movement, not just "undo"
        
        self.setZValue(-100)
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))
        
        self.setAspectLocked(lockAspect)
        
        self.border = border
        self.menu = ViewBoxMenu(self)
        
        self.register(name)
        if name is None:
            self.updateViewLists()
        
    def register(self, name):
        """
        Add this ViewBox to the registered list of views. 
        *name* will appear in the drop-down lists for axis linking in all other views.
        The same can be accomplished by initializing the ViewBox with the *name* attribute.
        """
        ViewBox.AllViews[self] = None
        if self.name is not None:
            del ViewBox.NamedViews[self.name]
        self.name = name
        if name is not None:
            ViewBox.NamedViews[name] = self
            ViewBox.updateAllViewLists()

    def unregister(self):
        del ViewBox.AllViews[self]
        if self.name is not None:
            del ViewBox.NamedViews[self.name]

    def close(self):
        self.unregister()

    def implements(self, interface):
        return interface == 'ViewBox'
        
        
    def getState(self, copy=True):
        state = self.state.copy()
        state['linkedViews'] = [(None if v is None else v.name) for v in state['linkedViews']]
        if copy:
            return deepcopy(self.state)
        else:
            return self.state
        
    def setState(self, state):
        state = state.copy()
        self.setXLink(state['linkedViews'][0])
        self.setYLink(state['linkedViews'][1])
        del state['linkedViews']
        
        self.state.update(state)
        self.updateMatrix()
        self.sigStateChanged.emit(self)


    def setMouseMode(self, mode):
        if mode not in [ViewBox.PanMode, ViewBox.RectMode]:
            raise Exception("Mode must be ViewBox.PanMode or ViewBox.RectMode")
        self.state['mouseMode'] = mode
        self.sigStateChanged.emit(self)

    #def toggleLeftAction(self, act):  ## for backward compatibility
        #if act.text() is 'pan':
            #self.setLeftButtonAction('pan')
        #elif act.text() is 'zoom':
            #self.setLeftButtonAction('rect')

    def setLeftButtonAction(self, mode='rect'):  ## for backward compatibility
        if mode.lower() == 'rect':
            self.setMouseMode(ViewBox.RectMode)
        elif mode.lower() == 'pan':
            self.setMouseMode(ViewBox.PanMode)
        else:
            raise Exception('graphicsItems:ViewBox:setLeftButtonAction: unknown mode = %s (Options are "pan" and "rect")' % mode)
            
    def innerSceneItem(self):
        return self.childGroup
    
    def setMouseEnabled(self, x=None, y=None):
        if x is not None:
            self.state['mouseEnabled'][0] = x
        if y is not None:
            self.state['mouseEnabled'][1] = y
        self.sigStateChanged.emit(self)
            
    def mouseEnabled(self):
        return self.state['mouseEnabled'][:]
    
    def addItem(self, item):
        if item.zValue() < self.zValue():
            item.setZValue(self.zValue()+1)
        item.setParentItem(self.childGroup)
        self.addedItems.append(item)
        self.updateAutoRange()
        #print "addItem:", item, item.boundingRect()
        
    def removeItem(self, item):
        try:
            self.addedItems.remove(item)
        except:
            pass
        self.scene().removeItem(item)
        self.updateAutoRange()

    def resizeEvent(self, ev):
        #self.setRange(self.range, padding=0)
        self.updateAutoRange()
        self.updateMatrix()
        self.sigStateChanged.emit(self)
        
    def viewRange(self):
        return [x[:] for x in self.state['viewRange']]  ## return copy

    def viewRect(self):
        """Return a QRectF bounding the region visible within the ViewBox"""
        try:
            vr0 = self.state['viewRange'][0]
            vr1 = self.state['viewRange'][1]
            return QtCore.QRectF(vr0[0], vr1[0], vr0[1]-vr0[0], vr1[1] - vr1[0])
        except:
            print "make qrectf failed:", self.state['viewRange']
            raise
    
    #def viewportTransform(self):
        ##return self.itemTransform(self.childGroup)[0]
        #return self.childGroup.itemTransform(self)[0]
    
    def targetRange(self):
        return [x[:] for x in self.state['targetRange']]  ## return copy
    
    def targetRect(self):  
        """
        Return the region which has been requested to be visible. 
        (this is not necessarily the same as the region that is *actually* visible--
        resizing and aspect ratio constraints can cause targetRect() and viewRect() to differ)
        """
        try:
            tr0 = self.state['targetRange'][0]
            tr1 = self.state['targetRange'][1]
            return QtCore.QRectF(tr0[0], tr1[0], tr0[1]-tr0[0], tr1[1] - tr1[0])
        except:
            print "make qrectf failed:", self.state['targetRange']
            raise

    def setRange(self, rect=None, xRange=None, yRange=None, padding=0.02, update=True, disableAutoRange=True):
        """
        Set the visible range of the ViewBox.
        Must specify at least one of *range*, *xRange*, or *yRange*. 
        
        Arguments:
            *rect* (QRectF)    - The full range that should be visible in the view box.
            *xRange* (min,max) - The range that should be visible along the x-axis.
            *yRange* (min,max) - The range that should be visible along the y-axis.
            *padding* (float)  - Expand the view by a fraction of the requested range
                                 By default, this value is 0.02 (2%)
        
        """
        changes = {}
        
        if rect is not None:
            changes = {0: [rect.left(), rect.right()], 1: [rect.top(), rect.bottom()]}
        if xRange is not None:
            changes[0] = xRange
        if yRange is not None:
            changes[1] = yRange

        if len(changes) == 0:
            raise Exception("Must specify at least one of rect, xRange, or yRange.")
        
        changed = [False, False]
        for ax, range in changes.iteritems():
            mn = min(range)
            mx = max(range)
            if mn == mx:   ## If we requested 0 range, try to preserve previous scale. Otherwise just pick an arbitrary scale.
                dy = self.state['viewRange'][ax][1] - self.state['viewRange'][ax][0]
                if dy == 0:
                    dy = 1
                mn -= dy*0.5
                mx += dy*0.5
                padding = 0.0
            if any(np.isnan([mn, mx])) or any(np.isinf([mn, mx])):
                raise Exception("Not setting range [%s, %s]" % (str(mn), str(mx)))
                
            p = (mx-mn) * padding
            mn -= p
            mx += p
            
            if self.state['targetRange'][ax] != [mn, mx]:
                self.state['targetRange'][ax] = [mn, mx]
                changed[ax] = True
            
        if any(changed) and disableAutoRange:
            if all(changed):
                ax = ViewBox.XYAxes
            elif changed[0]:
                ax = ViewBox.XAxis
            elif changed[1]:
                ax = ViewBox.YAxis
            self.enableAutoRange(ax, False)
                
                
        self.sigStateChanged.emit(self)
        
        if update:
            self.updateMatrix(changed)
            
        for ax, range in changes.iteritems():
            link = self.state['linkedViews'][ax]
            if link is not None:
                link.linkedViewChanged(self, ax)
        

            
    def setYRange(self, min, max, padding=0.02, update=True):
        self.setRange(yRange=[min, max], update=update, padding=padding)
        
    def setXRange(self, min, max, padding=0.02, update=True):
        self.setRange(xRange=[min, max], update=update, padding=padding)

    def autoRange(self, padding=0.02):
        """
        Set the range of the view box to make all children visible.
        """
        bounds = self.childrenBoundingRect()
        if bounds is not None:
            self.setRange(bounds, padding=padding)
            
            
    def scaleBy(self, s, center=None):
        """
        Scale by *s* around given center point (or center of view).
        *s* may be a Point or tuple (x, y)
        """
        scale = Point(s)
        if self.state['aspectLocked'] is not False:
            scale[0] = self.state['aspectLocked'] * scale[1]

        vr = self.targetRect()
        if center is None:
            center = Point(vr.center())
        else:
            center = Point(center)
        
        tl = center + (vr.topLeft()-center) * scale
        br = center + (vr.bottomRight()-center) * scale
       
        self.setRange(QtCore.QRectF(tl, br), padding=0)
        
    def translateBy(self, t):
        """
        Translate the view by *t*, which may be a Point or tuple (x, y).
        """
        t = Point(t)
        #if viewCoords:  ## scale from pixels
            #o = self.mapToView(Point(0,0))
            #t = self.mapToView(t) - o
        
        vr = self.targetRect()
        self.setRange(vr.translated(t), padding=0)
        
    def enableAutoRange(self, axis=None, enable=True):
        """
        Enable (or disable) auto-range for *axis*, which may be ViewBox.XAxis, ViewBox.YAxis, or ViewBox.XYAxes for both.
        When enabled, the axis will automatically rescale when items are added/removed or change their shape.
        The argument *enable* may optionally be a float (0.0-1.0) which indicates the fraction of the data that should
        be visible (this only works with items implementing a dataRange method, such as PlotDataItem).
        """
        #print "autorange:", axis, enable
        #if not enable:
            #import traceback
            #traceback.print_stack()
        if enable is True:
            enable = 1.0
        
        if axis is None:
            axis = ViewBox.XYAxes
        
        if axis == ViewBox.XYAxes or axis == 'xy':
            self.state['autoRange'][0] = enable
            self.state['autoRange'][1] = enable
        elif axis == ViewBox.XAxis or axis == 'x':
            self.state['autoRange'][0] = enable
        elif axis == ViewBox.YAxis or axis == 'y':
            self.state['autoRange'][1] = enable
        else:
            raise Exception('axis argument must be ViewBox.XAxis, ViewBox.YAxis, or ViewBox.XYAxes.')
        
        if enable:
            self.updateAutoRange()
        self.sigStateChanged.emit(self)

    def disableAutoRange(self, axis=None):
        self.enableAutoRange(axis, enable=False)

    def autoRangeEnabled(self):
        return self.state['autoRange'][:]

    def updateAutoRange(self):
        tr = self.viewRect()
        if not any(self.state['autoRange']):
            return
            
        fractionVisible = self.state['autoRange'][:]
        for i in [0,1]:
            if type(fractionVisible[i]) is bool:
                fractionVisible[i] = 1.0
        cr = self.childrenBoundingRect(frac=fractionVisible)
        wp = cr.width() * 0.02
        hp = cr.height() * 0.02
        cr = cr.adjusted(-wp, -hp, wp, hp)
        
        if self.state['autoRange'][0] is not False:
            tr.setLeft(cr.left())
            tr.setRight(cr.right())
        if self.state['autoRange'][1] is not False:
            tr.setTop(cr.top())
            tr.setBottom(cr.bottom())
            
        self.setRange(tr, padding=0, disableAutoRange=False)
        
    def setXLink(self, view):
        self.linkView(self.XAxis, view)
        
    def setYLink(self, view):
        self.linkView(self.YAxis, view)
        
        
    def linkView(self, axis, view):
        """
        Link X or Y axes of two views and unlink any previously connected axes. *axis* must be ViewBox.XAxis or ViewBox.YAxis.
        If view is None, the axis is left unlinked.
        """
        if isinstance(view, basestring):
            if view == '':
                view = None
            else:
                view = ViewBox.NamedViews[view]

        if hasattr(view, 'implements') and view.implements('ViewBoxWrapper'):
            view = view.getViewBox()

        ## used to connect/disconnect signals between a pair of views
        if axis == ViewBox.XAxis:
            signal = 'sigXRangeChanged'
            slot = self.linkedXChanged
        else:
            signal = 'sigYRangeChanged'
            slot = self.linkedYChanged


        oldLink = self.state['linkedViews'][axis]
        if oldLink is not None:
            getattr(oldLink, signal).disconnect(slot)
            
        self.state['linkedViews'][axis] = view
        
        if view is not None:
            getattr(view, signal).connect(slot)
            if view.autoRangeEnabled()[axis] is True:
                self.enableAutoRange(axis, False)
                slot()
            else:
                if self.autoRangeEnabled()[axis] is False:
                    slot()
            
        self.sigStateChanged.emit(self)
        
    def blockLink(self, b):
        self.linksBlocked = b  ## prevents recursive plot-change propagation

    def linkedXChanged(self):
        view = self.state['linkedViews'][0]
        self.linkedViewChanged(view, ViewBox.XAxis)

    def linkedYChanged(self):
        view = self.state['linkedViews'][0]
        self.linkedViewChanged(view, ViewBox.YAxis)
        

    def linkedViewChanged(self, view, axis):
        if self.linksBlocked:
            return
        
        vr = view.viewRect()
        vg = view.screenGeometry()
        if vg is None:
            return
            
        sg = self.screenGeometry()
        
        view.blockLink(True)
        try:
            if axis == ViewBox.XAxis:
                upp = float(vr.width()) / vg.width()
                x1 = vr.left() + (sg.x()-vg.x()) * upp
                x2 = x1 + sg.width() * upp
                self.enableAutoRange(ViewBox.XAxis, False)
                self.setXRange(x1, x2, padding=0)
            else:
                upp = float(vr.height()) / vg.height()
                x1 = vr.bottom() + (sg.y()-vg.y()) * upp
                x2 = x1 + sg.height() * upp
                self.enableAutoRange(ViewBox.YAxis, False)
                self.setYRange(x1, x2, padding=0)
        finally:
            view.blockLink(False)
        
        
    def screenGeometry(self):
        """return the screen geometry of the viewbox"""
        v = self.getViewWidget()
        if v is None:
            return None
        b = self.sceneBoundingRect()
        wr = v.mapFromScene(b).boundingRect()
        pos = v.mapToGlobal(v.pos())
        wr.adjust(pos.x(), pos.y(), pos.x(), pos.y())
        return wr
        
    

    def itemsChanged(self):
        ## called when items are added/removed from self.childGroup
        self.updateAutoRange()
        
    def itemBoundsChanged(self, item):
        self.updateAutoRange()

    def invertY(self, b=True):
        """
        By default, the positive y-axis points upward on the screen. Use invertY(True) to reverse the y-axis.
        """
        self.state['yInverted'] = b
        self.updateMatrix()
        self.sigStateChanged.emit(self)
        
    def setAspectLocked(self, lock=True, ratio=1):
        """
        If the aspect ratio is locked, view scaling must always preserve the aspect ratio.
        By default, the ratio is set to 1; x and y both have the same scaling.
        This ratio can be overridden (width/height), or use None to lock in the current ratio.
        """
        if not lock:
            self.state['aspectLocked'] = False
        else:
            vr = self.viewRect()
            currentRatio = vr.width() / vr.height()
            if ratio is None:
                ratio = currentRatio
            self.state['aspectLocked'] = ratio
            if ratio != currentRatio:  ## If this would change the current range, do that now
                #self.setRange(0, self.state['viewRange'][0][0], self.state['viewRange'][0][1])
                self.updateMatrix()
        self.sigStateChanged.emit(self)
        
    def childTransform(self):
        """
        Return the transform that maps from child(item in the childGroup) coordinates to local coordinates.
        (This maps from inside the viewbox to outside)
        """ 
        m = self.childGroup.transform()
        #m1 = QtGui.QTransform()
        #m1.translate(self.childGroup.pos().x(), self.childGroup.pos().y())
        return m #*m1

    def mapToView(self, obj):
        """Maps from the local coordinates of the ViewBox to the coordinate system displayed inside the ViewBox"""
        m = self.childTransform().inverted()[0]
        return m.map(obj)

    def mapFromView(self, obj):
        """Maps from the coordinate system displayed inside the ViewBox to the local coordinates of the ViewBox"""
        m = self.childTransform()
        return m.map(obj)

    def mapSceneToView(self, obj):
        """Maps from scene coordinates to the coordinate system displayed inside the ViewBox"""
        return self.mapToView(self.mapFromScene(obj))

    def mapViewToScene(self, obj):
        """Maps from the coordinate system displayed inside the ViewBox to scene coordinates"""
        return self.mapToScene(self.mapFromView(obj))
    
    def mapFromItemToView(self, item, obj):
        return self.mapSceneToView(item.mapToScene(obj))

    def mapFromViewToItem(self, item, obj):
        return item.mapFromScene(self.mapViewToScene(obj))

    def itemBoundingRect(self, item):
        """Return the bounding rect of the item in view coordinates"""
        return self.mapSceneToView(item.sceneBoundingRect()).boundingRect()
    
    #def viewScale(self):
        #vr = self.viewRect()
        ##print "viewScale:", self.range
        #xd = vr.width()
        #yd = vr.height()
        #if xd == 0 or yd == 0:
            #print "Warning: 0 range in view:", xd, yd
            #return np.array([1,1])
        
        ##cs = self.canvas().size()
        #cs = self.boundingRect()
        #scale = np.array([cs.width() / xd, cs.height() / yd])
        ##print "view scale:", scale
        #return scale

    def wheelEvent(self, ev, axis=None):
        mask = np.array(self.state['mouseEnabled'], dtype=np.float)
        if axis is not None and axis >= 0 and axis < len(mask):
            mv = mask[axis]
            mask[:] = 0
            mask[axis] = mv
        s = ((mask * 0.02) + 1) ** (ev.delta() * self.state['wheelScaleFactor']) # actual scaling factor
        
        center = Point(self.childGroup.transform().inverted()[0].map(ev.pos()))
        #center = ev.pos()
        
        self.scaleBy(s, center)
        self.sigRangeChangedManually.emit(self.state['mouseEnabled'])
        ev.accept()

        
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.raiseContextMenu(ev)
    
    def raiseContextMenu(self, ev):
        #print "viewbox.raiseContextMenu called."
        
        #menu = self.getMenu(ev)
        menu = self.getMenu(ev)
        self.scene().addParentContextMenus(self, menu, ev)
        #print "2:", [str(a.text()) for a in self.menu.actions()]
        pos = ev.screenPos()
        #pos2 = ev.scenePos()
        #print "3:", [str(a.text()) for a in self.menu.actions()]
        #self.sigActionPositionChanged.emit(pos2)

        menu.popup(QtCore.QPoint(pos.x(), pos.y()))
        #print "4:", [str(a.text()) for a in self.menu.actions()]
        
    def getMenu(self, ev):
        self._menuCopy = self.menu.copy()  ## temporary storage to prevent menu disappearing
        return self._menuCopy
        
    def getContextMenus(self, event):
        return self.menu.subMenus()
        #return [self.getMenu(event)]
        

    def mouseDragEvent(self, ev):
        ev.accept()  ## we accept all buttons
        
        pos = ev.pos()
        lastPos = ev.lastPos()
        dif = pos - lastPos
        dif = dif * -1

        ## Ignore axes if mouse is disabled
        mask = np.array(self.state['mouseEnabled'], dtype=np.float)

        ## Scale or translate based on mouse button
        if ev.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MidButton):
            if self.state['mouseMode'] == ViewBox.RectMode:
                if ev.isFinish():  ## This is the final move in the drag; change the view scale now
                    #print "finish"
                    self.rbScaleBox.hide()
                    #ax = QtCore.QRectF(Point(self.pressPos), Point(self.mousePos))
                    ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
                    ax = self.childGroup.mapRectFromParent(ax)
                    self.showAxRect(ax)
                    self.axHistoryPointer += 1
                    self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
                else:
                    ## update shape of scale box
                    self.updateScaleBox(ev.buttonDownPos(), ev.pos())
            else:
                tr = dif*mask
                tr = self.mapToView(tr) - self.mapToView(Point(0,0))
                self.translateBy(tr)
                self.sigRangeChangedManually.emit(self.state['mouseEnabled'])
        elif ev.button() & QtCore.Qt.RightButton:
            #print "vb.rightDrag"
            if self.state['aspectLocked'] is not False:
                mask[0] = 0
            
            dif = ev.screenPos() - ev.lastScreenPos()
            dif = np.array([dif.x(), dif.y()])
            dif[0] *= -1
            s = ((mask * 0.02) + 1) ** dif
            center = Point(self.childGroup.transform().inverted()[0].map(ev.buttonDownPos(QtCore.Qt.RightButton)))
            #center = Point(ev.buttonDownPos(QtCore.Qt.RightButton))
            self.scaleBy(s, center)
            self.sigRangeChangedManually.emit(self.state['mouseEnabled'])

    def keyPressEvent(self, ev):
        """
        This routine should capture key presses in the current view box.
        Key presses are used only when mouse mode is RectMode
        The following events are implemented:
        ctrl-A : zooms out to the default "full" view of the plot
        ctrl-+ : moves forward in the zooming stack (if it exists)
        ctrl-- : moves backward in the zooming stack (if it exists)
         
        """
        #print ev.key()
        #print 'I intercepted a key press, but did not accept it'
        
        ## not implemented yet ?
        #self.keypress.sigkeyPressEvent.emit()
        
        ev.accept()
        if ev.text() == '-':
            self.scaleHistory(-1)
        elif ev.text() in ['+', '=']:
            self.scaleHistory(1)
        elif ev.key() == QtCore.Qt.Key_Backspace:
            self.scaleHistory(len(self.axHistory))
        else:
            ev.ignore()

    def scaleHistory(self, d):
        ptr = max(0, min(len(self.axHistory)-1, self.axHistoryPointer+d))
        if ptr != self.axHistoryPointer:
            self.axHistoryPointer = ptr
            self.showAxRect(self.axHistory[ptr])
            

    def updateScaleBox(self, p1, p2):
        r = QtCore.QRectF(p1, p2)
        r = self.childGroup.mapRectFromParent(r)
        self.rbScaleBox.setPos(r.topLeft())
        self.rbScaleBox.resetTransform()
        self.rbScaleBox.scale(r.width(), r.height())
        self.rbScaleBox.show()

    def showAxRect(self, ax):
        self.setRange(ax.normalized()) # be sure w, h are correct coordinates
        self.sigRangeChangedManually.emit(self.state['mouseEnabled'])

    #def mouseRect(self):
        #vs = self.viewScale()
        #vr = self.state['viewRange']
        ## Convert positions from screen (view) pixel coordinates to axis coordinates 
        #ax = QtCore.QRectF(self.pressPos[0]/vs[0]+vr[0][0], -(self.pressPos[1]/vs[1]-vr[1][1]),
            #(self.mousePos[0]-self.pressPos[0])/vs[0], -(self.mousePos[1]-self.pressPos[1])/vs[1])
        #return(ax)

    def allChildren(self, item=None):
        """Return a list of all children and grandchildren of this ViewBox"""
        if item is None:
            item = self.childGroup
        
        children = [item]
        for ch in item.childItems():
            children.extend(self.allChildren(ch))
        return children
        
        
        
    def childrenBoundingRect(self, frac=None):
        """Return the bounding range of all children.
        [[xmin, xmax], [ymin, ymax]]
        Values may be None if there are no specific bounds for an axis.
        """
        
        #items = self.allChildren()
        items = self.addedItems
        
        #if item is None:
            ##print "children bounding rect:"
            #item = self.childGroup
            
        range = [None, None]
            
        for item in items:
            if not item.isVisible():
                continue
        
            #print "=========", item
            useX = True
            useY = True
            if hasattr(item, 'dataBounds'):
                if frac is None:
                    frac = (1.0, 1.0)
                xr = item.dataBounds(0, frac=frac[0])
                yr = item.dataBounds(1, frac=frac[1])
                if xr is None:
                    useX = False
                    xr = (0,0)
                if yr is None:
                    useY = False
                    yr = (0,0)
                
                bounds = QtCore.QRectF(xr[0], yr[0], xr[1]-xr[0], yr[1]-yr[0])
                #print "   item real:", bounds
            else:
                if int(item.flags() & item.ItemHasNoContents) > 0:
                    continue
                    #print "   empty"
                else:
                    bounds = item.boundingRect()
                    #bounds = [[item.left(), item.top()], [item.right(), item.bottom()]]
                #print "   item:", bounds
            #bounds = QtCore.QRectF(bounds[0][0], bounds[1][0], bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0])
            bounds = self.mapFromItemToView(item, bounds).boundingRect()
            #print "    ", bounds
            
            
            if not any([useX, useY]):
                continue
            
            if useX != useY:  ##   !=  means  xor
                ang = item.transformAngle()
                if ang == 0 or ang == 180:
                    pass
                elif ang == 90 or ang == 270:
                    tmp = useX
                    useY = useX
                    useX = tmp
                else:
                    continue  ## need to check for item rotations and decide how best to apply this boundary. 
            
            
            if useY:
                if range[1] is not None:
                    range[1] = [min(bounds.top(), range[1][0]), max(bounds.bottom(), range[1][1])]
                    #bounds.setTop(min(bounds.top(), chb.top()))
                    #bounds.setBottom(max(bounds.bottom(), chb.bottom()))
                else:
                    range[1] = [bounds.top(), bounds.bottom()]
                    #bounds.setTop(chb.top())
                    #bounds.setBottom(chb.bottom())
            if useX:
                if range[0] is not None:
                    range[0] = [min(bounds.left(), range[0][0]), max(bounds.right(), range[0][1])]
                    #bounds.setLeft(min(bounds.left(), chb.left()))
                    #bounds.setRight(max(bounds.right(), chb.right()))
                else:
                    range[0] = [bounds.left(), bounds.right()]
                    #bounds.setLeft(chb.left())
                    #bounds.setRight(chb.right())
        
        tr = self.targetRange()
        if range[0] is None:
            range[0] = tr[0]
        if range[1] is None:
            range[1] = tr[1]
            
        bounds = QtCore.QRectF(range[0][0], range[1][0], range[0][1]-range[0][0], range[1][1]-range[1][0])
        return bounds
            
        

    def updateMatrix(self, changed=None):
        if changed is None:
            changed = [False, False]
        #print "udpateMatrix:"
        #print "  range:", self.range
        tr = self.targetRect()
        bounds = self.rect() #boundingRect()
        #print bounds
        
        ## set viewRect, given targetRect and possibly aspect ratio constraint
        if self.state['aspectLocked'] is False or bounds.height() == 0:
            self.state['viewRange'] = [self.state['targetRange'][0][:], self.state['targetRange'][1][:]]
        else:
            viewRatio = bounds.width() / bounds.height()
            targetRatio = self.state['aspectLocked'] * tr.width() / tr.height()
            if targetRatio > viewRatio:  
                ## target is wider than view
                dy = 0.5 * (tr.width() / (self.state['aspectLocked'] * viewRatio) - tr.height())
                if dy != 0:
                    changed[1] = True
                self.state['viewRange'] = [self.state['targetRange'][0][:], [self.state['targetRange'][1][0] - dy, self.state['targetRange'][1][1] + dy]]
            else:
                dx = 0.5 * (tr.height() * viewRatio * self.state['aspectLocked'] - tr.width())
                if dx != 0:
                    changed[0] = True
                self.state['viewRange'] = [[self.state['targetRange'][0][0] - dx, self.state['targetRange'][0][1] + dx], self.state['targetRange'][1][:]]
        
        vr = self.viewRect()
        #print "  bounds:", bounds
        if vr.height() == 0 or vr.width() == 0:
            return
        scale = Point(bounds.width()/vr.width(), bounds.height()/vr.height())
        if not self.state['yInverted']:
            scale = scale * Point(1, -1)
        m = QtGui.QTransform()
        
        ## First center the viewport at 0
        #self.childGroup.resetTransform()
        #self.resetTransform()
        #center = self.transform().inverted()[0].map(bounds.center())
        center = bounds.center()
        #print "  transform to center:", center
        #if self.state['yInverted']:
            #m.translate(center.x(), -center.y())
            #print "  inverted; translate", center.x(), center.y()
        #else:
        m.translate(center.x(), center.y())
            #print "  not inverted; translate", center.x(), -center.y()
            
        ## Now scale and translate properly
        m.scale(scale[0], scale[1])
        st = Point(vr.center())
        #st = translate
        m.translate(-st[0], -st[1])
        
        self.childGroup.setTransform(m)
        #self.setTransform(m)
        #self.prepareGeometryChange()
        
        #self.currentScale = scale
        
        if changed[0]:
            self.sigXRangeChanged.emit(self, tuple(self.state['viewRange'][0]))
        if changed[1]:
            self.sigYRangeChanged.emit(self, tuple(self.state['viewRange'][1]))
        if any(changed):
            self.sigRangeChanged.emit(self, self.state['viewRange'])

    def paint(self, p, opt, widget):
        if self.border is not None:
            bounds = self.shape()
            p.setPen(self.border)
            #p.fillRect(bounds, QtGui.QColor(0, 0, 0))
            p.drawPath(bounds)

    def saveSvg(self):
        pass
        
    def saveImage(self):
        pass

    def savePrint(self):
        printer = QtGui.QPrinter()
        if QtGui.QPrintDialog(printer).exec_() == QtGui.QDialog.Accepted:
            p = QtGui.QPainter(printer)
            p.setRenderHint(p.Antialiasing)
            self.scene().render(p)
            p.end()

    def updateViewLists(self):
        def cmpViews(a, b):
            wins = 100 * cmp(a.window() is self.window(), b.window() is self.window())
            alpha = cmp(a.name, b.name)
            return wins + alpha
            
        ## make a sorted list of all named views
        nv = ViewBox.NamedViews.values()
        nv.sort(cmpViews)
        
        if self in nv:
            nv.remove(self)
        names = [v.name for v in nv]
        self.menu.setViewList(names)

    @staticmethod
    def updateAllViewLists():
        for v in ViewBox.AllViews:
            v.updateViewLists()
            



from ViewBoxMenu import ViewBoxMenu
