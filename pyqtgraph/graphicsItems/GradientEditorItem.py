import operator
import weakref

import numpy as np

from .. import functions as fn
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorMapMenu import ColorMapMenu
from .GraphicsWidget import GraphicsWidget
from .GradientPresets import Gradients

translate = QtCore.QCoreApplication.translate

__all__ = ['TickSliderItem', 'GradientEditorItem', 'addGradientListToDocstring']


def addGradientListToDocstring():
    """Decorator to add list of current pre-defined gradients to the end of a function docstring."""
    def dec(fn):
        if fn.__doc__ is not None:
            fn.__doc__ = fn.__doc__ + str(list(Gradients.keys())).strip('[').strip(']')
        return fn
    return dec



class TickSliderItem(GraphicsWidget):
    ## public class
    """**Bases:** :class:`GraphicsWidget <pyqtgraph.GraphicsWidget>`
    
    A rectangular item with tick marks along its length that can (optionally) be moved by the user."""
    
    sigTicksChanged = QtCore.Signal(object)
    sigTicksChangeFinished = QtCore.Signal(object)
    
    def __init__(self, orientation='bottom', allowAdd=True, allowRemove=True, **kargs):
        """
        ==============  =================================================================================
        **Arguments:**
        orientation     Set the orientation of the gradient. Options are: 'left', 'right'
                        'top', and 'bottom'.
        allowAdd        Specifies whether the user can add ticks.
        allowRemove     Specifies whether the user can remove new ticks.
        tickPen         Default is white. Specifies the color of the outline of the ticks.
                        Can be any of the valid arguments for :func:`mkPen <pyqtgraph.mkPen>`
        ==============  =================================================================================
        """
        ## public
        GraphicsWidget.__init__(self)
        self.orientation = orientation
        self.length = 100
        self.tickSize = 15
        self.ticks = {}
        self.maxDim = 20
        self.allowAdd = allowAdd
        self.allowRemove = allowRemove
        if 'tickPen' in kargs:
            self.tickPen = fn.mkPen(kargs['tickPen'])
        else:
            self.tickPen = fn.mkPen('w')
            
        self.orientations = {
            'left': (90, 1, 1), 
            'right': (90, 1, 1), 
            'top': (0, 1, -1), 
            'bottom': (0, 1, 1)
        }
        
        self.setOrientation(orientation)
        #self.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame | QtWidgets.QFrame.Shadow.Plain)
        #self.setBackgroundRole(QtGui.QPalette.ColorRole.NoRole)
        #self.setMouseTracking(True)
        
    #def boundingRect(self):
        #return self.mapRectFromParent(self.geometry()).normalized()
        
    #def shape(self):  ## No idea why this is necessary, but rotated items do not receive clicks otherwise.
        #p = QtGui.QPainterPath()
        #p.addRect(self.boundingRect())
        #return p
        
    def paint(self, p, opt, widget):
        #p.setPen(fn.mkPen('g', width=3))
        #p.drawRect(self.boundingRect())
        return
        
    def keyPressEvent(self, ev):
        ev.ignore()

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
            
    
    def setOrientation(self, orientation):
        ## public
        """Set the orientation of the TickSliderItem.
        
        ==============  ===================================================================
        **Arguments:**
        orientation     Options are: 'left', 'right', 'top', 'bottom'
                        The orientation option specifies which side of the slider the
                        ticks are on, as well as whether the slider is vertical ('right'
                        and 'left') or horizontal ('top' and 'bottom').
        ==============  ===================================================================
        """
        self.orientation = orientation
        self.setMaxDim()
        self.resetTransform()
        ort = orientation
        if ort == 'top':
            transform = QtGui.QTransform.fromScale(1, -1)
            transform.translate(0, -self.height())
            self.setTransform(transform)
        elif ort == 'left':
            transform = QtGui.QTransform()
            transform.rotate(270)
            transform.scale(1, -1)
            transform.translate(-self.height(), -self.maxDim)
            self.setTransform(transform)
        elif ort == 'right':
            transform = QtGui.QTransform()
            transform.rotate(270)
            transform.translate(-self.height(), 0)
            self.setTransform(transform)
        elif ort != 'bottom':
            raise Exception("%s is not a valid orientation. Options are 'left', 'right', 'top', and 'bottom'" %str(ort))
        
        tr = QtGui.QTransform.fromTranslate(self.tickSize/2., 0)
        self.setTransform(tr, True)
    
    def addTick(self, x, color=None, movable=True, finish=True):
        ## public
        """
        Add a tick to the item.
        
        ==============  ==================================================================
        **Arguments:**
        x               Position where tick should be added.
        color           Color of added tick. If color is not specified, the color will be
                        white.
        movable         Specifies whether the tick is movable with the mouse.
        ==============  ==================================================================
        """        
        
        if color is None:
            color = QtGui.QColor(255,255,255)
        tick = Tick([x*self.length, 0], color, movable, self.tickSize, pen=self.tickPen, removeAllowed=self.allowRemove)
        self.ticks[tick] = x
        tick.setParentItem(self)
        
        tick.sigMoving.connect(self.tickMoved)
        tick.sigMoved.connect(self.tickMoveFinished)
        tick.sigClicked.connect(self.tickClicked)
        
        self.sigTicksChanged.emit(self)
        
        if finish:
            self.sigTicksChangeFinished.emit(self)
        
        return tick
    
    def removeTick(self, tick, finish=True):
        ## public
        """
        Removes the specified tick.
        """
        del self.ticks[tick]
        tick.setParentItem(None)
        if self.scene() is not None:
            self.scene().removeItem(tick)
        
        self.sigTicksChanged.emit(self)
        
        if finish:
            self.sigTicksChangeFinished.emit(self)
    
    @QtCore.Slot(object, object)
    def tickMoved(self, tick, pos):
        #print "tick changed"
        ## Correct position of tick if it has left bounds.
        newX = min(max(0, pos.x()), self.length)
        pos.setX(newX)
        tick.setPos(pos)
        self.ticks[tick] = float(newX) / self.length
        
        self.sigTicksChanged.emit(self)
    
    @QtCore.Slot(object)
    def tickMoveFinished(self, tick):
        self.sigTicksChangeFinished.emit(self)
    
    def tickClicked(self, tick, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton and tick.removeAllowed:
            self.removeTick(tick)
    
    def widgetLength(self):
        if self.orientation in ['bottom', 'top']:
            return self.width()
        else:
            return self.height()
    
    def resizeEvent(self, ev):
        wlen = max(40, self.widgetLength())
        self.setLength(wlen-self.tickSize-2)
        self.setOrientation(self.orientation)
        #bounds = self.scene().itemsBoundingRect()
        #bounds.setLeft(min(-self.tickSize*0.5, bounds.left()))
        #bounds.setRight(max(self.length + self.tickSize, bounds.right()))
        #self.setSceneRect(bounds)
        #self.fitInView(bounds, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        
    def setLength(self, newLen):
        #private
        for t, x in list(self.ticks.items()):
            t.setPos(x * newLen + 1, t.pos().y())
        self.length = float(newLen)
        
    #def mousePressEvent(self, ev):
        #QtWidgets.QGraphicsView.mousePressEvent(self, ev)
        #self.ignoreRelease = False
        #for i in self.items(ev.pos()):
            #if isinstance(i, Tick):
                #self.ignoreRelease = True
                #break
        ##if len(self.items(ev.pos())) > 0:  ## Let items handle their own clicks
            ##self.ignoreRelease = True
        
    #def mouseReleaseEvent(self, ev):
        #QtWidgets.QGraphicsView.mouseReleaseEvent(self, ev)
        #if self.ignoreRelease:
            #return
            
        #pos = self.mapToScene(ev.pos())
            
        #if ev.button() == QtCore.Qt.MouseButton.LeftButton and self.allowAdd:
            #if pos.x() < 0 or pos.x() > self.length:
                #return
            #if pos.y() < 0 or pos.y() > self.tickSize:
                #return
            #pos.setX(min(max(pos.x(), 0), self.length))
            #self.addTick(pos.x()/self.length)
        #elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            #self.showMenu(ev)
            
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and self.allowAdd:
            pos = ev.pos()
            if pos.x() < 0 or pos.x() > self.length:
                return
            if pos.y() < 0 or pos.y() > self.tickSize:
                return
            pos.setX(min(max(pos.x(), 0), self.length))
            self.addTick(pos.x()/self.length)
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.showMenu(ev)

        #if  ev.button() == QtCore.Qt.MouseButton.RightButton:
            #if self.moving:
                #ev.accept()
                #self.setPos(self.startPosition)
                #self.moving = False
                #self.sigMoving.emit(self)
                #self.sigMoved.emit(self)
            #else:
                #pass
                #self.view().tickClicked(self, ev)
                ###remove

    def hoverEvent(self, ev):
        if (not ev.isExit()) and ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton):
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)
            ## show ghost tick
            #self.currentPen = fn.mkPen(255, 0,0)
        #else:
            #self.currentPen = self.pen
        #self.update()
        
    def showMenu(self, ev):
        pass

    def setTickColor(self, tick, color):
        """Set the color of the specified tick.
        
        ==============  ==================================================================
        **Arguments:**
        tick            Can be either an integer corresponding to the index of the tick
                        or a Tick object. Ex: if you had a slider with 3 ticks and you
                        wanted to change the middle tick, the index would be 1.
        color           The color to make the tick. Can be any argument that is valid for
                        :func:`mkBrush <pyqtgraph.mkBrush>`
        ==============  ==================================================================
        """
        tick = self.getTick(tick)
        tick.color = color
        tick.update()
        #tick.setBrush(QtGui.QBrush(QtGui.QColor(tick.color)))
        
        self.sigTicksChanged.emit(self)
        self.sigTicksChangeFinished.emit(self)

    def setTickValue(self, tick, val):
        ## public
        """
        Set the position (along the slider) of the tick.
        
        ==============   ==================================================================
        **Arguments:**
        tick             Can be either an integer corresponding to the index of the tick
                         or a Tick object. Ex: if you had a slider with 3 ticks and you
                         wanted to change the middle tick, the index would be 1.
        val              The desired position of the tick. If val is < 0, position will be
                         set to 0. If val is > 1, position will be set to 1.
        ==============   ==================================================================
        """
        tick = self.getTick(tick)
        val = min(max(0.0, val), 1.0)
        x = val * self.length
        pos = tick.pos()
        pos.setX(x)
        tick.setPos(pos)
        self.ticks[tick] = val
        
        self.update()
        self.sigTicksChanged.emit(self)
        self.sigTicksChangeFinished.emit(self)
        
    def tickValue(self, tick):
        ## public
        """Return the value (from 0.0 to 1.0) of the specified tick.
        
        ==============  ==================================================================
        **Arguments:**
        tick            Can be either an integer corresponding to the index of the tick
                        or a Tick object. Ex: if you had a slider with 3 ticks and you
                        wanted the value of the middle tick, the index would be 1.
        ==============  ==================================================================
        """
        tick = self.getTick(tick)
        return self.ticks[tick]
        
    def getTick(self, tick):
        ## public
        """Return the Tick object at the specified index.
        
        ==============  ==================================================================
        **Arguments:**
        tick            An integer corresponding to the index of the desired tick. If the
                        argument is not an integer it will be returned unchanged.
        ==============  ==================================================================
        """
        if type(tick) is int:
            tick = self.listTicks()[tick][0]
        return tick

    #def mouseMoveEvent(self, ev):
        #QtWidgets.QGraphicsView.mouseMoveEvent(self, ev)

    def listTicks(self):
        """Return a sorted list of all the Tick objects on the slider."""
        ## public
        ticks = sorted(self.ticks.items(), key=operator.itemgetter(1))
        return ticks


class GradientEditorItem(TickSliderItem):
    """
    **Bases:** :class:`TickSliderItem <pyqtgraph.TickSliderItem>`
    
    An item that can be used to define a color gradient. Implements common pre-defined gradients that are 
    customizable by the user. :class: `GradientWidget <pyqtgraph.GradientWidget>` provides a widget
    with a GradientEditorItem that can be added to a GUI. 
    
    ================================ ===========================================================
    **Signals:**
    sigGradientChanged(self)         Signal is emitted anytime the gradient changes. The signal 
                                     is emitted in real time while ticks are being dragged or 
                                     colors are being changed.
    sigGradientChangeFinished(self)  Signal is emitted when the gradient is finished changing.
    ================================ ===========================================================    
 
    """
    
    sigGradientChanged = QtCore.Signal(object)
    sigGradientChangeFinished = QtCore.Signal(object)
    
    def __init__(self, *args, **kargs):
        """
        Create a new GradientEditorItem. 
        All arguments are passed to :func:`TickSliderItem.__init__ <pyqtgraph.TickSliderItem.__init__>`
        
        ===============  =================================================================================
        **Arguments:**
        orientation      Set the orientation of the gradient. Options are: 'left', 'right'
                         'top', and 'bottom'.
        allowAdd         Default is True. Specifies whether ticks can be added to the item.
        tickPen          Default is white. Specifies the color of the outline of the ticks.
                         Can be any of the valid arguments for :func:`mkPen <pyqtgraph.mkPen>`
        ===============  =================================================================================
        """
        self.currentTick = None
        self.currentTickColor = None
        self.rectSize = 15
        self.gradRect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, self.rectSize, 100, self.rectSize))
        self.backgroundRect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, -self.rectSize, 100, self.rectSize))
        self.backgroundRect.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.DiagCrossPattern))
        self.colorMode = 'rgb'
        
        TickSliderItem.__init__(self, *args, **kargs)
        
        self.colorDialog = QtWidgets.QColorDialog()
        self.colorDialog.setOption(QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
        self.colorDialog.setOption(QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        
        self.colorDialog.currentColorChanged.connect(self.currentColorChanged)
        self.colorDialog.rejected.connect(self.currentColorRejected)
        self.colorDialog.accepted.connect(self.currentColorAccepted)
        
        self.backgroundRect.setParentItem(self)
        self.gradRect.setParentItem(self)
        
        self.setMaxDim(self.rectSize + self.tickSize)
        
        self.rgbAction = QtGui.QAction(translate("GradiantEditorItem", 'RGB'), self)
        self.rgbAction.setCheckable(True)
        self.rgbAction.triggered.connect(self._setColorModeToRGB)
        self.hsvAction = QtGui.QAction(translate("GradiantEditorItem", 'HSV'), self)
        self.hsvAction.setCheckable(True)
        self.hsvAction.triggered.connect(self._setColorModeToHSV)
            
        self.menu = ColorMapMenu(showGradientSubMenu=True, showColorMapSubMenus=True)
        self.menu.sigColorMapTriggered.connect(self.colorMapMenuClicked)
        self.menu.addSeparator()
        self.menu.addAction(self.rgbAction)
        self.menu.addAction(self.hsvAction)
        
        
        for t in list(self.ticks.keys()):
            self.removeTick(t)
        self.addTick(0, QtGui.QColor(0,0,0), True)
        self.addTick(1, QtGui.QColor(255,0,0), True)
        self.setColorMode('rgb')
        self.updateGradient()
        self.linkedGradients = {}
        
        self.sigTicksChanged.connect(self._updateGradientIgnoreArgs)
        self.sigTicksChangeFinished.connect(self.sigGradientChangeFinished)

    def showTicks(self, show=True):
        for tick in self.ticks.keys():
            if show:
                tick.show()
                orig = getattr(self, '_allowAdd_backup', None)
                if orig: 
                    self.allowAdd = orig
            else:
                self._allowAdd_backup = self.allowAdd
                self.allowAdd = False #block tick creation
                tick.hide()

    def setOrientation(self, orientation):
        ## public
        """
        Set the orientation of the GradientEditorItem. 
        
        ==============  ===================================================================
        **Arguments:**
        orientation     Options are: 'left', 'right', 'top', 'bottom'
                        The orientation option specifies which side of the gradient the
                        ticks are on, as well as whether the gradient is vertical ('right'
                        and 'left') or horizontal ('top' and 'bottom').
        ==============  ===================================================================
        """
        TickSliderItem.setOrientation(self, orientation)
        tr = QtGui.QTransform.fromTranslate(0, self.rectSize)
        self.setTransform(tr, True)
    
    def showMenu(self, ev):
        #private
        self.menu.popup(ev.screenPos().toQPoint())
    
    @QtCore.Slot(object)
    def colorMapMenuClicked(self, cmap):
        #private
        if cmap.name.startswith("preset-gradient:"):
            name = cmap.name.split(":")[1]
            self.loadPreset(name)
        else:
            self.setColorMap(cmap)
            self.showTicks(False)
        
    @addGradientListToDocstring()
    def loadPreset(self, name):
        """
        Load a predefined gradient. Currently defined gradients are: 
        """## TODO: provide image with names of defined gradients
        
        #global Gradients
        self.restoreState(Gradients[name])
    
    def setColorMode(self, cm):
        """
        Set the color mode for the gradient. Options are: 'hsv', 'rgb'
        
        """
        
        ## public
        if cm not in ['rgb', 'hsv']:
            raise Exception("Unknown color mode %s. Options are 'rgb' and 'hsv'." % str(cm))
        
        try:
            self.rgbAction.blockSignals(True)
            self.hsvAction.blockSignals(True)
            self.rgbAction.setChecked(cm == 'rgb')
            self.hsvAction.setChecked(cm == 'hsv')
        finally:
            self.rgbAction.blockSignals(False)
            self.hsvAction.blockSignals(False)
        self.colorMode = cm
        
        self.sigTicksChanged.emit(self)
        self.sigGradientChangeFinished.emit(self)

    @QtCore.Slot()
    def _setColorModeToRGB(self):
        self.setColorMode("rgb")

    @QtCore.Slot()
    def _setColorModeToHSV(self):
        self.setColorMode("hsv")

    def colorMap(self):
        """Return a ColorMap object representing the current state of the editor."""
        if self.colorMode == 'hsv':
            raise NotImplementedError('hsv colormaps not yet supported')
        pos = []
        color = []
        for t,x in self.listTicks():
            pos.append(x)
            c = t.color
            color.append(c.getRgb())
        return ColorMap(np.array(pos), np.array(color, dtype=np.ubyte))
        
    def updateGradient(self):
        #private
        self.gradient = self.getGradient()
        self.gradRect.setBrush(QtGui.QBrush(self.gradient))
        self.sigGradientChanged.emit(self)

    @QtCore.Slot(object)
    def _updateGradientIgnoreArgs(self, *args, **kwargs):
        self.updateGradient()

    def setLength(self, newLen):
        #private (but maybe public)
        TickSliderItem.setLength(self, newLen)
        self.backgroundRect.setRect(1, -self.rectSize, newLen, self.rectSize)
        self.gradRect.setRect(1, -self.rectSize, newLen, self.rectSize)
        self.sigTicksChanged.emit(self)
        
    @QtCore.Slot(QtGui.QColor)
    def currentColorChanged(self, color):
        #private
        if color.isValid() and self.currentTick is not None:
            self.setTickColor(self.currentTick, color)
            
    @QtCore.Slot()
    def currentColorRejected(self):
        #private
        self.setTickColor(self.currentTick, self.currentTickColor)
        
    @QtCore.Slot()
    def currentColorAccepted(self):
        self.sigGradientChangeFinished.emit(self)
        
    @QtCore.Slot(object, object)
    def tickClicked(self, tick, ev):
        #private
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.raiseColorDialog(tick)
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.raiseTickContextMenu(tick, ev)
            
    def raiseColorDialog(self, tick):
        if not tick.colorChangeAllowed:
            return
        self.currentTick = tick
        self.currentTickColor = tick.color
        self.colorDialog.setCurrentColor(tick.color)
        self.colorDialog.open()
        
    def raiseTickContextMenu(self, tick, ev):
        self.tickMenu = TickMenu(tick, self)
        self.tickMenu.popup(ev.screenPos().toQPoint())

    def tickMoveFinished(self, tick):
        self.sigGradientChangeFinished.emit(self)

    def getGradient(self):
        """Return a QLinearGradient object."""
        g = QtGui.QLinearGradient(QtCore.QPointF(0,0), QtCore.QPointF(self.length,0))
        if self.colorMode == 'rgb':
            ticks = self.listTicks()
            g.setStops([(x, QtGui.QColor(t.color)) for t,x in ticks])
        elif self.colorMode == 'hsv':  ## HSV mode is approximated for display by interpolating 10 points between each stop
            ticks = self.listTicks()
            stops = []
            stops.append((ticks[0][1], ticks[0][0].color))
            for i in range(1,len(ticks)):
                x1 = ticks[i-1][1]
                x2 = ticks[i][1]
                dx = (x2-x1) / 10.
                for j in range(1,10):
                    x = x1 + dx*j
                    stops.append((x, self.getColor(x)))
                stops.append((x2, self.getColor(x2)))
            g.setStops(stops)
        return g
        
    def getColor(self, x, toQColor=True):
        """
        Return a color for a given value.
        
        ==============  ==================================================================
        **Arguments:**
        x               Value (position on gradient) of requested color.
        toQColor        If true, returns a QColor object, else returns a (r,g,b,a) tuple.
        ==============  ==================================================================
        """
        ticks = self.listTicks()
        if x <= ticks[0][1]:
            c = ticks[0][0].color
            if toQColor:
                return QtGui.QColor(c)  # always copy colors before handing them out
            else:
                return c.getRgb()
        if x >= ticks[-1][1]:
            c = ticks[-1][0].color
            if toQColor:
                return QtGui.QColor(c)  # always copy colors before handing them out
            else:
                return c.getRgb()
            
        x2 = ticks[0][1]
        for i in range(1,len(ticks)):
            x1 = x2
            x2 = ticks[i][1]
            if x1 <= x and x2 >= x:
                break
                
        dx = (x2-x1)
        if dx == 0:
            f = 0.
        else:
            f = (x-x1) / dx
        c1 = ticks[i-1][0].color
        c2 = ticks[i][0].color
        if self.colorMode == 'rgb':
            r = c1.red() * (1.-f) + c2.red() * f
            g = c1.green() * (1.-f) + c2.green() * f
            b = c1.blue() * (1.-f) + c2.blue() * f
            a = c1.alpha() * (1.-f) + c2.alpha() * f
            if toQColor:
                return QtGui.QColor(int(r), int(g), int(b), int(a))
            else:
                return (r,g,b,a)
        elif self.colorMode == 'hsv':
            h1,s1,v1,_ = c1.getHsv()
            h2,s2,v2,_ = c2.getHsv()
            h = h1 * (1.-f) + h2 * f
            s = s1 * (1.-f) + s2 * f
            v = v1 * (1.-f) + v2 * f
            c = QtGui.QColor.fromHsv(int(h), int(s), int(v))
            if toQColor:
                return c
            else:
                return c.getRgb()
                    
    def getLookupTable(self, nPts, alpha=None):
        """
        Return an RGB(A) lookup table (ndarray). 
        
        ==============  ============================================================================
        **Arguments:**
        nPts            The number of points in the returned lookup table.
        alpha           True, False, or None - Specifies whether or not alpha values are included
                        in the table.If alpha is None, alpha will be automatically determined.
        ==============  ============================================================================
        """
        if alpha is None:
            alpha = self.usesAlpha()
        if alpha:
            table = np.empty((nPts,4), dtype=np.ubyte)
        else:
            table = np.empty((nPts,3), dtype=np.ubyte)
            
        for i in range(nPts):
            x = float(i)/(nPts-1)
            color = self.getColor(x, toQColor=False)
            table[i] = color[:table.shape[1]]
            
        return table
    
    def usesAlpha(self):
        """Return True if any ticks have an alpha < 255"""
        
        ticks = self.listTicks()
        for t in ticks:
            if t[0].color.alpha() < 255:
                return True
            
        return False
            
    def isLookupTrivial(self):
        """Return True if the gradient has exactly two stops in it: black at 0.0 and white at 1.0"""
        ticks = self.listTicks()
        if len(ticks) != 2:
            return False
        if ticks[0][1] != 0.0 or ticks[1][1] != 1.0:
            return False
        c1 = ticks[0][0].color.getRgb()
        c2 = ticks[1][0].color.getRgb()
        if c1 != (0,0,0,255) or c2 != (255,255,255,255):
            return False
        return True
        
    def addTick(self, x, color=None, movable=True, finish=True):
        """
        Add a tick to the gradient. Return the tick.
        
        ==============  ==================================================================
        **Arguments:**
        x               Position where tick should be added.
        color           Color of added tick. If color is not specified, the color will be
                        the color of the gradient at the specified position.
        movable         Specifies whether the tick is movable with the mouse.
        ==============  ==================================================================
        """
        
        if color is None:
            color = self.getColor(x)
        t = TickSliderItem.addTick(self, x, color=color, movable=movable, finish=finish)
        t.colorChangeAllowed = True
        
        return t
        
    def saveState(self):
        """
        Return a dictionary with parameters for rebuilding the gradient. Keys will include:
        
           - 'mode': hsv or rgb
           - 'ticks': a list of tuples (pos, (r,g,b,a))
        """
        ## public
        ticks = []
        for t in self.ticks:
            c = t.color
            ticks.append((self.ticks[t], c.getRgb()))
        state = {'mode': self.colorMode, 
                 'ticks': ticks,
                 'ticksVisible': next(iter(self.ticks)).isVisible()}
        return state
        
    def restoreState(self, state):
        """
        Restore the gradient specified in state.
        
        ==============  ====================================================================
        **Arguments:**
        state           A dictionary with same structure as those returned by
                        :func:`saveState <pyqtgraph.GradientEditorItem.saveState>`
                      
                        Keys must include:
                      
                            - 'mode': hsv or rgb
                            - 'ticks': a list of tuples (pos, (r,g,b,a))
        ==============  ====================================================================
        """
        ## public
        
        # Mass edit ticks without graphics update
        signalsBlocked = self.blockSignals(True)
        
        self.setColorMode(state['mode'])
        for t in list(self.ticks.keys()):
            self.removeTick(t, finish=False)
        for t in state['ticks']:
            c = QtGui.QColor(*t[1])
            self.addTick(t[0], c, finish=False)
        self.showTicks( state.get('ticksVisible', 
                                  next(iter(self.ticks)).isVisible()) )
        
        # Close with graphics update
        self.blockSignals(signalsBlocked)
        self.sigTicksChanged.emit(self)
        self.sigGradientChangeFinished.emit(self)
        
    def setColorMap(self, cm):
        # Mass edit ticks without graphics update
        signalsBlocked = self.blockSignals(True)
        
        self.setColorMode('rgb')
        for t in list(self.ticks.keys()):
            self.removeTick(t, finish=False)
        colors = cm.getColors(mode='qcolor')
        for i in range(len(cm.pos)):
            x = cm.pos[i]
            c = colors[i]
            self.addTick(x, c, finish=False)
        
        # Close with graphics update
        self.blockSignals(signalsBlocked)
        self.sigTicksChanged.emit(self)
        self.sigGradientChangeFinished.emit(self)

    def linkGradient(self, slaveGradient, connect=True):
        if connect:
            fn = lambda g, slave=slaveGradient:slave.restoreState(
                                                     g.saveState())
            self.linkedGradients[id(slaveGradient)] = fn
            self.sigGradientChanged.connect(fn)
            self.sigGradientChanged.emit(self)
        else:
            fn = self.linkedGradients.get(id(slaveGradient), None)
            if fn:
                self.sigGradientChanged.disconnect(fn)


class Tick(QtWidgets.QGraphicsWidget):  ## NOTE: Making this a subclass of GraphicsObject instead results in
                                    ## activating this bug: https://bugreports.qt-project.org/browse/PYSIDE-86
    ## private class

    # When making Tick a subclass of QtWidgets.QGraphicsObject as origin,
    # ..GraphicsScene.items(self, *args) will get Tick object as a
    # class of QtGui.QMultimediaWidgets.QGraphicsVideoItem in python2.7-PyQt5(5.4.0)

    sigMoving = QtCore.Signal(object, object)
    sigMoved = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object)
    
    def __init__(self, pos, color, movable=True, scale=10, pen='w', removeAllowed=True):
        self.movable = movable
        self.moving = False
        self.scale = scale
        self.color = color
        self.pen = fn.mkPen(pen)
        self.hoverPen = fn.mkPen(255,255,0)
        self.currentPen = self.pen
        self.removeAllowed = removeAllowed
        self.pg = QtGui.QPainterPath(QtCore.QPointF(0,0))
        self.pg.lineTo(QtCore.QPointF(-scale/3**0.5, scale))
        self.pg.lineTo(QtCore.QPointF(scale/3**0.5, scale))
        self.pg.closeSubpath()
        
        QtWidgets.QGraphicsWidget.__init__(self)
        self.setPos(pos[0], pos[1])
        if self.movable:
            self.setZValue(1)
        else:
            self.setZValue(0)

    def boundingRect(self):
        return self.pg.boundingRect()
    
    def shape(self):
        return self.pg

    def paint(self, p, *args):
        p.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillPath(self.pg, fn.mkBrush(self.color))
        
        p.setPen(self.currentPen)
        p.drawPath(self.pg)


    def mouseDragEvent(self, ev):
        if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
            ev.accept()
            
            if not self.moving:
                return
                
            newPos = self.cursorOffset + self.mapToParent(ev.pos())
            newPos.setY(self.pos().y())
            
            self.setPos(newPos)
            self.sigMoving.emit(self, newPos)
            if ev.isFinish():
                self.moving = False
                self.sigMoved.emit(self)

    def mouseClickEvent(self, ev):
        ev.accept()
        if ev.button() == QtCore.Qt.MouseButton.RightButton and self.moving:
            self.setPos(self.startPosition)
            self.moving = False
            self.sigMoving.emit(self, self.startPosition)
            self.sigMoved.emit(self)
        else:
            self.sigClicked.emit(self, ev)

    def hoverEvent(self, ev):
        if (not ev.isExit()) and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)
            self.currentPen = self.hoverPen
        else:
            self.currentPen = self.pen
        self.update()
        

class TickMenu(QtWidgets.QMenu):
    
    def __init__(self, tick, sliderItem):
        QtWidgets.QMenu.__init__(self)
        
        self.tick = weakref.ref(tick)
        self.sliderItem = weakref.ref(sliderItem)
        
        self.removeAct = self.addAction(translate("GradientEditorItem", "Remove Tick"), lambda: self.sliderItem().removeTick(tick))
        if (not self.tick().removeAllowed) or len(self.sliderItem().ticks) < 3:
            self.removeAct.setEnabled(False)
            
        positionMenu = self.addMenu(translate("GradientEditorItem", "Set Position"))
        w = QtWidgets.QWidget()
        l = QtWidgets.QGridLayout()
        w.setLayout(l)
        
        value = sliderItem.tickValue(tick)
        self.fracPosSpin = SpinBox()
        self.fracPosSpin.setOpts(value=value, bounds=(0.0, 1.0), step=0.01, decimals=2)
        #self.dataPosSpin = SpinBox(value=dataVal)
        #self.dataPosSpin.setOpts(decimals=3, siPrefix=True)
                
        l.addWidget(QtWidgets.QLabel(f"{translate('GradiantEditorItem', 'Position')}:"), 0,0)
        l.addWidget(self.fracPosSpin, 0, 1)
        #l.addWidget(QtWidgets.QLabel("Position (data units):"), 1, 0)
        #l.addWidget(self.dataPosSpin, 1,1)
        
        #if self.sliderItem().dataParent is None:
        #    self.dataPosSpin.setEnabled(False)
        
        a = QtWidgets.QWidgetAction(self)
        a.setDefaultWidget(w)
        positionMenu.addAction(a)        
        
        self.fracPosSpin.sigValueChanging.connect(self.fractionalValueChanged)
        #self.dataPosSpin.valueChanged.connect(self.dataValueChanged)
        
        colorAct = self.addAction(translate("Context Menu", "Set Color"), lambda: self.sliderItem().raiseColorDialog(self.tick()))
        if not self.tick().colorChangeAllowed:
            colorAct.setEnabled(False)

    def fractionalValueChanged(self, x):
        self.sliderItem().setTickValue(self.tick(), self.fracPosSpin.value())
        #if self.sliderItem().dataParent is not None:
        #    self.dataPosSpin.blockSignals(True)
        #    self.dataPosSpin.setValue(self.sliderItem().tickDataValue(self.tick()))
        #    self.dataPosSpin.blockSignals(False)
            
    #def dataValueChanged(self, val):
    #    self.sliderItem().setTickValue(self.tick(), val, dataUnits=True)
    #    self.fracPosSpin.blockSignals(True)
    #    self.fracPosSpin.setValue(self.sliderItem().tickValue(self.tick()))
    #    self.fracPosSpin.blockSignals(False)
