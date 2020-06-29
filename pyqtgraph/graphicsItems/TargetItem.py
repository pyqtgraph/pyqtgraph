from ..Qt import QtGui, QtCore
import numpy as np
from ..Point import Point
from .. import functions as fn
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
from .TextItem import TextItem
from .ScatterPlotItem import Symbols


class TargetItem(UIGraphicsItem):
    """

    **Bases:** :class:`UIGraphicsItem <pyqtgraph.UIGraphicsItem>`
    Displays a draggable target of any available symbol (defaulting to a 
    crosshair).
    =============================== ===================================================
    **Signals:**
    sigPositionChangeFinished(self)
    sigPositionChanged(self)
    =============================== ===================================================

    The size of TargetItem will remain fixed on screen even as the view is zoomed.
    Includes an optional text label.
    """
    # sigDragged = QtCore.Signal(object)
    sigPositionChanged = QtCore.Signal(object)  # self
    sigPositionChangeFinished = QtCore.Signal(object)  # self

    # def __init__(self, movable=True, radii=(5, 10, 10), pen=(255, 255, 0), brush=(0, 0, 255, 100)):
    #     GraphicsObject.__init__(self)
    #     self._bounds = None
    #     self._radii = radii
    #     self._picture = None
    #     self.movable = movable
    #     self.moving = False
    #     self.label = None
    #     self.labelAngle = 0
    #     self.pen = fn.mkPen(pen)
    #     self.brush = fn.mkBrush(brush)

    def __init__(self, pos=None, radius=5, symbol='s', pen=None, brush=None,
                 movable=True, hoverPen=None, hoverBrush=None, label=None,
                 labelOpts=None, name=None, parent=None):
        """
        =============== ==================================================================
        **Arguments:**
        pos             Position of the target. This can be a list of QPointF or a list
                        of floats
        radius          Size of the target in pixel
        symbol          String that defines the shape of the target.  It can take any
                        compatible value from the symbols representation
        pen             Pen to use when drawing line. Can be any arguments that are valid
                        for :func:`mkPen <pyqtgraph.mkPen>`. Default pen is transparent
                        yellow.
        brush           Defines the brush that fill the target. Can be any arguments
                        that is valid for :func:`mkBrush<pyqtgraph.mkBrush>`. Default
                        is transparent blue.
        movable         If True, the target can be dragged to a new position by the user.
        hoverPen        Pen to use when drawing target when hovering over it. Can be any
                        arguments that are valid for :func:`mkPen <pyqtgraph.mkPen>`.
                        Default pen is red.
        hoverBrush      Brush to use to fill the target when hovering over it. Can be any
                        arguments that is valid for :func:`mkBrush<pyqtgraph.mkBrush>`.
                        Default is transparent blue.
        label           Text to be displayed in a label attached to the target, or
                        None to show no label (default is None). May optionally
                        include formatting strings to display the target value.
        labelOpts       A dict of keyword arguments to use when constructing the
                        text label. See :class:`TargetLabel`.
        name            Name of the item
        =============== ==================================================================
        """
        # UIGraphicsItem.__init__(self, parent=parent)
        super().__init__(parent)
        self.radius = radius
        # self.type = target
        self.symbol = symbol
        self.label = None

        self.movable = movable
        self.moving = False
        self.mouseHovering = False

        self._name = name

        self._pos = [0, 0]

        if pos is None:
            pos = [0, 0]

        if pen is None:
            pen = (200, 200, 100)
        self.setPen(pen)
        if hoverPen is None:
            hoverPen = (255, 0, 0)
        self.setHoverPen(hoverPen)

        if brush is None:
            brush = (0, 0, 255, 50)
        self.setBrush(brush)
        if hoverBrush is None:
            hoverBrush = (0, 0, 255, 100)
        self.setHoverBrush(hoverBrush)

        self.currentPen = self.pen
        self.currentBrush = self.brush

        # self.buildPath()
        self._shape = None

        # if label is not None:
        #     labelOpts = {} if labelOpts is None else labelOpts
        #     self.label = TargetLabel(self, text=label, **labelOpts)

        self.setPos(pos)
        self.setLabel(label, labelOpts)

    def setPos(self, pos):
        if type(pos) in [list, tuple]:
            newPos = pos
        elif isinstance(pos, QtCore.QPointF):
            newPos = [pos.x(), pos.y()]
        if self._pos != newPos:
            self._pos = newPos
            UIGraphicsItem.setPos(self, Point(self._pos))
            self.sigPositionChanged.emit(self)
    
    def position(self):
        return self._pos
    
    def setLabel(self, label, labelOpts):
        if label is not None:
            labelOptions = {} if labelOpts is None else labelOpts
            if self.label is None:
                self.label = TargetLabel(self, text=label, **labelOpts)
                self.label.setParentItem(self)
            else:
                self.label.setText(label)
                # self._updateLabel()
        else:
            if self.label is not None:
                self.label.scene().removeItem(self.label)
                self.label = None

    def setPen(self, *args, **kwargs):
        """Set the pen for drawing the target. Allowable arguments are any that
        are valid for :func:`mkPen <pyqtgraph.mkPen>`."""
        self.pen = fn.mkPen(*args, **kwargs)
        if not self.mouseHovering:
            self.currentPen = self.pen
            self.update()

    def setHoverPen(self, *args, **kwargs):
        """Set the pen for drawing the target when hovering over it. Allowable
        arguments are any that are valid for
        :func:`mkPen <pyqtgraph.mkPen>`."""
        self.hoverPen = fn.mkPen(*args, **kwargs)
        if self.mouseHovering:
            self.currentPen = self.hoverPen
            self.update()

    def setBrush(self, *args, **kwargs):
        """Set the brush that fills the target. Allowable arguments are any that
        are valid for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.brush = fn.mkBrush(*args, **kwargs)
        if not self.mouseHovering:
            self.currentBrush = self.brush
            self.update()

    def setHoverBrush(self, *args, **kwargs):
        """Set the brush that fills the target when hovering over it. Allowable
        arguments are any that are valid for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.hoverBrush = fn.mkBrush(*args, **kwargs)
        if self.mouseHovering:
            self.currentBrush = self.hoverBrush
            self.update()

    def setMovable(self, m):
        """Set whether the line is movable by the user."""
        self.movable = m
        self.setAcceptHoverEvents(m)

    # def setLabel(self, label):
    #     if label is None:
    #         if self.label is not None:
    #             self.label.scene().removeItem(self.label)
    #             self.label = None
    #     else:
    #         if self.label is None:
    #             self.label = TextItem()
    #             self.label.setParentItem(self)
    #         self.label.setText(label)
    #         self._updateLabel()

    # def setLabelAngle(self, angle):
    #     self.labelAngle = angle
    #     self._updateLabel()

    # def boundingRect(self):
    #     if self._picture is None:
    #         self._drawPicture()
    #     return self._bounds

    def boundingRect(self):
        return self.shape().boundingRect()
    
    # def dataBounds(self, axis, frac=1.0, orthoRange=None):
    #     return [0, 0]

    # def viewTransformChanged(self):
    #     self._picture = None
    #     self.prepareGeometryChange()
    #     self._updateLabel()

    def viewTransformChanged(self):
        """
        Called whenever the transformation matrix of the view has changed.
        (eg, the view range has changed or the view was resized)
        """
        GraphicsObject.viewTransformChanged(self)
        self._shape = None  # invalidate shape, recompute later if requested.
        self.update()

    # def _updateLabel(self):
    #     if self.label is None:
    #         return

    #     # find an optimal location for text at the given angle
    #     angle = self.labelAngle * np.pi / 180.
    #     lbr = self.label.boundingRect()
    #     center = lbr.center()
    #     a = abs(np.sin(angle) * lbr.height()*0.5)
    #     b = abs(np.cos(angle) * lbr.width()*0.5)
    #     r = max(self._radii) + 2 + max(a, b)
    #     pos = self.mapFromScene(self.mapToScene(QtCore.QPointF(0, 0)) + r * QtCore.QPointF(np.cos(angle), -np.sin(angle)) - center)
    #     self.label.setPos(pos)

    # def paint(self, p, *args):
    #     if self._picture is None:
    #         self._drawPicture()
    #     self._picture.play(p)

    def paint(self, p, *args):
        p.setBrush(self.currentBrush)
        p.setPen(self.currentPen)

        symbol = Symbols[self.symbol]
        p.drawPath(symbol)
    

    # def _drawPicture(self):
    #     self._picture = QtGui.QPicture()
    #     p = QtGui.QPainter(self._picture)
    #     p.setRenderHint(p.Antialiasing)
        
    #     # Note: could do this with self.pixelLength, but this is faster.
    #     o = self.mapToScene(QtCore.QPointF(0, 0))
    #     px = abs(1.0 / (self.mapToScene(QtCore.QPointF(1, 0)) - o).x())
    #     py = abs(1.0 / (self.mapToScene(QtCore.QPointF(0, 1)) - o).y())
        
    #     r, w, h = self._radii
    #     w = w * px
    #     h = h * py
    #     rx = r * px
    #     ry = r * py
    #     rect = QtCore.QRectF(-rx, -ry, rx*2, ry*2)
    #     p.setPen(self.pen)
    #     p.setBrush(self.brush)
    #     p.drawEllipse(rect)
    #     p.drawLine(Point(-w, 0), Point(w, 0))
    #     p.drawLine(Point(0, -h), Point(0, h))
    #     p.end()
        
    #     bx = max(w, rx)
    #     by = max(h, ry)
    #     self._bounds = QtCore.QRectF(-bx, -by, bx*2, by*2)

    def shape(self):
        if self._shape is None:
            s = self.generateShape()
            if s is None:
                return Symbols[self.symbol]
            self._shape = s
            self.prepareGeometryChange()  ## beware--this can cause the view to adjust, which would immediately invalidate the shape.
        return self._shape

    def mouseDragEvent(self, ev):
        if not self.movable:
            return
        if ev.button() == QtCore.Qt.LeftButton:
            if ev.isStart():
                self.moving = True
                self.targetOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
            ev.accept()
            
            if not self.moving:
                return
                
            self.setPos(self.targetOffset + self.mapToParent(ev.pos()))
            if ev.isFinish():
                self.moving = False
                self.sigDragged.emit(self)

    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.moving = False
            self.sigPositionChanged.emit(self)
            self.sigPositionChangeFinished.emit(self)

    # def hoverEvent(self, ev):
    #     if self.movable:
    #         ev.acceptDrags(QtCore.Qt.LeftButton)
    
    def hoverEvent(self, ev):
        if self.movable and (not ev.isExit()) and ev.acceptDrags(QtCore.Qt.LeftButton):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)

    def setMouseHover(self, hover):
        ## Inform the item that the mouse is(not) hovering over it
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        if hover:
            self.currentBrush = self.hoverBrush
            self.currentPen = self.hoverPen
        else:
            self.currentBrush = self.brush
            self.currentPen = self.pen
        self.update()


    def generateShape(self):
        dt = self.deviceTransform()
        if dt is None:
            self._shape = Symbols[self.symbol]
            return None
        v = dt.map(QtCore.QPointF(1, 0)) - dt.map(QtCore.QPointF(0, 0))
        va = np.arctan2(v.y(), v.x())
        dti = fn.invertQTransform(dt)
        devPos = dt.map(QtCore.QPointF(0, 0))
        tr = QtGui.QTransform()
        tr.translate(devPos.x(), devPos.y())
        tr.rotate(va * 180. / 3.1415926)
        return dti.map(tr.map(Symbols[self.symbol]))
    
    # from CursorItem
    # def buildPath(self):
    #     size = self.radius
    #     self.path = QtGui.QPainterPath()
    #     if self.type == 's':
    #         ang = np.pi/4
    #         sides = 4
    #     elif self.type == 'c':
    #         ang = 0
    #         sides = 24
    #     dt = 2*np.pi/sides
    #     for i in range(0, sides+1):
    #         x = size*cos(ang)
    #         y = size*sin(ang)
    #         ang += dt
    #         if i == 0:
    #             self.path.moveTo(x, y)
    #         else:
    #             self.path.lineTo(x, y)


class TargetLabel(TextItem):

    """
    A TextItem that attaches itself to a TargetTargetItem.
    This class extends TextItem with the following features :
    * Automatically positions adjacent to the target at a fixed position.
    * Automatically reformats text when the target location has changed.
    =============== ==================================================================
    **Arguments:**
    target          The TargetItem to which this label will be attached.
    text            String to display in the label. May contain two {value, value}
                    formatting strings to display the current value of the target.
    =============== ==================================================================
    All extra keyword arguments are passed to TextItem.
    """

    def __init__(self, target, text="", **kwds):
        self.target = target
        self.format = text

        TextItem.__init__(self, **kwds)
        self.setParentItem(target)
        self.anchor = Point(-0.25, 1.25)
        self.valueChanged()

        self.target.sigPositionChanged.connect(self.valueChanged)

    def valueChanged(self):
        x, y = self.target.position()
        self.setText(self.format.format(x, y))