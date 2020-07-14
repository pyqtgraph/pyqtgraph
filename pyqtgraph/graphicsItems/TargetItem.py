from ..Qt import QtGui, QtCore
import numpy as np
from ..Point import Point
from .. import functions as fn
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
from .TextItem import TextItem


class TargetItem(UIGraphicsItem):
    """Draws a draggable target symbol (circle plus crosshair).

    The size of TargetItem will remain fixed on screen even as the view is zoomed.
    Includes an optional text label.
    """
    sigDragged = QtCore.Signal(object)
    sigPositionChanged = QtCore.Signal(object)

    def __init__(self, pos=None, movable=True, radii=(5, 10, 10), pen=None, hoverPen=None, brush=None, hoverBrush=None):
        UIGraphicsItem.__init__(self)
        self._bounds = None
        self._radii = radii
        self._picture = None
        self.movable = movable
        self.moving = False
        self.label = None
        self.labelAngle = 0

        self.mouseHovering = False

        if pen is None:
            pen = (255, 255, 0)
        self.setPen(pen)

        if hoverPen is None:
            hoverPen = (255, 0, 255)
        self.setHoverPen(hoverPen)

        if brush is None:
            brush = (0, 0, 255, 50)
        self.setBrush(brush)

        if hoverBrush is None:
            hoverBrush = (0, 255, 255, 100)
        self.setHoverBrush(hoverBrush)

        self.currentPen = self.pen
        self.currentBrush = self.brush

        self._shape = None
        self.buildPath()

        self._pos = [0, 0]
        if pos is None:
            pos = [0, 0]
        self.setPos(pos)


    def setPos(self, pos):
        if isinstance(pos, (list, tuple)):
            newPos = pos
        elif isinstance(pos, QtCore.QPointF):
            newPos = [pos.x(), pos.y()]
        if self._pos != newPos:
            self._pos = newPos
            GraphicsObject.setPos(self, Point(self._pos))
            self.sigPositionChanged.emit(self)

    def setBrush(self, *args, **kwargs):
        """Set the brush that fills the cursor. Allowable arguments are any that
        are valid for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.brush = fn.mkBrush(*args, **kwargs)
        if not self.mouseHovering:
            self.currentBrush = self.brush
            self.update()

    def setHoverBrush(self, *args, **kwargs):
        """Set the brush that fills the cursor when hovering over it. Allowable
        arguments are any that are valid for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.hoverBrush = fn.mkBrush(*args, **kwargs)
        if self.mouseHovering:
            self.currentBrush = self.hoverBrush
            self.update()
    

    def setPen(self, *args, **kwargs):
        """Set the pen for drawing the cursor. Allowable arguments are any that
        are valid for :func:`mkPen <pyqtgraph.mkPen>`."""
        self.pen = fn.mkPen(*args, **kwargs)
        if not self.mouseHovering:
            self.currentPen = self.pen
            self.update()

    def setHoverPen(self, *args, **kwargs):
        """Set the pen for drawing the cursor when hovering over it. Allowable
        arguments are any that are valid for
        :func:`mkPen <pyqtgraph.mkPen>`."""
        self.hoverPen = fn.mkPen(*args, **kwargs)
        if self.mouseHovering:
            self.currentPen = self.hoverPen
            self.update()

    def setLabel(self, label):
        if label is None:
            if self.label is not None:
                self.label.scene().removeItem(self.label)
                self.label = None
        else:
            if self.label is None:
                self.label = TextItem()
                self.label.setParentItem(self)
            self.label.setText(label)
            self._updateLabel()

    def setLabelAngle(self, angle):
        if self.labelAngle != angle:
            self.labelAngle = angle
            self._updateLabel()

    def boundingRect(self):
        return self._bounds
    
    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        return [0, 0]

    def viewTransformChanged(self):
        self._picture = None
        self.prepareGeometryChange()
        self._updateLabel()

    def _updateLabel(self):
        if self.label is None:
            return

        # find an optimal location for text at the given angle
        angle = self.labelAngle * np.pi / 180.
        lbr = self.label.boundingRect()
        center = lbr.center()
        a = abs(np.sin(angle) * lbr.height()*0.5)
        b = abs(np.cos(angle) * lbr.width()*0.5)
        r = max(self._radii) + 2 + max(a, b)
        pos = self.mapFromScene(self.mapToScene(QtCore.QPointF(0, 0)) + r * QtCore.QPointF(np.cos(angle), -np.sin(angle)) - center)
        self.label.setPos(pos)

    def paint(self, p, *args):
        p.setPen(self.currentPen)
        p.setBrush(self.currentBrush)
        p.drawPath(self.shape())
    
    def shape(self):
        if self._shape is None:
            s = self.generateShape()
            if s is None:
                return self.path
            self._shape = s
            self.prepareGeometryChange()  ## beware--this can cause the view to adjust, which would immediately invalidate the shape.
        return self._shape

    def buildPath(self):
        self.path = p = QtGui.QPainterPath()
        o = self.mapToScene(QtCore.QPointF(0, 0))
        dx = (self.mapToScene(QtCore.QPointF(1, 0)) - o).x()
        dy = (self.mapToScene(QtCore.QPointF(0, 1)) - o).y()
        if dx == 0 or dy == 0:
            p.end()
            self._bounds = QtCore.QRectF()
            return

        r, w, h = self._radii
        rect = QtCore.QRectF(-r, -r, r*2, r*2)
        p.addEllipse(rect)
        p.moveTo(-w, 0)
        p.lineTo(w, 0)
        p.moveTo(0, -h)
        p.lineTo(0, h)
        self._bounds = QtCore.QRectF(-w, -h, w*2, h*2)
    
    def generateShape(self):
        dt = self.deviceTransform()
        if dt is None:
            self._shape = self.path
            return None
        v = dt.map(QtCore.QPointF(1, 0)) - dt.map(QtCore.QPointF(0, 0))
        va = np.arctan2(v.y(), v.x())
        dti = fn.invertQTransform(dt)
        devPos = dt.map(QtCore.QPointF(0, 0))
        tr = QtGui.QTransform()
        tr.translate(devPos.x(), devPos.y())
        tr.rotate(va * 180. / 3.1415926)
        return dti.map(tr.map(self.path))

    def mouseDragEvent(self, ev):
        if not self.movable:
            return
        if ev.button() == QtCore.Qt.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
            ev.accept()
            
            if not self.moving:
                return
                
            self.setPos(self.cursorOffset + self.mapToParent(ev.pos()))
            if ev.isFinish():
                self.moving = False
                self.sigDragged.emit(self)

    def setMouseHover(self, hover):
        ## Inform the item that the mouse is(not) hovering over it
        if self.mouseHovering is hover:
            return
        self.mouseHovering = hover
        if hover:
            self.currentBrush = self.hoverBrush
            self.currentPen = self.hoverPen
        else:
            self.currentBrush = self.brush
            self.currentPen = self.pen
        self.update()

    def hoverEvent(self, ev):
        if self.movable and (not ev.isExit()) and ev.acceptDrags(QtCore.Qt.LeftButton):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)

    def viewTransformChanged(self):
        """
        Called whenever the transformation matrix of the view has changed.
        (eg, the view range has changed or the view was resized)
        """
        GraphicsObject.viewTransformChanged(self)
        self._shape = None  # invalidate shape, recompute later if requested.
        self.update()