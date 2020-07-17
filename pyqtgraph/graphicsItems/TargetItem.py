from math import sin, cos

from ..Qt import QtGui, QtCore
import numpy as np
from ..Point import Point
from .. import functions as fn
from .GraphicsObject import GraphicsObject
from .TextItem import TextItem


def makeTarget(radii=(5, 10, 10)):
    path = QtGui.QPainterPath()
    r, w, h = radii
    rect = QtCore.QRectF(-r, -r, r*2, r*2)
    path.addEllipse(rect)
    path.moveTo(-w, 0)
    path.lineTo(w, 0)
    path.moveTo(0, -h)
    path.lineTo(0, h)
    return path

class TargetItem(GraphicsObject):
    """Draws a draggable target symbol (circle plus crosshair).

    The size of TargetItem will remain fixed on screen even as the view is zoomed.
    Includes an optional text label.
    """
    sigPositionChanged = QtCore.Signal(object)
    sigPositionChangeFinished = QtCore.Signal(object)

    def __init__(self, pos=None, movable=True, pen=None, hoverPen=None, brush=None, hoverBrush=None, path=None, label=None, labelOpts=None, labelAngle=None):
        GraphicsObject.__init__(self)
        self._bounds = None
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

        self._pos = (0, 0)
        if pos is None:
            pos = (0, 0)
        self.setPos(pos)

        if path is None:
            path = makeTarget()
        self.setPath(path)

        if label is not None:
            self.setLabel(label, labelOpts)
        
        if labelAngle is not None:
            self.label.angle = labelAngle
            self._updateLabel()

        # This is the CursorLabel method
        # if label is not None:
        #     labelOpts = {} if labelOpts is None else labelOpts
        #     self.label = TargetLabel(self, text=label, **labelOpts)


    def setPos(self, pos):
        if isinstance(pos, (list, tuple)):
            newPos = tuple(pos)
        elif isinstance(pos, QtCore.QPointF):
            newPos = (pos.x(), pos.y())
        else:
            raise TypeError
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

    def setLabel(self, label=None, labelOpts=None):
        if label is None and self.label is not None:
            self.label.scene().removeItem(self.label)
            self.label = None
        else:
            if self.label is None:
                labelOpts = {} if labelOpts is None else labelOpts
                self.label = TargetLabel(self, label, **labelOpts)
            else:
                self.label.setText(label)
            self._updateLabel()
            # self.label.update()

    def setLabelAngle(self, angle):
        if self.label is None:
            return
        if self.label.angle != angle:
            self.label.angle = angle
            self._updateLabel()
            # self.label.update()

    def boundingRect(self):
        return self._bounds
    
    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        return [0, 0]

    def paint(self, p, *args):
        p.setPen(self.currentPen)
        p.setBrush(self.currentBrush)
        p.drawPath(self.shape())

    def setPath(self, path):
        o = self.mapToScene(QtCore.QPointF(0, 0))
        dx = (self.mapToScene(QtCore.QPointF(1, 0)) - o).x()
        dy = (self.mapToScene(QtCore.QPointF(0, 1)) - o).y()
        if dx == 0 or dy == 0:
            self._bounds = QtCore.QRectF()
            return
        self._path = path
        self._bounds = path.boundingRect()
        return None

    def shape(self):
        if self._shape is None:
            s = self.generateShape()
            if s is None:
                return self._path
            self._shape = s
            self.prepareGeometryChange()  ## beware--this can cause the view to adjust, which would immediately invalidate the shape.
        return self._shape
    
    def generateShape(self):
        dt = self.deviceTransform()
        if dt is None:
            self._shape = self._path
            return None
        v = dt.map(QtCore.QPointF(1, 0)) - dt.map(QtCore.QPointF(0, 0))
        va = np.arctan2(v.y(), v.x())
        dti = fn.invertQTransform(dt)
        devPos = dt.map(QtCore.QPointF(0, 0))
        tr = QtGui.QTransform()
        tr.translate(devPos.x(), devPos.y())
        tr.rotate(va * 180. / 3.1415926)
        return dti.map(tr.map(self._path))

    def mouseDragEvent(self, ev):
        if not self.movable or int(ev.button() & QtCore.Qt.LeftButton) == 0:
            return
        ev.accept()
        if ev.isStart():
            self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
            self.moving = True

        if not self.moving:
            return

        self.setPos(self.cursorOffset+self.mapToParent(ev.pos()))

        if ev.isFinish():
            self.moving = False
            self.sigPositionChangeFinished.emit(self)
    
    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.moving = False
            self.sigPositionChanged.emit(self)
            self.sigPositionChangeFinished.emit(self)

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
        self._updateLabel()
        # self.prepareGeometryChange()
        # if self.label is not None:
        #     self._updateLabel()

    def position(self):
        return self._pos

    def _updateLabel(self):
        if self.label is None:
            return
        self.label.updatePosition()

class TargetLabel(TextItem):
    """
    A TextItem that attaches itself to a CursorItem.
    This class extends TextItem with the following features :
    * Automatically positions adjacent to the cursor at a fixed position.
    * Automatically reformats text when the cursor location has changed.
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
        self.angle = 0
        self.valueChanged()
        self.target.sigPositionChanged.connect(self.valueChanged)

    def valueChanged(self):
        x, y = self.target.position()
        self.setText(self.format.format(x, y))


    def updatePosition(self):
        angle = self.angle * np.pi / 180.
        labelBoundingRect = self.boundingRect()
        center = labelBoundingRect.center()
        print(labelBoundingRect)
        a = abs(sin(angle) * labelBoundingRect.height()*0.5)
        b = abs(cos(angle) * labelBoundingRect.width()*0.5)
        targetItemBoundingRect = self.target._path.boundingRect()
        r = max(targetItemBoundingRect.width(), targetItemBoundingRect.height()) + 2 + max(a, b)
        # pos = self.mapFromScene(self.mapToScene(QtCore.QPointF(0, 0)) + r * QtCore.QPointF(cos(angle), -sin(angle)) - center)
        pos = r * QtCore.QPointF(cos(angle), -sin(angle)) - center
        print(f"center: ({center.x()}, {center.y()}) \t a: {a} \t b: {b} \t r: {r} \t pos: ({pos.x()}, {pos.y()})")
        self.setPos(pos)
