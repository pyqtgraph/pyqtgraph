from ..Qt import QtGui, QtCore
import numpy as np
from ..Point import Point
from .. import functions as fn
from .GraphicsObject import GraphicsObject
from .TextItem import TextItem


class TargetItem(GraphicsObject):
    """Draws a draggable target symbol (circle plus crosshair).

    The size of TargetItem will remain fixed on screen even as the view is zoomed.
    Includes an optional text label.
    """
    sigDragged = QtCore.Signal(object)

    def __init__(self, movable=True, radii=(5, 10, 10), pen=(255, 255, 0), brush=(0, 0, 255, 100)):
        GraphicsObject.__init__(self)
        self._bounds = None
        self._radii = radii
        self._picture = None
        self.movable = movable
        self.moving = False
        self.label = None
        self.labelAngle = 0
        self.pen = fn.mkPen(pen)
        self.brush = fn.mkBrush(brush)

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
        self.labelAngle = angle
        self._updateLabel()

    def boundingRect(self):
        if self._picture is None:
            self._drawPicture()
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
        if self._picture is None:
            self._drawPicture()
        self._picture.play(p)

    def _drawPicture(self):
        self._picture = QtGui.QPicture()
        p = QtGui.QPainter(self._picture)
        p.setRenderHint(p.Antialiasing)
        
        # Note: could do this with self.pixelLength, but this is faster.
        o = self.mapToScene(QtCore.QPointF(0, 0))
        px = abs(1.0 / (self.mapToScene(QtCore.QPointF(1, 0)) - o).x())
        py = abs(1.0 / (self.mapToScene(QtCore.QPointF(0, 1)) - o).y())
        
        r, w, h = self._radii
        w = w * px
        h = h * py
        rx = r * px
        ry = r * py
        rect = QtCore.QRectF(-rx, -ry, rx*2, ry*2)
        p.setPen(self.pen)
        p.setBrush(self.brush)
        p.drawEllipse(rect)
        p.drawLine(Point(-w, 0), Point(w, 0))
        p.drawLine(Point(0, -h), Point(0, h))
        p.end()
        
        bx = max(w, rx)
        by = max(h, ry)
        self._bounds = QtCore.QRectF(-bx, -by, bx*2, by*2)

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

    def hoverEvent(self, ev):
        if self.movable:
            ev.acceptDrags(QtCore.Qt.LeftButton)

