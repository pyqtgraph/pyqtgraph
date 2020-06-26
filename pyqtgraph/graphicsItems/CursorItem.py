# -*- coding: utf-8 -*-

from ..Qt import QtGui, QtCore
from .UIGraphicsItem import UIGraphicsItem
from .GraphicsObject import GraphicsObject
from .TextItem import TextItem
from .. import functions as fn
from ..Point import Point
import numpy as np
from math import cos, sin


__all__ = ['CursorItem', 'CursorLabel']


class CursorItem(UIGraphicsItem):
    """
    **Bases:** :class:`UIGraphicsItem <pyqtgraph.UIGraphicsItem>`
    Displays a cursor.
    This cursor may be dragged to indicate a position in data coordinates.
    =============================== ===================================================
    **Signals:**
    sigPositionChangeFinished(self)
    sigPositionChanged(self)
    =============================== ===================================================
    """

    sigPositionChanged = QtCore.pyqtSignal(object)  # self
    sigPositionChangeFinished = QtCore.pyqtSignal(object)  # self

    def __init__(self, pos=None, radius=5, cursor='s', pen=None, brush=None,
                 movable=True, hoverPen=None, hoverBrush=None, label=None,
                 labelOpts=None, name=None, parent=None):
        """
        =============== ==================================================================
        **Arguments:**
        pos             Position of the cursor. This can be a list of QPointF or a list
                        of floats
        radius          Size of the cursor in pixel
        cursor          String that defines the shape of the cursor (can take the
                        following values for the moment : 's' (square), 'c' (circle))
        pen             Pen to use when drawing line. Can be any arguments that are valid
                        for :func:`mkPen <pyqtgraph.mkPen>`. Default pen is transparent
                        yellow.
        brush           Defines the brush that fill the cursor. Can be any arguments
                        that is valid for :func:`mkBrush<pyqtgraph.mkBrush>`. Default
                        is transparent blue.
        movable         If True, the cursor can be dragged to a new position by the user.
        hoverPen        Pen to use when drawing cursor when hovering over it. Can be any
                        arguments that are valid for :func:`mkPen <pyqtgraph.mkPen>`.
                        Default pen is red.
        hoverBrush      Brush to use to fill the cursor when hovering over it. Can be any
                        arguments that is valid for :func:`mkBrush<pyqtgraph.mkBrush>`.
                        Default is transparent blue.
        label           Text to be displayed in a label attached to the cursor, or
                        None to show no label (default is None). May optionally
                        include formatting strings to display the cursor value.
        labelOpts       A dict of keyword arguments to use when constructing the
                        text label. See :class:`CursorLabel`.
        name            Name of the item
        =============== ==================================================================
        """
        UIGraphicsItem.__init__(self, parent=parent)
        self.radius = radius
        self.type = cursor

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

        self.buildPath()
        self._shape = None

        if label is not None:
            labelOpts = {} if labelOpts is None else labelOpts
            self.label = CursorLabel(self, text=label, **labelOpts)

        self.setPos(pos)

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

    def setMovable(self, m):
        """Set whether the line is movable by the user."""
        self.movable = m
        self.setAcceptHoverEvents(m)

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
        # self.prepareGeometryChange()

        if ev.isFinish():
            self.moving = False
            self.sigPositionChangeFinished.emit(self)

    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.moving = False
            self.sigPositionChanged.emit(self)
            self.sigPositionChangeFinished.emit(self)

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

    def boundingRect(self):
        return self.shape().boundingRect()

    def paint(self, p, *args):
        p.setBrush(self.currentBrush)
        p.setPen(self.currentPen)
        p.drawPath(self.shape())

    def shape(self):
        if self._shape is None:
            s = self.generateShape()
            if s is None:
                return self.path
            self._shape = s
            self.prepareGeometryChange()  ## beware--this can cause the view to adjust, which would immediately invalidate the shape.
        return self._shape

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

    def buildPath(self):
        size = self.radius
        self.path = QtGui.QPainterPath()
        if self.type == 's':
            ang = np.pi/4
            sides = 4
        elif self.type == 'c':
            ang = 0
            sides = 24
        dt = 2*np.pi/sides
        for i in range(0, sides+1):
            x = size*cos(ang)
            y = size*sin(ang)
            ang += dt
            if i == 0:
                self.path.moveTo(x, y)
            else:
                self.path.lineTo(x, y)

    def viewTransformChanged(self):
        """
        Called whenever the transformation matrix of the view has changed.
        (eg, the view range has changed or the view was resized)
        """
        GraphicsObject.viewTransformChanged(self)
        self._shape = None  # invalidate shape, recompute later if requested.
        self.update()

    def setName(self, name):
        self._name = name

    def name(self):
        return self._name


class CursorLabel(TextItem):

    """
    A TextItem that attaches itself to a CursorItem.
    This class extends TextItem with the following features :
    * Automatically positions adjacent to the cursor at a fixed position.
    * Automatically reformats text when the cursor location has changed.
    =============== ==================================================================
    **Arguments:**
    cursor          The CursorItem to which this label will be attached.
    text            String to display in the label. May contain two {value, value}
                    formatting strings to display the current value of the cursor.
    =============== ==================================================================
    All extra keyword arguments are passed to TextItem.
    """

    def __init__(self, cursor, text="", **kwds):
        self.cursor = cursor
        self.format = text

        TextItem.__init__(self, **kwds)
        self.setParentItem(cursor)
        self.anchor = Point(-0.25, 1.25)
        self.valueChanged()

        self.cursor.sigPositionChanged.connect(self.valueChanged)

    def valueChanged(self):
        x, y = self.cursor.position()
        self.setText(self.format.format(x, y))