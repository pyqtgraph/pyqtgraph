from ..Qt import QtGui, QtCore
from ..Point import Point
from .UIGraphicsItem import UIGraphicsItem
from .TextItem import TextItem
from .. import functions as fn
import numpy as np
import weakref
import math


__all__ = ['InfiniteLine']


def _calcLine(pos, angle, xmin, ymin, xmax, ymax):
    """
    Evaluate the location of the points that delimitates a line into a viewbox
    described by x and y ranges. Depending on the angle value, pos can be a
    float (if angle=0 and 90) or a list of float (x and y coordinates).
    Could be possible to beautify this piece of code.
    New in verson 0.9.11
    """
    if angle == 0:
        x1, y1, x2, y2 = xmin, pos, xmax, pos
    elif angle == 90:
        x1, y1, x2, y2 = pos, ymin, pos, ymax
    else:
        x0, y0 = pos
        tana = math.tan(angle*math.pi/180)
        y1 = tana*(xmin-x0) + y0
        y2 = tana*(xmax-x0) + y0
        if angle > 0:
            y1 = max(y1, ymin)
            y2 = min(y2, ymax)
        else:
            y1 = min(y1, ymax)
            y2 = max(y2, ymin)
        x1 = (y1-y0)/tana + x0
        x2 = (y2-y0)/tana + x0
    p1 = Point(x1, y1)
    p2 = Point(x2, y2)
    return p1, p2


class InfiniteLine(UIGraphicsItem):
    """
    **Bases:** :class:`UIGraphicsItem <pyqtgraph.UIGraphicsItem>`

    Displays a line of infinite length.
    This line may be dragged to indicate a position in data coordinates.

    =============================== ===================================================
    **Signals:**
    sigDragged(self)
    sigPositionChangeFinished(self)
    sigPositionChanged(self)
    =============================== ===================================================

    Major changes have been performed in this class since version 0.9.11. The
    number of methods in the public API has been increased, but the already
    existing methods can be used in the same way.
    """

    sigDragged = QtCore.Signal(object)
    sigPositionChangeFinished = QtCore.Signal(object)
    sigPositionChanged = QtCore.Signal(object)

    def __init__(self, pos=None, angle=90, pen=None, movable=False, bounds=None,
                 hoverPen=None, label=False, textColor=None, textFill=None,
                 textLocation=0.05, textShift=0.5, textFormat="{:.3f}",
                 unit=None, name=None):
        """
        =============== ==================================================================
        **Arguments:**
        pos             Position of the line. This can be a QPointF or a single value for
                        vertical/horizontal lines.
        angle           Angle of line in degrees. 0 is horizontal, 90 is vertical.
        pen             Pen to use when drawing line. Can be any arguments that are valid
                        for :func:`mkPen <pyqtgraph.mkPen>`. Default pen is transparent
                        yellow.
        movable         If True, the line can be dragged to a new position by the user.
        hoverPen        Pen to use when drawing line when hovering over it. Can be any
                        arguments that are valid for :func:`mkPen <pyqtgraph.mkPen>`.
                        Default pen is red.
        bounds          Optional [min, max] bounding values. Bounds are only valid if the
                        line is vertical or horizontal.
        label           if True, a label is displayed next to the line to indicate its
                        location in data coordinates
        textColor       color of the label. Can be any argument fn.mkColor can understand.
        textFill        A brush to use when filling within the border of the text.
        textLocation    A float [0-1] that defines the location of the text.
        textShift       A float [0-1] that defines when the text shifts from one side to
                        another.
        textFormat      Any new python 3 str.format() format.
        unit            If not None, corresponds to the unit to show next to the label
        name            If not None, corresponds to the name of the object
        =============== ==================================================================
        """

        UIGraphicsItem.__init__(self)

        if bounds is None:              ## allowed value boundaries for orthogonal lines
            self.maxRange = [None, None]
        else:
            self.maxRange = bounds
        self.moving = False
        self.mouseHovering = False

        self.angle = ((angle+45) % 180) - 45
        if textColor is None:
            textColor = (200, 200, 200)
        self.textColor = textColor
        self.location = textLocation
        self.shift = textShift
        self.label = label
        self.format = textFormat
        self.unit = unit
        self._name = name

        self.anchorLeft = (1., 0.5)
        self.anchorRight = (0., 0.5)
        self.anchorUp = (0.5, 1.)
        self.anchorDown = (0.5, 0.)
        self.text = TextItem(fill=textFill)
        self.text.setParentItem(self) # important
        self.p = [0, 0]

        if pen is None:
            pen = (200, 200, 100)

        self.setPen(pen)

        if hoverPen is None:
            self.setHoverPen(color=(255,0,0), width=self.pen.width())
        else:
            self.setHoverPen(hoverPen)
        self.currentPen = self.pen

        self.setMovable(movable)

        if pos is None:
            pos = Point(0,0)
        self.setPos(pos)

        if (self.angle == 0 or self.angle == 90) and self.label:
            self.text.show()
        else:
            self.text.hide()


    def setMovable(self, m):
        """Set whether the line is movable by the user."""
        self.movable = m
        self.setAcceptHoverEvents(m)

    def setBounds(self, bounds):
        """Set the (minimum, maximum) allowable values when dragging."""
        self.maxRange = bounds
        self.setValue(self.value())

    def setPen(self, *args, **kwargs):
        """Set the pen for drawing the line. Allowable arguments are any that are valid
        for :func:`mkPen <pyqtgraph.mkPen>`."""
        self.pen = fn.mkPen(*args, **kwargs)
        if not self.mouseHovering:
            self.currentPen = self.pen
            self.update()

    def setHoverPen(self, *args, **kwargs):
        """Set the pen for drawing the line while the mouse hovers over it.
        Allowable arguments are any that are valid
        for :func:`mkPen <pyqtgraph.mkPen>`.

        If the line is not movable, then hovering is also disabled.

        Added in version 0.9.9."""
        self.hoverPen = fn.mkPen(*args, **kwargs)
        if self.mouseHovering:
            self.currentPen = self.hoverPen
            self.update()

    def setAngle(self, angle):
        """
        Takes angle argument in degrees.
        0 is horizontal; 90 is vertical.

        Note that the use of value() and setValue() changes if the line is
        not vertical or horizontal.
        """
        self.angle = ((angle+45) % 180) - 45   ##  -45 <= angle < 135
        # self.resetTransform()   # no longer needed since version 0.9.11
        # self.rotate(self.angle) # no longer needed since version 0.9.11
        if (self.angle == 0 or self.angle == 90) and self.label:
            self.text.show()
        else:
            self.text.hide()
        self.update()

    def setPos(self, pos):

        if type(pos) in [list, tuple]:
            newPos = pos
        elif isinstance(pos, QtCore.QPointF):
            newPos = [pos.x(), pos.y()]
        else:
            if self.angle == 90:
                newPos = [pos, 0]
            elif self.angle == 0:
                newPos = [0, pos]
            else:
                raise Exception("Must specify 2D coordinate for non-orthogonal lines.")

        ## check bounds (only works for orthogonal lines)
        if self.angle == 90:
            if self.maxRange[0] is not None:
                newPos[0] = max(newPos[0], self.maxRange[0])
            if self.maxRange[1] is not None:
                newPos[0] = min(newPos[0], self.maxRange[1])
        elif self.angle == 0:
            if self.maxRange[0] is not None:
                newPos[1] = max(newPos[1], self.maxRange[0])
            if self.maxRange[1] is not None:
                newPos[1] = min(newPos[1], self.maxRange[1])

        if self.p != newPos:
            self.p = newPos
            # UIGraphicsItem.setPos(self, Point(self.p)) # thanks Sylvain!
            self.update()
            self.sigPositionChanged.emit(self)

    def getXPos(self):
        return self.p[0]

    def getYPos(self):
        return self.p[1]

    def getPos(self):
        return self.p

    def value(self):
        """Return the value of the line. Will be a single number for horizontal and
        vertical lines, and a list of [x,y] values for diagonal lines."""
        if self.angle%180 == 0:
            return self.getYPos()
        elif self.angle%180 == 90:
            return self.getXPos()
        else:
            return self.getPos()

    def setValue(self, v):
        """Set the position of the line. If line is horizontal or vertical, v can be
        a single value. Otherwise, a 2D coordinate must be specified (list, tuple and
        QPointF are all acceptable)."""
        self.setPos(v)

    ## broken in 4.7
    #def itemChange(self, change, val):
        #if change in [self.ItemScenePositionHasChanged, self.ItemSceneHasChanged]:
            #self.updateLine()
            #print "update", change
            #print self.getBoundingParents()
        #else:
            #print "ignore", change
        #return GraphicsObject.itemChange(self, change, val)

    def boundingRect(self):
        br = UIGraphicsItem.boundingRect(self) # directly in viewBox coordinates
        # we need to limit the boundingRect to the appropriate value.
        val = self.value()
        if self.angle == 0: # horizontal line
            self._p1, self._p2 = _calcLine(val, 0, *br.getCoords())
            px = self.pixelLength(direction=Point(1,0), ortho=True)  ## get pixel length orthogonal to the line
            if px is None:
                px = 0
            w = (max(4, self.pen.width()/2, self.hoverPen.width()/2)+1) * px
            o1, o2 = _calcLine(val-w, 0, *br.getCoords())
            o3, o4 = _calcLine(val+w, 0, *br.getCoords())
        elif self.angle == 90: # vertical line
            self._p1, self._p2 = _calcLine(val, 90, *br.getCoords())
            px = self.pixelLength(direction=Point(0,1), ortho=True)  ## get pixel length orthogonal to the line
            if px is None:
                px = 0
            w = (max(4, self.pen.width()/2, self.hoverPen.width()/2)+1) * px
            o1, o2 = _calcLine(val-w, 90, *br.getCoords())
            o3, o4 = _calcLine(val+w, 90, *br.getCoords())
        else: # oblique line
            self._p1, self._p2 = _calcLine(val, self.angle, *br.getCoords())
            pxy = self.pixelLength(direction=Point(0,1), ortho=True)
            if pxy is None:
                pxy = 0
            wy = (max(4, self.pen.width()/2, self.hoverPen.width()/2)+1) * pxy
            pxx = self.pixelLength(direction=Point(1,0), ortho=True)
            if pxx is None:
                pxx = 0
            wx = (max(4, self.pen.width()/2, self.hoverPen.width()/2)+1) * pxx
            o1, o2 = _calcLine([val[0]-wy, val[1]-wx], self.angle, *br.getCoords())
            o3, o4 = _calcLine([val[0]+wy, val[1]+wx], self.angle, *br.getCoords())
        self._polygon = QtGui.QPolygonF([o1, o2, o4, o3])
        br = self._polygon.boundingRect()
        return br.normalized()


    def shape(self):
        # returns a QPainterPath. Needed when the item is non rectangular if
        # accurate mouse click detection is required.
        # New in version 0.9.11
        qpp = QtGui.QPainterPath()
        qpp.addPolygon(self._polygon)
        return qpp

    def paint(self, p, *args):
        br = self.boundingRect()
        p.setPen(self.currentPen)
        p.drawLine(self._p1, self._p2)

    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        if axis == 0:
            return None   ## x axis should never be auto-scaled
        else:
            return (0,0)

    def mouseDragEvent(self, ev):
        if self.movable and ev.button() == QtCore.Qt.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.value() - ev.buttonDownPos()
                self.startPosition = self.value()
            ev.accept()

            if not self.moving:
                return

            self.setPos(self.cursorOffset + ev.pos())
            self.prepareGeometryChange() # new in version 0.9.11
            self.sigDragged.emit(self)
            if ev.isFinish():
                self.moving = False
                self.sigPositionChangeFinished.emit(self)

    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.setPos(self.startPosition)
            self.moving = False
            self.sigDragged.emit(self)
            self.sigPositionChangeFinished.emit(self)

    def hoverEvent(self, ev):
        if (not ev.isExit()) and self.movable and ev.acceptDrags(QtCore.Qt.LeftButton):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)

    def setMouseHover(self, hover):
        ## Inform the item that the mouse is (not) hovering over it
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        if hover:
            self.currentPen = self.hoverPen
        else:
            self.currentPen = self.pen
        self.update()

    def update(self):
        # new in version 0.9.11
        UIGraphicsItem.update(self)
        br = UIGraphicsItem.boundingRect(self) # directly in viewBox coordinates
        xmin, ymin, xmax, ymax = br.getCoords()
        if self.angle == 90:  # vertical line
            diffX = xmax-xmin
            diffMin = self.value()-xmin
            limInf = self.shift*diffX
            ypos = ymin+self.location*(ymax-ymin)
            if diffMin < limInf:
                self.text.anchor = Point(self.anchorRight)
            else:
                self.text.anchor = Point(self.anchorLeft)
            fmt = " x = " + self.format
            if self.unit is not None:
                fmt = fmt + self.unit
            self.text.setText(fmt.format(self.value()), color=self.textColor)
            self.text.setPos(self.value(), ypos)
        elif self.angle == 0:  # horizontal line
            diffY = ymax-ymin
            diffMin = self.value()-ymin
            limInf = self.shift*(ymax-ymin)
            xpos = xmin+self.location*(xmax-xmin)
            if diffMin < limInf:
                self.text.anchor = Point(self.anchorUp)
            else:
                self.text.anchor = Point(self.anchorDown)
            fmt = " y = " + self.format
            if self.unit is not None:
                fmt = fmt + self.unit
            self.text.setText(fmt.format(self.value()), color=self.textColor)
            self.text.setPos(xpos, self.value())

    def showLabel(self, state):
        """
        Display or not the label indicating the location of the line in data
        coordinates.

        ==============   ==============================================
        **Arguments:**
        state            If True, the label is shown. Otherwise, it is hidden.
        ==============   ==============================================
        """
        if state:
            self.text.show()
        else:
            self.text.hide()
        self.update()

    def setTextLocation(self, param):
        """
        Set the location of the label. param is a list of two values.
        param[0] defines the location of the label along the axis and
        param[1] defines the shift value (defines the condition where the
        label shifts from one side of the line to the other one).
        New in version 0.9.11
        ==============   ==============================================
        **Arguments:**
        param              list of parameters.
        ==============   ==============================================
        """
        if len(param) != 2: # check that the input data are correct
            return
        self.location = np.clip(param[0], 0, 1)
        self.shift = np.clip(param[1], 0, 1)
        self.update()

    def setFormat(self, format):
        """
        Set the format of the label used to indicate the location of the line.


        ==============   ==============================================
        **Arguments:**
        format           Any format compatible with the new python
                         str.format() format style.
        ==============   ==============================================
        """
        self.format = format
        self.update()

    def setUnit(self, unit):
        """
        Set the unit of the label used to indicate the location of the line.


        ==============   ==============================================
        **Arguments:**
        unit             Any string.
        ==============   ==============================================
        """
        self.unit = unit
        self.update()

    def setName(self, name):
        self._name = name

    def name(self):
        return self._name
