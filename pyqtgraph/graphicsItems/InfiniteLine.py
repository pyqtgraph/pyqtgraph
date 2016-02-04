from ..Qt import QtGui, QtCore
from ..Point import Point
from .GraphicsObject import GraphicsObject
#from UIGraphicsItem import UIGraphicsItem
from .TextItem import TextItem
from .ViewBox import ViewBox
from .. import functions as fn
import numpy as np
import weakref


__all__ = ['InfiniteLine']


class InfiniteLine(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    Displays a line of infinite length.
    This line may be dragged to indicate a position in data coordinates.

    =============================== ===================================================
    **Signals:**
    sigDragged(self)
    sigPositionChangeFinished(self)
    sigPositionChanged(self)
    =============================== ===================================================
    """

    sigDragged = QtCore.Signal(object)
    sigPositionChangeFinished = QtCore.Signal(object)
    sigPositionChanged = QtCore.Signal(object)

    def __init__(self, pos=None, angle=90, pen=None, movable=False, bounds=None,
                 hoverPen=None, label=False, textColor=None, textFill=None,
                 textLocation=[0.05,0.5], textFormat="{:.3f}",
                 suffix=None, name='InfiniteLine'):
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
        textLocation    list  where list[0] defines the location of the text (if
                        vertical, a 0 value means that the textItem is on the bottom
                        axis, and a 1 value means that thet TextItem is on the top
                        axis, same thing if horizontal) and list[1] defines when the
                        text shifts from one side to the other side of the line.
        textFormat      Any new python 3 str.format() format.
        suffix          If not None, corresponds to the unit to show next to the label
        name            name of the item
        =============== ==================================================================
        """

        GraphicsObject.__init__(self)

        if bounds is None:              ## allowed value boundaries for orthogonal lines
            self.maxRange = [None, None]
        else:
            self.maxRange = bounds
        self.moving = False
        self.setMovable(movable)
        self.mouseHovering = False
        self.p = [0, 0]
        self.setAngle(angle)

        if textColor is None:
            textColor = (200, 200, 100)
        self.textColor = textColor
        self.textFill = textFill
        self.textLocation = textLocation
        self.suffix = suffix

        if (self.angle == 0 or self.angle == 90) and label:
            self.textItem = TextItem(fill=textFill)
            self.textItem.setParentItem(self)
        else:
            self.textItem = None

        self.anchorLeft = (1., 0.5)
        self.anchorRight = (0., 0.5)
        self.anchorUp = (0.5, 1.)
        self.anchorDown = (0.5, 0.)

        if pos is None:
            pos = Point(0,0)
        self.setPos(pos)

        if pen is None:
            pen = (200, 200, 100)
        self.setPen(pen)
        if hoverPen is None:
            self.setHoverPen(color=(255,0,0), width=self.pen.width())
        else:
            self.setHoverPen(hoverPen)
        self.currentPen = self.pen

        self.format = textFormat

        self._name = name

        # Cache complex value for drawing speed-up
        self._line = None
        self._boundingRect = None

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
        self.resetTransform()
        self.rotate(self.angle)
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
            self._invalidateCache()
            GraphicsObject.setPos(self, Point(self.p))

            if self.textItem is not None and self.getViewBox() is not None and isinstance(self.getViewBox(), ViewBox):
                self.updateTextPosition()

            self.update()
            self.sigPositionChanged.emit(self)

    def updateTextPosition(self):
        """
        Update the location of the textItem. Called only if a textItem is
        requested and if the item has already been added to a PlotItem.
        """
        rangeX, rangeY = self.getViewBox().viewRange()
        xmin, xmax = rangeX
        ymin, ymax = rangeY
        if self.angle == 90:  # vertical line
            diffMin = self.value()-xmin
            limInf = self.textLocation[1]*(xmax-xmin)
            ypos = ymin+self.textLocation[0]*(ymax-ymin)
            if diffMin < limInf:
                self.textItem.anchor = Point(self.anchorRight)
            else:
                self.textItem.anchor = Point(self.anchorLeft)
            fmt = " x = " + self.format
            if self.suffix is not None:
                fmt = fmt + self.suffix
            self.textItem.setText(fmt.format(self.value()), color=self.textColor)
        elif self.angle == 0:  # horizontal line
            diffMin = self.value()-ymin
            limInf = self.textLocation[1]*(ymax-ymin)
            xpos = xmin+self.textLocation[0]*(xmax-xmin)
            if diffMin < limInf:
                self.textItem.anchor = Point(self.anchorUp)
            else:
                self.textItem.anchor = Point(self.anchorDown)
            fmt = " y = " + self.format
            if self.suffix is not None:
                fmt = fmt + self.suffix
            self.textItem.setText(fmt.format(self.value()), color=self.textColor)

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

    def _invalidateCache(self):
        self._line = None
        self._boundingRect = None

    def boundingRect(self):
        if self._boundingRect is None:
            #br = UIGraphicsItem.boundingRect(self)
            br = self.viewRect()
            ## add a 4-pixel radius around the line for mouse interaction.

            px = self.pixelLength(direction=Point(1,0), ortho=True)  ## get pixel length orthogonal to the line
            if px is None:
                px = 0
            w = (max(4, self.pen.width()/2, self.hoverPen.width()/2)+1) * px
            br.setBottom(-w)
            br.setTop(w)
            br = br.normalized()
            self._boundingRect = br
            self._line = QtCore.QLineF(br.right(), 0.0, br.left(), 0.0)
        return self._boundingRect

    def paint(self, p, *args):
        p.setPen(self.currentPen)
        p.drawLine(self._line)

    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        if axis == 0:
            return None   ## x axis should never be auto-scaled
        else:
            return (0,0)

    def mouseDragEvent(self, ev):
        if self.movable and ev.button() == QtCore.Qt.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
            ev.accept()

            if not self.moving:
                return

            self.setPos(self.cursorOffset + self.mapToParent(ev.pos()))
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

    def viewTransformChanged(self):
        """
        Called whenever the transformation matrix of the view has changed.
        (eg, the view range has changed or the view was resized)
        """
        self._invalidateCache()

        if self.getViewBox() is not None and isinstance(self.getViewBox(), ViewBox) and self.textItem is not None:
            self.updateTextPosition()
        #GraphicsObject.viewTransformChanged(self)

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
            self.textItem = TextItem(fill=self.textFill)
            self.textItem.setParentItem(self)
            self.viewTransformChanged()
        else:
            self.textItem = None


    def setTextLocation(self, loc):
        """
        Set the parameters that defines the location of the textItem with respect
        to a specific axis. If the line is vertical, the location is based on the
        normalized range of the yaxis. Otherwise, it is based on the normalized
        range of the xaxis.
        loc[0] defines the location of the text along the infiniteLine
        loc[1] defines the location when the label shifts from one side of then
        infiniteLine to the other.
        """
        self.textLocation = [np.clip(loc[0], 0, 1), np.clip(loc[1], 0, 1)]
        self.update()

    def setName(self, name):
        self._name = name

    def name(self):
        return self._name
