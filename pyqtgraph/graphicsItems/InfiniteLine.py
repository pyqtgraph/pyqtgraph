from ..Qt import QtGui, QtCore
from ..Point import Point
from .GraphicsObject import GraphicsObject
from .TextItem import TextItem
from .ViewBox import ViewBox
from .. import functions as fn
import numpy as np
import weakref


__all__ = ['InfiniteLine', 'InfLineLabel']


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
                 hoverPen=None, label=None, labelOpts=None, span=(0, 1), markers=None, 
                 name=None):
        """
        =============== ==================================================================
        **Arguments:**
        pos             Position of the line. This can be a QPointF or a single value for
                        vertical/horizontal lines.
        angle           Angle of line in degrees. 0 is horizontal, 90 is vertical.
        pen             Pen to use when drawing line. Can be any arguments that are valid
                        for :func:`mkPen <pyqtgraph.mkPen>`. Default pen is transparent
                        yellow.
        hoverPen        Pen to use when the mouse cursor hovers over the line. 
                        Only used when movable=True.
        movable         If True, the line can be dragged to a new position by the user.
        bounds          Optional [min, max] bounding values. Bounds are only valid if the
                        line is vertical or horizontal.
        hoverPen        Pen to use when drawing line when hovering over it. Can be any
                        arguments that are valid for :func:`mkPen <pyqtgraph.mkPen>`.
                        Default pen is red.
        label           Text to be displayed in a label attached to the line, or
                        None to show no label (default is None). May optionally
                        include formatting strings to display the line value.
        labelOpts       A dict of keyword arguments to use when constructing the
                        text label. See :class:`InfLineLabel`.
        span            Optional tuple (min, max) giving the range over the view to draw
                        the line. For example, with a vertical line, use span=(0.5, 1)
                        to draw only on the top half of the view.
        markers         List of (marker, position, size) tuples, one per marker to display
                        on the line. See the addMarker method.
        name            Name of the item
        =============== ==================================================================
        """
        self._boundingRect = None

        self._name = name

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
        
        self.span = span
        self.currentPen = self.pen

        self.markers = []
        self._maxMarkerSize = 0
        if markers is not None:
            for m in markers:
                self.addMarker(*m)
                
        # Cache variables for managing bounds
        self._endPoints = [0, 1] # 
        self._bounds = None
        self._lastViewSize = None
        
        if label is not None:
            labelOpts = {} if labelOpts is None else labelOpts
            self.label = InfLineLabel(self, text=label, **labelOpts)

    def setMovable(self, m):
        """Set whether the line is movable by the user."""
        self.movable = m
        self.setAcceptHoverEvents(m)

    def setBounds(self, bounds):
        """Set the (minimum, maximum) allowable values when dragging."""
        self.maxRange = bounds
        self.setValue(self.value())
        
    def bounds(self):
        """Return the (minimum, maximum) values allowed when dragging.
        """
        return self.maxRange[:]
        
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
        # If user did not supply a width, then copy it from pen
        widthSpecified = ((len(args) == 1 and 
                           (isinstance(args[0], QtGui.QPen) or
                           (isinstance(args[0], dict) and 'width' in args[0]))
                          ) or 'width' in kwargs)
        self.hoverPen = fn.mkPen(*args, **kwargs)
        if not widthSpecified:
            self.hoverPen.setWidth(self.pen.width())
            
        if self.mouseHovering:
            self.currentPen = self.hoverPen
            self.update()
        
    def addMarker(self, marker, position=0.5, size=10.0):
        """Add a marker to be displayed on the line. 
        
        ============= =========================================================
        **Arguments**
        marker        String indicating the style of marker to add:
                      '<|', '|>', '>|', '|<', '<|>', '>|<', '^', 'v', 'o'
        position      Position (0.0-1.0) along the visible extent of the line
                      to place the marker. Default is 0.5.
        size          Size of the marker in pixels. Default is 10.0.
        ============= =========================================================
        """
        path = QtGui.QPainterPath()
        if marker == 'o': 
            path.addEllipse(QtCore.QRectF(-0.5, -0.5, 1, 1))
        if '<|' in marker:
            p = QtGui.QPolygonF([Point(0.5, 0), Point(0, -0.5), Point(-0.5, 0)])
            path.addPolygon(p)
            path.closeSubpath()
        if '|>' in marker:
            p = QtGui.QPolygonF([Point(0.5, 0), Point(0, 0.5), Point(-0.5, 0)])
            path.addPolygon(p)
            path.closeSubpath()
        if '>|' in marker:
            p = QtGui.QPolygonF([Point(0.5, -0.5), Point(0, 0), Point(-0.5, -0.5)])
            path.addPolygon(p)
            path.closeSubpath()
        if '|<' in marker:
            p = QtGui.QPolygonF([Point(0.5, 0.5), Point(0, 0), Point(-0.5, 0.5)])
            path.addPolygon(p)
            path.closeSubpath()
        if '^' in marker:
            p = QtGui.QPolygonF([Point(0, -0.5), Point(0.5, 0), Point(0, 0.5)])
            path.addPolygon(p)
            path.closeSubpath()
        if 'v' in marker:
            p = QtGui.QPolygonF([Point(0, -0.5), Point(-0.5, 0), Point(0, 0.5)])
            path.addPolygon(p)
            path.closeSubpath()
        
        self.markers.append((path, position, size))
        self._maxMarkerSize = max([m[2] / 2. for m in self.markers])
        self.update()

    def clearMarkers(self):
        """ Remove all markers from this line.
        """
        self.markers = []
        self._maxMarkerSize = 0
        self.update()
        
    def setAngle(self, angle):
        """
        Takes angle argument in degrees.
        0 is horizontal; 90 is vertical.

        Note that the use of value() and setValue() changes if the line is
        not vertical or horizontal.
        """
        self.angle = angle #((angle+45) % 180) - 45   ##  -45 <= angle < 135
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
    
    def setSpan(self, mn, mx):
        if self.span != (mn, mx):
            self.span = (mn, mx)
            self.update()

    def _invalidateCache(self):
        self._boundingRect = None

    def _computeBoundingRect(self):
        #br = UIGraphicsItem.boundingRect(self)
        vr = self.viewRect()  # bounds of containing ViewBox mapped to local coords.
        if vr is None:
            return QtCore.QRectF()
        
        ## add a 4-pixel radius around the line for mouse interaction.
        
        px = self.pixelLength(direction=Point(1,0), ortho=True)  ## get pixel length orthogonal to the line
        if px is None:
            px = 0
        pw = max(self.pen.width() / 2, self.hoverPen.width() / 2)
        w = max(4, self._maxMarkerSize + pw) + 1
        w = w * px
        br = QtCore.QRectF(vr)
        br.setBottom(-w)
        br.setTop(w)

        length = br.width()
        left = br.left() + length * self.span[0]
        right = br.left() + length * self.span[1]
        br.setLeft(left)
        br.setRight(right)
        br = br.normalized()
        
        vs = self.getViewBox().size()
        
        if self._bounds != br or self._lastViewSize != vs:
            self._bounds = br
            self._lastViewSize = vs
            self.prepareGeometryChange()
        
        self._endPoints = (left, right)
        self._lastViewRect = vr
        
        return self._bounds

    def boundingRect(self):
        if self._boundingRect is None:
            self._boundingRect = self._computeBoundingRect()
        return self._boundingRect

    def paint(self, p, *args):
        p.setRenderHint(p.Antialiasing)
        
        left, right = self._endPoints
        pen = self.currentPen
        pen.setJoinStyle(QtCore.Qt.MiterJoin)
        p.setPen(pen)
        p.drawLine(Point(left, 0), Point(right, 0))
        
        
        if len(self.markers) == 0:
            return
        
        # paint markers in native coordinate system
        tr = p.transform()
        p.resetTransform()
        
        start = tr.map(Point(left, 0))
        end = tr.map(Point(right, 0))
        up = tr.map(Point(left, 1))
        dif = end - start
        length = Point(dif).length()
        angle = np.arctan2(dif.y(), dif.x()) * 180 / np.pi
        
        p.translate(start)
        p.rotate(angle)
        
        up = up - start
        det = up.x() * dif.y() - dif.x() * up.y()
        p.scale(1, 1 if det > 0 else -1)
        
        p.setBrush(fn.mkBrush(self.currentPen.color()))
        #p.setPen(fn.mkPen(None))
        tr = p.transform()
        for path, pos, size in self.markers:
            p.setTransform(tr)
            x = length * pos
            p.translate(x, 0)
            p.scale(size, size)
            p.drawPath(path)
        
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
        
    def setName(self, name):
        self._name = name

    def name(self):
        return self._name


class InfLineLabel(TextItem):
    """
    A TextItem that attaches itself to an InfiniteLine.
    
    This class extends TextItem with the following features:
    
    * Automatically positions adjacent to the line at a fixed position along
      the line and within the view box.
    * Automatically reformats text when the line value has changed.
    * Can optionally be dragged to change its location along the line.
    * Optionally aligns to its parent line.

    =============== ==================================================================
    **Arguments:**
    line            The InfiniteLine to which this label will be attached.
    text            String to display in the label. May contain a {value} formatting
                    string to display the current value of the line.
    movable         Bool; if True, then the label can be dragged along the line.
    position        Relative position (0.0-1.0) within the view to position the label
                    along the line.
    anchors         List of (x,y) pairs giving the text anchor positions that should
                    be used when the line is moved to one side of the view or the
                    other. This allows text to switch to the opposite side of the line
                    as it approaches the edge of the view. These are automatically
                    selected for some common cases, but may be specified if the 
                    default values give unexpected results.
    =============== ==================================================================
    
    All extra keyword arguments are passed to TextItem. A particularly useful
    option here is to use `rotateAxis=(1, 0)`, which will cause the text to
    be automatically rotated parallel to the line.
    """
    def __init__(self, line, text="", movable=False, position=0.5, anchors=None, **kwds):
        self.line = line
        self.movable = movable
        self.moving = False
        self.orthoPos = position  # text will always be placed on the line at a position relative to view bounds
        self.format = text
        self.line.sigPositionChanged.connect(self.valueChanged)
        self._endpoints = (None, None)
        if anchors is None:
            # automatically pick sensible anchors
            rax = kwds.get('rotateAxis', None)
            if rax is not None:
                if tuple(rax) == (1,0):
                    anchors = [(0.5, 0), (0.5, 1)]
                else:
                    anchors = [(0, 0.5), (1, 0.5)]
            else:
                if line.angle % 180 == 0:
                    anchors = [(0.5, 0), (0.5, 1)]
                else:
                    anchors = [(0, 0.5), (1, 0.5)]
            
        self.anchors = anchors
        TextItem.__init__(self, **kwds)
        self.setParentItem(line)
        self.valueChanged()

    def valueChanged(self):
        if not self.isVisible():
            return
        value = self.line.value()
        self.setText(self.format.format(value=value))
        self.updatePosition()

    def getEndpoints(self):
        # calculate points where line intersects view box
        # (in line coordinates)
        if self._endpoints[0] is None:
            lr = self.line.boundingRect()
            pt1 = Point(lr.left(), 0)
            pt2 = Point(lr.right(), 0)
            
            if self.line.angle % 90 != 0:
                # more expensive to find text position for oblique lines.
                view = self.getViewBox()
                if not self.isVisible() or not isinstance(view, ViewBox):
                    # not in a viewbox, skip update
                    return (None, None)
                p = QtGui.QPainterPath()
                p.moveTo(pt1)
                p.lineTo(pt2)
                p = self.line.itemTransform(view)[0].map(p)
                vr = QtGui.QPainterPath()
                vr.addRect(view.boundingRect())
                paths = vr.intersected(p).toSubpathPolygons(QtGui.QTransform())
                if len(paths) > 0:
                    l = list(paths[0])
                    pt1 = self.line.mapFromItem(view, l[0])
                    pt2 = self.line.mapFromItem(view, l[1])
            self._endpoints = (pt1, pt2)
        return self._endpoints
    
    def updatePosition(self):
        # update text position to relative view location along line
        self._endpoints = (None, None)
        pt1, pt2 = self.getEndpoints()
        if pt1 is None:
            return
        pt = pt2 * self.orthoPos + pt1 * (1-self.orthoPos)
        self.setPos(pt)
        
        # update anchor to keep text visible as it nears the view box edge
        vr = self.line.viewRect()
        if vr is not None:
            self.setAnchor(self.anchors[0 if vr.center().y() < 0 else 1])
        
    def setVisible(self, v):
        TextItem.setVisible(self, v)
        if v:
            self.updateText()
            self.updatePosition()
            
    def setMovable(self, m):
        """Set whether this label is movable by dragging along the line.
        """
        self.movable = m
        self.setAcceptHoverEvents(m)
        
    def setPosition(self, p):
        """Set the relative position (0.0-1.0) of this label within the view box
        and along the line. 
        
        For horizontal (angle=0) and vertical (angle=90) lines, a value of 0.0
        places the text at the bottom or left of the view, respectively. 
        """
        self.orthoPos = p
        self.updatePosition()
        
    def setFormat(self, text):
        """Set the text format string for this label.
        
        May optionally contain "{value}" to include the lines current value
        (the text will be reformatted whenever the line is moved).
        """
        self.format = text
        self.valueChanged()
        
    def mouseDragEvent(self, ev):
        if self.movable and ev.button() == QtCore.Qt.LeftButton:
            if ev.isStart():
                self._moving = True
                self._cursorOffset = self._posToRel(ev.buttonDownPos())
                self._startPosition = self.orthoPos
            ev.accept()

            if not self._moving:
                return

            rel = self._posToRel(ev.pos())
            self.orthoPos = np.clip(self._startPosition + rel - self._cursorOffset, 0, 1)
            self.updatePosition()
            if ev.isFinish():
                self._moving = False

    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.orthoPos = self._startPosition
            self.moving = False

    def hoverEvent(self, ev):
        if not ev.isExit() and self.movable:
            ev.acceptDrags(QtCore.Qt.LeftButton)

    def viewTransformChanged(self):
        self.updatePosition()
        TextItem.viewTransformChanged(self)

    def _posToRel(self, pos):
        # convert local position to relative position along line between view bounds
        pt1, pt2 = self.getEndpoints()
        if pt1 is None:
            return 0
        view = self.getViewBox()
        pos = self.mapToParent(pos)
        return (pos.x() - pt1.x()) / (pt2.x()-pt1.x())
