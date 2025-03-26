from .. import debug
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .InfiniteLine import InfiniteLine

__all__ = ['LinearRegionItem']

class LinearRegionItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    
    Used for marking a horizontal or vertical region in plots.
    The region can be dragged and is bounded by lines which can be dragged individually.
    
    ===============================  =============================================================================
    **Signals:**
    sigRegionChangeFinished(self)    Emitted when the user has finished dragging the region (or one of its lines)
                                     and when the region is changed programatically.
    sigRegionChanged(self)           Emitted while the user is dragging the region (or one of its lines)
                                     and when the region is changed programatically.
    ===============================  =============================================================================
    """
    
    sigRegionChangeFinished = QtCore.Signal(object)
    sigRegionChanged = QtCore.Signal(object)
    Vertical = 0
    Horizontal = 1
    _orientation_axis = {
        Vertical: 0,
        Horizontal: 1,
        'vertical': 0,
        'horizontal': 1,
        }
    
    def __init__(self, values=(0, 1), orientation='vertical', brush=None, pen=None,
                 hoverBrush=None, hoverPen=None, movable=True, bounds=None, 
                 span=(0, 1), swapMode='sort', clipItem=None):
        """Create a new LinearRegionItem.
        
        ==============  =====================================================================
        **Arguments:**
        values          A list of the positions of the lines in the region. These are not
                        limits; limits can be set by specifying bounds.
        orientation     Options are 'vertical' or 'horizontal'
                        The default is 'vertical', indicating that the region is bounded
                        by vertical lines.
        brush           Defines the brush that fills the region. Can be any arguments that
                        are valid for :func:`mkBrush <pyqtgraph.mkBrush>`. Default is
                        transparent blue.
        pen             The pen to use when drawing the lines that bound the region.
        hoverBrush      The brush to use when the mouse is hovering over the region.
        hoverPen        The pen to use when the mouse is hovering over the region.
        movable         If True, the region and individual lines are movable by the user; if
                        False, they are static.
        bounds          Optional [min, max] bounding values for the region
        span            Optional [min, max] giving the range over the view to draw
                        the region. For example, with a vertical line, use
                        ``span=(0.5, 1)`` to draw only on the top half of the
                        view.
        swapMode        Sets the behavior of the region when the lines are moved such that
                        their order reverses:

                          * "block" means the user cannot drag one line past the other
                          * "push" causes both lines to be moved if one would cross the other
                          * "sort" means that lines may trade places, but the output of
                            getRegion always gives the line positions in ascending order.
                          * None means that no attempt is made to handle swapped line
                            positions.

                        The default is "sort".
        clipItem        An item whose bounds will be used to limit the region bounds.
                        This is useful when a LinearRegionItem is added on top of an
                        :class:`~pyqtgraph.ImageItem` or
                        :class:`~pyqtgraph.PlotDataItem` and the visual region should
                        not extend beyond its range. This overrides ``bounds``.
        ==============  =====================================================================
        """
        
        GraphicsObject.__init__(self)
        self.orientation = orientation
        self.blockLineSignal = False
        self.moving = False
        self.mouseHovering = False
        self.span = span
        self.swapMode = swapMode
        self.clipItem = clipItem

        self._boundingRectCache = None
        self._clipItemBoundsCache = None
        
        # note LinearRegionItem.Horizontal and LinearRegionItem.Vertical
        # are kept for backward compatibility.
        lineKwds = dict(
            movable=movable,
            bounds=bounds,
            span=span,
            pen=pen,
            hoverPen=hoverPen,
        )
            
        if orientation in ('horizontal', LinearRegionItem.Horizontal):
            self.lines = [
                # rotate lines to 180 to preserve expected line orientation 
                # with respect to region. This ensures that placing a '<|' 
                # marker on lines[0] causes it to point left in vertical mode
                # and down in horizontal mode. 
                InfiniteLine(QtCore.QPointF(0, values[0]), angle=0, **lineKwds), 
                InfiniteLine(QtCore.QPointF(0, values[1]), angle=0, **lineKwds)]
            tr = QtGui.QTransform.fromScale(1, -1)
            self.lines[0].setTransform(tr, True)
            self.lines[1].setTransform(tr, True)
        elif orientation in ('vertical', LinearRegionItem.Vertical):
            self.lines = [
                InfiniteLine(QtCore.QPointF(values[0], 0), angle=90, **lineKwds), 
                InfiniteLine(QtCore.QPointF(values[1], 0), angle=90, **lineKwds)]
        else:
            raise Exception("Orientation must be 'vertical' or 'horizontal'.")
        
        for l in self.lines:
            l.setParentItem(self)
            l.sigPositionChangeFinished.connect(self.lineMoveFinished)
        self.lines[0].sigPositionChanged.connect(self._line0Moved)
        self.lines[1].sigPositionChanged.connect(self._line1Moved)
            
        if brush is None:
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, 50))
        self.setBrush(brush)
        
        if hoverBrush is None:
            c = self.brush.color()
            c.setAlpha(min(c.alpha() * 2, 255))
            hoverBrush = fn.mkBrush(c)
        self.setHoverBrush(hoverBrush)
        
        self.setMovable(movable)

    def getRegion(self):
        """Return the values at the edges of the region."""

        r = (self.lines[0].value(), self.lines[1].value())
        if self.swapMode == 'sort':
            return (min(r), max(r))
        else:
            return r

    def setRegion(self, rgn):
        """Set the values for the edges of the region.
        
        ==============   ==============================================
        **Arguments:**
        rgn              A list or tuple of the lower and upper values.
        ==============   ==============================================
        """
        if self.lines[0].value() == rgn[0] and self.lines[1].value() == rgn[1]:
            return
        self.blockLineSignal = True
        self.lines[0].setValue(rgn[0])
        self.lines[1].setValue(rgn[1])
        self.lineMoved(0)
        self.blockLineSignal = False
        self.lineMoved(1)
        self.lineMoveFinished()

    def setBrush(self, *br, **kargs):
        """Set the brush that fills the region. Can have any arguments that are valid
        for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.brush = fn.mkBrush(*br, **kargs)
        self.currentBrush = self.brush

    def setHoverBrush(self, *br, **kargs):
        """Set the brush that fills the region when the mouse is hovering over.
        Can have any arguments that are valid
        for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.hoverBrush = fn.mkBrush(*br, **kargs)

    def setBounds(self, bounds):
        """Set ``(min, max)`` bounding values for the region.

        The current position is only affected it is outside the new bounds. See
        :func:`~pyqtgraph.LinearRegionItem.setRegion` to set the position of the region.

        Use ``(None, None)`` to disable bounds.
        """
        if self.clipItem is not None:
            self.setClipItem(None)
        self._setBounds(bounds)

    def _setBounds(self, bounds):
        # internal impl so user-facing setBounds can clear clipItem and clipItem can
        # set bounds without clearing itself
        for line in self.lines:
            line.setBounds(bounds)

    def setMovable(self, m=True):
        """Set lines to be movable by the user, or not. If lines are movable, they will 
        also accept HoverEvents."""
        for line in self.lines:
            line.setMovable(m)
        self.movable = m
        self.setAcceptHoverEvents(m)

    def setSpan(self, mn, mx):
        if self.span == (mn, mx):
            return
        self.span = (mn, mx)
        for line in self.lines:
            line.setSpan(mn, mx)
        self.update()

    def setClipItem(self, item=None):
        """Set an item to which the region is bounded.

        If ``None``, bounds are disabled.
        """
        self.clipItem = item
        self._clipItemBoundsCache = None
        if item is None:
            self._setBounds((None, None))
        if item is not None:
            self._updateClipItemBounds()

    def _updateClipItemBounds(self):
        # set region bounds corresponding to clipItem
        item_vb = self.clipItem.getViewBox()
        if item_vb is None:
            return

        item_bounds = item_vb.childrenBounds(items=(self.clipItem,))
        if item_bounds == self._clipItemBoundsCache or None in item_bounds:
            return

        self._clipItemBoundsCache = item_bounds

        if self.orientation in ('horizontal', LinearRegionItem.Horizontal):
            self._setBounds(item_bounds[1])
        else:
            self._setBounds(item_bounds[0])

    def boundingRect(self):
        br = QtCore.QRectF(self.viewRect())  # bounds of containing ViewBox mapped to local coords.

        if self.clipItem is not None:
            self._updateClipItemBounds()

        rng = self.getRegion()
        if self.orientation in ('vertical', LinearRegionItem.Vertical):
            br.setLeft(rng[0])
            br.setRight(rng[1])
            length = br.height()
            br.setBottom(br.top() + length * self.span[1])
            br.setTop(br.top() + length * self.span[0])
        else:
            br.setTop(rng[0])
            br.setBottom(rng[1])
            length = br.width()
            br.setRight(br.left() + length * self.span[1])
            br.setLeft(br.left() + length * self.span[0])

        br = br.normalized()
        
        if self._boundingRectCache != br:
            self._boundingRectCache = br
            self.prepareGeometryChange()
        
        return br
        
    def paint(self, p, *args):
        profiler = debug.Profiler()  # noqa: profiler does prints on GC
        p.setBrush(self.currentBrush)
        p.setPen(fn.mkPen(None))
        p.drawRect(self.boundingRect())

    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        if axis == self._orientation_axis[self.orientation]:
            return self.getRegion()
        else:
            return None

    def lineMoved(self, i):
        if self.blockLineSignal:
            return

        # lines swapped
        if self.lines[0].value() > self.lines[1].value():
            if self.swapMode == 'block':
                self.lines[i].setValue(self.lines[1-i].value())
            elif self.swapMode == 'push':
                self.lines[1-i].setValue(self.lines[i].value())
        
        self.prepareGeometryChange()
        self.sigRegionChanged.emit(self)

    @QtCore.Slot()
    def _line0Moved(self):
        self.lineMoved(0)

    @QtCore.Slot()
    def _line1Moved(self):
        self.lineMoved(1)

    @QtCore.Slot()
    def lineMoveFinished(self):
        self.sigRegionChangeFinished.emit(self)

    def mouseDragEvent(self, ev):
        if not self.movable or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ev.accept()
        
        if ev.isStart():
            bdp = ev.buttonDownPos()
            self.cursorOffsets = [l.pos() - bdp for l in self.lines]
            self.startPositions = [l.pos() for l in self.lines]
            self.moving = True
            
        if not self.moving:
            return
            
        self.blockLineSignal = True  # only want to update once
        for i, l in enumerate(self.lines):
            l.setPos(self.cursorOffsets[i] + ev.pos())
        self.prepareGeometryChange()
        self.blockLineSignal = False
        
        if ev.isFinish():
            self.moving = False
            self.sigRegionChangeFinished.emit(self)
        else:
            self.sigRegionChanged.emit(self)
            
    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            for i, l in enumerate(self.lines):
                l.setPos(self.startPositions[i])
            self.moving = False
            self.sigRegionChanged.emit(self)
            self.sigRegionChangeFinished.emit(self)

    def hoverEvent(self, ev):
        if self.movable and (not ev.isExit()) and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
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
        else:
            self.currentBrush = self.brush
        self.update()
