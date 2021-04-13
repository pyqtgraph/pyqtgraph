from math import atan2, pi

from ..Qt import QtGui, QtCore
from ..Point import Point
from .. import functions as fn
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
from .TextItem import TextItem
from .ScatterPlotItem import Symbols, makeCrosshair
from .ViewBox import ViewBox
import string
import warnings


class TargetItem(UIGraphicsItem):
    """Draws a draggable target symbol (circle plus crosshair).

    The size of TargetItem will remain fixed on screen even as the view is zoomed.
    Includes an optional text label.
    """

    sigPositionChanged = QtCore.Signal(object)
    sigPositionChangeFinished = QtCore.Signal(object)

    def __init__(
        self,
        pos=None,
        size=10,
        radii=None,
        symbol="crosshair",
        pen=None,
        hoverPen=None,
        brush=None,
        hoverBrush=None,
        movable=True,
        label=None,
        labelOpts=None,
    ):
        r"""
        Parameters
        ----------
        pos : list, tuple, QPointF, QPoint, Optional
            Initial position of the symbol.  Default is (0, 0)
        size : int
            Size of the symbol in pixels.  Default is 10.
        radii : tuple of int
            Deprecated.  Gives size of crosshair in screen pixels.
        pen : QPen, tuple, list or str
            Pen to use when drawing line. Can be any arguments that are valid
            for :func:`~pyqtgraph.mkPen`. Default pen is transparent yellow.
        brush : QBrush, tuple, list, or str
            Defines the brush that fill the symbol. Can be any arguments that
            is valid for :func:`~pyqtgraph.mkBrush`. Default is transparent
            blue.
        movable : bool
            If True, the symbol can be dragged to a new position by the user.
        hoverPen : QPen, tuple, list, or str
            Pen to use when drawing symbol when hovering over it. Can be any
            arguments that are valid for :func:`~pyqtgraph.mkPen`. Default pen
            is red.
        hoverBrush : QBrush, tuple, list or str
            Brush to use to fill the symbol when hovering over it. Can be any
            arguments that is valid for :func:`~pyqtgraph.mkBrush`. Default is
            transparent blue.
        symbol : QPainterPath or str
            QPainterPath to use for drawing the target, should be centered at
            ``(0, 0)`` with ``max(width, height) == 1.0``.  Alternatively a string
            which can be any symbol accepted by
            :func:`~pyqtgraph.ScatterPlotItem.setData`
        label : bool, str or callable, optional
            Text to be displayed in a label attached to the symbol, or None to
            show no label (default is None). May optionally include formatting
            strings to display the symbol value, or a callable that accepts x
            and y as inputs.  If True, the label is ``x = {: >.3n}\ny = {: >.3n}``
            False or None will result in no text being displayed
        labelOpts : dict
            A dict of keyword arguments to use when constructing the text
            label. See :class:`TargetLabel` and :class:`~pyqtgraph.TextItem`
        """
        super().__init__(self)
        self.movable = movable
        self.moving = False
        self._label = None
        self.mouseHovering = False

        if radii is not None:
            warnings.warn(
                "'radii' is now deprecated, and will be removed in 0.13.0. Use 'size' "
                "parameter instead",
                DeprecationWarning,
                stacklevel=2,
            )
            symbol = makeCrosshair(*radii)
            size = 1

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

        self._pos = Point(0, 0)
        if pos is None:
            pos = Point(0, 0)
        self.setPos(pos)

        if isinstance(symbol, str):
            try:
                self._path = Symbols[symbol]
            except KeyError:
                raise KeyError("symbol name found in available Symbols")
        elif isinstance(symbol, QtGui.QPainterPath):
            self._path = symbol
        else:
            raise TypeError("Unknown type provided as symbol")

        self.scale = size
        self.setPath(self._path)
        self.setLabel(label, labelOpts)

    @property
    def sigDragged(self):
        warnings.warn(
            "'sigDragged' has been deprecated and will be removed in 0.13.0.  Use "
            "`sigPositionChanged` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sigPositionChangeFinished

    def setPos(self, pos):
        """Method to set the position to ``(x, y)`` within the plot view

        Parameters
        ----------
        pos : tuple, list, QPointF, QPoint, or pg.Point
            Container that consists of ``(x, y)`` representation of where the
            TargetItem should be placed

        Raises
        ------
        TypeError
            If the type of ``pos`` does not match the known types to extract
            coordinate info from, a TypeError is raised
        """
        if isinstance(pos, Point):
            newPos = pos
        elif isinstance(pos, (tuple, list)):
            newPos = Point(pos)
        elif isinstance(pos, (QtCore.QPointF, QtCore.QPoint)):
            newPos = Point(pos.x(), pos.y())
        else:
            raise TypeError
        if self._pos != newPos:
            self._pos = newPos
            super().setPos(self._pos)
            self.sigPositionChanged.emit(self)

    def setBrush(self, *args, **kwargs):
        """Set the brush that fills the symbol. Allowable arguments are any that
        are valid for :func:`~pyqtgraph.mkBrush`.
        """
        self.brush = fn.mkBrush(*args, **kwargs)
        if not self.mouseHovering:
            self.currentBrush = self.brush
            self.update()

    def setHoverBrush(self, *args, **kwargs):
        """Set the brush that fills the symbol when hovering over it. Allowable
        arguments are any that are valid for :func:`~pyqtgraph.mkBrush`.
        """
        self.hoverBrush = fn.mkBrush(*args, **kwargs)
        if self.mouseHovering:
            self.currentBrush = self.hoverBrush
            self.update()

    def setPen(self, *args, **kwargs):
        """Set the pen for drawing the symbol. Allowable arguments are any that
        are valid for :func:`~pyqtgraph.mkPen`."""
        self.pen = fn.mkPen(*args, **kwargs)
        if not self.mouseHovering:
            self.currentPen = self.pen
            self.update()

    def setHoverPen(self, *args, **kwargs):
        """Set the pen for drawing the symbol when hovering over it. Allowable
        arguments are any that are valid for
        :func:`~pyqtgraph.mkPen`."""
        self.hoverPen = fn.mkPen(*args, **kwargs)
        if self.mouseHovering:
            self.currentPen = self.hoverPen
            self.update()

    def boundingRect(self):
        return self.shape().boundingRect()

    def paint(self, p, *_):
        p.setPen(self.currentPen)
        p.setBrush(self.currentBrush)
        p.drawPath(self.shape())

    def setPath(self, path):
        if path != self._path:
            self._path = path
            self._shape = None
        return None

    def shape(self):
        if self._shape is None:
            s = self.generateShape()
            if s is None:
                return self._path
            self._shape = s

            # beware--this can cause the view to adjust
            # which would immediately invalidate the shape.
            self.prepareGeometryChange()
        return self._shape

    def generateShape(self):
        dt = self.deviceTransform()
        if dt is None:
            self._shape = self._path
            return None
        v = dt.map(QtCore.QPointF(1, 0)) - dt.map(QtCore.QPointF(0, 0))
        dti = fn.invertQTransform(dt)
        devPos = dt.map(QtCore.QPointF(0, 0))
        tr = QtGui.QTransform()
        tr.translate(devPos.x(), devPos.y())
        va = atan2(v.y(), v.x())
        tr.rotate(va * 180.0 / pi)
        tr.scale(self.scale, self.scale)
        return dti.map(tr.map(self._path))

    def mouseDragEvent(self, ev):
        if not self.movable or int(ev.button() & QtCore.Qt.LeftButton) == 0:
            return
        ev.accept()
        if ev.isStart():
            self.symbolOffset = self.pos() - self.mapToView(ev.buttonDownPos())
            self.moving = True

        if not self.moving:
            return
        self.setPos(self.symbolOffset + self.mapToView(ev.pos()))

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
        # Inform the item that the mouse is(not) hovering over it
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
        GraphicsObject.viewTransformChanged(self)
        self._shape = None  # invalidate shape, recompute later if requested.
        self.update()

    def pos(self):
        """Provides the current position of the TargetItem

        Returns
        -------
        Point
            pg.Point of the current position of the TargetItem
        """
        return self._pos

    def label(self):
        """Provides the TargetLabel if it exists

        Returns
        -------
        TargetLabel or None
            If a TargetLabel exists for this TargetItem, return that, otherwise
            return None
        """
        return self._label

    def setLabel(self, text=None, labelOpts=None):
        """Method to call to enable or disable the TargetLabel for displaying text

        Parameters
        ----------
        text : Callable or str, optional
            Details how to format the text, by default None
            If None, do not show any text next to the TargetItem
            If Callable, then the label will display the result of ``text(x, y)``
            If a fromatted string, then the output of ``text.format(x, y)`` will be
            displayed
            If a non-formatted string, then the text label will display ``text``, by
            default None
        labelOpts : dictionary, optional
            These arguments are passed on to :class:`~pyqtgraph.TextItem`
        """
        if not text:
            if self._label is not None and self._label.scene() is not None:
                # remove the label if it's already added
                self._label.scene().removeItem(self._label)
            self._label = None
        else:
            # provide default text if text is True
            if text is True:
                # convert to default value or empty string
                text = "x = {: .3n}\ny = {: .3n}"

            labelOpts = {} if labelOpts is None else labelOpts
            if self._label is not None:
                self._label.scene().removeItem(self._label)
            self._label = TargetLabel(self, text=text, **labelOpts)

    def setLabelAngle(self, angle):
        warnings.warn(
            "TargetItem.setLabelAngle is deprecated and will be removed in 0.13.0."
            "Use TargetItem.label().setAngle() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.label() is not None and angle != self.label().angle:
            self.label().setAngle(angle)
        return None


class TargetLabel(TextItem):
    """A TextItem that attaches itself to a TargetItem.

    This class extends TextItem with the following features :
    * Automatically positions adjacent to the symbol at a fixed position.
    * Automatically reformats text when the symbol location has changed.

    Parameters
    ----------
    target : TargetItem
        The TargetItem to which this label will be attached to.
    text : str or callable, Optional
        Governs the text displayed, can be a fixed string or a format string
        that accepts the x, and y position of the target item; or be a callable
        method that accepts a tuple (x, y) and returns a string to be displayed.
        If None, an empty string is used.  Default is None
    offset : tuple or list or QPointF or QPoint
        Position to set the anchor of the TargetLabel away from the center of
        the target in pixels, by default it is (20, 0).
    anchor : tuple, list, QPointF or QPoint
        Position to rotate the TargetLabel about, and position to set the
        offset value to see :class:`~pyqtgraph.TextItem` for more inforation.
    kwargs : dict of arguments that are passed on to
        :class:`~pyqtgraph.TextItem` constructor, excluding text parameter
    """

    def __init__(
        self,
        target,
        text="",
        offset=(20, 0),
        anchor=(0, 0.5),
        **kwargs,
    ):
        if isinstance(offset, Point):
            self.offset = offset
        elif isinstance(offset, (tuple, list)):
            self.offset = Point(*offset)
        elif isinstance(offset, (QtCore.QPoint, QtCore.QPointF)):
            self.offset = Point(offset.x(), offset.y())
        else:
            raise TypeError("Offset parameter is the wrong data type")

        super().__init__(anchor=anchor, **kwargs)
        self.setParentItem(target)
        self.target = target
        self.setFormat(text)

        self.target.sigPositionChanged.connect(self.valueChanged)
        self.valueChanged()

    def format(self):
        return self._format

    def setFormat(self, text):
        """Method to set how the TargetLabel should display the text.  This
        method should be called from TargetItem.setLabel directly.

        Parameters
        ----------
        text : Callable or str
            Details how to format the text.
            If Callable, then the label will display the result of ``text(x, y)``
            If a fromatted string, then the output of ``text.format(x, y)`` will be
            displayed
            If a non-formatted string, then the text label will display ``text``
        """
        if not callable(text):
            parsed = list(string.Formatter().parse(text))
            if parsed and parsed[0][1] is not None:
                self.setProperty("formattableText", True)
            else:
                self.setText(text)
                self.setProperty("formattableText", False)
        else:
            self.setProperty("formattableText", False)
        self._format = text
        self.valueChanged()

    def valueChanged(self):
        x, y = self.target.pos()
        if self.property("formattableText"):
            self.setText(self._format.format(float(x), float(y)))
        elif callable(self._format):
            self.setText(self._format(x, y))

    def viewTransformChanged(self):
        viewbox = self.getViewBox()
        if isinstance(viewbox, ViewBox):
            viewPixelSize = viewbox.viewPixelSize()
            scaledOffset = QtCore.QPointF(
                self.offset.x() * viewPixelSize[0], self.offset.y() * viewPixelSize[1]
            )
            self.setPos(scaledOffset)
        return super().viewTransformChanged()

    def mouseClickEvent(self, ev):
        return self.parentItem().mouseClickEvent(ev)

    def mouseDragEvent(self, ev):
        targetItem = self.parentItem()
        if not targetItem.movable or int(ev.button() & QtCore.Qt.LeftButton) == 0:
            return
        ev.accept()
        if ev.isStart():
            targetItem.symbolOffset = targetItem.pos() - self.mapToView(
                ev.buttonDownPos()
            )
            targetItem.moving = True

        if not targetItem.moving:
            return
        targetItem.setPos(targetItem.symbolOffset + self.mapToView(ev.pos()))

        if ev.isFinish():
            targetItem.moving = False
            targetItem.sigPositionChangeFinished.emit(self)
