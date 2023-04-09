import math
from typing import Any, List, Optional, Tuple, TypedDict, Union
import warnings


from .. import functions as fn
from .. import configStyle
from ..style.core import (
    ConfigColorHint,
    ConfigKeyHint,
    ConfigValueHint,
    initItemStyle)
from ..icons import invisibleEye
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .BarGraphItem import BarGraphItem
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
from .LabelItem import LabelItem
from .PlotItem import PlotItem
from .PlotDataItem import PlotDataItem
from .ScatterPlotItem import ScatterPlotItem, drawSymbol
from ..GraphicsScene.mouseEvents import HoverEvent, MouseDragEvent, MouseClickEvent

__all__ = ['LegendItem', 'ItemSample']


optsHint = TypedDict('optsHint',
                     {'offset'         : float,
                      'labelcolor'     : ConfigColorHint,
                      'labelfontsize'  : float,
                      'labelfontweight': str,
                      'labelalign'     : str,
                      'facecolor'      : ConfigColorHint,
                      'edgecolor'      : ConfigColorHint,
                      'frameon'        : bool},
                     total=False)
# kwargs are not typed because mypy has not yet included Unpack[Typeddict]
SampleTypeHint = Optional[Union[GraphicsWidget, "ItemSample"]]

class LegendItem(GraphicsWidgetAnchor, GraphicsWidget):
    """
    Displays a legend used for describing the contents of a plot.

    LegendItems are most commonly created by calling :meth:`PlotItem.addLegend
    <pyqtgraph.PlotItem.addLegend>`.

    Note that this item should *not* be added directly to a PlotItem (via
    :meth:`PlotItem.addItem <pyqtgraph.PlotItem.addItem>`). Instead, make it a
    direct descendant of the PlotItem::

        legend.setParentItem(plotItem)

    """

    sampleType : Any # didn't find better...

    def __init__(self, size: Optional[Tuple[float, float]]=None,
                       horSpacing: float=25,
                       verSpacing: float=0,
                       colCount: int=1,
                       sampleType: SampleTypeHint=None,
                       **kwargs):
        """
        Constructs a new legend item.

        Parameters
        ----------
        size :
            Specifies the fixed size (width, height) of the legend.
            If this argument is omitted, the legend will automatically resize to
            fit its contents.
            Default None
        horSpacing :
            Specifies the spacing between the line symbol and the label.
        verSpacing :
            Specifies the spacing between individual entries of the legendvertically.
            (Can also be negative to have them really close)
            Default 0
        colCount :
            Specifies the integer number of columns that the legend should be
            divided into. The number of rows will be calculated based on this argument.
            This is useful for plots with many curves displayed simultaneously.
            Default 1
        sampleType :
            Customizes the item sample class of the `LegendItem`.
            Default None
        *kwargs:
            Style options , see setStyles() for accepted style parameters.
        """


        GraphicsWidget.__init__(self)
        GraphicsWidgetAnchor.__init__(self)
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations)
        self.layout = QtWidgets.QGraphicsGridLayout()
        self.layout.setVerticalSpacing(verSpacing)
        self.layout.setHorizontalSpacing(horSpacing)

        self.setLayout(self.layout)
        self.items: List[Tuple[SampleTypeHint, LabelItem]] = []
        self.size = size
        self.columnCount = colCount
        self.rowCount = 1
        if size is not None:
            self.setGeometry(QtCore.QRectF(0, 0, size[0], size[1]))

        if sampleType is not None:
            if not issubclass(sampleType, GraphicsWidget):
                raise RuntimeError("Only classes of type `GraphicsWidgets` "
                                   "are allowed as `sampleType`")
            self.sampleType = sampleType
        else:
            self.sampleType = ItemSample

        self.opts: optsHint = {}
        # Get default stylesheet
        initItemStyle(self, 'legendItem', configStyle)
        # Update style if needed
        if len(kwargs)>0:
            self.setStyles(**kwargs)

        # Deprecated attributes
        self._brush: Optional[QtGui.QBrush] = None
        self._pen:   Optional[QtGui.QPen]   = None


    ##############################################################
    #
    #                   Style methods
    #
    ##############################################################

    # All style applicable to labelItem contained inside the LegendItem
    ## labelColor
    def getLabelColor(self) -> ConfigColorHint:
        """
        Get the color used for the item labels.
        """
        return self.opts['labelcolor']

    def setLabelColor(self, color: ConfigColorHint):
        """
        Set the color used for the item labels.

        Parameters
        ----------
        color : ConfigColorHint
            The QColor to set for the item labels.
        """
        if hasattr(self, 'labelcolor'):
            update = self.opts['labelcolor']==color
        else:
            update = False
        self.opts['labelcolor'] = color
        for _, label in self.items:
            label.setStyle('color', self.opts['labelcolor'])
        if update:
            self.update()

    ## labelFontsize
    def getLabelFontsize(self) -> float:
        """
        Get the font size used for the item labels.
        """
        return self.opts['labelfontsize']

    def setLabelFontsize(self, size: float) -> None:
        """
        Set the font weight used for the item labels.

        Parameters
        ----------
        weight : str
            The font weight to set for the item labels.
        """
        self.opts['labelfontsize'] = size
        for _, label in self.items:
            label.setStyle('fontsize', self.opts['labelfontsize'])

        self.update()

    ## labelFontweight
    def getLabelFontweight(self) -> str:
        """
        Get the font weight used for the item labels.

        Returns
        -------
        str
            The font weight used for the item labels.
        """
        return self.opts['labelfontweight']

    def setLabelFontweight(self, weight: str) -> None:
        """
        Set the font weight used for the item labels.

        Parameters
        ----------
        align : str
            The font weight to set for the item labels.
            Must be "normal", "bold", "bolder", or "lighter".

        Raises
        ------
        ValueError
            If the given `weight` parameter is not one of "bold", "bolder", or "lighter".
        """
        self.opts['labelfontweight'] = weight
        for _, label in self.items:
            label.setStyle('fontweight', self.opts['labelfontweight'])

        self.update()

    ## labelAlign
    def getLabelAlign(self) -> str:
        """
        Get the alignment of the item labels.

        Returns
        -------
        str
            The current alignment of the item labels, which can be one of
            'left', 'center', or 'right'.
        """
        return self.opts['labelalign']

    def setLabelAlign(self, align: str) -> None:
        """
        Set the alignment of the item labels.

        Parameters
        ----------
        align : str
            The new alignment for the item labels. This can be one of 'left',
            'center', or 'right'.

        Raises
        ------
        ValueError
            If the given `align` argument is not a string or not one of 'left',
            'center', or 'right'.
        """

        if isinstance(align, str):
            if align not in ('left', 'center', 'right'):
                raise ValueError('Given "align" argument:{} must be "left", "center", or "right".'.format(align))
        else:
            raise ValueError('align argument:{} is not a string'.format(align))

        self.opts['labelalign'] = align
        for _, label in self.items:
            label.setStyle('align', self.opts['labelalign'])

        self.update()

    # Style applicable to LegendItem
    ## offset
    def setOffset(self, offset: Tuple[float, float]):
        """
        Set the offset position of the legend relative to the parent.

        Parameters
        ----------
        offset : Tuple[float, float]
            The new offset position of the legend relative to the parent. This should
            be a tuple of two float values.

        Raises
        ------
        ValueError
            If the given `offset` argument is not a tuple or contains any values
            that are not float values.
        """
        if not isinstance(offset, tuple):
            if not all((isinstance(i, float)for i in offset)):
                raise ValueError('offset argument:{} is not a tuple of float'.format(offset))
        self.opts['offset'] = offset

    def getOffset(self):
        """
        Get the offset position of the legend relative to the parent.
        """
        return self.opts['offset']

    ## facecolor
    def getFacecolor(self):
        return self.opts['facecolor']

    def setFacecolor(self, color: ConfigColorHint):
        self.opts['facecolor'] = color

    ## edgecolor
    def getEdgecolor(self):
        return self.opts['edgecolor']

    def setEdgecolor(self, color: ConfigColorHint):
        self.opts['edgecolor'] = color

    ## frame
    def getFrameon(self):
        return self.opts['frameon']

    def setFrameon(self, frameon: bool):
        self.opts['frameon'] = frameon

    ## Since we keep two parameters for the same task, these 2 methods sort
    # that
    def _getBrush(self):
        if self._brush is None:
            return fn.mkBrush(self.opts['facecolor'])
        else:
            return self._brush

    def _getPen(self):
        if self._pen is None:
            return fn.mkPen(self.opts['edgecolor'])
        else:
            return self._pen

    def setStyle(self, attr: ConfigKeyHint,
                       value: ConfigValueHint) -> None:
        """
        Set a single style property.

        Parameters
        ----------
        attr : ConfigKeyHint
            The name of the style parameter to change.
            See `setStyles()` for a list of all accepted style parameters.
        value : ConfigValueHint
            The new value for the specified style parameter.

        See Also
        --------
        setStyles : Set multiple style properties at once.
        stylesheet : Qt Style Sheets reference.

        Examples
        --------
        >>> setStyle('color', 'red')
        """

        # If the attr is a valid entry of the stylesheet
        if attr in (key[11:] for key in configStyle.keys() if key[:10]=='legendItem'):
            fun = getattr(self, 'set{}{}'.format(attr[:1].upper(), attr[1:]))
            fun(value)
        else:
            raise ValueError('Your "attr" argument: "{}" is not recognized'.format(value))

    def setStyles(self, **kwargs) -> None:
        """
        Set the style of the LegendItem.

        Parameters
        ----------
        offset : tuple of float or None, optional
            Specifies the offset position relative to the legend's parent.
            Positive values offset from the left or top; negative values offset
            from the right or bottom. If offset is None, the legend must be
            anchored manually by calling anchor() or positioned by calling setPos().
        labelColor: ConfigColorHint or None, optional
            Color to use when drawing legend text.
            Any single argument accepted by :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        labelFontsize : float or None, optional
            Size to use when drawing legend text in pt.
        labelFontweight : {'normal', 'bold', 'bolder' 'lighter'} or None, optional
            Label weight.
        labelFontstyle : {'normal', 'italic' 'oblique'} or None, optional
            Label style.
        labelAlign : {'left', 'center', 'right'} or None, optional
            Label alignment.
        facecolor : ConfigColorHint or None, optional
            Background color of the legend.
        frameon : bool or None, optional
            If True, draw a frame around the legend.
        edgecolor : ConfigColorHint or None, optional
            Color of the frame.

        Notes
        -----
        The parameters that are not provided will not be modified.

        Examples
        --------
        >>> setStyles(offset=(10, 20), labelColor='w', labelFontsize=12,
                      labelFontweight='bold', labelAlign='center')
        """
        for k, v in kwargs.items():
            self.setStyle(k, v)

    ##############################################################
    #
    #                   Deprecated style methods
    #
    ##############################################################

    ## offset
    def offset(self):
        """Get the offset position relative to the parent."""

        warnings.warn('Method "offset" is deprecated. Use "getOffset" instead',
                      DeprecationWarning,
                      stacklevel=2)
        return self.opts['offset']


    ## labelTextColor
    def labelTextColor(self):
        """Get the QColor used for the item labels."""

        warnings.warn('Method "labelTextColor" is deprecated. Use "getLabelColor" instead',
                      DeprecationWarning,
                      stacklevel=2)
        return self.opts['labelcolor']

    def setLabelTextColor(self, *args, **kargs):
        """Set the color of the item labels.

        Accepts the same arguments as :func:`~pyqtgraph.mkColor`.
        """

        warnings.warn('Method "setLabelTextColor" is deprecated. Use "setLabelColor" instead',
                       DeprecationWarning,
                      stacklevel=2)

        self.opts['labelcolor'] = fn.mkColor(*args, **kargs)
        for sample, label in self.items:
            label.setStyle('color', self.opts['labelcolor'])

        self.update()

    ## labelTextSize
    def labelTextSize(self) -> float:
        """Get the `labelTextSize` used for the item labels."""

        warnings.warn('Method "labelTextSize" is deprecated. Use "getLabelFontsize" instead',
                      DeprecationWarning,
                      stacklevel=2)
        return self.opts['labelfontsize']

    def setLabelTextSize(self, size: float) -> None:
        """
        Set the `size` of the item labels.

        Args:
            size : font-size in pt.
        """

        warnings.warn('Method "setLabelTextSize" is deprecated. Use "setLabelFontsize" instead',
                      DeprecationWarning,
                      stacklevel=2)
        self.opts['labelfontsize'] = size
        for _, label in self.items:
            label.setStyle('size', self.opts['labelfontsize'])

        self.update()

    ## brush
    def brush(self):

        warnings.warn('Method "brush" is deprecated. Use "setFacecolor" instead',
                      DeprecationWarning,
                      stacklevel=2)
        """Get the QBrush used to draw the legend background."""
        return self._brush

    def setBrush(self, *args, **kargs):
        """Set the brush used to draw the legend background.

        Accepts the same arguments as :func:`~pyqtgraph.mkBrush`.
        """

        warnings.warn('Method "setBrush" is deprecated. Use "getFacecolor" instead',
                      DeprecationWarning,
                      stacklevel=2)
        brush = fn.mkBrush(*args, **kargs)
        if self._brush == brush:
            return
        self._brush = brush

        self.update()

    ## pen
    def pen(self):
        """Get the QPen used to draw the border around the legend."""

        warnings.warn('Method "pen" is deprecated. Use "getEdgecolor" instead',
                      DeprecationWarning,
                      stacklevel=2)
        return self._pen

    def setPen(self, *args, **kargs):
        """Set the pen used to draw a border around the legend.
        Accepts the same arguments as :func:`~pyqtgraph.mkPen`.
        """

        warnings.warn('Method "setPen" is deprecated. Use "setEdgecolor" instead',
                      DeprecationWarning,
                      stacklevel=2)
        self._pen = fn.mkPen(*args, **kargs)

        self.update()

    ##############################################################
    #
    #                   Item
    #
    ##############################################################

    def setSampleType(self, sample: SampleTypeHint) -> None:
        """
        Set the new sample item claspes
        """
        if sample is self.sampleType:
            return

        # Clear the legend, but before create a list of items
        items = list(self.items)
        self.sampleType = sample
        self.clear()

        # Refill the legend with the item list and new sample item
        for sample, label in items:
            plot_item = sample.item
            plot_name = label.text
            self.addItem(plot_item, plot_name)

        self.updateSize()

    # Not sure about the return type
    def setParentItem(self, p: PlotItem) -> Optional[Any]:
        """Set the parent."""
        ret = GraphicsWidget.setParentItem(self, p)
        if self.opts['offset'] is not None:
            offset = Point(self.opts['offset'])
            anchorx = 1 if offset[0] <= 0 else 0
            anchory = 1 if offset[1] <= 0 else 0
            anchor = (anchorx, anchory)
            self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)

        return ret

    def addItem(self, item: PlotDataItem,
                      name: str) -> None:
        """
        Add a new entry to the legend.

        Parameters
        ----------
        item
            A :class:`~pyqtgraph.PlotDataItem` from which the line and point
            style of the item will be determined or an instance of ItemSample
            (or a subclass), allowing the item display to be customized.
        name
            The title to display for this item. Simple HTML allowed.
        """
        label = LabelItem(name,
                          color=self.opts['labelcolor'],
                          align=self.opts['labelalign'],
                          fontsize=self.opts['labelfontsize'],
                          fontweight=self.opts['labelfontweight'])
        if isinstance(item, self.sampleType):
            sample = item
        else:
            sample = self.sampleType(item)
        self.items.append((sample, label))
        self._addItemToLayout(sample, label)
        self.updateSize()

    def _addItemToLayout(self, sample: SampleTypeHint,
                               label: LabelItem) -> None:
        """
        Add a sample and its corresponding label into the legend.
        """
        col = self.layout.columnCount()
        row = self.layout.rowCount()
        if row:
            row -= 1
        nCol = self.columnCount * 2
        # FIRST ROW FULL
        if col == nCol:
            for col in range(0, nCol, 2):
                # FIND RIGHT COLUMN
                if not self.layout.itemAt(row, col):
                    break
            else:
                if col + 2 == nCol:
                    # MAKE NEW ROW
                    col = 0
                    row += 1
        self.layout.addItem(sample, row, col)
        self.layout.addItem(label, row, col + 1)
        # Keep rowCount in sync with the number of rows if items are added
        self.rowCount = max(self.rowCount, row + 1)

    def setColumnCount(self, columnCount: int) -> None:
        """
        Change the orientation of all items of the legend
        """
        if columnCount != self.columnCount:
            self.columnCount = columnCount

            self.rowCount = math.ceil(len(self.items) / columnCount)
            for i in range(self.layout.count() - 1, -1, -1):
                self.layout.removeAt(i)  # clear layout
            for sample, label in self.items:
                self._addItemToLayout(sample, label)
            self.updateSize()

    def getLabel(self, plotItem: PlotItem) -> Optional[LabelItem]:
        """
        Return the labelItem inside the legend for a given plotItem.
        Return None if no Label.

        The label-text can be changed via labelItem.setText
        """
        out = [(it, lab) for it, lab in self.items if it.item == plotItem]
        try:
            return out[0][1]
        except IndexError:
            return None

    def _removeItemFromLayout(self, *args):
        for item in args:
            self.layout.removeItem(item)
            item.close()
            # Normally, the item is automatically removed from
            # its scene when it gets destroyed.
            # this doesn't happen on current versions of
            # PySide (5.15.x, 6.3.x) and results in a leak.
            scene = item.scene()
            if scene:
                scene.removeItem(item)

    def removeItem(self, item: Union[SampleTypeHint, LabelItem]) -> None:
        """
        Removes one item from the legend.

        Parameters
        ----------
        item
            The item to remove or its name.
        """
        for sample, label in self.items:
            if sample.item is item or label.text == item:
                self.items.remove((sample, label))  # remove from itemlist
                self._removeItemFromLayout(sample, label)
                self.updateSize()  # redraw box
                return  # return after first match

    def clear(self) -> None:
        """Remove all items from the legend."""
        for sample, label in self.items:
            self._removeItemFromLayout(sample, label)

        self.items = []
        self.updateSize()

    def updateSize(self) -> None:
        if self.size is not None:
            return
        height = 0
        width = 0
        for row in range(self.layout.rowCount()):
            row_height = 0
            col_width = 0
            for col in range(self.layout.columnCount()):
                item = self.layout.itemAt(row, col)
                if item:
                    col_width += item.width() + 3
                    row_height = max(row_height, item.height())
            width = max(width, col_width)
            height += row_height
        self.setGeometry(0, 0, width, height)
        return

    def boundingRect(self) -> QtCore.QRectF:
        """
        Return the item rectangle coordinates
        """
        return QtCore.QRectF(0, 0, self.width(), self.height())

    def paint(self, p: QtGui.QPainter, *args) -> None:
        if self.opts['frameon']:
            p.setPen(self._getPen())
            p.setBrush(self._getBrush())
            p.drawRect(self.boundingRect())

    def hoverEvent(self, ev: HoverEvent) -> None:
        """
        Call when pointer is hovering the LegenItem.
        """
        ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)

    def mouseDragEvent(self, ev: MouseDragEvent) -> None:
        """
        Call when the LegenItem is being dragged.
        """
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            dpos = ev.pos() - ev.lastPos()
            self.autoAnchor(self.pos() + dpos)


class ItemSample(GraphicsWidget):
    """Class responsible for drawing a single item in a LegendItem (sans label)
    """

    def __init__(self, item: PlotDataItem) -> None:
        GraphicsWidget.__init__(self)
        self.item = item

    def boundingRect(self) -> QtCore.QRectF:
        """
        Return the item rectangle coordinates.
        Currently, hardcoded to be (0, 0, 20, 20),
        """
        return QtCore.QRectF(0, 0, 20, 20)

    def paint(self, p: QtGui.QPainter, *args) -> None:
        opts = self.item.opts
        if opts.get('antialias'):
            p.setRenderHint(p.RenderHint.Antialiasing)

        visible = self.item.isVisible()
        if not visible:
            icon = invisibleEye.qicon
            p.drawPixmap(QtCore.QPoint(1, 1), icon.pixmap(18, 18))
            return

        if not isinstance(self.item, ScatterPlotItem):
            p.setPen(fn.mkPen(opts['pen']))
            p.drawLine(0, 11, 20, 11)

            if (opts.get('fillLevel', None) is not None and
                    opts.get('fillBrush', None) is not None):
                p.setBrush(fn.mkBrush(opts['fillBrush']))
                p.setPen(fn.mkPen(opts['pen']))
                p.drawPolygon(QtGui.QPolygonF(
                    [QtCore.QPointF(2, 18), QtCore.QPointF(18, 2),
                     QtCore.QPointF(18, 18)]))

        symbol = opts.get('symbol', None)
        if symbol is not None:
            if isinstance(self.item, PlotDataItem):
                opts = self.item.scatter.opts
            p.translate(10, 10)
            drawSymbol(p, symbol, opts['size'], fn.mkPen(opts['pen']),
                       fn.mkBrush(opts['brush']))

        if isinstance(self.item, BarGraphItem):
            p.setBrush(fn.mkBrush(opts['brush']))
            p.drawRect(QtCore.QRectF(2, 2, 18, 18))

    def mouseClickEvent(self, ev: MouseClickEvent) -> None:
        """
        Use the mouseClick ev to toggle the visibility of the plotItem
        """
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            visible = self.item.isVisible()
            self.item.setVisible(not visible)

        ev.accept()
        self.update()
