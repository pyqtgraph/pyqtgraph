# -*- coding: utf-8 -*-
from .GraphicsWidget import GraphicsWidget
from .LabelItem import LabelItem
from ..Qt import QtGui, QtCore
from .. import functions as fn
from ..Point import Point
from .ScatterPlotItem import ScatterPlotItem, drawSymbol
from .PlotDataItem import PlotDataItem
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
from .BarGraphItem import BarGraphItem

__all__ = ['LegendItem', 'LegendStyle']


class LegendStyle:
    Classic = 0
    Toggle = 1


class LegendItem(GraphicsWidget, GraphicsWidgetAnchor):
    """
    Displays a legend used for describing the contents of a plot.

    LegendItems are most commonly created by calling :meth:`PlotItem.addLegend
    <pyqtgraph.PlotItem.addLegend>`.

    Note that this item should *not* be added directly to a PlotItem (via
    :meth:`PlotItem.addItem <pyqtgraph.PlotItem.addItem>`). Instead, make it a
    direct descendant of the PlotItem::

        legend.setParentItem(plotItem)

    """

    def __init__(self, size=None, offset=None, horSpacing=25, verSpacing=0,
                 pen=None, brush=None, labelTextColor=None, frame=True,
                 labelTextSize='9pt', rowCount=1, colCount=1,
                 itemStyle=LegendStyle.Classic, **kwargs):
        """
        ==============  ===============================================================
        **Arguments:**
        size            Specifies the fixed size (width, height) of the legend. If
                        this argument is omitted, the legend will automatically resize
                        to fit its contents.
        offset          Specifies the offset position relative to the legend's parent.
                        Positive values offset from the left or top; negative values
                        offset from the right or bottom. If offset is None, the
                        legend must be anchored manually by calling anchor() or
                        positioned by calling setPos().
        horSpacing      Specifies the spacing between the line symbol and the label.
        verSpacing      Specifies the spacing between individual entries of the legend
                        vertically. (Can also be negative to have them really close)
        pen             Pen to use when drawing legend border. Any single argument
                        accepted by :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        brush           QBrush to use as legend background filling. Any single argument
                        accepted by :func:`mkBrush <pyqtgraph.mkBrush>` is allowed.
        labelTextColor  Pen to use when drawing legend text. Any single argument
                        accepted by :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        labelTextSize   Size to use when drawing legend text. Accepts CSS style
                        string arguments, e.g. '9pt'.
        itemStyle       Fixed style of the legend. Defaults to 0 (LegendStyle.Classic).
        ==============  ===============================================================

        """
        GraphicsWidget.__init__(self)
        GraphicsWidgetAnchor.__init__(self)
        self.setFlag(self.ItemIgnoresTransformations)
        self.layout = QtGui.QGraphicsGridLayout()
        self.layout.setVerticalSpacing(verSpacing)
        self.layout.setHorizontalSpacing(horSpacing)

        self.setLayout(self.layout)
        self.items = []
        self.size = size
        self.offset = offset
        self.frame = frame
        self.columnCount = colCount
        self.rowCount = rowCount
        self.curRow = 0
        if size is not None:
            self.setGeometry(QtCore.QRectF(0, 0, self.size[0], self.size[1]))

        self.opts = {
            'pen': fn.mkPen(pen),
            'brush': fn.mkBrush(brush),
            'labelTextColor': labelTextColor,
            'labelTextSize': labelTextSize,
            'offset': offset,
            'itemStyle': itemStyle
        }
        self.opts.update(kwargs)

    def setItemStyle(self, style):
        """Set the new legend item style"""
        if style == self.opts['itemStyle']:
            return

        # Clear the legend, but before create a list of items
        items = list(self.items)
        self.opts['itemStyle'] = style
        self.clear()

        # Refill the legend with the item list and new item style
        for sample, _ in items:
            plot_item = sample.item
            self.addItem(plot_item, plot_item.name())

        self.updateSize()

    def itemStyle(self):
        """Get the `itemStyle` of the `LegendItem`"""
        return self.opts['itemStyle']

    def offset(self):
        """Get the offset position relative to the parent."""
        return self.opts['offset']

    def setOffset(self, offset):
        """Set the offset position relative to the parent."""
        self.opts['offset'] = offset

        offset = Point(self.opts['offset'])
        anchorx = 1 if offset[0] <= 0 else 0
        anchory = 1 if offset[1] <= 0 else 0
        anchor = (anchorx, anchory)
        self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)

    def pen(self):
        """Get the QPen used to draw the border around the legend."""
        return self.opts['pen']

    def setPen(self, *args, **kargs):
        """Set the pen used to draw a border around the legend.

        Accepts the same arguments as :func:`~pyqtgraph.mkPen`.
        """
        pen = fn.mkPen(*args, **kargs)
        self.opts['pen'] = pen

        self.update()

    def brush(self):
        """Get the QBrush used to draw the legend background."""
        return self.opts['brush']

    def setBrush(self, *args, **kargs):
        """Set the brush used to draw the legend background.

        Accepts the same arguments as :func:`~pyqtgraph.mkBrush`.
        """
        brush = fn.mkBrush(*args, **kargs)
        if self.opts['brush'] == brush:
            return
        self.opts['brush'] = brush

        self.update()

    def labelTextColor(self):
        """Get the QColor used for the item labels."""
        return self.opts['labelTextColor']

    def setLabelTextColor(self, *args, **kargs):
        """Set the color of the item labels.

        Accepts the same arguments as :func:`~pyqtgraph.mkColor`.
        """
        self.opts['labelTextColor'] = fn.mkColor(*args, **kargs)
        for sample, label in self.items:
            label.setAttr('color', self.opts['labelTextColor'])

        self.update()

    def labelTextSize(self):
        """Get the `labelTextSize` used for the item labels."""
        return self.opts['labelTextSize']

    def setLabelTextSize(self, size):
        """Set the `size` of the item labels.

        Accepts the CSS style string arguments, e.g. '8pt'.
        """
        self.opts['labelTextSize'] = size
        for _, label in self.items:
            label.setAttr('size', self.opts['labelTextSize'])

        self.update()

    def setParentItem(self, p):
        """Set the parent."""
        ret = GraphicsWidget.setParentItem(self, p)
        if self.opts['offset'] is not None:
            offset = Point(self.opts['offset'])
            anchorx = 1 if offset[0] <= 0 else 0
            anchory = 1 if offset[1] <= 0 else 0
            anchor = (anchorx, anchory)
            self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)
        return ret

    def addItem(self, item, name):
        """
        Add a new entry to the legend.

        ==============  ========================================================
        **Arguments:**
        item            A :class:`~pyqtgraph.PlotDataItem` from which the line
                        and point style of the item will be determined or an
                        instance of ItemSample (or a subclass), allowing the
                        item display to be customized.
        title           The title to display for this item. Simple HTML allowed.
        ==============  ========================================================
        """
        label = LabelItem(name, color=self.opts['labelTextColor'],
                          justify='left', size=self.opts['labelTextSize'])
        if isinstance(item, ItemSample):
            sample = item
        else:
            item_class = _get_legend_item_class(self.opts['itemStyle'])
            sample = item_class(item)
        self.items.append((sample, label))
        self._addItemToLayout(sample, label)
        self.updateSize()

    def _addItemToLayout(self, sample, label):
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
            if col + 2 == nCol:
                # MAKE NEW ROW
                col = 0
                row += 1
        self.layout.addItem(sample, row, col)
        self.layout.addItem(label, row, col + 1)

    def setColumnCount(self, columnCount):
        """change the orientation of all items of the legend
        """
        if columnCount != self.columnCount:
            self.columnCount = columnCount
            self.rowCount = int(len(self.items) / columnCount)
            for i in range(self.layout.count() - 1, -1, -1):
                self.layout.removeAt(i)  # clear layout
            for sample, label in self.items:
                self._addItemToLayout(sample, label)
            self.updateSize()

    def getLabel(self, plotItem):
        """Return the labelItem inside the legend for a given plotItem

        The label-text can be changed via labenItem.setText
        """
        out = [(it, lab) for it, lab in self.items if it.item == plotItem]
        try:
            return out[0][1]
        except IndexError:
            return None

    def removeItem(self, item):
        """Removes one item from the legend.

        ==============  ========================================================
        **Arguments:**
        item            The item to remove or its name.
        ==============  ========================================================
        """
        for sample, label in self.items:
            if sample.item is item or label.text == item:
                self.items.remove((sample, label))  # remove from itemlist
                self.layout.removeItem(sample)  # remove from layout
                sample.close()  # remove from drawing
                self.layout.removeItem(label)
                label.close()
                self.updateSize()  # redraq box
                return  # return after first match

    def clear(self):
        """Remove all items from the legend."""
        for sample, label in self.items:
            self.layout.removeItem(sample)
            sample.close()
            self.layout.removeItem(label)
            label.close()

        self.items = []
        self.updateSize()

    def updateSize(self):
        if self.size is not None:
            return
        height = 0
        width = 0
        for row in range(self.layout.rowCount()):
            row_height = 0
            col_witdh = 0
            for col in range(self.layout.columnCount()):
                item = self.layout.itemAt(row, col)
                if item:
                    col_witdh += item.width() + 3
                    row_height = max(row_height, item.height())
            width = max(width, col_witdh)
            height += row_height
        self.setGeometry(0, 0, width, height)
        return

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.width(), self.height())

    def paint(self, p, *args):
        if self.frame:
            p.setPen(self.opts['pen'])
            p.setBrush(self.opts['brush'])
            p.drawRect(self.boundingRect())

    def hoverEvent(self, ev):
        ev.acceptDrags(QtCore.Qt.LeftButton)

    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            dpos = ev.pos() - ev.lastPos()
            self.autoAnchor(self.pos() + dpos)


class ItemSample(GraphicsWidget):
    """Base Class responsible for drawing a single item in a LegendItem

    This has to be subclassed to draw custom graphics in a Legend.
    """

    def __init__(self, item):
        GraphicsWidget.__init__(self)
        self.item = item

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 20, 20)

    def paint(self, p, *args):
        raise NotImplementedError


class ClassicSample(ItemSample):
    """Generic legend item to handle plot data items

    The `ClassicSample` is the default style of a LegendItem. Considers the
    style and type of the plot to draw a colored symbol for the legend sample.
    """

    def __init__(self, item):
        ItemSample.__init__(self, item)

    def paint(self, p, *args):
        opts = self.item.opts

        if opts.get('antialias'):
            p.setRenderHint(p.Antialiasing)

        if not isinstance(self.item, ScatterPlotItem):
            p.setPen(fn.mkPen(opts['pen']))
            p.drawLine(0, 11, 20, 11)

            if (opts.get('fillLevel', None) is not None and
                    opts.get('fillBrush', None) is not None):
                p.setBrush(fn.mkBrush(opts['fillBrush']))
                p.setPen(fn.mkPen(opts['fillBrush']))
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


class ToggleSample(ItemSample):
    """The ToggleSample can toggle the visibility of the plot item"""

    def __init__(self, item):
        ItemSample.__init__(self, item)

    def paint(self, p, *args):
        pen, brush = get_toggle_pen_brush(self.item)
        visible = self.item.isVisible()
        brush = fn.mkBrush(
            QtGui.QColor(211, 211, 211)) if not visible else brush
        p.setPen(pen)
        p.setBrush(brush)
        p.drawRect(0, 0, 10, 14)

    def mouseClickEvent(self, event):
        """Use the mouse click with the left button to toggle the visibility
        """
        if event.button() == QtCore.Qt.LeftButton:
            visible = self.item.isVisible()
            self.item.setVisible(not visible)

        self.update()
        event.accept()


def get_toggle_pen_brush(item):
    """Retrieve a pen and brush for a legend sample for plot item `item`"""
    item = item
    opts = item.opts
    if isinstance(item, PlotDataItem) and opts.get('symbol', None):
        opts = item.scatter.opts
        pen = fn.mkPen(opts['pen'])
        brush = fn.mkBrush(opts['brush'])

        return pen, brush

    if isinstance(item, ScatterPlotItem):
        if (opts.get('fillLevel', None) is not None and
                opts.get('fillBrush', None) is not None):
            brush = fn.mkBrush(opts['fillBrush'])
            pen = fn.mkPen(opts['fillBrush'])
        else:
            pen = fn.mkPen(opts['pen'])
            brush = fn.mkBrush(opts['brush'])

        return pen, brush

    if isinstance(item, BarGraphItem):
        brush = fn.mkBrush(opts['brush'])
        pen = fn.mkPen(opts['brush'])

        return pen, brush

    pen = fn.mkPen(opts['pen'])
    brush = fn.mkBrush(pen.color())

    return pen, brush


def _get_legend_item_class(style):
    """Return a legend item style according to input `style`"""
    item_style = {
        LegendStyle.Classic: ClassicSample,
        LegendStyle.Toggle: ToggleSample
    }

    return item_style.get(style, ClassicSample)
