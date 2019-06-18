from .GraphicsWidget import GraphicsWidget
from .LabelItem import LabelItem
from ..Qt import QtGui, QtCore
from .. import functions as fn
from ..Point import Point
from .ScatterPlotItem import ScatterPlotItem, drawSymbol
from .PlotDataItem import PlotDataItem
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
__all__ = ['LegendItem']

class LegendItem(GraphicsWidget, GraphicsWidgetAnchor):
    """
    Displays a legend used for describing the contents of a plot.
    LegendItems are most commonly created by calling PlotItem.addLegend().

    Note that this item should not be added directly to a PlotItem. Instead,
    Make it a direct descendant of the PlotItem::

        legend.setParentItem(plotItem)

    """
    def __init__(self, size=None, offset=None, horSpacing=25, verSpacing=0, pen=None, brush=None, labelTextColor=None, **kwargs):
        """
        ==============  ===============================================================
        **Arguments:**
        size            Specifies the fixed size (width, height) of the legend. If
                        this argument is omitted, the legend will autimatically resize
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
        if size is not None:
            self.setGeometry(QtCore.QRectF(0, 0, self.size[0], self.size[1]))
        
        self._numItems = 0

        self.opts = {
            'pen': fn.mkPen(pen),
            'brush': fn.mkBrush(brush),
            'labelTextColor': labelTextColor,
            'offset': offset,
        }

        self.opts.update(kwargs)
       
    def offset(self):
        return self.opts['offset']

    def setOffset(self, offset):
        self.opts['offset'] = offset

        offset = Point(self.opts['offset'])
        anchorx = 1 if offset[0] <= 0 else 0
        anchory = 1 if offset[1] <= 0 else 0
        anchor = (anchorx, anchory)
        self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)
        
    def pen(self):
        return self.opts['pen']

    def setPen(self, *args, **kargs):
        """
        Sets the pen used to draw lines between points.
        *pen* can be a QPen or any argument accepted by
        :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>`
        """
        pen = fn.mkPen(*args, **kargs)
        self.opts['pen'] = pen

    def brush(self):
        return self.opts['brush']

    def setBrush(self, *args, **kargs):
        brush = fn.mkBrush(*args, **kargs)
        if self.opts['brush'] == brush:
            return
        self.opts['brush'] = brush

    def labelTextColor(self):
        return self.opts['labelTextColor']

    def setLabelTextColor(self, *args, **kargs):
        """
        Sets the color of the label text.
        *pen* can be a QPen or any argument accepted by
        :func:`pyqtgraph.mkColor() <pyqtgraph.mkPen>`
        """
        self.opts['labelTextColor'] = fn.mkColor(*args, **kargs)
        for sample, label in self.items:
            label.setAttr('color', self.opts['labelTextColor'])
        
    def setParentItem(self, p):
        ret = GraphicsWidget.setParentItem(self, p)
        if self.offset is not None:
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
        item            A PlotDataItem from which the line and point style
                        of the item will be determined or an instance of
                        ItemSample (or a subclass), allowing the item display
                        to be customized.
        title           The title to display for this item. Simple HTML allowed.
        ==============  ========================================================
        """
        label = LabelItem(name, color=self.opts['labelTextColor'], justify='left')
        if isinstance(item, ItemSample):
            sample = item
        else:
            sample = ItemSample(item)

        self.items.append((sample, label))
        self.layout.addItem(sample, self._numItems, 0)
        self.layout.addItem(label, self._numItems, 1)
        self._numItems += 1
        self.updateSize()

        height = max(sample.minimumHeight(), label.minimumHeight())
        self.layout.setRowMaximumHeight(row, height)
        self.layout.setColumnSpacing(0, 10)
        
        self.updateSize()
    
    def removeItem(self, item):
        """
        Removes one item from the legend. 

        ==============  ========================================================
        **Arguments:**
        item            The item to remove or its name.
        ==============  ========================================================
        """
        for sample, label in self.items:
            if sample.item is item or label.text == item:
                self.items.remove( (sample, label) )    # remove from itemlist
                self.layout.removeItem(sample)          # remove from layout
                sample.close()                          # remove from drawing
                self.layout.removeItem(label)
                label.close()
                self.updateSize()                       # redraq box

    def updateSize(self):
        if self.size is not None:
            return

        self.setGeometry(0, 0, 0, 0)
    
    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.width(), self.height())
    
    def paint(self, p, *args):
        p.setPen(self.opts['pen'])
        p.setBrush(self.opts['brush'])
        p.drawRect(self.boundingRect())

    def hoverEvent(self, ev):
        ev.acceptDrags(QtCore.Qt.LeftButton)
        
    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            dpos = ev.pos() - ev.lastPos()
            self.autoAnchor(self.pos() + dpos)

    def clear(self):
        """Clears legend."""
        for sample, label in self.items:
            self.layout.removeItem(sample)
            self.layout.removeItem(label)
        while len(self.items) > 0:
            self.items.pop()

class ItemSample(GraphicsWidget):
    """ Class responsible for drawing a single item in a LegendItem (sans label).
    
    This may be subclassed to draw custom graphics in a Legend.
    """
    ## Todo: make this more generic; let each item decide how it should be represented.
    def __init__(self, item):
        GraphicsWidget.__init__(self)
        self.item = item

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 20, 20)

    def paint(self, p, *args):
        opts = self.item.opts

        if opts['antialias']:
            p.setRenderHint(p.Antialiasing)

        if not isinstance(self.item, ScatterPlotItem):
            p.setPen(fn.mkPen(opts['pen']))
            p.drawLine(-7, 11, 7, 11)

        symbol = opts.get('symbol', None)
        if symbol is not None:
            if isinstance(self.item, PlotDataItem):
                opts = self.item.scatter.opts

            pen = fn.mkPen(opts['pen'])
            brush = fn.mkBrush(opts['brush'])
            size = opts['size']

            p.translate(10, 10)
            path = drawSymbol(p, symbol, size, pen, brush)
        
        
        
        
