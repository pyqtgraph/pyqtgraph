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
    def __init__(self, size=None, offset=None, drawFrame=True):
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
        ==============  ===============================================================
        
        """
        
        
        GraphicsWidget.__init__(self)
        GraphicsWidgetAnchor.__init__(self)
        self.setFlag(self.ItemIgnoresTransformations)
        self.layout = QtGui.QGraphicsGridLayout()
        self.setLayout(self.layout)
        self.items = []
        self.size = size
        self.offset = offset
        self.drawFrame = drawFrame
        self.columnCount = 1
        self.rowCount = 1
        self.curRow = 0
        if size is not None:
            self.setGeometry(QtCore.QRectF(0, 0, self.size[0], self.size[1]))
    
    def setParentItem(self, p):
        ret = GraphicsWidget.setParentItem(self, p)
        if self.offset is not None:
            offset = Point(self.offset)
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
        label = LabelItem(name)
        if isinstance(item, ItemSample):
            sample = item
        else:
            sample = ItemSample(item)    
        self.items.append((sample, label))
        self._addItemToLayout(sample, label)
        self.updateSize()

    def _addItemToLayout(self, sample, label):
        col = self.layout.columnCount()
        row = self.layout.rowCount()
        if row:
            row -= 1
        nCol = self.columnCount*2
        #FIRST ROW FULL
        if col == nCol:
            for col in range(0,nCol,2):
                #FIND RIGHT COLUMN
                if not self.layout.itemAt(row, col):
                    break
            if col+2 == nCol:
                #MAKE NEW ROW
                col = 0
                row += 1
        self.layout.addItem(sample, row, col)
        self.layout.addItem(label, row, col+1)

    def setColumnCount(self, columnCount):
        '''
        change the orientation of all items of the legend 
        '''
        if columnCount != self.columnCount:
            self.columnCount = columnCount
            self.rowCount = int(len(self.items)/columnCount)
            for i in range(self.layout.count()-1,-1,-1):
                self.layout.removeAt(i)  #clear layout
            for sample, label in self.items:
                self._addItemToLayout(sample, label) 
            self.updateSize()
    
    def getLabel(self, plotItem):
        """
        return the labelItem inside the legend for a given plotItem
        the label-text can be changed via labenItem.setText
        """
        for i in self.items:
            if i[0].item == plotItem:
                return i[1]

    def removeItem(self, name):
        """
        Removes one item from the legend. 

        ==============  ========================================================
        **Arguments:**
        title           The title displayed for this item.
        ==============  ========================================================
        """
        # Thanks, Ulrich!
        # cycle for a match
        for sample, label in self.items:
            if label.text == name:  # hit
                self.items.remove( (sample, label) )    # remove from itemlist
                self.layout.removeItem(sample)          # remove from layout
                sample.close()                          # remove from drawing
                self.layout.removeItem(label)
                label.close()
                self.updateSize()                       # redraq box

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
    
    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.width(), self.height())
    
    def paint(self, p, *args):
        if self.drawFrame:
            p.setPen(fn.mkPen(255,255,255,100))
            p.setBrush(fn.mkBrush(100,100,100,50))
            p.drawRect(self.boundingRect())

    def hoverEvent(self, ev):
        ev.acceptDrags(QtCore.Qt.LeftButton)
        
    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            dpos = ev.pos() - ev.lastPos()
            self.autoAnchor(self.pos() + dpos)
        
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
        #p.setRenderHint(p.Antialiasing)  # only if the data is antialiased.
        opts = self.item.opts
        
        if opts.get('fillLevel',None) is not None and opts.get('fillBrush',None) is not None:
            p.setBrush(fn.mkBrush(opts['fillBrush']))
            p.setPen(fn.mkPen(None))
            p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(2,18), QtCore.QPointF(18,2), QtCore.QPointF(18,18)]))
        
        if not isinstance(self.item, ScatterPlotItem):
            p.setPen(fn.mkPen(opts['pen']))
            p.drawLine(2, 18, 18, 2)
        
        symbol = opts.get('symbol', None)
        if symbol is not None:
            if isinstance(self.item, PlotDataItem):
                opts = self.item.scatter.opts
                
            pen = fn.mkPen(opts['pen'])
            brush = fn.mkBrush(opts['brush'])
            size = opts['size']
            
            p.translate(10,10)
            path = drawSymbol(p, symbol, size, pen, brush)
        
        
        
        
