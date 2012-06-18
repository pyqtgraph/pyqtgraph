from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.functions as fn
from .GraphicsWidget import GraphicsWidget
## Must be imported at the end to avoid cyclic-dependency hell:
from .ViewBox import ViewBox
from .PlotItem import PlotItem
from .LabelItem import LabelItem

__all__ = ['GraphicsLayout']
class GraphicsLayout(GraphicsWidget):
    """
    Used for laying out GraphicsWidgets in a grid.
    This is usually created automatically as part of a :class:`GraphicsWindow <pyqtgraph.GraphicsWindow>` or :class:`GraphicsLayoutWidget <pyqtgraph.GraphicsLayoutWidget>`.
    """


    def __init__(self, parent=None, border=None):
        GraphicsWidget.__init__(self, parent)
        if border is True:
            border = (100,100,100)
        self.border = border
        self.layout = QtGui.QGraphicsGridLayout()
        self.setLayout(self.layout)
        self.items = {}
        self.rows = {}
        self.currentRow = 0
        self.currentCol = 0
    
    def nextRow(self):
        """Advance to next row for automatic item placement"""
        self.currentRow += 1
        self.currentCol = 0
        
    def nextColumn(self, colspan=1):
        """Advance to next column, while returning the current column number 
        (generally only for internal use--called by addItem)"""
        self.currentCol += colspan
        return self.currentCol-colspan
        
    def nextCol(self, *args, **kargs):
        """Alias of nextColumn"""
        return self.nextColumn(*args, **kargs)
        
    def addPlot(self, row=None, col=None, rowspan=1, colspan=1, **kargs):
        """
        Create a PlotItem and place it in the next available cell (or in the cell specified)
        All extra keyword arguments are passed to :func:`PlotItem.__init__ <pyqtgraph.PlotItem.__init__>`
        Returns the created item.
        """
        plot = PlotItem(**kargs)
        self.addItem(plot, row, col, rowspan, colspan)
        return plot
        
    def addViewBox(self, row=None, col=None, rowspan=1, colspan=1, **kargs):
        """
        Create a ViewBox and place it in the next available cell (or in the cell specified)
        All extra keyword arguments are passed to :func:`ViewBox.__init__ <pyqtgraph.ViewBox.__init__>`
        Returns the created item.
        """
        vb = ViewBox(**kargs)
        self.addItem(vb, row, col, rowspan, colspan)
        return vb
        
    def addLabel(self, text=' ', row=None, col=None, rowspan=1, colspan=1, **kargs):
        """
        Create a LabelItem with *text* and place it in the next available cell (or in the cell specified)
        All extra keyword arguments are passed to :func:`LabelItem.__init__ <pyqtgraph.LabelItem.__init__>`
        Returns the created item.
        """
        text = LabelItem(text, **kargs)
        self.addItem(text, row, col, rowspan, colspan)
        return text
        
    def addLayout(self, row=None, col=None, rowspan=1, colspan=1, **kargs):
        """
        Create an empty GraphicsLayout and place it in the next available cell (or in the cell specified)
        All extra keyword arguments are passed to :func:`GraphicsLayout.__init__ <pyqtgraph.GraphicsLayout.__init__>`
        Returns the created item.
        """
        layout = GraphicsLayout(**kargs)
        self.addItem(layout, row, col, rowspan, colspan)
        return layout
        
    def addItem(self, item, row=None, col=None, rowspan=1, colspan=1):
        """
        Add an item to the layout and place it in the next available cell (or in the cell specified).
        The item must be an instance of a QGraphicsWidget subclass.
        """
        if row is None:
            row = self.currentRow
        if col is None:
            col = self.nextCol(colspan)
            
        if row not in self.rows:
            self.rows[row] = {}
        self.rows[row][col] = item
        self.items[item] = (row, col)
        
        self.layout.addItem(item, row, col, rowspan, colspan)

    def getItem(self, row, col):
        """Return the item in (*row*, *col*)"""
        return self.row[row][col]

    def boundingRect(self):
        return self.rect()
        
    def paint(self, p, *args):
        if self.border is None:
            return
        p.setPen(fn.mkPen(self.border))
        for i in self.items:
            r = i.mapRectToParent(i.boundingRect())
            p.drawRect(r)
    
    def itemIndex(self, item):
        for i in range(self.layout.count()):
            if self.layout.itemAt(i).graphicsItem() is item:
                return i
        raise Exception("Could not determine index of item " + str(item))
    
    def removeItem(self, item):
        """Remove *item* from the layout."""
        ind = self.itemIndex(item)
        self.layout.removeAt(ind)
        self.scene().removeItem(item)
        r,c = self.items[item]
        del self.items[item]
        del self.rows[r][c]
        self.update()
    
    def clear(self):
        items = []
        for i in list(self.items.keys()):
            self.removeItem(i)


