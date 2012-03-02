from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.functions as fn
from GraphicsWidget import GraphicsWidget

__all__ = ['GraphicsLayout']
class GraphicsLayout(GraphicsWidget):
    """
    Used for laying out GraphicsWidgets in a grid.
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
        
    def nextCol(self, colspan=1):
        """Advance to next column, while returning the current column number 
        (generally only for internal use--called by addItem)"""
        self.currentCol += colspan
        return self.currentCol-colspan
        
    def addPlot(self, row=None, col=None, rowspan=1, colspan=1, **kargs):
        from PlotItem import PlotItem
        plot = PlotItem(**kargs)
        self.addItem(plot, row, col, rowspan, colspan)
        return plot
        
    def addViewBox(self, row=None, col=None, rowspan=1, colspan=1, **kargs):
        vb = ViewBox(**kargs)
        self.addItem(vb, row, col, rowspan, colspan)
        return vb
        

    def addItem(self, item, row=None, col=None, rowspan=1, colspan=1):
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
        ind = self.itemIndex(item)
        self.layout.removeAt(ind)
        self.scene().removeItem(item)
        r,c = self.items[item]
        del self.items[item]
        del self.rows[r][c]
        self.update()
    
    def clear(self):
        items = []
        for i in self.items.keys():
            self.removeItem(i)


## Must be imported at the end to avoid cyclic-dependency hell:
from ViewBox import ViewBox
from PlotItem import PlotItem
