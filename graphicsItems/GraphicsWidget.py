from pyqtgraph.Qt import QtGui, QtCore  
from pyqtgraph.GraphicsScene import GraphicsScene
from GraphicsItem import GraphicsItem

__all__ = ['GraphicsWidget']

class GraphicsWidget(GraphicsItem, QtGui.QGraphicsWidget):
    def __init__(self, *args, **kargs):
        """
        **Bases:** :class:`GraphicsItem <pyqtgraph.GraphicsItem>`, :class:`QtGui.QGraphicsWidget`
        
        Extends QGraphicsWidget with several helpful methods and workarounds for PyQt bugs. 
        Most of the extra functionality is inherited from :class:`GraphicsItem <pyqtgraph.GraphicsItem>`.
        """
        QtGui.QGraphicsWidget.__init__(self, *args, **kargs)
        GraphicsItem.__init__(self)
        GraphicsScene.registerObject(self)  ## workaround for pyqt bug in graphicsscene.items()

    #def getMenu(self):
        #pass
        
    def setFixedHeight(self, h):
        self.setMaximumHeight(h)
        self.setMinimumHeight(h)

    def setFixedWidth(self, h):
        self.setMaximumWidth(h)
        self.setMinimumWidth(h)
        
    def height(self):
        return self.geometry().height()
    
    def width(self):
        return self.geometry().width()

    def boundingRect(self):
        br = self.mapRectFromParent(self.geometry()).normalized()
        #print "bounds:", br
        return br
        
    def shape(self):  ## No idea why this is necessary, but rotated items do not receive clicks otherwise.
        p = QtGui.QPainterPath()
        p.addRect(self.boundingRect())
        #print "shape:", p.boundingRect()
        return p


