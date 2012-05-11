from pyqtgraph.Qt import QtGui, QtCore  
from .GraphicsItem import GraphicsItem

__all__ = ['GraphicsObject']
class GraphicsObject(GraphicsItem, QtGui.QGraphicsObject):
    """
    **Bases:** :class:`GraphicsItem <pyqtgraph.graphicsItems.GraphicsItem>`, :class:`QtGui.QGraphicsObject`

    Extension of QGraphicsObject with some useful methods (provided by :class:`GraphicsItem <pyqtgraph.graphicsItems.GraphicsItem>`)
    """
    def __init__(self, *args):
        QtGui.QGraphicsObject.__init__(self, *args)
        GraphicsItem.__init__(self)
        
        
