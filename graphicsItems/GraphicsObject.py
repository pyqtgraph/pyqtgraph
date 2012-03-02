from pyqtgraph.Qt import QtGui, QtCore  
from GraphicsItemMethods import GraphicsItemMethods

__all__ = ['GraphicsObject']
class GraphicsObject(GraphicsItemMethods, QtGui.QGraphicsObject):
    """Extends QGraphicsObject with a few important functions. 
    (Most of these assume that the object is in a scene with a single view)
    
    This class also generates a cache of the Qt-internal addresses of each item
    so that GraphicsScene.items() can return the correct objects (this is a PyQt bug)
    
    Note: most of the extended functionality is inherited from GraphicsItemMethods,
    which is shared between GraphicsObject and GraphicsWidget.
    """
    def __init__(self, *args):
        QtGui.QGraphicsObject.__init__(self, *args)
        GraphicsItemMethods.__init__(self)
        
        
