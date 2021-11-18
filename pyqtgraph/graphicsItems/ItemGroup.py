from ..Qt import QtCore
from .GraphicsObject import GraphicsObject

__all__ = ['ItemGroup']
class ItemGroup(GraphicsObject):
    """
    Replacement for QGraphicsItemGroup
    """
    
    def __init__(self, *args):
        GraphicsObject.__init__(self, *args)
        self.setFlag(self.GraphicsItemFlag.ItemHasNoContents)
    
    def boundingRect(self):
        return QtCore.QRectF()
        
    def paint(self, *args):
        pass
    
    def addItem(self, item):
        item.setParentItem(self)
