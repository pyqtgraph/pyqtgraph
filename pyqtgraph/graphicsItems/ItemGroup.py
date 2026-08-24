__all__ = ['ItemGroup']

from ..Qt import QtCore, QtWidgets
from .GraphicsObject import GraphicsObject


class ItemGroup(GraphicsObject):
    """
    Replacement for QGraphicsItemGroup
    """
    
    def __init__(self, *args):
        GraphicsObject.__init__(self, *args)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemHasNoContents)
    
    def boundingRect(self):
        return QtCore.QRectF()
        
    def paint(self, *args):
        pass
    
    def addItem(self, item):
        item.setParentItem(self)
