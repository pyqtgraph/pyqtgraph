
from ..Qt import QtGui, QtCore, USE_PYSIDE
if not USE_PYSIDE:
    import sip
#from .GraphicsObject import GraphicsObject
from .GraphicsItem import GraphicsItem

from ..QtNativeUtils import GraphicsObject
from ..QtNativeUtils import ItemGroup


'''
__all__ = ['ItemGroup']
class ItemGroup(GraphicsObject):
    """
    Replacement for QGraphicsItemGroup
    """
    
    _qtBaseClass = GraphicsObject
    def __init__(self, parent=None):
        GraphicsObject.__init__(self, parent=parent)
        #GraphicsItem.__init__(self)
        if hasattr(self, "ItemHasNoContents"):
            self.setFlag(self.ItemHasNoContents)
    
    def boundingRect(self):
        return QtCore.QRectF()
        
    def paint(self, *args):
        pass
    
    def addItem(self, item):
        item.setParentItem(self)
'''
