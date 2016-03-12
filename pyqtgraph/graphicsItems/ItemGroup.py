
from ..Qt import QtGui, QtCore, USE_PYSIDE
if not USE_PYSIDE:
    import sip
#from .GraphicsObject import GraphicsObject
from .GraphicsItem import GraphicsItem

from ..QtNativeUtils import QGraphicsObject2

__all__ = ['ItemGroup']
class ItemGroup(GraphicsItem, QGraphicsObject2):
    """
    Replacement for QGraphicsItemGroup
    """
    
    _qtBaseClass = QGraphicsObject2
    def __init__(self, parent=None):
        self.__inform_view_on_changes = True
        QGraphicsObject2.__init__(self, parent=parent)
        self.setFlag(self.ItemSendsGeometryChanges)
        GraphicsItem.__init__(self)
        if hasattr(self, "ItemHasNoContents"):
            self.setFlag(self.ItemHasNoContents)
    
    def boundingRect(self):
        return QtCore.QRectF()
        
    def paint(self, *args):
        pass
    
    def addItem(self, item):
        item.setParentItem(self)

    def itemChange(self, change, value):
        ret = QtGui.QGraphicsObject.itemChange(self, change, value)
        if change in [self.ItemParentHasChanged, self.ItemSceneHasChanged]:
            self.parentIsChanged()
        try:
            inform_view_on_change = self.__inform_view_on_changes
        except AttributeError:
            # It's possible that the attribute was already collected when the itemChange happened
            # (if it was triggered during the gc of the object).
            pass
        else:
            if inform_view_on_change and change in [self.ItemPositionHasChanged, self.ItemTransformHasChanged]:
                self.informViewBoundsChanged()

        ## workaround for pyqt bug:
        ## http://www.riverbankcomputing.com/pipermail/pyqt/2012-August/031818.html
        if not USE_PYSIDE and change == self.ItemParentChange and isinstance(ret, QtGui.QGraphicsItem):
            ret = sip.cast(ret, QtGui.QGraphicsItem)

        return ret



