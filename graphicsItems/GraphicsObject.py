from pyqtgraph.Qt import QtGui, QtCore  
from .GraphicsItem import GraphicsItem

__all__ = ['GraphicsObject']
class GraphicsObject(GraphicsItem, QtGui.QGraphicsObject):
    """
    **Bases:** :class:`GraphicsItem <pyqtgraph.graphicsItems.GraphicsItem>`, :class:`QtGui.QGraphicsObject`

    Extension of QGraphicsObject with some useful methods (provided by :class:`GraphicsItem <pyqtgraph.graphicsItems.GraphicsItem>`)
    """
    _qtBaseClass = QtGui.QGraphicsObject
    def __init__(self, *args):
        QtGui.QGraphicsObject.__init__(self, *args)
        self.setFlag(self.ItemSendsGeometryChanges)
        GraphicsItem.__init__(self)
        
    def itemChange(self, change, value):
        ret = QtGui.QGraphicsObject.itemChange(self, change, value)
        if change in [self.ItemParentHasChanged, self.ItemSceneHasChanged]:
            self._updateView()
        if change in [self.ItemPositionHasChanged, self.ItemTransformHasChanged]:
            self.informViewBoundsChanged()
        return ret
