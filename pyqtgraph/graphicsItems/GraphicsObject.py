from ..Qt import QT_LIB, QtWidgets

if QT_LIB.startswith('PyQt'):
    from ..Qt import sip

from .GraphicsItem import GraphicsItem

__all__ = ['GraphicsObject']
class GraphicsObject(GraphicsItem, QtWidgets.QGraphicsObject):
    """
    **Bases:** :class:`GraphicsItem <pyqtgraph.graphicsItems.GraphicsItem>`, :class:`QtWidgets.QGraphicsObject`

    Extension of QGraphicsObject with some useful methods (provided by :class:`GraphicsItem <pyqtgraph.graphicsItems.GraphicsItem>`)
    """
    _qtBaseClass = QtWidgets.QGraphicsObject
    def __init__(self, *args):
        self.__inform_view_on_changes = True
        QtWidgets.QGraphicsObject.__init__(self, *args)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges)
        GraphicsItem.__init__(self)
        
    def itemChange(self, change, value):
        ret = super().itemChange(change, value)
        if change in [self.GraphicsItemChange.ItemParentHasChanged, self.GraphicsItemChange.ItemSceneHasChanged]:
            import types
            if isinstance(self.parentChanged, types.MethodType):
                self.parentChanged()
            else:
                # workaround PySide6 6.2.2 issue https://bugreports.qt.io/browse/PYSIDE-1730
                getattr(self.__class__, 'parentChanged')(self)
        try:
            inform_view_on_change = self.__inform_view_on_changes
        except AttributeError:
            # It's possible that the attribute was already collected when the itemChange happened
            # (if it was triggered during the gc of the object).
            pass
        else:
            if inform_view_on_change and change in [self.GraphicsItemChange.ItemPositionHasChanged, self.GraphicsItemChange.ItemTransformHasChanged]:
                self.informViewBoundsChanged()
            
        ## workaround for pyqt bug:
        ## http://www.riverbankcomputing.com/pipermail/pyqt/2012-August/031818.html
        if QT_LIB == 'PyQt5' and change == self.GraphicsItemChange.ItemParentChange and isinstance(ret, QtWidgets.QGraphicsItem):
            ret = sip.cast(ret, QtWidgets.QGraphicsItem)

        return ret
