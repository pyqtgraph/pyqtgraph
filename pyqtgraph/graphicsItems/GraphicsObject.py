from ..Qt import QtGui, QtCore, QT_LIB
if QT_LIB.startswith('PyQt'):
    from ..Qt import sip
from .GraphicsItem import GraphicsItem
from .. import functions as fn

__all__ = ['GraphicsObject']
DEBUG_REDRAW = False
    
class GraphicsObject(GraphicsItem, QtGui.QGraphicsObject):
    """
    **Bases:** :class:`GraphicsItem <pyqtgraph.graphicsItems.GraphicsItem>`, :class:`QtGui.QGraphicsObject`

    Extension of QGraphicsObject with some useful methods (provided by :class:`GraphicsItem <pyqtgraph.graphicsItems.GraphicsItem>`)
    """
    _qtBaseClass = QtGui.QGraphicsObject
    def __init__(self, *args):
        self.__inform_view_on_changes = True
        QtGui.QGraphicsObject.__init__(self, *args)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges)
        GraphicsItem.__init__(self)
        fn.STYLE_REGISTRY.graphStyleChanged.connect(self.updateGraphStyle)
        
    def itemChange(self, change, value):
        ret = super().itemChange(change, value)
        if change in [self.GraphicsItemChange.ItemParentHasChanged, self.GraphicsItemChange.ItemSceneHasChanged]:
            self.parentChanged()
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
        if QT_LIB == 'PyQt5' and change == self.GraphicsItemChange.ItemParentChange and isinstance(ret, QtGui.QGraphicsItem):
            ret = sip.cast(ret, QtGui.QGraphicsItem)

        return ret

    # Slot for graphStyleChanged signal emitted by StyleRegistry, omitted decorator: @QtCore.Slot()
    def updateGraphStyle(self):
        """ called to trigger redraw after all registered colors have been updated """
        # self._boundingRect = None
        self.update()
        if DEBUG_REDRAW: print('  GraphicsObject: redraw after style change:', self)