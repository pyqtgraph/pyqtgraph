from ..Qt import QtGui, QtCore, QT_LIB
if QT_LIB.startswith('PyQt'):
    from ..Qt import sip
from .GraphicsItem import GraphicsItem
from .. import functions as fn

__all__ = ['GraphicsObject']
DEBUG = False
    
class GraphicsObject(GraphicsItem, QtGui.QGraphicsObject):
    """
    **Bases:** :class:`GraphicsItem <pyqtgraph.graphicsItems.GraphicsItem>`, :class:`QtGui.QGraphicsObject`

    Extension of QGraphicsObject with some useful methods (provided by :class:`GraphicsItem <pyqtgraph.graphicsItems.GraphicsItem>`)
    """
    _qtBaseClass = QtGui.QGraphicsObject
    def __init__(self, *args):
        self.__inform_view_on_changes = True
        QtGui.QGraphicsObject.__init__(self, *args)
        self.setFlag(self.ItemSendsGeometryChanges)
        GraphicsItem.__init__(self)
        # fn.NAMED_COLOR_MANAGER.paletteChangeSignal.connect(self.styleChange)
        fn.NAMED_COLOR_MANAGER.paletteHasChangedSignal.connect(self.styleHasChanged)
        
    def itemChange(self, change, value):
        ret = super().itemChange(change, value)
        if change in [self.ItemParentHasChanged, self.ItemSceneHasChanged]:
            self.parentChanged()
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
        if QT_LIB in ['PyQt4', 'PyQt5'] and change == self.ItemParentChange and isinstance(ret, QtGui.QGraphicsItem):
            ret = sip.cast(ret, QtGui.QGraphicsItem)

        return ret


    # @QT_CORE_SLOT(dict)
    # def styleChange(self, color_dict):
    #     """ stub function called after Palette.apply(), specific reactions to palette redefinitions execute here """
    #     print('style change request:', self, type(color_dict))
        
    @QtCore.Slot() # qt.py equates this to pyqtSlot for PyQt
    def styleHasChanged(self):
        """ called to trigger redraw after all named colors have been updated """
        # self._boundingRect = None
        self.update()
        if DEBUG: print('  GraphicsObject: redraw after style change:', self)