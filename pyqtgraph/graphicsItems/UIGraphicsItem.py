from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject

__all__ = ['UIGraphicsItem']
class UIGraphicsItem(GraphicsObject):
    """
    Base class for graphics items with boundaries relative to a GraphicsView or ViewBox.
    The purpose of this class is to allow the creation of GraphicsItems which live inside 
    a scalable view, but whose boundaries will always stay fixed relative to the view's boundaries.
    For example: GridItem, InfiniteLine
    
    The view can be specified on initialization or it can be automatically detected when the item is painted.
    
    NOTE: Only the item's boundingRect is affected; the item is not transformed in any way. Use viewRangeChanged
    to respond to changes in the view.
    """
    
    #sigViewChanged = QtCore.Signal(object)  ## emitted whenever the viewport coords have changed
    
    def __init__(self, bounds=None, parent=None):
        """
        ============== =============================================================================
        **Arguments:**
        bounds         QRectF with coordinates relative to view box. The default is QRectF(0,0,1,1),
                       which means the item will have the same bounds as the view.
        ============== =============================================================================
        """
        GraphicsObject.__init__(self, parent)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
            
        if bounds is None:
            self._bounds = QtCore.QRectF(0, 0, 1, 1)
        else:
            self._bounds = bounds
            
        self._boundingRect = None
        self._updateView()
        
    def paint(self, *args):
        ## check for a new view object every time we paint.
        #self.updateView()
        pass
    
    def itemChange(self, change, value):
        ret = GraphicsObject.itemChange(self, change, value)
            
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged:
            self.setNewBounds()
        return ret

    def boundingRect(self):
        if self._boundingRect is None:
            br = self.viewRect()
            if br is None:
                return QtCore.QRectF()
            else:
                self._boundingRect = br
        return QtCore.QRectF(self._boundingRect)
    
    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        """Called by ViewBox for determining the auto-range bounds.
        By default, UIGraphicsItems are excluded from autoRange."""
        return None

    @QtCore.Slot()
    def viewRangeChanged(self):
        """Called when the view widget/viewbox is resized/rescaled"""
        self.setNewBounds()
        self.update()
        
    def setNewBounds(self):
        """Update the item's bounding rect to match the viewport"""
        self._boundingRect = None  ## invalidate bounding rect, regenerate later if needed.
        self.prepareGeometryChange()

    def setPos(self, *args):
        GraphicsObject.setPos(self, *args)
        self.setNewBounds()
        
    def mouseShape(self):
        """Return the shape of this item after expanding by 2 pixels"""
        shape = self.shape()
        ds = self.mapToDevice(shape)
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(2)
        ds2 = stroker.createStroke(ds).united(ds)
        return self.mapFromDevice(ds2)
