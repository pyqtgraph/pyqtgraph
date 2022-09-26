from ..Qt import QtGui, QtWidgets
from .GraphicsItem import GraphicsItem

__all__ = ['GraphicsWidget']


class GraphicsWidget(GraphicsItem, QtWidgets.QGraphicsWidget):
    
    _qtBaseClass = QtWidgets.QGraphicsWidget

    def __init__(self, *args, **kwargs):
        """
        **Bases:** :class:`GraphicsItem <pyqtgraph.GraphicsItem>`, :class:`QtWidgets.QGraphicsWidget`
        
        Extends QGraphicsWidget with several helpful methods and workarounds for PyQt bugs. 
        Most of the extra functionality is inherited from :class:`GraphicsItem <pyqtgraph.GraphicsItem>`.
        """
        QtWidgets.QGraphicsWidget.__init__(self, *args, **kwargs)
        GraphicsItem.__init__(self)

        # cache bounding rect and geometry
        self._boundingRectCache = self._previousGeometry = None
        self._painterPathCache = None
        self.geometryChanged.connect(self._resetCachedProperties)

        # done by GraphicsItem init
        # GraphicsScene.registerObject(self)  # workaround for pyqt bug in GraphicsScene.items()

    # Removed due to https://bugreports.qt-project.org/browse/PYSIDE-86
    # def itemChange(self, change, value):
    #     # BEWARE: Calling QGraphicsWidget.itemChange can lead to crashing!
    #     # ret = QtWidgets.QGraphicsWidget.itemChange(self, change, value)  # segv occurs here
    #     # The default behavior is just to return the value argument, so we'll do that
    #     # without calling the original method.
    #     ret = value
    #     if change in [self.ItemParentHasChanged, self.ItemSceneHasChanged]:
    #         self._updateView()
    #     return ret

    def _resetCachedProperties(self):
        self._boundingRectCache = self._previousGeometry = None
        self._painterPathCache = None

    def setFixedHeight(self, h):
        self.setMaximumHeight(h)
        self.setMinimumHeight(h)

    def setFixedWidth(self, h):
        self.setMaximumWidth(h)
        self.setMinimumWidth(h)

    def height(self):
        return self.geometry().height()

    def width(self):
        return self.geometry().width()

    def boundingRect(self):
        geometry = self.geometry()
        if geometry != self._previousGeometry:
            self._painterPathCache = None
            br = self.mapRectFromParent(geometry).normalized()
            self._boundingRectCache = br
            self._previousGeometry = geometry
        else:
            br = self._boundingRectCache
        return br

    def shape(self):
        p = self._painterPathCache
        if p is None:
            self._painterPathCache = p = QtGui.QPainterPath()
            p.addRect(self.boundingRect())
        return p
