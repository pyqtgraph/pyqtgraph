from ..Qt import QtWidgets
from .GraphicsItem import GraphicsItem
from ..GraphicsScene.GraphicsScene import GraphicsScene
from typing import TYPE_CHECKING

__all__ = ['GraphicsWidget']


class GraphicsWidget(GraphicsItem, QtWidgets.QGraphicsWidget):
    def __init__(self, *args, **kwargs):
        """
        **Bases:** :class:`GraphicsItem <pyqtgraph.GraphicsItem>`, :class:`QtWidgets.QGraphicsWidget`
        
        Extends QGraphicsWidget with several helpful methods.
        Most of the extra functionality is inherited from :class:`GraphicsItem <pyqtgraph.GraphicsItem>`.
        """
        QtWidgets.QGraphicsWidget.__init__(self, *args, **kwargs)
        GraphicsItem.__init__(self)

    if TYPE_CHECKING:
        def scene(self) -> GraphicsScene: ...

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

    # The default implementations of boundingRect() and shape()
    # provided by QGraphicsWidget are sufficient unless your
    # subclass sets its own transform.
    # Sample implementations to override in your subclass are shown below.

    # def boundingRect(self):
    #     return self.mapRectFromParent(self.geometry()).normalized()

    # def shape(self):
    #     path = QtGui.QPainterPath()
    #     path.addRect(self.boundingRect())
    #     return path
