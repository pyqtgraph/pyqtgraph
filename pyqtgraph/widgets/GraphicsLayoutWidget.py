from ..Qt import QtGui
from ..graphicsItems.GraphicsLayout import GraphicsLayout
from .GraphicsView import GraphicsView

__all__ = ['GraphicsLayoutWidget']
class GraphicsLayoutWidget(GraphicsView):
    """
    Convenience class consisting of a :class:`GraphicsView
    <pyqtgraph.GraphicsView>` with a single :class:`GraphicsLayout
    <pyqtgraph.GraphicsLayout>` as its central item.

    This class wraps several methods from its internal GraphicsLayout:
    :func:`nextRow <pyqtgraph.GraphicsLayout.nextRow>`
    :func:`nextColumn <pyqtgraph.GraphicsLayout.nextColumn>`
    :func:`addPlot <pyqtgraph.GraphicsLayout.addPlot>`
    :func:`addViewBox <pyqtgraph.GraphicsLayout.addViewBox>`
    :func:`addItem <pyqtgraph.GraphicsLayout.addItem>`
    :func:`getItem <pyqtgraph.GraphicsLayout.getItem>`
    :func:`addLabel <pyqtgraph.GraphicsLayout.addLabel>`
    :func:`addLayout <pyqtgraph.GraphicsLayout.addLayout>`
    :func:`removeItem <pyqtgraph.GraphicsLayout.removeItem>`
    :func:`itemIndex <pyqtgraph.GraphicsLayout.itemIndex>`
    :func:`clear <pyqtgraph.GraphicsLayout.clear>`
    """
    def __init__(self, parent=None, **kargs):
        GraphicsView.__init__(self, parent)
        self.ci = GraphicsLayout(**kargs)
        for n in ['nextRow', 'nextCol', 'nextColumn', 'addPlot', 'addViewBox', 'addItem', 'getItem', 'addLayout', 'addLabel', 'removeItem', 'itemIndex', 'clear']:
            setattr(self, n, getattr(self.ci, n))
        self.setCentralItem(self.ci)
