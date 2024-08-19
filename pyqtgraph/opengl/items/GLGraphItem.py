import numpy as np

from ... import functions as fn
from ...Qt import QtCore, QtGui
from ..GLGraphicsItem import GLGraphicsItem
from .GLScatterPlotItem import GLScatterPlotItem
from .GLLinePlotItem import GLLinePlotItem

__all__ = ['GLGraphItem']

class GLGraphItem(GLGraphicsItem):
    """A GLGraphItem displays graph information as
    a set of nodes connected by lines (as in 'graph theory', not 'graphics').
    Useful for drawing networks, trees, etc.
    """

    def __init__(self, parentItem=None, **kwds):
        super().__init__()
        glopts = kwds.pop('glOptions', 'translucent')

        self.edges = None
        self.edgeColor = QtGui.QColor(QtCore.Qt.GlobalColor.white)
        self.edgeWidth = 1.0

        self.lineplot = GLLinePlotItem(parentItem=self, glOptions=glopts, mode='lines')
        self.scatter = GLScatterPlotItem(parentItem=self, glOptions=glopts)
        self.setParentItem(parentItem)
        self.setData(**kwds)

    def setData(self, **kwds):
        """
        Change the data displayed by the graph. 

        Parameters
        ----------
        edges : np.ndarray
            2D array of shape (M, 2) of connection data, each row contains
            indexes of two nodes that are connected.  Dtype must be integer
            or unsigned.
        edgeColor: color_like, optional
            The color to draw edges. Accepts the same arguments as 
            :func:`~pyqtgraph.mkColor()`.  If None, no edges will be drawn.
            Default is (1.0, 1.0, 1.0, 0.5).
        edgeWidth: float, optional
            Value specifying edge width.  Default is 1.0
        nodePositions : np.ndarray
            2D array of shape (N, 3), where each row represents the x, y, z
            coordinates for each node
        nodeColor : np.ndarray or float or color_like, optional
            2D array of shape (N, 4) of dtype float32, where each row represents
            the R, G, B, A values in range of 0-1, or for the same color for all
            nodes, provide either QColor type or input for 
            :func:`~pyqtgraph.mkColor()`
        nodeSize : np.ndarray or float or int
            Either 2D numpy array of shape (N, 1) where each row represents the
            size of each node, or if a scalar, apply the same size to all nodes
        **kwds
            All other keyword arguments are given to
            :meth:`GLScatterPlotItem.setData() <pyqtgraph.opengl.GLScatterPlotItem.setData>`
            to affect the appearance of nodes (pos, color, size, pxMode, etc.)
        
        Raises
        ------
        TypeError
            When dtype of edges dtype is not unisnged or integer dtype
        """

        if 'edges' in kwds:
            self.edges = kwds.pop('edges')
            if self.edges.dtype.kind not in 'iu':
                raise TypeError("edges array must have int or unsigned dtype.")
        if 'edgeColor' in kwds:
            edgeColor = kwds.pop('edgeColor')
            self.edgeColor = fn.mkColor(edgeColor) if edgeColor is not None else None
        if 'edgeWidth' in kwds:
            self.edgeWidth = kwds.pop('edgeWidth')
        if 'nodePositions' in kwds:
            kwds['pos'] = kwds.pop('nodePositions')
        if 'nodeColor' in kwds:
            kwds['color'] = kwds.pop('nodeColor')
        if 'nodeSize' in kwds:
            kwds['size'] = kwds.pop('nodeSize')

        self.scatter.setData(**kwds)

        kwdLines = dict(width=self.edgeWidth)
        if self.scatter.pos is None or self.edges is None or self.edgeColor is None:
            kwdLines['pos'] = np.zeros((0, 3), dtype=np.float32)
            kwdLines['color'] = (0, 0, 0, 0)
        else:
            kwdLines['pos'] = self.scatter.pos[self.edges].reshape((-1, 3))
            kwdLines['color'] = self.edgeColor
        self.lineplot.setData(**kwdLines)

        self.update()
