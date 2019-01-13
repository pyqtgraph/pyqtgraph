from OpenGL.GL import *
from ..GLGraphicsItem import GLGraphicsItem
from .GLScatterPlotItem import GLScatterPlotItem
from ...Qt import QtGui
from ... import functions as fn


__all__ = ['GLGraphItem']

class GLGraphItem(GLGraphicsItem):
    """A GLGraphItem displays graph information as
    a set of nodes connected by lines (as in 'graph theory', not 'graphics').
    Useful for drawing networks, trees, etc.
    """

    def __init__(self, **kwds):
        GLGraphicsItem.__init__(self)

        self.edges = None
        self.edgeColor = (1.0, 1.0, 1.0, 0.5)
        self.edgeWidth = 1

        self.scatter = GLScatterPlotItem(parentItem=self)
        self.setData(**kwds)


    def setData(self, **kwds):
        """
        Change the data displayed by the graph. 

        ==============  =======================================================================
        **Arguments:**
        edges           (M,2) array of connection data. Each row contains indexes
                        of two nodes that are connected.
        edgeColor       The color to use when drawing lines between connected
                        nodes. AMay be one of:
                        * an array [red, green blue, alpha]
                        * None (to disable edge drawing)
                        * [not available yet] a record array of length M
                          with fields (red, green, blue, alpha, width). Note
                          that using this option may have a significant performance
                          cost.
        edgeWidth       float specifying edge width

        nodePositions   (N,3) array of the positions of each node in the graph.
                        (overwrites pos)
        nodeColor       (N,4) array of floats (0.0-1.0) specifying spot colors 
                        OR a tuple of floats specifying a single color for all spots.
                        (overwrites color)
        nodeSize        (N,) array of floats specifying spot sizes 
                        OR a single value to apply to all spots. (overwrites size)
        ``**opts``      All other keyword arguments are given to
                        :func:`GLScatterPlotItem.setData() <pyqtgraph.GLScatterPlotItem.setData>`
                        to affect the appearance of nodes (pos, color, size, pxMode, etc.)
        ==============  =======================================================================
        """
        if 'edges' in kwds:
            self.edges = kwds.pop('edges')
            if self.edges.dtype.kind not in 'iu':
                raise Exception("edges array must have int or unsigned type.")
        if 'edgeColor' in kwds:
            self.edgeColor = kwds.pop('edgeColor')
        if 'edgeWidth' in kwds:
            self.edgeWidth = kwds.pop('edgeWidth')
        
        if 'nodePositions' in kwds:
            kwds['pos'] = kwds.pop('nodePositions')
        if 'nodeColor' in kwds:
            kwds['color'] = kwds.pop('nodeColor')
        if 'nodeSize' in kwds:
            kwds['size'] = kwds.pop('nodeSize')
        self.scatter.setData(**kwds)


    def initializeGL(self):
        self.scatter.initializeGL()


    def paint(self):
        if self.scatter.pos is not None \
                and self.edges is not None \
                and self.edgeColor is not None:
            verts = self.scatter.pos
            edges = self.edges.flatten()
            glEnableClientState(GL_VERTEX_ARRAY)
            try:
                if isinstance(self.edgeColor, QtGui.QColor):
                    glColor4f(*fn.glColor(self.edgeColor))
                else:
                    glColor4f(*self.edgeColor)
                glLineWidth(self.edgeWidth)

                glVertexPointerf(verts)
                glDrawElements(GL_LINES, edges.shape[0], GL_UNSIGNED_INT, edges)
            finally:
                glDisableClientState(GL_VERTEX_ARRAY)
