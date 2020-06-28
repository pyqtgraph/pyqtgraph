from __future__ import division

from ..Qt import QtGui, QtCore
import numpy as np
from .. import functions as fn
from .. import debug as debug
from .GraphicsObject import GraphicsObject
from ..Point import Point
from .. import getConfigOption
from .GradientEditorItem import Gradients
from ..colormap import ColorMap

try:
    from collections.abc import Callable
except ImportError:
    # fallback for python < 3.3
    from collections import Callable

__all__ = ['PColorMeshItem']


class PColorMeshItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    Create a pseudocolor plot with convex polygons.
    """

    sigImageChanged = QtCore.Signal()
    sigRemoveRequested = QtCore.Signal(object)  # self; emitted when 'remove' is selected from context menu


    def __init__(self, x=None, y=None, z=None,
                 cmap='viridis', edgecolors=None):
        """


        Parameters
        ----------
        x, y : np.ndarray
            2D array containing the coordinates of the polygons
        z : np.ndarray
            2D array containing the value which will be maped into the polygons
            colors.
        cmap : str, default 'viridis
            Colormap used to map the z value to colors.
        edgecolors : dict , default None
            The color of the edges of the polygons.
            Default None means no edges.
            The dict may contains any arguments accepted by :func:`mkColor() <pyqtgraph.mkColor>.
            Example:
                mkPen(color='w', width=2)
        """
        GraphicsObject.__init__(self)

        self.x = x
        self.y = y
        self.z = z

        self.qpicture = None  ## rendered image for display
        
        self.axisOrder = getConfigOption('imageAxisOrder')

        self.edgecolors = edgecolors
        if cmap in Gradients.keys():
            self.cmap = cmap
        else:
            raise NameError('Undefined colormap')
            
        # If some data have been sent we directly display it
        if x is not None and y is not None and z is not None:
            self.setData(x, y, z)


    def setData(self, x, y, z):
        

        # We test of the view has changed
        if np.any(self.x != x) or np.any(self.y != y) or np.any(self.z != z):
            self.informViewBoundsChanged()

        # Replace data
        self.x = x
        self.y = y
        self.z = z

        profile = debug.Profiler()

        self.qpicture = QtGui.QPicture()
        p = QtGui.QPainter(self.qpicture)
        
        # We set the pen of all polygons once
        if self.edgecolors is None:
            p.setPen(QtGui.QColor(0, 0, 0, 0))
        else:
            p.setPen(fn.mkPen(self.edgecolors))

        ## Prepare colormap
        # First we get the LookupTable
        pos   = [i[0] for i in Gradients[self.cmap]['ticks']]
        color = [i[1] for i in Gradients[self.cmap]['ticks']]
        cmap  = ColorMap(pos, color)
        lut   = cmap.getLookupTable(0.0, 1.0, 256)
        # Second we associate each z value, that we normalize, to the lut
        norm  = z - z.min()
        norm = norm/norm.max()
        norm  = (norm*(len(lut)-1)).astype(int)
        
        # Go through all the data and draw the polygons accordingly
        for xi in range(z.shape[0]):
            for yi in range(z.shape[1]):
                
                # Set the color of the polygon first
                # print(xi, yi, norm[xi][yi])
                c = lut[norm[xi][yi]]
                p.setBrush(QtGui.QColor(c[0], c[1], c[2]))

                # DrawConvexPlygon is faster
                p.drawConvexPolygon(QtCore.QPointF(x[xi][yi],     y[xi][yi]),
                                    QtCore.QPointF(x[xi+1][yi],   y[xi+1][yi]),
                                    QtCore.QPointF(x[xi+1][yi+1], y[xi+1][yi+1]),
                                    QtCore.QPointF(x[xi][yi+1],   y[xi][yi+1]))


        p.end()
        self.update()



    def paint(self, p, *args):
        profile = debug.Profiler()
        if self.z is None:
            return

        profile('p.drawPicture')
        p.drawPicture(0, 0, self.qpicture)



    def setBorder(self, b):
        self.border = fn.mkPen(b)
        self.update()



    def width(self):
        if self.x is None:
            return None
        return np.max(self.x)



    def height(self):
        if self.y is None:
            return None
        return np.max(self.y)




    def boundingRect(self):

        if self.z is None:
            return QtCore.QRectF(0., 0., 0., 0.)
        return QtCore.QRectF(0., 0., float(self.width()), float(self.height()))
