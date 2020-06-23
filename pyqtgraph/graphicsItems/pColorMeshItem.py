from __future__ import division

from ..Qt import QtGui, QtCore
import numpy as np
from .. import functions as fn
from .. import debug as debug
from .GraphicsObject import GraphicsObject
from ..Point import Point
from .. import getConfigOption

try:
    from collections.abc import Callable
except ImportError:
    # fallback for python < 3.3
    from collections import Callable

__all__ = ['pColorMeshItem']


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class pColorMeshItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    TODO
    """

    sigImageChanged = QtCore.Signal()
    sigRemoveRequested = QtCore.Signal(object)  # self; emitted when 'remove' is selected from context menu


    def __init__(self, x=None, y=None, z=None, cmap=None):
        """
        See :func:`setImage <pyqtgraph.ImageItem.setImage>` for all allowed initialization arguments.
        """
        GraphicsObject.__init__(self)

        self.x = x
        self.y = y
        self.z = z
        self.qpicture = None  ## rendered image for display

        self.axisOrder = getConfigOption('imageAxisOrder')


        if cmap is None:
            self.cmap = plt.cm.viridis

        if x is not None and y is not None and z is not None:
            self.setData(x, y, z)


    def setData(self, x, y, z):
        ## pre-computing a QPicture object allows paint() to run much more quickly, 
        ## rather than re-drawing the shapes every time.
        profile = debug.Profiler()

        self.qpicture = QtGui.QPicture()
        p = QtGui.QPainter(self.qpicture)
        p.setPen(fn.mkPen('w'))
        
        xfn = z.shape[0]
        yfn = z.shape[1]

        norm = Normalize(vmin=z.min(), vmax=z.max())
        for xi in range(xfn):
            for yi in range(yfn):
                
                p.drawConvexPolygon(QtCore.QPointF(x[xi][yi],     y[xi][yi]),
                                    QtCore.QPointF(x[xi+1][yi],   y[xi+1][yi]),
                                    QtCore.QPointF(x[xi+1][yi+1], y[xi+1][yi+1]),
                                    QtCore.QPointF(x[xi][yi+1],   y[xi][yi+1]))

                c = self.cmap(norm(z[xi][yi]))[:-1]
                p.setBrush(QtGui.QColor(c[0]*255, c[1]*255, c[2]*255))
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
        return len(self.x)



    def height(self):
        if self.y is None:
            return None
        return len(self.y)




    def boundingRect(self):

        if self.z is None:
            return QtCore.QRectF(0., 0., 0., 0.)
        return QtCore.QRectF(0., 0., float(self.width()), float(self.height()))