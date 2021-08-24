# -*- coding: utf-8 -*-
import itertools
from ..Qt import QtGui, QtCore
from .. import Qt
import numpy as np
from .. import functions as fn
from .GraphicsObject import GraphicsObject
from .GradientEditorItem import Gradients # List of colormaps
from .. import colormap


__all__ = ['PColorMeshItem']


if Qt.QT_LIB.startswith('PyQt'):
    wrapinstance = Qt.sip.wrapinstance
else:
    wrapinstance = Qt.shiboken.wrapInstance


class QuadInstances:
    def __init__(self):
        self.polys = []

    def alloc(self, size):
        self.polys.clear()

        # 2 * (size + 1) vertices, (x, y)
        arr = np.empty((2 * (size + 1), 2), dtype=np.float64)
        ptrs = list(map(wrapinstance,
            itertools.count(arr.ctypes.data, arr.strides[0]),
            itertools.repeat(QtCore.QPointF, arr.shape[0])))

        # arrange into 2 rows, (size + 1) vertices
        points = [ptrs[:len(ptrs)//2], ptrs[len(ptrs)//2:]]
        self.arr = arr.reshape((2, -1, 2))

        # pre-create quads from those 2 rows of QPointF(s)
        for j in range(size):
            bl, tl = points[0][j:j+2]
            br, tr = points[1][j:j+2]
            poly = (bl, br, tr, tl)
            self.polys.append(poly)

    def array(self, size):
        if size != len(self.polys):
            self.alloc(size)
        return self.arr

    def instances(self):
        return self.polys


class PColorMeshItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    """


    def __init__(self, *args, **kwargs):
        """
        Create a pseudocolor plot with convex polygons.

        Call signature:

        ``PColorMeshItem([x, y,] z, **kwargs)``

        x and y can be used to specify the corners of the quadrilaterals.
        z must be used to specified to color of the quadrilaterals.

        Parameters
        ----------
        x, y : np.ndarray, optional, default None
            2D array containing the coordinates of the polygons
        z : np.ndarray
            2D array containing the value which will be mapped into the polygons
            colors.
            If x and y is None, the polygons will be displaced on a grid
            otherwise x and y will be used as polygons vertices coordinates as::

                (x[i+1, j], y[i+1, j])           (x[i+1, j+1], y[i+1, j+1])
                                    +---------+
                                    | z[i, j] |
                                    +---------+
                    (x[i, j], y[i, j])           (x[i, j+1], y[i, j+1])

            "ASCII from: <https://matplotlib.org/3.2.1/api/_as_gen/
                         matplotlib.pyplot.pcolormesh.html>".
        cmap : pg.ColorMap, default 'viridis'
            Colormap used to map the z value to colors.
        edgecolors : dict, default None
            The color of the edges of the polygons.
            Default None means no edges.
            The dict may contains any arguments accepted by :func:`mkColor() <pyqtgraph.mkColor>`.
            Example:

                ``mkPen(color='w', width=2)``

        antialiasing : bool, default False
            Whether to draw edgelines with antialiasing.
            Note that if edgecolors is None, antialiasing is always False.
        """

        GraphicsObject.__init__(self)

        self.qpicture = None  ## rendered picture for display
        self.x = None
        self.y = None
        self.z = None
        
        self.edgecolors = kwargs.get('edgecolors', None)
        self.antialiasing = kwargs.get('antialiasing', False)
        
        self.cmap = kwargs.get('cmap', None)
        if self.cmap is None:
            self.cmap = colormap.get('viridis')

        # Allow specifying str for backwards compatibility
        # but this will only use colormaps from Gradients.
        # Note that the colors will be wrong for the hsv colormaps.
        if isinstance(self.cmap, str):
            if self.cmap in Gradients:
                pos, color = zip(*Gradients[self.cmap]['ticks'])
                self.cmap  = colormap.ColorMap(pos, color)
            else:
                raise NameError('Undefined colormap, should be one of the following: '+', '.join(['"'+i+'"' for i in Gradients.keys()])+'.')

        self.lut = self.cmap.getLookupTable(nPts=256, mode=self.cmap.QCOLOR)

        self.quads = QuadInstances()

        # If some data have been sent we directly display it
        if len(args)>0:
            self.setData(*args)


    def _prepareData(self, args):
        """
        Check the shape of the data.
        Return a set of 2d array x, y, z ready to be used to draw the picture.
        """

        # User didn't specified data
        if len(args)==0:

            self.x = None
            self.y = None
            self.z = None
            
        # User only specified z
        elif len(args)==1:
            # If x and y is None, the polygons will be displaced on a grid
            x = np.arange(0, args[0].shape[0]+1, 1)
            y = np.arange(0, args[0].shape[1]+1, 1)
            self.x, self.y = np.meshgrid(x, y, indexing='ij')
            self.z = args[0]

        # User specified x, y, z
        elif len(args)==3:

            # Shape checking
            if args[0].shape[0] != args[2].shape[0]+1 or args[0].shape[1] != args[2].shape[1]+1:
                raise ValueError('The dimension of x should be one greater than the one of z')
            
            if args[1].shape[0] != args[2].shape[0]+1 or args[1].shape[1] != args[2].shape[1]+1:
                raise ValueError('The dimension of y should be one greater than the one of z')
        
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]

        else:
            ValueError('Data must been sent as (z) or (x, y, z)')


    def setData(self, *args):
        """
        Set the data to be drawn.

        Parameters
        ----------
        x, y : np.ndarray, optional, default None
            2D array containing the coordinates of the polygons
        z : np.ndarray
            2D array containing the value which will be mapped into the polygons
            colors.
            If x and y is None, the polygons will be displaced on a grid
            otherwise x and y will be used as polygons vertices coordinates as::
                
                (x[i+1, j], y[i+1, j])           (x[i+1, j+1], y[i+1, j+1])
                                    +---------+
                                    | z[i, j] |
                                    +---------+
                    (x[i, j], y[i, j])           (x[i, j+1], y[i, j+1])

            "ASCII from: <https://matplotlib.org/3.2.1/api/_as_gen/
                         matplotlib.pyplot.pcolormesh.html>".
        """

        # Prepare data
        self._prepareData(args)

        # Has the view bounds changed
        shapeChanged = False
        if self.qpicture is None:
            shapeChanged = True
        elif len(args)==1:
            if args[0].shape[0] != self.x[:,1][-1] or args[0].shape[1] != self.y[0][-1]:
                shapeChanged = True
        elif len(args)==3:
            if np.any(self.x != args[0]) or np.any(self.y != args[1]):
                shapeChanged = True

        self.qpicture = QtGui.QPicture()
        p = QtGui.QPainter(self.qpicture)
        # We set the pen of all polygons once
        if self.edgecolors is None:
            p.setPen(QtCore.Qt.PenStyle.NoPen)
        else:
            p.setPen(fn.mkPen(self.edgecolors))
            if self.antialiasing:
                p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                

        ## Prepare colormap
        # First we get the LookupTable
        lut = self.lut
        # Second we associate each z value, that we normalize, to the lut
        scale = len(lut) - 1
        rng = self.z.ptp()
        if rng == 0:
            rng = 1
        norm = fn.rescaleData(self.z, scale / rng, self.z.min(),
            dtype=int, clip=(0, len(lut)-1))

        brush = QtGui.QBrush(QtCore.Qt.BrushStyle.SolidPattern)

        if Qt.QT_LIB.startswith('PyQt'):
            drawConvexPolygon = lambda x : p.drawConvexPolygon(*x)
        else:
            drawConvexPolygon = p.drawConvexPolygon

        memory = self.quads.array(self.z.shape[1])
        polys = self.quads.instances()

        # Go through all the data and draw the polygons accordingly
        for i in range(self.z.shape[0]):
            # populate 2 rows of values into points
            memory[..., 0] = self.x[i:i+2, :]
            memory[..., 1] = self.y[i:i+2, :]

            colors = [lut[z] for z in norm[i].tolist()]

            for color, poly in zip(colors, polys):
                brush.setColor(color)
                p.setBrush(brush)
                drawConvexPolygon(poly)

        p.end()
        self.update()

        self.prepareGeometryChange()
        if shapeChanged:
            self.informViewBoundsChanged()



    def paint(self, p, *args):
        if self.z is None:
            return

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
        if self.qpicture is None:
            return QtCore.QRectF(0., 0., 0., 0.)
        return QtCore.QRectF(self.qpicture.boundingRect())
