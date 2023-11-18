import numpy as np

from .. import Qt, colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject

__all__ = ['PColorMeshItem']


class QuadInstances:
    def __init__(self):
        self.nrows = -1
        self.ncols = -1
        self.pointsarray = Qt.internals.PrimitiveArray(QtCore.QPointF, 2)
        self.resize(0, 0)

    def resize(self, nrows, ncols):
        if nrows == self.nrows and ncols == self.ncols:
            return

        self.nrows = nrows
        self.ncols = ncols

        # (nrows + 1) * (ncols + 1) vertices, (x, y)
        self.pointsarray.resize((nrows+1)*(ncols+1))
        points = self.pointsarray.instances()
        # points is a flattened list of a 2d array of
        # QPointF(s) of shape (nrows+1, ncols+1)

        # pre-create quads from those instances of QPointF(s).
        # store the quads as a flattened list of a 2d array
        # of polygons of shape (nrows, ncols)
        polys = []
        for r in range(nrows):
            for c in range(ncols):
                bl = points[(r+0)*(ncols+1)+(c+0)]
                tl = points[(r+0)*(ncols+1)+(c+1)]
                br = points[(r+1)*(ncols+1)+(c+0)]
                tr = points[(r+1)*(ncols+1)+(c+1)]
                poly = (bl, br, tr, tl)
                polys.append(poly)
        self.polys = polys

    def ndarray(self):
        return self.pointsarray.ndarray()

    def instances(self):
        return self.polys


class PColorMeshItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    """

    sigLevelsChanged = QtCore.Signal(object)  # emits tuple with levels (low,high) when color levels are changed.

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

            "ASCII from: <https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.pcolormesh.html>".
        colorMap : pyqtgraph.ColorMap
            Colormap used to map the z value to colors.
            default ``pyqtgraph.colormap.get('viridis')``
        levels: tuple, optional, default None
            Sets the minimum and maximum values to be represented by the colormap (min, max). 
            Values outside this range will be clipped to the colors representing min or max.
            ``None`` disables the limits, meaning that the colormap will autoscale 
            the next time ``setData()`` is called with new data.
        enableAutoLevels: bool, optional, default True
            Causes the colormap levels to autoscale whenever ``setData()`` is called. 
            It is possible to override this value on a per-change-basis by using the
            ``autoLevels`` keyword argument when calling ``setData()``.
            If ``enableAutoLevels==False`` and ``levels==None``, autoscaling will be 
            performed once when the first z data is supplied. 
        edgecolors : dict, optional
            The color of the edges of the polygons.
            Default None means no edges.
            Only cosmetic pens are supported.
            The dict may contains any arguments accepted by :func:`mkColor() <pyqtgraph.mkColor>`.
            Example: ``mkPen(color='w', width=2)``
        antialiasing : bool, default False
            Whether to draw edgelines with antialiasing.
            Note that if edgecolors is None, antialiasing is always False.
        """

        GraphicsObject.__init__(self)

        self.qpicture = None  ## rendered picture for display
        self.x = None
        self.y = None
        self.z = None
        self._dataBounds = None

        self.edgecolors = kwargs.get('edgecolors', None)
        if self.edgecolors is not None:
            self.edgecolors = fn.mkPen(self.edgecolors)
            # force the pen to be cosmetic. see discussion in
            # https://github.com/pyqtgraph/pyqtgraph/pull/2586
            self.edgecolors.setCosmetic(True)
        self.antialiasing = kwargs.get('antialiasing', False)
        self.levels = kwargs.get('levels', None)
        self._defaultAutoLevels = kwargs.get('enableAutoLevels', True)
        
        if 'colorMap' in kwargs:
            cmap = kwargs.get('colorMap')
            if not isinstance(cmap, colormap.ColorMap):
                raise ValueError('colorMap argument must be a ColorMap instance')
            self.cmap = cmap
        else:
            self.cmap = colormap.get('viridis')

        self.lut_qcolor = self.cmap.getLookupTable(nPts=256, mode=self.cmap.QCOLOR)

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

            self._dataBounds = None
            
        # User only specified z
        elif len(args)==1:
            # If x and y is None, the polygons will be displaced on a grid
            x = np.arange(0, args[0].shape[0]+1, 1)
            y = np.arange(0, args[0].shape[1]+1, 1)
            self.x, self.y = np.meshgrid(x, y, indexing='ij')
            self.z = args[0]

            self._dataBounds = ((x[0], x[-1]), (y[0], y[-1]))

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

            xmn, xmx = np.min(self.x), np.max(self.x)
            ymn, ymx = np.min(self.y), np.max(self.y)
            self._dataBounds = ((xmn, xmx), (ymn, ymx))

        else:
            raise ValueError('Data must been sent as (z) or (x, y, z)')


    def setData(self, *args, **kwargs):
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
        autoLevels: bool, optional
            If set, overrides the value of ``enableAutoLevels``
        """
        old_bounds = self._dataBounds
        self._prepareData(args)
        boundsChanged = old_bounds != self._dataBounds

        self._rerender(
            autoLevels=kwargs.get('autoLevels', self._defaultAutoLevels)
        )

        if boundsChanged:
            self.prepareGeometryChange()
            self.informViewBoundsChanged()

        self.update()

    def _rerender(self, *, autoLevels):
        self.qpicture = None
        if self.z is not None:
            if (self.levels is None) or autoLevels:
                # Autoscale colormap
                z_min = self.z.min()
                z_max = self.z.max()
                self.setLevels( (z_min, z_max), update=False)
            self.qpicture = self._drawPicture()

    def _drawPicture(self) -> QtGui.QPicture:
        # on entry, the following members are all valid: x, y, z, levels
        # this function does not alter any state (besides using self.quads)

        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        # We set the pen of all polygons once
        if self.edgecolors is None:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
        else:
            painter.setPen(self.edgecolors)
            if self.antialiasing:
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        ## Prepare colormap
        # First we get the LookupTable
        lut = self.lut_qcolor
        # Second we associate each z value, that we normalize, to the lut
        scale = len(lut) - 1
        lo, hi = self.levels[0], self.levels[1]
        rng = hi - lo
        if rng == 0:
            rng = 1
        norm = fn.rescaleData(self.z, scale / rng, lo, dtype=int, clip=(0, len(lut)-1))

        if Qt.QT_LIB.startswith('PyQt'):
            drawConvexPolygon = lambda x : painter.drawConvexPolygon(*x)
        else:
            drawConvexPolygon = painter.drawConvexPolygon

        self.quads.resize(self.z.shape[0], self.z.shape[1])
        memory = self.quads.ndarray()
        memory[..., 0] = self.x.ravel()
        memory[..., 1] = self.y.ravel()
        polys = self.quads.instances()

        # group indices of same coloridx together
        color_indices, counts = np.unique(norm, return_counts=True)
        sorted_indices = np.argsort(norm, axis=None)

        offset = 0
        for coloridx, cnt in zip(color_indices, counts):
            indices = sorted_indices[offset:offset+cnt]
            offset += cnt
            painter.setBrush(lut[coloridx])
            for idx in indices:
                drawConvexPolygon(polys[idx])

        painter.end()
        return picture


    def setLevels(self, levels, update=True):
        """
        Sets color-scaling levels for the mesh. 
        
        Parameters
        ----------
            levels: tuple
                ``(low, high)`` 
                sets the range for which values can be represented in the colormap.
            update: bool, optional
                Controls if mesh immediately updates to reflect the new color levels.
        """
        self.levels = levels
        self.sigLevelsChanged.emit(levels)
        if update:
            self._rerender(autoLevels=False)
            self.update()

    def getLevels(self):
        """
        Returns a tuple containing the current level settings. See :func:`~setLevels`.
        The format is ``(low, high)``.
        """
        return self.levels


    
    def setLookupTable(self, lut, update=True):
        self.cmap = None    # invalidate since no longer consistent with lut
        self.lut_qcolor = lut[:]
        if update:
            self._rerender(autoLevels=False)
            self.update()

    def getColorMap(self):
        return self.cmap

    def setColorMap(self, cmap):
        self.setLookupTable(cmap.getLookupTable(nPts=256, mode=cmap.QCOLOR), update=True)
        self.cmap = cmap

    def enableAutoLevels(self):
        self._defaultAutoLevels = True

    def disableAutoLevels(self):
        self._defaultAutoLevels = False

    def paint(self, p, *args):
        if self.qpicture is not None:
            p.drawPicture(0, 0, self.qpicture)

    def width(self):
        if self._dataBounds is None:
            return 0
        bounds = self._dataBounds[0]
        return bounds[1]-bounds[0]

    def height(self):
        if self._dataBounds is None:
            return 0
        bounds = self._dataBounds[1]
        return bounds[1]-bounds[0]

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if self._dataBounds is None:
            return (None, None)
        return self._dataBounds[ax]

    def pixelPadding(self):
        # pen is known to be cosmetic
        pen = self.edgecolors
        no_pen = (pen is None) or (pen.style() == QtCore.Qt.PenStyle.NoPen)
        return 0 if no_pen else (pen.widthF() or 1) * 0.5

    def boundingRect(self):
        xmn, xmx = self.dataBounds(ax=0)
        if xmn is None or xmx is None:
            return QtCore.QRectF()
        ymn, ymx = self.dataBounds(ax=1)
        if ymn is None or ymx is None:
            return QtCore.QRectF()

        px = py = 0
        pxPad = self.pixelPadding()
        if pxPad > 0:
            # determine length of pixel in local x, y directions
            px, py = self.pixelVectors()
            px = 0 if px is None else px.length()
            py = 0 if py is None else py.length()
            # return bounds expanded by pixel size
            px *= pxPad
            py *= pxPad

        return QtCore.QRectF(xmn-px, ymn-py, (2*px)+xmx-xmn, (2*py)+ymx-ymn)
