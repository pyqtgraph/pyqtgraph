import itertools
import warnings

import numpy as np

from .. import Qt, colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GradientEditorItem import Gradients  # List of colormaps
from .GraphicsObject import GraphicsObject

__all__ = ['PColorMeshItem']


class QuadInstances:
    def __init__(self):
        self.polys = []

    def alloc(self, size):
        self.polys.clear()

        # 2 * (size + 1) vertices, (x, y)
        arr = np.empty((2 * (size + 1), 2), dtype=np.float64)
        ptrs = list(map(Qt.compat.wrapinstance,
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
            each time ``setData()`` is called - unless ``enableAutoLevels=False``.
        enableAutoLevels: bool, optional, default True
            Causes the colormap levels to autoscale whenever ``setData()`` is called. 
            When enableAutoLevels is set to True, it is still possible to disable autoscaling
            on a per-change-basis by using ``autoLevels=False`` when calling ``setData()``.
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
        self.enableautolevels = kwargs.get('enableAutoLevels', True)
        
        if 'colorMap' in kwargs:
            cmap = kwargs.get('colorMap')
            if not isinstance(cmap, colormap.ColorMap):
                raise ValueError('colorMap argument must be a ColorMap instance')
            self.cmap = cmap
        elif 'cmap' in kwargs:
            # legacy unadvertised argument for backwards compatibility.
            # this will only use colormaps from Gradients.
            # Note that the colors will be wrong for the hsv colormaps.
            warnings.warn(
                "The parameter 'cmap' will be removed in a version of PyQtGraph released after Nov 2022.",
                DeprecationWarning, stacklevel=2
            )
            cmap = kwargs.get('cmap')
            if not isinstance(cmap, str) or cmap not in Gradients:
                raise NameError('Undefined colormap, should be one of the following: '+', '.join(['"'+i+'"' for i in Gradients.keys()])+'.')
            pos, color = zip(*Gradients[cmap]['ticks'])
            self.cmap = colormap.ColorMap(pos, color)
        else:
            self.cmap = colormap.get('viridis')

        lut_qcolor = self.cmap.getLookupTable(nPts=256, mode=self.cmap.QCOLOR)
        self.lut_qbrush = [QtGui.QBrush(x) for x in lut_qcolor]

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
            ValueError('Data must been sent as (z) or (x, y, z)')


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
        autoLevels: bool, optional, default True
            When set to True, PColorMeshItem will automatically select levels
            based on the minimum and maximum values encountered in the data along the z axis.
            The minimum and maximum levels are mapped to the lowest and highest colors 
            in the colormap. The autoLevels parameter is ignored if ``enableAutoLevels is False`` 
        """
        autoLevels = kwargs.get('autoLevels', True)

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

        if len(args)==0:
            # No data was received.
            if self.z is None:
                # No data is currently displayed, 
                # so other settings (like colormap) can not be updated
                return
        else:
            # Got new data. Prepare it for plotting
            self._prepareData(args)


        self.qpicture = QtGui.QPicture()
        painter = QtGui.QPainter(self.qpicture)
        # We set the pen of all polygons once
        if self.edgecolors is None:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
        else:
            painter.setPen(self.edgecolors)
            if self.antialiasing:
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                

        ## Prepare colormap
        # First we get the LookupTable
        lut = self.lut_qbrush
        # Second we associate each z value, that we normalize, to the lut
        scale = len(lut) - 1
        # Decide whether to autoscale the colormap or use the same levels as before
        if (self.levels is None) or (self.enableautolevels and autoLevels):
            # Autoscale colormap 
            z_min = self.z.min()
            z_max = self.z.max()
            self.setLevels( (z_min, z_max), update=False)
        else:
            # Use consistent colormap scaling
            z_min = self.levels[0]
            z_max = self.levels[1]
        rng = z_max - z_min
        if rng == 0:
            rng = 1
        norm = fn.rescaleData(self.z, scale / rng, z_min,
            dtype=int, clip=(0, len(lut)-1))

        if Qt.QT_LIB.startswith('PyQt'):
            drawConvexPolygon = lambda x : painter.drawConvexPolygon(*x)
        else:
            drawConvexPolygon = painter.drawConvexPolygon

        memory = self.quads.array(self.z.shape[1])
        polys = self.quads.instances()

        # Go through all the data and draw the polygons accordingly
        for i in range(self.z.shape[0]):
            # populate 2 rows of values into points
            memory[..., 0] = self.x[i:i+2, :]
            memory[..., 1] = self.y[i:i+2, :]

            brushes = [lut[z] for z in norm[i].tolist()]

            for brush, poly in zip(brushes, polys):
                painter.setBrush(brush)
                drawConvexPolygon(poly)

        painter.end()
        self.update()

        self.prepareGeometryChange()
        if shapeChanged:
            self.informViewBoundsChanged()



    def _updateDisplayWithCurrentState(self, *args, **kargs):
        ## Used for re-rendering mesh from self.z.
        ## For example when a new colormap is applied, or the levels are adjusted

        defaults = {
            'autoLevels': False,
        }
        defaults.update(kargs)
        return self.setData(*args, **defaults)



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
            self._updateDisplayWithCurrentState()



    def getLevels(self):
        """
        Returns a tuple containing the current level settings. See :func:`~setLevels`.
        The format is ``(low, high)``.
        """
        return self.levels


    
    def setLookupTable(self, lut, update=True):
        if lut is not self.lut_qbrush:
            self.lut_qbrush = [QtGui.QBrush(x) for x in lut]
            if update:
                self._updateDisplayWithCurrentState()



    def getColorMap(self):
        return self.cmap



    def enableAutoLevels(self):
        self.enableautolevels = True



    def disableAutoLevels(self):
        self.enableautolevels = False



    def paint(self, p, *args):
        if self.z is None:
            return

        p.drawPicture(0, 0, self.qpicture)


    def setBorder(self, b):
        self.border = fn.mkPen(b)
        self.update()



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
