import math
import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt

from .. import Qt
from .. import functions as fn
from .. import mkBrush, mkPen
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .HistogramLUTItem import HistogramLUTItem

__all__ = ['NonUniformImage']

class NonUniformImage(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    GraphicsObject displaying an image with non-uniform sample points. It's
    commonly used to display 2-d or slices of higher dimensional data that
    have a regular but non-uniform grid e.g. measurements or simulation results.

    """


    def __init__(self, x, y, z, border=None, **kwargs):
        """
        Create an image plot by drawing rectangles with specified x and y
        boundaries.

        Call signature

        ``NonUniformImage(x, y, z, border)``

        Parameters
        ----------
        x, y : array_like
            1D array_like of monotonically increasing values of where the centers
            of the 'pixels' should be. Exception for first and last element values
            in the array that represent the boundaries.
        z : array_like
            2D array_like where the shape must equal (x.size, y.size). Value
            represents the color that should be used::

                                   x[i]        x[i+1]
                                    |            |
                            +-------------+-------------+
                            |   z[i, j]   |  z[i+1, j]  |- y[j]
                            +-------------+-------------+
                            |  z[i, j-1]  | z[i+1, j-1] |- y[j-1]
                            +-------------+-------------+

        border : color_like, optional
            Argument that can be relayed to :func:`mkPen <pyqtgraph.mkPen>`
            that represents the border of the border drawn around the
            NonUniformImageItem.  Default is None

        See Also
        --------
        :class:`PColorMeshItem <pyqtgraph.PColorMeshItem>` - This item provides
        similar functionality but instead of rectangles, it uses Polygons for
        more customizable geometry.
        """

        super().__init__()

                # convert to numpy arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1-d arrays.")

        self._dataBounds = None

        border = kwargs.get("edgecolors")
        if border is None:
            self.edgecolors = border
        else:
            self.edgecolors = fn.mkPen(border)
            # force the pen to be cosmetic. see discussion in
            # https://github.com/pyqtgraph/pyqtgraph/pull/2586
            self.edgecolors.setCosmetic(True)

        self.antialiasing = kwargs.get("antialiasing", False)

        if np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0):
            raise ValueError("The values in x and y must be monotonically increasing.")

        if len(z.shape) != 2 or z.shape != (x.size, y.size):
            raise ValueError("The length of x and y must match the shape of z.")


        # default colormap (black - white)
        self.cmap = ColorMap(pos=[0.0, 1.0], color=[(0, 0, 0), (255, 255, 255)])

        self.data = (x, y, z)
        self._dataBounds = ((x[0], x[-1]), (y[0], y[-1]))
        self.levels = None
        self.lut = None
        self.picture = None

        self.update()

    def setLookupTable(self, lut, update=True, **kwargs):
        # backwards compatibility hack
        if isinstance(lut, HistogramLUTItem):
            warnings.warn(
                "NonUniformImage::setLookupTable(HistogramLUTItem) is deprecated "
                "and will be removed in a future version of PyQtGraph. "
                "use HistogramLUTItem::setImageItem(NonUniformImage) instead",
                DeprecationWarning, stacklevel=2
            )
            lut.setImageItem(self)
            return

        self.lut = lut
        self.picture = None
        if update:
            self.update()

    def setColorMap(self, cmap):
        """
        Set the colormap for the NonUniformImage

        Parameters
        ----------
        cmap
            Specify the colormap to be used.
        """
        self.cmap = cmap
        self.picture = None
        self.update()

    def getHistogram(self, **kwds):
        """
        Returns x and y arrays containing the histogram values for the current image.
        For an explanation of the return format, see numpy.histogram().
        """
        # hist = np.histogram(self.z, **kwds)
        z = self.data[2]
        z = z[np.isfinite(z)]
        hist = np.histogram(z, **kwds)

        return hist[1][:-1], hist[0]

    def setLevels(self, levels):
        self.levels = levels
        self.picture = None
        self.update()

    def getLevels(self):
        if self.levels is None:
            z = self.data[2]
            z = z[np.isfinite(z)]
            self.levels = z.min(), z.max()
        return self.levels

    def generatePicture(self):
        x, y, z = self.data

        # pad x and y so that we don't need to special-case the edges
        x = np.pad(x, 1, mode='edge')
        y = np.pad(y, 1, mode='edge')

        x = (x[:-1] + x[1:]) / 2
        y = (y[:-1] + y[1:]) / 2

        X, Y = np.meshgrid(x[:-1], y[:-1], indexing='ij')
        W, H = np.meshgrid(np.diff(x), np.diff(y), indexing='ij')
        Z = z

        # get colormap, lut has precedence over cmap
        if self.lut is None:
            lut = self.cmap.getLookupTable(nPts=256)
        elif callable(self.lut):
            lut = self.lut(z)
        else:
            lut = self.lut

        # normalize and quantize
        mn, mx = self.getLevels()
        rng = mx - mn
        if rng == 0:
            rng = 1
        scale = len(lut) / rng
        Z = fn.rescaleData(Z, scale, mn, dtype=int, clip=(0, len(lut)-1))

        # replace nans positions with invalid lut index
        invalid_coloridx = len(lut)
        Z[np.isnan(z)] = invalid_coloridx

        # pre-allocate to the largest array needed
        color_indices, counts = np.unique(Z, return_counts=True)
        rectarray = Qt.internals.PrimitiveArray(QtCore.QRectF, 4)
        rectarray.resize(counts.max())

        # sorted_indices effectively groups together the
        # (flattened) indices of the same coloridx together.
        sorted_indices = np.argsort(Z, axis=None)
        for arr in X, Y, W, H:
            arr.shape = -1      # in-place unravel

        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        painter.setPen(fn.mkPen(None))

        # draw the tiles grouped by coloridx
        offset = 0
        for coloridx, cnt in zip(color_indices, counts):
            if coloridx == invalid_coloridx:
                continue
            indices = sorted_indices[offset:offset+cnt]
            offset += cnt
            rectarray.resize(cnt)
            memory = rectarray.ndarray()
            memory[:,0] = X[indices]
            memory[:,1] = Y[indices]
            memory[:,2] = W[indices]
            memory[:,3] = H[indices]

            brush = fn.mkBrush(lut[coloridx])
            painter.setBrush(brush)
            painter.drawRects(*rectarray.drawargs())

        if self.edgecolors is not None:
            painter.setPen(self.edgecolors)
            painter.setBrush(fn.mkBrush(None))
            painter.drawRect(self.boundingRect())

        painter.end()

    def paint(self, p, *args):
        if self.picture is None:
            self.generatePicture()
        p.drawPicture(0, 0, self.picture)


    def pixelPadding(self):
        pen = self.edgecolors
        no_pen = (pen is None) or (pen.style() == QtCore.Qt.PenStyle.NoPen)
        return 0 if no_pen else (pen.widthF() or 1) * 0.5

    def boundingRect(self):
        xmin, xmax = self.dataBounds(ax=0)
        if xmin is None or xmax is None:
            return QtCore.QRectF()
        ymin, ymax = self.dataBounds(ax=1)
        if ymin is None or ymax is None:
            return QtCore.QRectF()
        px = py = 0
        pxPad = self.pixelPadding()
        if pxPad > 0:
            px, py = self.pixelVectors()
            px = 0 if px is None else px.length()
            py = 0 if py is None else py.length()

            px *= pxPad
            py *= pxPad
        return QtCore.QRectF(
            xmin-px,
            ymin-py,
            (2 * px) + xmax - xmin,
            (2 * py) + ymax - ymin
        )

    def width(self):
        if self._dataBounds is None:
            return 0
        x_bounds = self._dataBounds[0]
        return x_bounds[1] - x_bounds[0]

    def height(self):
        if self._dataBounds is None:
            return 0
        y_bounds = self._dataBounds[1]
        return y_bounds[1] - y_bounds[0]

    def dataBounds(
            self,
            ax: int,
            frac: float = 1.0,
            orthoRange: Optional[tuple[float, float]] = None
    ) -> tuple[Optional[float], Optional[float]]:
        return (None, None) if self._dataBounds is None else self._dataBounds[ax]

    def _updateDisplayWithCurrentState(self, *args, **kwargs):
        defaults = {
            'autolevels': False
        }
        defaults.update(kwargs)
        return self.setData(*args, **defaults)
