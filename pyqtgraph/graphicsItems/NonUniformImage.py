import warnings
import numpy as np

from .. import functions as fn
from ..colormap import ColorMap
from .. import Qt
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
    def __init__(self, x, y, z, border=None):

        GraphicsObject.__init__(self)

        # convert to numpy arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        if x.ndim != 1 or y.ndim != 1:
            raise Exception("x and y must be 1-d arrays.")

        if np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0):
            raise Exception("The values in x and y must be monotonically increasing.")

        if len(z.shape) != 2 or z.shape != (x.size, y.size):
            raise Exception("The length of x and y must match the shape of z.")

        # default colormap (black - white)
        self.cmap = ColorMap(None, [0.0, 1.0])
        self.lut = self.cmap.getLookupTable(nPts=256)

        self.data = (x, y, z)
        self.levels = None
        self.border = border
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

        self.cmap = None    # invalidate since no longer consistent with lut
        self.lut = lut
        self.picture = None
        if update:
            self.update()

    def setColorMap(self, cmap):
        self.setLookupTable(cmap.getLookupTable(nPts=256), update=True)
        self.cmap = cmap

    def getHistogram(self, **kwds):
        """Returns x and y arrays containing the histogram values for the current image.
        For an explanation of the return format, see numpy.histogram().
        """

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

        # get colormap
        if callable(self.lut):
            lut = self.lut(z)
        else:
            lut = self.lut

        if lut is None:
            # lut can be None for a few reasons:
            # 1) self.lut(z) can also return None on error
            # 2) if a trivial gradient is being used, HistogramLUTItem calls
            #    setLookupTable(None) as an optimization for ImageItem
            cmap = ColorMap(None, [0.0, 1.0])
            lut = cmap.getLookupTable(nPts=256)

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

        if self.border is not None:
            painter.setPen(self.border)
            painter.setBrush(fn.mkBrush(None))
            painter.drawRect(self.boundingRect())

        painter.end()

    def paint(self, p, *args):
        if self.picture is None:
            self.generatePicture()
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        x, y, _ = self.data
        return QtCore.QRectF(x[0], y[0], x[-1]-x[0], y[-1]-y[0])
