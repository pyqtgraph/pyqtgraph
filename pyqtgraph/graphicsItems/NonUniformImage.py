import math

import numpy as np

from .. import functions as fn
from .. import mkBrush, mkPen
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject

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
        self.cmap = ColorMap(pos=[0.0, 1.0], color=[(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)])

        self.data = (x, y, z)
        self.lut = None
        self.border = border
        self.generatePicture()

    def setLookupTable(self, lut, autoLevel=False):
        lut.sigLevelsChanged.connect(self.generatePicture)
        lut.gradient.sigGradientChanged.connect(self.generatePicture)
        self.lut = lut

        if autoLevel:
            _, _, z = self.data
            f = z[np.isfinite(z)]
            lut.setLevels(f.min(), f.max())

        self.generatePicture()

    def setColorMap(self, cmap):
        self.cmap = cmap
        self.generatePicture()

    def getHistogram(self, **kwds):
        """Returns x and y arrays containing the histogram values for the current image.
        For an explanation of the return format, see numpy.histogram().
        """

        z = self.data[2]
        z = z[np.isfinite(z)]
        hist = np.histogram(z, **kwds)

        return hist[1][:-1], hist[0]

    def generatePicture(self):

        x, y, z = self.data

        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(mkPen(None))

        # normalize
        if self.lut is not None:
            mn, mx = self.lut.getLevels()
        else:
            f = z[np.isfinite(z)]
            mn = f.min()
            mx = f.max()

        # draw the tiles
        for i in range(x.size):
            for j in range(y.size):

                value = z[i, j]

                if np.isneginf(value):
                    value = 0.0
                elif np.isposinf(value):
                    value = 1.0
                elif math.isnan(value):
                    continue  # ignore NaN
                else:
                    value = (value - mn) / (mx - mn)  # normalize

                if self.lut:
                    color = self.lut.gradient.getColor(value)
                else:
                    color = self.cmap.mapToQColor(value)

                p.setBrush(mkBrush(color))

                # left, right, bottom, top
                l = x[0] if i == 0 else (x[i - 1] + x[i]) / 2
                r = (x[i] + x[i + 1]) / 2 if i < x.size - 1 else x[-1]
                b = y[0] if j == 0 else (y[j - 1] + y[j]) / 2
                t = (y[j] + y[j + 1]) / 2 if j < y.size - 1 else y[-1]

                p.drawRect(QtCore.QRectF(l, t, r - l, b - t))

        if self.border is not None:
            p.setPen(self.border)
            p.setBrush(fn.mkBrush(None))
            p.drawRect(self.boundingRect())

        p.end()

        self.update()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        x, y, _ = self.data
        return QtCore.QRectF(x[0], y[0], x[-1]-x[0], y[-1]-y[0])
