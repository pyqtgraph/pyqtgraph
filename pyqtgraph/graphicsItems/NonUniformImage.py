__all__ = ['NonUniformImage']

import warnings

import numpy as np
import numpy.typing as npt

from .. import Qt, colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .HistogramLUTItem import HistogramLUTItem


class NonUniformImage(GraphicsObject):
    """
    GraphicsObject displaying an image with non-uniform sample points. 

    Create an image plot by drawing rectangles with specified x and y
    boundaries. The correct interpretation is that the provided coordinates are
    supposed to be the sampled data points, while the drawn polygons are shaded
    to the nearest coordinate z value.

    It can be used to display 2D or slices of higher dimensional data that have
    a regular but non uniform grid such as measurements or simulation results.

    Parameters
    ----------
    x, y : array_like, optional, default None
        1D array_like of monotonically increasing values of where the centers
        of the 'pixels' should be. Exception for first and last element values
        in the array that represent the boundaries. A value of None can be
        used to instantiate the GraphicsObject without applying any data.
    z : array_like, optional, default None
        2D array_like where the shape must equal (x.size, y.size). Value
        will be mapped into the rectangles colors. A value of None can be
        used to instantiate the GraphicsObject without applying any data.
    colorMap
        Colormap used to map the z value to colors. If None is specified,
        the viridis colormap will be used.
    levels
        Sets the minimum and maximum values to be represented by the colormap
        (min, max). Values outside this range will be clipped to the colors
        representing min or max. ``None`` disables the limits, meaning the
        colormap will autoscale the next time ``setImage()`` is called with new
        data.
    edgecolors : color_like, optional, default None
        The color of the edges of the polygons. Argument is relayed to
        :func:`mkPen <pyqtgraph.mkPen>` to construct the pen used to draw the
        edges.
    border : color_like, optional, default None
        Argument that can be relayed to :func:`mkPen <pyqtgraph.mkPen>`
        that represents the border of the border drawn around the
        NonUniformImageItem.
    antialiasing
        Whether to draw edgelines with antialiasing. As lines are vertical and
        horizontal, it is highly suggested to leave to False. If rotation is applied
        consider enabling.

    See Also
    --------
    :class:`~pyqtgraph.PColorMeshItem`
        This item provides similar functionality but instead of rectangles, it uses
        Polygons for more customizable geometry.
    """
    def __init__(
        self,
        x: npt.ArrayLike | None = None,
        y: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
        colorMap: colormap.ColorMap | None = None,
        levels: tuple[float, float] | None = None,
        edgecolors=None,
        border=None,
        antialiasing: bool = False,
    ):
        super().__init__()
        self.setEdgeColor(edgecolors, update=False)
        self.setBorder(border, update=False)
        self.antialiasing = antialiasing
        if colorMap is None:
            # if None is specified, use black - white colormap
            self.cmap = colormap.get('viridis')
        else:
            self.cmap = colorMap
        self._levels = levels
        self._picture = QtGui.QPicture()

        # set default bounds to be between 0 and 1 in both the x and y axis
        self._dataBounds = ((0.0, 1.0), (0.0, 1.0))
        self._data = (x, y, z)
        self.setImage(x, y, z, levels=levels, colorMap=colorMap)
        self.update()

    def setImage(
        self,
        x: npt.ArrayLike | None=None,
        y: npt.ArrayLike | None=None,
        z: npt.ArrayLike | None=None,
        colorMap: colormap.ColorMap | None=None,
        levels: tuple[float, float] | None=None,
        edgecolors=None,
        border=None,
        antialiasing: bool | None=None,
    ):
        """
        Update the plot with new data or parameters.

        Parameters
        ----------
        x, y : array_like, optional, default None
            1D array_like of monotonically increasing values of where the centers
            of the 'pixels' should be. Exception for first and last element values
            in the array that represent the boundaries. A value of None can be
            used to instantiate the GraphicsObject without applying any data.
        z : array_like, optional, default None
            2D array_like where the shape must equal (x.size, y.size). Value
            will be mapped into the rectangles colors. A value of None can be
            used to instantiate the GraphicsObject without applying any data.
        colorMap
            Colormap used to map the z value to colors. If None is specified,
            the viridis colormap will be used.
        levels
            Sets the minimum and maximum values to be represented by the colormap
            (min, max). Values outside this range will be clipped to the colors
            representing min or max. ``None`` disables the limits, meaning the
            colormap will autoscale the next time ``setImage()`` is called with new
            data.
        edgecolors : color_like, optional, default None
            The color of the edges of the polygons. Argument is relayed to
            :func:`mkPen <pyqtgraph.mkPen>` to construct the pen used to draw the
            edges.
        border : color_like, optional, default None
            Argument that can be relayed to :func:`mkPen <pyqtgraph.mkPen>`
            that represents the border of the border drawn around the
            NonUniformImageItem.
        antialiasing
            Whether to draw edgelines with antialiasing. As lines are vertical and
            horizontal, it is highly suggested to leave to False. If rotation is applied
            consider enabling.
        """

        if edgecolors is not None:
            self.setEdgeColor(edgecolors)

        if border is not None:
            self.setBorder(border)
        
        if antialiasing is not None:
            self.antialiasing = antialiasing

        if colorMap is not None:
            self.setColorMap(colorMap)

        if levels is not None:
            self.setLevels(levels)

        if any(data is None for data in (x, y, z)):
            return None

        # convert to numpy arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        # check shape of data
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1-d arrays.")
        
        if len(z.shape) != 2 or z.shape != (x.size, y.size):
            raise ValueError("The length of x and y must match the shape of z.")

        # ensure values are monotonically increasing
        if np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0):
            raise ValueError("The values in x and y must be monotonically increasing.")
        
        self._dataBounds = ((x[0], x[-1]), (y[0], y[-1]))
        self._data = (x, y, z)
        return None

    def setEdgeColor(self, edgecolors, update=True):
        """
        Set the pen used to draw the edges of the rectangles.

        Parameters
        ----------
        edgecolors : color_like
            Value passed to :func:`pyqtgraph.mkPen` to construct the Pen used to
            draw the edges of the rectangles. To disable, pass None or a QPen
            instance with the ``NoPen`` attribute of :class:`QtCore.Qt.PenStyle` set.
        update : bool, optional, default True
            Queue a redraw of the image.
        """
        self._edgecolors = fn.mkPen(edgecolors)
        # force the pen to be cosmetic. see discussion in
        # https://github.com/pyqtgraph/pyqtgraph/pull/2586
        self._edgecolors.setCosmetic(True)
        if update:
            self.update()


    def edgeColor(self) -> QtGui.QPen:
        """
        Get the pen that is used to draw the edges of the rectangles.

        Returns
        -------
        QPen
            Pen instance used to draw the boundaries between rectangles.
        """
        return self._edgecolors

    def setLookupTable(self, lut: list[QtGui.QColor] | npt.ArrayLike, update=True):
        """
        Set the image to use the specified lookup table.

        Parameters
        ----------
        lut : numpy.ndarray or list of QColor
            Lookup table to apply for coloring the image.
        update : bool, optional, default True
            Queue a redraw of the image.

        Warns
        -----
        DeprecationWarning
            If a :class:`~pyqtgraph.HistogramLUTItem` is passed in, a deprecation 
            warning will be emitted. Using a :class:`~pyqtgraph.HistogramLUTItem`
            as the LUT will stop being supported.  Use
            ``HistogramLUTItem.setImageItem(NonUniformImage)`` instead.
        """

        # backwards compatibility hack
        if isinstance(lut, HistogramLUTItem):
            warnings.warn(
                "NonUniformImage::setLookupTable(HistogramLUTItem) is deprecated "
                "and will be removed in a future version of PyQtGraph. "
                "use HistogramLUTItem.setImageItem(NonUniformImage) instead",
                DeprecationWarning, stacklevel=2
            )
            lut.setImageItem(self)
            return

        self.cmap = None    # invalidate since no longer consistent with lut
        self.lut = lut
        self._picture.swap(QtGui.QPicture())
        if update:
            self.update()

    def setBorder(self, border, update=True):
        """
        Set the pen to draw the border for the NonUniformImage.

        Parameters
        ----------
        border : color_like
            :class:`QPen` or parameters for a QPen constructed by 
            :func:`mkPen <pyqtgraph.mkPen>` that will be used to draw the border
            around the NonUniformImage.  To disable the border, pass None or a QPen
            instance with the ``NoPen`` attribute of :class:`QtCore.Qt.PenStyle` set.
        update : bool, optional, default True
            Queue a redraw of the image.
        """

        self._border = fn.mkPen(border)
        # force the pen to be cosmetic. see discussion in
        # https://github.com/pyqtgraph/pyqtgraph/pull/2586
        self._border.setCosmetic(True)
        if update:
            self.update()

    def border(self) -> QtGui.QPen:
        """
        Get the pen used to draw the border.

        Returns
        -------
        QPen
            The :class:`QtGui.QPen` instance used to draw the border.
        """
        return self._border

    def setColorMap(self, cmap, update=True):
        """
        Set the colormap for the NonUniformImage.

        Parameters
        ----------
        cmap : pyqtgraph.ColorMap or None
            Specify the colormap to be used.
        update : bool, optional, default True
            Queue a redraw of the image.
        """
        if cmap is None:
            cmap = colormap.ColorMap(None, [0.0, 1.0])
        self.setLookupTable(cmap.getLookupTable(nPts=256), update=True)
        self.cmap = cmap
        self.lut = cmap.getLookupTable(nPts=256)
        if update:
            self.update()

    def colorMap(self) -> colormap.ColorMap | None:
        """
        Get the ColorMap instance used.

        Returns
        -------
        pyqtgraph.ColorMap or None
            The colormap instance used.  If None, that means that a lookup table
            was applied directly via :meth:`~pyqtgraph.NonUniformImage.setLookupTable`.
        """
        return self.cmap


    def getHistogram(self, **kwds):
        warnings.warn(
            "NonUniformImage::getHistogram is deprecated, use NonUniformImage::histogram instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.histogram(**kwds)

    def histogram(self, **kwargs):
        """
        Get histogram of finite z values.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are relayed to :func:`numpy.histogram`.

        Returns
        -------
        hist : numpy.ndarray
            The value of the histogram.
        bin_edges : numpy.ndarray
            The bin edges in an array of dtype float.

        Raises
        ------
        ValueError
            If the z-data has not been set.

        See Also
        --------
        :func:`numpy.histogram`
            Function used to compute the histogram.
        """
        z = self._data[2]
        if z is None:
            raise ValueError("No 'z' value set to take the histogram of.")
        z = z[np.isfinite(z)]
        hist = np.histogram(z, **kwargs)
        return hist[1][:-1], hist[0]


    def setLevels(self, levels: tuple[float, float], update=True):
        """
        Set the levels used in the image.

        Parameters
        ----------
        levels : tuple of float, float
            Lower and upper limits of the values that the colormap will consider.
        update : bool, optional, default True
            Queue a redraw of the image.
        """
        self._levels = levels
        self._picture.swap(QtGui.QPicture())
        if update:
            self.update()

    def levels(self) -> tuple[float, float]:
        """
        Get the levels used in the image.

        Returns
        -------
        tuple of float, float
            The maximum and mininum values corresponding to the color of the images.
            If no image data is set, returns (0.0, 1.0).
        """
        if self._levels is None:
            # self._levels is not yet set...
            z = self._data[2]
            if z is not None:
                z = z[np.isfinite(z)]
                self._levels = z.min(), z.max()
            else:
                self._levels = (float('nan'), float('nan'))
        return self._levels

    def getLevels(self):
        warnings.warn(
            "NonUniformImage::getLevels is deprecated, use NonUniformImage::levels instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.levels()

    def generatePicture(self):
        x, y, z = self._data

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
            cmap = colormap.ColorMap(None, [0.0, 1.0])
            lut = cmap.getLookupTable(nPts=256)

        # normalize and quantize
        mn, mx = self.levels()
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

        painter = QtGui.QPainter(self._picture)
        painter.setPen(self._edgecolors)
        if self.antialiasing:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

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

        if self._border is not None:
            painter.setPen(self._border)
            painter.setBrush(fn.mkBrush(None))
            painter.drawRect(self.boundingRect())
        painter.end()

    def paint(self, p, *args):
        if self._picture.isNull():
            self.generatePicture()
        self._picture.play(p)

    def pixelPadding(self):
        pen = self._edgecolors
        no_pen = (pen is None) or (pen.style() == QtCore.Qt.PenStyle.NoPen)
        return 0 if no_pen else (pen.widthF() or 1) * 0.5

    def boundingRect(self):
        """
        Rectangle containing the image.

        Returns
        -------
        QRectF
            The bounding rectangle of the NonUniformImageItem. Takes into account
            pixel padding due to the :class:`QPen` width.
        """
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
            orthoRange: tuple[float, float] | None = None
    ) -> tuple[float, float] | tuple[None, None]:
        return (None, None) if self._dataBounds is None else self._dataBounds[ax]
