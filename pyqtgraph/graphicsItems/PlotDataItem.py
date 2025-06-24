import math
import warnings
import bisect

from typing import TypedDict

import numpy as np

from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject
from .PlotCurveItem import PlotCurveItem
from .ScatterPlotItem import ScatterPlotItem

__all__ = ['PlotDataItem']


# For type-hints, but cannot be utilized with setData or __init__ until
# typing.Unpack is available in the library
class MetaKeywordArgs(TypedDict):
    name: str


class PointStyleKeywordArgs(TypedDict):
    symbol: str | QtGui.QPainterPath | list[str | QtGui.QPainterPath] | None
    symbolPen: fn.color_like | QtGui.QPen | list[fn.color_like | QtGui.QPen] | None
    symbolBrush: fn.color_like | QtGui.QBrush | list[fn.color_like | QtGui.QBrush] | None
    symbolSize: int | list[int]
    pxMode: bool


class LineStyleKeywordArgs(TypedDict):
    connect: str | np.ndarray
    pen: fn.color_like | QtGui.QPen | None
    shadowPen: fn.color_like | QtGui.QPen | None
    fillLevel: float | None
    fillOutline: bool
    fillBrush: fn.color_like | QtGui.QBrush | None
    stepMode: str | None


class OptimizationKeywordArgs(TypedDict):
    useCache: bool
    antialias: bool
    downsample: int
    downsampleMethod: str
    autoDownsample: bool
    clipToView: bool
    dynamicRangeLimit: float | None
    dynamicRangeHyst: float
    skipFiniteCheck: bool
    useDownsamplingCache: bool
    downsamplingCacheSize: int


class PlotDataset:
    """
    Holds collected information for a plottable dataset.
    
    Numpy arrays containing x and y coordinates are available as ``dataset.x`` and
    ``dataset.y``.
    
    After a search has been performed, typically during a call to
    :meth:`dataRect <pyqtgraph.PlotDataset.dataRect>`, ``dataset.containsNonfinite``
    is ``True`` if any coordinate values are non-finite (e.g. ``NaN`` or ``Inf``) or 
    ``False`` if all values are finite. If no search has been performed yet,
    `dataset.containsNonfinite` is ``None``.

    Parameters
    ----------
    x : np.ndarray
        Coordinates for `x` data points.
    y : np.ndarray
        Coordinates for `y` data points.
    xAllFinite : bool or None, default None
        Label for `x` data points, indicating if all values are finite, or not, and
        unknown if ``None``.
    yAllFinite : bool or None, default None
        Label for `y` data points, indicating if all values are finite, or not, and
        unknown if ``None``.
    connect : np.ndarray or None, default None
        Array of boolean values indicating if points are connected. This is only
        populated if the PlotDataItem's `connect` argument is set to a numpy array.
        Otherwise, will be None.

    Warnings
    --------
    :orphan:
    .. warning:: 
        
        This class is intended for internal use of :class:`~pyqtgraph.PlotDataItem`.
        The interface may change without warning.  It is not considered part of the
        public API.
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xAllFinite: bool | None = None,
        yAllFinite: bool | None = None,
        connect: np.ndarray | None = None
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.xAllFinite = xAllFinite
        self.yAllFinite = yAllFinite
        self.connect = connect
        self._dataRect = None

        if isinstance(x, np.ndarray) and x.dtype.kind in 'iu':
            self.xAllFinite = True
        if isinstance(y, np.ndarray) and y.dtype.kind in 'iu':
            self.yAllFinite = True

    @property
    def containsNonfinite(self) -> bool | None:
        if self.xAllFinite is None or self.yAllFinite is None:
            # don't know for sure yet
            return None
        return not (self.xAllFinite and self.yAllFinite)

    def _updateDataRect(self):
        """ 
        Identify plottable bounds and presence of non-finite data.
        """
        if self.y is None or self.x is None:
            return None
        xmin, xmax, self.xAllFinite = self._getArrayBounds(self.x, self.xAllFinite)
        ymin, ymax, self.yAllFinite = self._getArrayBounds(self.y, self.yAllFinite)
        self._dataRect = QtCore.QRectF(
            QtCore.QPointF(xmin, ymin),
            QtCore.QPointF(xmax, ymax)
        )

    def _getArrayBounds(
        self,
        arr: np.ndarray,
        all_finite: bool | None
    ) -> tuple[float, float, bool]:
        # here all_finite could be [None, False, True]
        if not all_finite:  # This may contain NaN or inf values.
            # We are looking for the bounds of the plottable data set. Infinite and Nan
            # are ignored.
            selection = np.isfinite(arr)
            # True if all values are finite, False if there are any non-finites
            all_finite = bool(selection.all())
            if not all_finite:
                arr = arr[selection]
        
        # here all_finite could be [False, True]
        try:
            amin = np.min( arr )  # find minimum of all finite values
            amax = np.max( arr )  # find maximum of all finite values
        except ValueError:  # is raised when there are no finite values
            amin = np.nan
            amax = np.nan
        return amin, amax, all_finite

    def dataRect(self) -> QtCore.QRectF | None:
        """
        Get the bounding rectangle for the finite subset of data.

        If there is an active mapping function, such as logarithmic scaling, then bounds
        represent the mapped data.

        Returns
        -------
        :class:`QRectF` or None
            The bounding rect of the data in view-space.  Will return ``None`` if there
            is no data or if all `x` and `y` values are ``NaN``.
        """
        if self._dataRect is None: 
            self._updateDataRect()
        return self._dataRect

    def applyLogMapping(self, logMode: tuple[bool, bool]):
        """
        Apply a log_10 map transformation if requested to the respective axis.

        This replaces the internal data. Values of ``-inf`` resulting from zeros in the
        original dataset are replaced by ``np.nan``.
        
        Parameters
        ----------
        logMode : tuple of bool
            A ``True`` value requests log-scale mapping for the `x` and then `y` axis.
        """
        if logMode[0]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.x = np.log10(self.x)
            non_finites = ~np.isfinite( self.x )
            if non_finites.any():
                self.x[non_finites] = np.nan  # set all non-finite values to NaN
                all_x_finite = False
            else:
                all_x_finite = True
            self.xAllFinite = all_x_finite

        if logMode[1]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.y = np.log10(self.y)
            non_finites = ~np.isfinite( self.y )
            if non_finites.any():
                self.y[non_finites] = np.nan  # set all non-finite values to NaN
                all_y_finite = False
            else:
                all_y_finite = True
            self.yAllFinite = all_y_finite


class PlotDataItem(GraphicsObject):
    """
    PlotDataItem is PyQtGraph's primary way to plot 2D data.
    
    It provides a unified interface for displaying plot curves, scatter plots, or both.
    The library's convenience functions such as :func:`pyqtgraph.plot` create
    PlotDataItem objects.

    .. code-block::

            o-------------o---------------o---------------o
            ^             ^               ^               ^
          point         point           point           point

    The scatter plot renders a symbol for each point of the data. The curve connects
    adjacent points. This is realized by a combination of a
    :class:`~pyqtgraph.ScatterPlotItem` and a :class:`~pyqtgraph.PlotCurveItem`.
    Although these classes can be used individually, :class:`~pyqtgraph.PlotDataItem`
    is the recommended way to interact with them.

    PlotDataItem contains methods to transform the original data:
      
    * :meth:`setDerivativeMode`
    * :meth:`setPhasemapMode`
    * :meth:`setFftMode`
    * :meth:`setLogMode`
    * :meth:`setSubtractMeanMode`

    It can pre-process large data sets to accelerate plotting:
    
    * :meth:`setDownsampling`
    * :meth:`setClipToView`
    
    PlotDataItem's performance is usually sufficient for real-time interaction even for
    large numbers of points. If you do encounter performance issues, consider the
    following.

    * Use a :class:`QPen` with ``width=1``. All wider pens cause a loss in performance.
      This loss can be partially mitigated by using fully opaque colors 
      (``alphaF=1.0``), solid lines, and no anti-aliasing. 
    * For scatter plots that use multiple pen or brush settings, passing a list of
      string representation to `symbolPen` or `symbolBrush` creates many internal
      :class:`QPen` and :class:`QBrush` objects. Instead, create the needed
      :class:`QPen` and :class:`QBrush` objects in advance and pass these as a list
      instead. This lets a smaller number of stored instances be reused.
    * If you know that all points in your data set will have numerical, finite values,
      :meth:`setSkipFiniteCheck` can disable a check to identify points that require
      special treatment.  
    * When passing `x` and `y` data to :meth:`PlotDataItem.setData`, use
      :class:`numpy.ndarray` instead of python's built-in lists.

    **Bases:** :class:`~pyqtgraph.GraphicsObject`

    Parameters
    ----------
    *args : tuple, optional
        Arguments representing the `x` and `y` data to be drawn. The following are
        examples for initializing data.

        * ``PlotDataItem(x, y)`` - `x` and `y` are array_like coordinate values.
        * ``PlotDataItem(x=x, y=y)`` - same as above, but using keyword arguments.
        * ``PlotDataItem(y)`` - `y` values only, `x` is automatically set to
          ``np.arange(len(y))``.
        * ``PlotDataItem(np.ndarray((N, 2)))`` - single :class:`numpy.ndarray` with
          shape ``(N, 2)``, where `x` is given by ``data[:, 0]`` and `y` by
          ``data[:, 1]``.
    
        Data can also be initialized with spot-style, per-point arguments.

        * ``PlotDataItem(recarray)`` - :class:`numpy.recarray` record array with
          ``dtype=[('x', float), ('y', float), ...]``
        * ``PlotDataItem(list[dict[str, value]])`` - list of dictionaries, where
          each dictionary provides information for a single point. Dictionaries can
          include coordinate information associated with the `x` and `y` keys.
        * ``PlotDataItem(dict[str, array_like])`` - dictionary of lists, where each key
          corresponds to a keyword argument, and the associated list or array_like
          structure specifies a value for each point. All dictionary items must provide
          the same length of data. `x` and `y` keys can be included to specify
          coordinates.
        
        When using spot-style arguments, it is always possible to give coordinate data
        separately through the `x` and `y` keyword arguments.
    
    **kwargs : dict, optional
        The supported keyword arguments can be grouped into several categories:

        *Point Style Keyword Arguments*, see
        :meth:`ScatterPlotItem.setData <pyqtgraph.ScatterPlotItem.setData>` for more
        information.
    
        =========== ====================================================================
        Property    Description
        =========== ====================================================================
        symbol      ``str``, :class:`QPainterPath`,
                     
                    list of ``str`` or :class:`QPainterPath`,
                    
                    or ``None``, default ``None``

                    The symbol to use for drawing points, or a list specifying a symbol
                    for each point. If used, ``str`` needs to be a string that
                    :class:`~pyqtgraph.ScatterPlotItem` will recognize. ``None``
                    disables the scatter plot.
        
        symbolPen   :class:`QPen`, or arguments accepted by
                    :func:`mkPen <pyqtgraph.mkPen>`,

                    list of :class:`QPen`, 
                    or arguments to :func:`mkPen <pyqtgraph.mkPen>`,

                    or ``None``, default ``(200, 200, 200)``
                    
                    Outline pen for drawing points, or a list specifying a pen for each
                    point.
        
        symbolBrush :class:`QBrush`, or arguments accepted by
                    :func:`mkBrush <pyqtgraph.mkBrush>`,

                    or list of :class:`QBrush`
                    or arguments to :func:`mkBrush <pyqtgraph.mkBrush>`
                    
                    default ``(50, 50, 150)``

                    Brush for filling points, or a list specifying a brush for each
                    point.
        
        symbolSize  ``int`` or ``list[int]``, default ``10``

                    Diameter of the symbols, or array-like list of diameters. Diameter
                    is either in pixels or data-space coordinates depending on the value
                    of `pxMode`.
        
        pxMode      ``bool``, default ``True``

                    If ``True``, the `symbolSize` represents the diameter in pixels. If
                    ``False``, the `symbolSize` represents the diameter in data
                    coordinates.
        =========== ====================================================================

        *Line Style Keyword Arguments*
        
        =========== ====================================================================
        Property    Description
        =========== ====================================================================
        connect     ``{ 'auto', 'finite', 'all', 'pairs', (N,) ndarray }``, default
                    ``'auto'``
                    
                    Normally, the curve connects each point in sequence. Any non-finite,
                    non-plottable values such as ``NaN`` result in a gap. The
                    ``connect`` argument modifies this behavior.
                    
                    - ``'finite'`` and ``'auto'`` both give the normal behavior. The 
                      default ``auto`` mode enables PlotDataItem to avoid some
                      repeated tests for non-finite values in 
                      :class:`~pyqtgraph.PlotCurveItem`.
                    - ``'all'`` - ignores any non-finite values to plot an uninterrupted
                      curve.  
                    - ``'pairs'`` - generates one line segment for each successive pair
                      of points.
                    - :class:`~numpy.ndarray` - Individual connections can be specified
                      by an array of length `N`, matching the number of points. After
                      casting to Boolean, a value of ``True`` causes the respective
                      point to be connected to the next.

        stepMode    ``{ 'left', 'right', 'center' }`` or ``None``, default ``None``
                    
                    If specified and not ``None``, a stepped curve is drawn.
                    
                    - ``'left'``- the specified points each describe the left edge of a
                      step.
                    - ``'right'``- the specified points each describe the right edge of
                      a step. 
                    - ``'center'``- the x coordinates specify the location of the step
                      boundaries. This mode is commonly used for histograms. Note that
                      it requires an additional `x` value, such that
                      ``len(x) = len(y) + 1``.
                    - ``None`` - Render the curve normally, and not as a step curve.
                    
        pen         :class:`QPen`, arguments accepted by :func:`mkPen <pyqtgraph.mkPen>`,
                    or ``None``, default is a 1px thick solid line of color 
                    ``(200, 200, 200)``
                    
                    Pen for drawing the lines between points. Use ``None`` to
                    disable line drawing.
                         
        shadowPen   :class:`QPen`, arguments accepted by :func:`mkPen <pyqtgraph.mkPen>`,
                    or ``None``, default ``None``
          
                    Pen for drawing a secondary line behind the primary line.
                    Typically used for highlighting or to increase contrast when drawing
                    over background elements.
                    
        fillLevel   ``float`` or ``None``, default ``None``

                    If set, the area between the curve and the value of fillLevel is
                    filled. Use ``None`` to disable.
        
        fillBrush   :class:`QBrush`, ``None`` or args accepted by
                    :func:`mkBrush <pyqtgraph.mkBrush>`, default ``None``
                    
                    Brush used to fill the area specified by `fillLevel`.

        fillOutline ``bool``, default ``False``

                    ``True`` draws an outline surrounding the area specified
                    by `fillLevel`, using the plot's `pen` and `shadowPen`.
        
        =========== ====================================================================

        *Optimization Keyword Arguments*

        =================== ============================================================
        Property            Description
        =================== ============================================================
        useCache            ``bool``, default ``True``

                            Generated point graphics of the scatter plot are cached to
                            improve performance.  Setting this to ``False`` can improve
                            image quality in some situations.

        antialias           ``bool``, default inherited from
                            ``pyqtgraph.getConfigOption('antialias')``

                            Disabling antialiasing can improve performance. In some
                            cases, in particular when ``pxMode=True``, points will be 
                            rendered with antialiasing regardless of this setting.

        autoDownsample      ``bool``, default ``False``

                            Resample the data before plotting to avoid plotting multiple
                            line segments per pixel. This can improve performance when
                            viewing very high-density data, but increases initial
                            overhead and memory usage. See :meth:`setDownsampling` for
                            more information.

        downsample          ``int``, default ``1``

                            Resample the data before plotting, reducing the number of 
                            displayed elements by the specified factor.
                            See :meth:`setDownsampling` for more information.

        downsampleMethod    ``str``, default ``'peak'``

                            Method for downsampling data. See
                            :meth:`setDownsampling` for more information.

        clipToView          ``bool``, default ``False``

                            Clip the data to only the visible range on the x-axis.
                            See :meth:`setClipToView` for more information.

        dynamicRangeLimit   ``float``, default ``1e6``

                            Limit off-screen y positions of data points. ``None``
                            disables the limiting. This can increase performance but may
                            cause plots to disappear at high levels of magnification. 
                            See :meth:`setDynamicRangeLimit` for more information.

        dynamicRangeHyst    ``float``, default ``3.0``
        
                            Permit vertical zoom to change up to the given hysteresis
                            factor before the limit calculation is repeated. See
                            :meth:`setDynamicRangeLimit` for more information.

        skipFiniteCheck     ``bool``, default ``False``

                            If ``True``, the special handling of non-finite values such as
                            ``NaN`` in :class:`~pyqtgraph.PlotCurveItem` is skipped.
                            This speeds up the plot, but creates error or causes the
                            plotting to fail entirely if any such values are present.
                            If ``connect='auto'``, PlotDataItem manages the check and
                            this item will be overridden.

        useDownsamplingCache ``bool``, default ``True``

                            Use cache instead of "real-time" computation of downsampled signal.
                            See

        downsamplingCacheSize ``int``, default ``20000``


        =================== ============================================================

        *Meta Keyword Arguments*

        =========== ====================================================================
        Property    Description
        =========== ====================================================================
        name        ``str`` or ``None``, default ``None``

                    Name of item for use in the plot legend.
        =========== ====================================================================

    Attributes
    ----------
    curve : :class:`~pyqtgraph.PlotCurveItem`
        The underlying Graphics Object used to represent the curve.
    scatter : :class:`~pyqtgraph.ScatterPlotItem`
        The underlying Graphics Object used to the points along the curve.
    xData : numpy.ndarray or None
        The numpy array representing x-axis data. ``None`` if no data has been added.
    yData : numpy.ndarray or None
        The numpy array representing y-axis data. ``None`` if no data has been added.

    Signals
    -------
    sigPlotChanged : Signal
        Emits when the data in this item is updated.
    sigClicked : Signal
        Emits when the item is clicked. This signal sends the
        :class:`~pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent`.
    sigPointsClicked : Signal
        Emits when a plot point is clicked. Sends the list of points under the
        mouse, as well as the
        :class:`~pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent`.
    sigPointsHovered : Signal
        Emits when a plot point is hovered over. Sends the list of points under the
        mouse, as well as the :class:`~pyqtgraph.GraphicsScene.mouseEvents.HoverEvent`.
    
    See Also
    --------
    :func:`~pyqtgraph.arrayToQPath`
        Function used to convert :class:`numpy.ndarray` to :class:`QPainterPath`.

    Notes
    -----
    The fastest performance results for drawing lines that have a :class:`QPen` width of
    1 pixel. If drawing a 1 pixel thick line, PyQtGraph converts the `x` and `y` data to
    a :class:`QPainterPath` that is rendered.
    
    The render performance of :class:`QPainterPath` when using a :class:`QPen` that has
    a width greater than 1 is quite poor, but PyQtGraph can fall back to constructing an
    array of :class:`QLine` objects representing each line segment.  Using
    :meth:`QPainter.drawLines <QPainter.drawLines>`, PyQtGraph is able to draw lines
    with thickness greater than 1 pixel with a smaller performance penalty.  
    
    For the :meth:`QPainter.drawLines <QPainter.drawLines>` method to work, some other
    factors need to be present.

    * ``pen.style() == QtCore.Qt.PenStyle.SolidLine``
    * ``pen.isSolid() is True``
    * ``pen.color().alphaF() == 1.0``
    * ``pyqtgraph.getConfigOption('antialias') is False``

    If using lines with a thickness greater than 4 pixel, the :class:`QPen` instance
    will be modified such that ``pen.capStyle() == QtCore.Qt.PenCapStyle.RoundCap``.
    There is a small additional performance penalty with this change.
    """

    sigPlotChanged = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object)
    sigPointsClicked = QtCore.Signal(object, object, object)
    sigPointsHovered = QtCore.Signal(object, object, object)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemHasNoContents)
        # Original data, mapped data, and data processed for display is now all held in
        # PlotDataset objects.
        # The convention throughout PlotDataItem is that a PlotDataset is only
        # instantiated if valid data is available.
        # will hold a PlotDataset for the original data, accessed by getOriginalData()
        self._dataset        = None
        # will hold a PlotDataset for data after mapping transforms (e.g. log scale)
        self._datasetMapped  = None
        # will hold a PlotDataset for data downsampled and limited for display,
        # accessed by getData()
        self._datasetDisplay = None
        self.curve = PlotCurveItem()
        self.scatter = ScatterPlotItem()
        self.curve.setParentItem(self)
        self.scatter.setParentItem(self)

        self.curve.sigClicked.connect(self.sigClicked)
        self.scatter.sigClicked.connect(self.scatterClicked)
        self.scatter.sigHovered.connect(self.sigPointsHovered)
        
        # update-required notifications are handled through properties to allow future 
        # management through the QDynamicPropertyChangeEvent sent on any change.
        self.setProperty('xViewRangeWasChanged', False)
        self.setProperty('yViewRangeWasChanged', False)
        self.setProperty('styleWasChanged', True)  # force initial update

        # holds last clipping points of dynamic range limiter
        self._drlLastClip = (0.0, 0.0)
        self._adsLastValue = 1
        # self.clear()
        self.opts = {
            # defaults to 'all', unless overridden to 'finite' for log-scaling
            'connect': 'auto',
            'skipFiniteCheck': False,
            'subtractMeanMode': False,
            'fftMode': False,
            'logMode': [False, False],
            'derivativeMode': False,
            'phasemapMode': False,
            'alphaHint': 1.0,
            'alphaMode': False,

            'pen': (200,200,200),
            'shadowPen': None,
            'fillLevel': None,
            'fillOutline': False,
            'fillBrush': None,
            'stepMode': None,

            'symbol': None,
            'symbolSize': 10,
            'symbolPen': (200, 200, 200),
            'symbolBrush': (50, 50, 150),
            'pxMode': True,

            'antialias': getConfigOption('antialias'),
            'pointMode': None,

            'useCache': True,
            'downsample': 1,
            'autoDownsample': False,
            'downsampleMethod': 'peak',
            'autoDownsampleFactor': 5.0,  # draw ~5 samples per pixel
            'useDownsamplingCache': True,
            'downsamplingCacheSize': 20000,  # Number of samples after downsampling (with autodownsample and caching)
            'minSampPerPxForCache': 2.0,  # Draw at least this many samples per pixel when using cache
            'clipToView': False,
            'dynamicRangeLimit': 1e6,
            'dynamicRangeHyst': 3.0,
            'data': None,
        }
        self.setCurveClickable(kwargs.get('clickable', False))
        self.setData(*args, **kwargs)
    
    # Fix "NotImplementedError: QGraphicsObject.paint() is abstract and must be overridden"
    def paint(self, *args):
        ...
    
    # Compatibility with direct property access to previous xData and yData structures:
    @property
    def xData(self):
        if self._dataset is None: return None
        return self._dataset.x
        
    @property
    def yData(self):
        if self._dataset is None: return None
        return self._dataset.y

    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self) -> str | None:
        """
        Get the name attribute if set.

        Returns
        -------
        str or None
            The name that represents this item in the legend.
        """
        return self.opts.get('name')

    def setCurveClickable(self, state: bool, width: int | None = None):
        """
        Set the attribute for the curve being clickable.

        Parameters
        ----------
        state : bool
            Set the curve to be clickable.
        width : int 
            The distance tolerance margin in pixels to recognize the mouse click.
        """
        self.curve.setClickable(state, width)

    def curveClickable(self) -> bool:
        """
        Get the attribute if the curve is clickable.

        Returns
        -------
        bool
            Return if the curve is set to be clickable.
        """
        return self.curve.clickable

    def boundingRect(self):
        return QtCore.QRectF()  # let child items handle this

    def setPos(self, x, y):
        # super().setPos(x, y)
        GraphicsObject.setPos(self, x, y)
        # to update viewRect:
        self.viewTransformChanged()
        # to update displayed point sets, e.g. when clipping (which uses viewRect):
        self.viewRangeChanged()

    def setAlpha(self, alpha: float, auto: bool):
        """
        Set the opacity of the item to the value passed in.

        Parameters
        ----------
        alpha : float
            Value passed to :meth:`QGraphicsItem.setOpacity`.
        auto : bool
            Receives the ``autoAlpha`` value from a parent
            :class:`~pyqtgraph.PlotItem`, but has no function within PlotDataItem
            itself.

        See Also
        --------
        :meth:`QGraphicsItem.setOpacity <QGraphicsItem.setOpacity>`
            This is the Qt method that the value is relayed to.
        """
        if self.opts['alphaHint'] == alpha and self.opts['alphaMode'] == auto:
            return
        self.opts['alphaHint'] = alpha
        self.opts['alphaMode'] = auto
        self.setOpacity(alpha)

    def setFftMode(self, state: bool):
        """
        Enable FFT mode.

        FFT mode enables mapping the data by a fast Fourier transform.  If the `x`
        values are not equidistant, the data set is resampled at equal intervals.

        Parameters
        ----------
        state : bool
            To enable or disable FFT mode.
        """
        if self.opts['fftMode'] == state:
            return
        self.opts['fftMode'] = state
        self._reloadYValues()

    def setLogMode(self, xState: bool, yState: bool):
        """
        Enable log mode per axis.

        When the log mode is enabled for the respective axis, a mapping according to
        ``mapped = np.log10(value)`` is applied to the data. For each negative or zero
        value, this results in a ``NaN`` value.

        Parameters
        ----------
        xState : bool
            Enable log mode on the x-axis.
        yState : bool
            Enable log mode on the y-axis.
        """
        if self.opts['logMode'] == [xState, yState]:
            return
        self.opts['logMode'] = [xState, yState]
        self._reloadYValues()

    def setSubtractMeanMode(self, state: bool):
        """
        Enable mean value subtraction mode.

        In mean value subtraction mode, the data is mapped according to ``y_mapped = y - mean(y)``.

        Parameters
        ----------
        state : bool
            Enable mean subtraction mode.
        """
        if self.opts['subtractMeanMode'] == state:
            return
        self.opts['subtractMeanMode'] = state
        self._reloadYValues()

    def setDerivativeMode(self, state: bool):
        """
        Enable derivative mode.

        In derivative mode, the data is mapped according to ``y_mapped = dy / dx``,
        with `dx` and `dy` representing the difference between adjacent `x` and `y`
        values.

        Parameters
        ----------
        state : bool
            Enable derivative mode.
        """
        if self.opts['derivativeMode'] == state:
            return
        self.opts['derivativeMode'] = state
        self._reloadYValues()

    def setPhasemapMode(self, state: bool):
        """
        Enable phase map mode.

        In phase map mode, the data undergoes a mapping where ``x_mapped = y`` and
        ``y_mapped = dy / dx``, where the numerical derivative of the data is plotted
        over the original `y` values.

        Parameters
        ----------
        state : bool
            This enabled phase map mode.
        """
        if self.opts['phasemapMode'] == state:
            return
        self.opts['phasemapMode'] = state
        self._reloadYValues()

    def setPen(self, *args, **kwargs):
        """
        Set the primary pen used to draw lines between points.

        Parameters
        ----------
        *args : tuple or None
            :class:`QPen`, or parameters for a QPen constructed by 
            :func:`mkPen <pyqtgraph.mkPen>`. Use ``None`` to disable drawing of lines.
        **kwargs : dict
            Alternative specification of arguments directed to
            :func:`mkPen <pyqtgraph.mkPen>`.
        """
        pen = fn.mkPen(*args, **kwargs)
        self.opts['pen'] = pen
        self.updateItems(styleUpdate=True)

    def setShadowPen(self, *args, **kwargs):
        """
        Set the shadow pen used to draw lines between points.

        The shadow pen is often used for enhancing contrast or emphasizing data. The
        line is drawn behind the primary pen and should generally have a greater width
        than the primary pen.

        Parameters
        ----------
        *args : tuple or None
            :class:`QPen`, or parameters for a QPen constructed by 
            :func:`mkPen <pyqtgraph.mkPen>`. Use ``None`` to disable the shadow pen.
        **kwargs : dict
            Alternative specification of arguments directed to
            :func:`mkPen <pyqtgraph.mkPen>`.
        """
        if args and args[0] is None:
            pen = None
        else:
            pen = fn.mkPen(*args, **kwargs)
        self.opts['shadowPen'] = pen
        self.updateItems(styleUpdate=True)

    def setFillBrush(self, *args, **kwargs):
        """
        Set the :class:`QBrush` used to fill the area under the curve.
         
        Use :meth:`setFillLevel` to enable filling and set the boundary value. 

        Parameters
        ----------
        *args : tuple
            :class:`QBrush`, or parameters for a QBrush constructed by
            :func:`mkBrush <pyqtgraph.mkBrush>`. Also accepts a color specifier
            understood by :func:`mkColor <pyqtgraph.mkColor>`.
        **kwargs : dict
            Alternative specification of arguments directed to
            :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        if args and args[0] is None:
            brush = None
        else:
            brush = fn.mkBrush(*args, **kwargs)
        if self.opts['fillBrush'] == brush:
            return
        self.opts['fillBrush'] = brush
        self.updateItems(styleUpdate=True)

    def setBrush(self, *args, **kwargs):
        """
        An alias to :meth:`setFillBrush`.

        Parameters
        ----------
        *args : tuple
            :class:`QBrush`, or parameters for a QBrush constructed by
            :func:`mkBrush <pyqtgraph.mkBrush>`. Also accepts a color specifier
            understood by :func:`mkColor <pyqtgraph.mkColor>`.
        **kwargs : dict
            Alternative specification of arguments directed to
            :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.setFillBrush(*args, **kwargs)

    def setFillLevel(self, level: float | None):
        """
        Enable filling the area under the curve and set its boundary.

        Parameters
        ----------
        level : float or None
            The value that the fill from the curve is drawn to. Use ``None`` to disable
            the filling.

        See Also
        --------
        :class:`pyqtgraph.FillBetweenItem`
            This
            :class:`~pyqtgraph.GraphicsItem` creates a filled in region between two
            curves.
        """
        if self.opts['fillLevel'] == level:
            return
        self.opts['fillLevel'] = level
        self.updateItems(styleUpdate=True)

    def setSymbol(
        self,
        symbol: str | QtGui.QPainterPath | list[str | QtGui.QPainterPath]
    ):
        """
        Set the symbol or symbols for drawing the points.

        See :meth:`pyqtgraph.ScatterPlotItem.setSymbol` for a full list of accepted
        arguments.

        Parameters
        ----------
        symbol : str or :class:`QPainterPath` or list
            Symbol to draw as the points. If of type ``list``, it must be the same
            length as the number of points, and every element must be a recognized
            string or of type :class:`QPainterPath`. Use ``None`` to disable the scatter
            plot.
        
        See Also
        --------
        :meth:`pyqtgraph.ScatterPlotItem.setSymbol`
            Recognized symbols are detailed in the description of this method.
        """
        if self.opts['symbol'] == symbol:
            return
        self.opts['symbol'] = symbol
        self.updateItems(styleUpdate=True)

    def setSymbolPen(self, *args, **kwargs):
        """
        Set the :class:`QPen` used to draw symbols.
        
        Setting a different :class:`QPen` per point is not supported by this function.

        Parameters
        ----------
        *args : tuple
            :class:`QPen`, or parameters for a QPen constructed by 
            :func:`mkPen <pyqtgraph.mkPen>`.
        **kwargs : dict
            Alternative specification of arguments directed to
            :func:`mkPen <pyqtgraph.mkPen>`.
        """
        pen = fn.mkPen(*args, **kwargs)
        if self.opts['symbolPen'] == pen:
            return
        self.opts['symbolPen'] = pen
        self.updateItems(styleUpdate=True)

    def setSymbolBrush(self, *args, **kwargs):
        """
        Set the :class:`QBrush` used to fill symbols.
        
        Setting a different :class:`QBrush` per point is not supported by this function.

        Parameters
        ----------
        *args : tuple
            :class:`QBrush`, or parameters for a QBrush constructed by
            :func:`mkBrush <pyqtgraph.mkBrush>`. Also accepts a color specifier
            understood by :func:`mkColor <pyqtgraph.mkColor>`.
        **kwargs : dict
            Alternative specification of arguments directed to
            :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        brush = fn.mkBrush(*args, **kwargs)
        if self.opts['symbolBrush'] == brush:
            return
        self.opts['symbolBrush'] = brush
        #self.scatter.setSymbolBrush(brush)
        self.updateItems(styleUpdate=True)

    def setSymbolSize(self, size: int):
        """
        Set the symbol size or sizes.

        Parameters
        ----------
        size : int | list[int]
            Diameter of the symbols, or array-like list of diameters. Diameter is
            either in pixels or data-space coordinates depending on the value of
            `pxMode`.
        """
        if self.opts['symbolSize'] == size:
            return
        self.opts['symbolSize'] = size
        self.updateItems(styleUpdate=True)

    def setDownsampling(
        self,
        ds: int | None = None,
        auto: bool | None = None,
        method: str = 'peak'
    ):
        """
        Set the downsampling mode.
        
        Downsampling reduces the number of samples drawn to increase performance.

        Parameters
        ----------
        ds : int or None, default None
            Reduce the number of displayed data points by a factor `N=ds`. To disable,
            set ``ds=1``.
        auto : bool or None, default None
            If ``True``, automatically pick `ds` based on visible range.
        method : { 'subsample', 'mean', 'peak' }, default 'peak'
            Specify the method of the downsampling calculation.
            
            * `subsample` - Downsample by taking the first of `N` samples. This method
              is the fastest, but least accurate.
            * `mean` - Downsample by taking the mean of `N` samples.
            * `peak` - Downsample by drawing a saw wave that follows the min and max of
              the original data. This method produces the best visual representation of
              the data but is slower.
        """
        changed = False
        if ds is not None and self.opts['downsample'] != ds:
            changed = True
            self.opts['downsample'] = ds

        if auto is not None and self.opts['autoDownsample'] != auto:
            changed = True
            self.opts['autoDownsample'] = auto

        if method is not None and self.opts['downsampleMethod'] != method:
            changed = True
            self.opts['downsampleMethod'] = method

        if changed:
            self._reloadYValues()

    def setDownsamplingCacheMode(self, useCache: bool = True, cacheSize: int = 20000):
        """
        If downsampling is enabled, this method sets the use of cache for downsampling.
        Downsampling with cache reduces CPU load while changing view (zooming), since
        the downsampled signal will not have to be re-calculated each time the plot is
        re-drawn. For fixed downsampling, caching has no drawbacks except a minor increase
        in memory usage. For auto-downsampling, there is a tradeoff between CPU-use for
        downsampling and the number of samples shown. Experimenting to find the best settings
        for your use case is encouraged.

        Parameters
        ----------
        useCache : bool, default True
            `True` to used downsampling cache, `False` to not use cache.
        cacheSize: int, default 20000
            Number of samples to store in the cache. This is also the number of
            samples actually drawn when the full data is in view when using
            downsampling cache with autoDownsample. When there are many
            PlotDataItems with long sequences in a Plot, it may be advantageous
            to reduce the downsampleCache size to reduce the number of samples
            drawn on screen at one time, at the cost of somewhat reduced zooming
            performance at higher zoom-levels. This setting has no effect when
            `autoDownsample` is off.
        """
        changed = False
        if self.opts["useDownsamplingCache"] != useCache:
            changed = True
            self.opts["useDownsamplingCache"] = useCache
        if cacheSize != self.opts["downsamplingCacheSize"]:
            changed = True
            self.opts["downsamplingCacheSize"] = cacheSize
        if changed:
            self._reloadYValues()

    def setClipToView(self, state: bool):
        """
        Clip the displayed data to the visible range of the x-axis.

        This setting can result in significant performance improvements. 

        Parameters
        ----------
        state : bool
            Enable clipping the displayed data set to the visible x-axis range.
        """
        if self.opts['clipToView'] == state:
            return
        self.opts['clipToView'] = state
        self._datasetDisplay = None  # invalidate display data
        self.updateItems(styleUpdate=False)

    def setDynamicRangeLimit(self, limit: float | None = 1e06, hysteresis: float = 3.):
        """
        Limit the off-screen positions of data points at large magnification.

        This is intended to work around an upstream Qt issue:
        When zoomed closely into plots with a much larger range of data, plots can fail 
        to display entirely because they are incorrectly determined to be off-screen. 
        The dynamic range limiting avoids this error by repositioning far-off points.
        At default settings, points are restricted to ±10⁶ times the viewport height. 

        Parameters
        ----------
        limit : float or None, default 1e+06
            Any data outside the range of ``limit * hysteresis`` will be constrained to
            the limit value. All values are relative to the viewport height. ``None``
            disables the check for a minimal increase in performance.
        hysteresis : float, default 3.0
            Hysteresis factor that controls how much change in zoom level (in terms of 
            the visible y-axis range) is allowed before recalculating.
        
        Notes
        -----
        See https://github.com/pyqtgraph/pyqtgraph/issues/1676 for an example
        of the issue this method addresses.
        """
        hysteresis = max(hysteresis, 1.0)
        self.opts['dynamicRangeHyst'] = hysteresis

        if limit == self.opts['dynamicRangeLimit']:
            return  # avoid update if there is no change
        self.opts['dynamicRangeLimit'] = limit  # can be None
        self._datasetDisplay = None  # invalidate display data
        self.updateItems(styleUpdate=False)
        
    def setSkipFiniteCheck(self, skipFiniteCheck: bool):
        """
        Toggle performance option to bypass the finite check.

        This option improves performance if it is known that the `x` and `y` data passed
        to ``PlotDataItem`` will never contain any non-finite values. If the data does
        contain any non-finite values (such as ``NaN`` or ``Inf``) while this flag is
        set, unpredictable behavior will occur. The data might not be plotted, or there
        might be significant performance impact.
        
        In the default ``connect='auto'`` mode, PlotDataItem will apply this setting
        automatically.

        Parameters
        ----------
        skipFiniteCheck : bool
            Skip the :obj:`numpy.isfinite` check for the input arrays.

        See Also
        --------
        numpy.isfinite
            NumPy function used to identify if there are non-finite values in the `x`
            and `y` data.
        :func:`~pyqtgraph.arrayToQPath`
            Function to create :class:`QPainterPath` which is rendered on the screen
            from numpy arrays.
        """
        self.opts['skipFiniteCheck'] = skipFiniteCheck

    def setData(
        self,
        *args,
        **kwargs
    ):
        """
        Clear any data displayed by this item and display new data.

        Parameters
        ----------
        *args : tuple
            See :class:`PlotDataItem` description for supported arguments.
        **kwargs : dict
            See :class:`PlotDataItem` description for supported arguments.

        Raises
        ------
        TypeError
            Raised when an invalid type was passed in for `x` or `y` data.

        See Also
        --------
        :class:`PlotDataItem`
            The arguments accepted by :meth:`setData` are the same used during 
            initialization, and are listed in the opening section.
        :func:`~pyqtgraph.arrayToQPath`
            Explains the constructions of the draw paths.
        """
        profiler = debug.Profiler()
        y = None
        x = None
        if len(args) == 1:
            data = args[0]
            dt = dataType(data)
            if dt == 'empty':
                pass
            elif dt == 'listOfValues':
                y = np.array(data)
            elif dt == 'Nx2array':
                x = data[:, 0]
                y = data[:, 1]
            elif dt == 'recarray':
                if "x" in data.dtype.names:
                    x = data["x"]
                if "y" in data.dtype.names:
                    y = data["y"]
            elif dt == 'dictOfLists':
                if 'x' in data:
                    x = np.array(data['x'])
                if 'y' in data:
                    y = np.array(data['y'])
            elif dt == 'listOfDicts':
                if 'x' in data[0]:
                    x = np.array([d.get('x',None) for d in data])
                if 'y' in data[0]:
                    y = np.array([d.get('y',None) for d in data])
                for k in [
                    'data', 'symbolSize', 'symbolPen', 'symbolBrush', 'symbolShape'
                ]:
                    if k in data[0]:
                        kwargs[k] = [d.get(k) for d in data]
            else:
                raise TypeError('Invalid data type %s' % type(data))

        elif len(args) == 2:
            seq = ('listOfValues', 'empty')
            dtyp = dataType(args[0]), dataType(args[1])
            if dtyp[0] not in seq or dtyp[1] not in seq:
                raise TypeError(
                    (
                        'When passing two unnamed arguments, both must be a list or '
                        'array of values. (got %s, %s)'
                        % (str(type(args[0])), str(type(args[1])))
                    )
                )
            if not isinstance(args[0], np.ndarray):
                x = np.array(args[0])
            else:
                x = args[0].view(np.ndarray)
            if not isinstance(args[1], np.ndarray):
                y = np.array(args[1])
            else:
                y = args[1].view(np.ndarray)

        if 'x' in kwargs:
            x = kwargs['x']
        if 'y' in kwargs:
            y = kwargs['y']

        profiler('interpret data')
        # pull in all style arguments.
        # Use self.opts to fill in anything not present in kwargs.

        if 'name' in kwargs:
            self.opts['name'] = kwargs['name']
            self.setProperty('styleWasChanged', True)

        if 'connect' in kwargs:
            self.opts['connect'] = kwargs['connect']
            self.setProperty('styleWasChanged', True)
            
        if 'skipFiniteCheck' in kwargs:
            self.opts['skipFiniteCheck'] = kwargs['skipFiniteCheck']

        # if symbol pen/brush are given with no previously set symbol,
        # then assume symbol is 'o'
        if (
            'symbol' not in kwargs
            and (
                'symbolPen' in kwargs
                or 'symbolBrush' in kwargs
                or 'symbolSize' in kwargs
            ) and self.opts['symbol'] is None
        ):
            kwargs['symbol'] = 'o'

        if 'brush' in kwargs:
            kwargs['fillBrush'] = kwargs['brush']

        for k in list(self.opts.keys()):
            if k in kwargs:
                self.opts[k] = kwargs[k]
                self.setProperty('styleWasChanged', True)
        #curveArgs = {}
        #for k in ['pen', 'shadowPen', 'fillLevel', 'brush']:
            #if k in kwargs:
                #self.opts[k] = kwargs[k]
            #curveArgs[k] = self.opts[k]

        #scatterArgs = {}
        #for k,v in [('symbolPen','pen'), ('symbolBrush','brush'), ('symbol','symbol')]:
            #if k in kwargs:
                #self.opts[k] = kwargs[k]
            #scatterArgs[v] = self.opts[k]

        if y is None or len(y) == 0:  # empty data is represented as None
            yData = None
        else:  # actual data is represented by ndarray
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            yData = y.view(np.ndarray)
            if x is None:
                x = np.arange(len(y))
                
        if x is None or len(x) == 0:  # empty data is represented as None
            xData = None
        else:  # actual data is represented by ndarray
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            xData = x.view(np.ndarray)

        if xData is None or yData is None:
            self._dataset = None
        else:
            self._dataset = PlotDataset( xData, yData )

        profiler('set data')

        self._reloadYValues(styleUpdate=self.property("styleWasChanged"))
        
        # items have been updated
        self.setProperty('styleWasChanged', False)
        profiler('update items')

        self.sigPlotChanged.emit(self)
        profiler('emit')

    def updateItems(self, styleUpdate: bool = True):
        """
        Update the displayed curve and scatter plot.

        This method is called internally to redraw the curve and scatter plot when the
        data or graphics style has been updated. It is not usally necessary to call this
        from user code. 

        Parameters
        ----------
        styleUpdate : bool, default True
            Indicates if the style was updated in addition to the data.
        """

        # override styleUpdate request and always enforce update until we have a
        # better solution for:
        # - ScatterPlotItem losing per-point style information
        # - PlotDataItem performing multiple unnecessary setData calls on initialization
        # See: https://github.com/pyqtgraph/pyqtgraph/pull/1653
        if not styleUpdate:
            styleUpdate = True

        curveArgs = {}
        scatterArgs = {}

        if styleUpdate:  # repeat style arguments only when changed
            for k, v in [
                ('pen', 'pen'),
                ('shadowPen', 'shadowPen'),
                ('fillLevel', 'fillLevel'),
                ('fillOutline', 'fillOutline'),
                ('fillBrush', 'brush'),
                ('antialias', 'antialias'),
                ('connect', 'connect'),
                ('stepMode', 'stepMode'),
                ('skipFiniteCheck', 'skipFiniteCheck')
            ]:
                if k in self.opts:
                    curveArgs[v] = self.opts[k]

            for k, v in [
                ('symbolPen', 'pen'),
                ('symbolBrush', 'brush'),
                ('symbol', 'symbol'),
                ('symbolSize', 'size'),
                ('data', 'data'),
                ('pxMode', 'pxMode'),
                ('antialias', 'antialias'),
                ('useCache', 'useCache')
            ]:
                if k in self.opts:
                    scatterArgs[v] = self.opts[k]

        dataset = self._getDisplayDataset()
        if dataset is None:  # then we have nothing to show
            self.curve.hide()
            self.scatter.hide()
            return

        x = dataset.x
        y = dataset.y
        if dataset.connect is not None:
            curveArgs['connect'] = dataset.connect
        #scatterArgs['mask'] = self.dataMask
        if (
            self.opts['pen'] is not None
            or (
                self.opts['fillBrush'] is not None and
                self.opts['fillLevel'] is not None
            )
        ):  # draw if visible...
            # auto-switch to indicate non-finite values as interruptions in the curve:
            if (
                isinstance(curveArgs['connect'], str) and
                curveArgs['connect'] == 'auto'
            ):  # connect can also take a boolean array
                if dataset.containsNonfinite is False:
                    # all points can be connected, and no further check is needed.
                    curveArgs['connect'] = 'all'
                    curveArgs['skipFiniteCheck'] = True
                else:   # True or None
                    # True: (we checked and found non-finites)
                    #   don't connect non-finites
                    # None: (we haven't performed a check for non-finites yet)
                    #   use connect='finite' in case there are non-finites.
                    curveArgs['connect'] = 'finite'
                    curveArgs['skipFiniteCheck'] = False
            self.curve.setData(x=x, y=y, **curveArgs)
            self.curve.show()
        else:  # ...hide if not.
            self.curve.hide()

        if self.opts['symbol'] is not None:  # draw if visible...
            if self.opts.get('stepMode') == "center":
                x = 0.5 * (x[:-1] + x[1:])                
            self.scatter.setData(x=x, y=y, **scatterArgs)
            self.scatter.show()
        else:  # ...hide if not.
            self.scatter.hide()

    def getOriginalDataset(self) -> tuple[None, None] | tuple[np.ndarray, np.ndarray]:
        """
        Get the numpy array representation of the data provided to PlotDataItem.

        Returns
        -------
        xData : np.ndarray or None
            Representation of the original x-axis data.
        yData : np.ndarray or None
            Representation of the original y-axis data.

        See Also
        --------
        :meth:`getData`
            This method returns the transformed data displayed on the screen instead.
        """
        dataset = self._dataset
        return (None, None) if dataset is None else (dataset.x, dataset.y)

    def _getDisplayDataset(self) -> PlotDataset | None:
        """
        Get data suitable for display as a :class:`PlotDataset`.

        Warnings
        --------
        This method is not considered part of the public API.

        Returns
        ------- 
        :class:`PlotDataset`
            Data suitable for display (including mapping and data reduction) as
            ``dataset.x`` and ``dataset.y``.
        """
        if self._dataset is None:
            return None
        # Return cached processed dataset if available and still valid:
        if (
            self._datasetDisplay is not None and
            not (self.property('xViewRangeWasChanged') and self.opts['clipToView']) and
            not (self.property('xViewRangeWasChanged') and self.opts['autoDownsample']) and
            not (self.property('yViewRangeWasChanged') and self.opts['dynamicRangeLimit'] is not None)
        ):
            return self._datasetDisplay

        # Apply data mapping functions if mapped dataset is not yet available: 
        if self._datasetMapped is None:
            x = self._dataset.x
            y = self._dataset.y
            if y.dtype == bool:
                y = y.astype(np.uint8)
            if x.dtype == bool:
                x = x.astype(np.uint8)
            if self.opts['subtractMeanMode']:
                y = y - np.mean(y)
            if self.opts['fftMode']:
                x, y = self._fourierTransform(x, y)
                # Ignore the first bin for fft data if we have a logx scale
                if self.opts['logMode'][0]:
                    x = x[1:]
                    y = y[1:]
            if self.opts['derivativeMode']:  # plot dV/dt
                y = np.diff(self._dataset.y) / np.diff(self._dataset.x)
                x = x[:-1]
            if self.opts['phasemapMode']:  # plot dV/dt vs V
                x = self._dataset.y[:-1]
                y = np.diff(self._dataset.y) / np.diff(self._dataset.x)

            dataset = PlotDataset(
                x,
                y,
                self._dataset.xAllFinite,
                self._dataset.yAllFinite
            )
            
            if True in self.opts['logMode']:
                # Apply log scaling for x and/or y-axis
                dataset.applyLogMapping( self.opts['logMode'] )

            self._datasetMapped = dataset
        
        # apply processing that affects the on-screen display of data:
        x = self._datasetMapped.x
        y = self._datasetMapped.y
        xAllFinite = self._datasetMapped.xAllFinite
        yAllFinite = self._datasetMapped.yAllFinite

        view = self.getViewBox()
        if view is None:
            view_range = None
        else:
            view_range = view.viewRect()  # this is always up-to-date
        if view_range is None:
            view_range = self.viewRect()

        self._ds = self.opts['downsample']
        if not isinstance(self._ds, int):
            self._ds = 1

        if self.opts['autoDownsample']:
            # this option presumes that x-values have uniform spacing
            x_finite = x if xAllFinite else x[np.isfinite(x)]
            self._ds = self._getAutoDownsampleFactor(x_finite, view_range)

        connect = (
            self.opts['connect']
            if isinstance(self.opts['connect'], np.ndarray)
            else None
        )
        if self.opts['clipToView']:
            if view is None or view.autoRangeEnabled()[0]:
                pass  # no ViewBox to clip to, or view will autoscale to data range.
            else:
                # clip-to-view always presumes that x-values are in increasing order
                if view_range is not None and len(x) > 1:
                    x, y, connect = self._clipToView(
                        x,
                        y,
                        connect,
                        view_range.left(),
                        view_range.right(),
                        shift=self._ds,
                    )

        if self._ds > 1:
            x, y, connect = self._downsample(x, y, connect)

        if self.opts['dynamicRangeLimit'] is not None and view_range is not None:
            data_range = self._datasetMapped.dataRect()
            if data_range is not None:
                view_height = view_range.height()
                limit = self.opts['dynamicRangeLimit']
                hyst = self.opts['dynamicRangeHyst']
                # never clip data if it fits into +/- (extended) limit * view height
                if (
                    # note that "bottom" is the larger number, and "top" is the smaller
                    # one. Never clip if the view does not show anything and would cause
                    # division by zero
                    view_height > 0
                    # never clip if all data is too small to see
                    and not data_range.bottom() < view_range.top()
                    # never clip if all data is too large to see
                    and not data_range.top() > view_range.bottom()
                    and data_range.height() > 2 * hyst * limit * view_height
                ):
                    cache_is_good = False
                    # check if cached display data can be reused:
                    if self._datasetDisplay is not None:
                        # top is minimum value, bottom is maximum value
                        # how many multiples of the current view height does the clipped
                        # plot extend to the top and bottom?
                        top_exc = (
                            -(self._drlLastClip[0] - view_range.bottom()) / view_height
                        )
                        bot_exc = (
                            self._drlLastClip[1] - view_range.top()
                        ) / view_height
                        if (
                            limit / hyst <= top_exc <= limit * hyst
                            and limit / hyst <= bot_exc <= limit * hyst
                        ):
                            # restore cached values
                            x = self._datasetDisplay.x
                            y = self._datasetDisplay.y
                            cache_is_good = True
                    if not cache_is_good:
                        min_val = view_range.bottom() - limit * view_height
                        max_val = view_range.top() + limit * view_height
                        y = fn.clip_array(y, min_val, max_val)
                        self._drlLastClip = (min_val, max_val)
        self._datasetDisplay = PlotDataset(x, y, xAllFinite, yAllFinite, connect)
        self.setProperty('xViewRangeWasChanged', False)
        self.setProperty('yViewRangeWasChanged', False)

        return self._datasetDisplay

    def _reloadYValues(self, styleUpdate=False):
        """Invalidate display data and update items."""
        self._datasetMapped = None  # invalidate mapped data
        self._datasetDisplay = None  # invalidate display data
        self._adsLastValue = 1  # reset auto-downsample value
        self._downsampling_cache_x = None  # Invalidate downsampling cache
        self._downsampling_cache_y = None
        self._downsampling_cache_connect = None
        self._cache_downsampling_factor = 1
        self.updateItems(styleUpdate=styleUpdate)
        self._buildDownsamplingCache()
        self.informViewBoundsChanged()

    def _buildDownsamplingCache(self):
        """Build a cache of downsampled data."""
        if not self.opts["useDownsamplingCache"]:
            return
        if self._datasetMapped is None:
            return
        x = self._datasetMapped.x
        y = self._datasetMapped.y
        connect = self._datasetMapped.connect
        if not self.opts["autoDownsample"]:
            if self.opts["downsample"] == 1:
                return
            self._cache_downsampling_factor = self.opts["downsample"]
        else:
            self._cache_downsampling_factor = max(
                1, int(len(x) / self.opts["downsamplingCacheSize"])
            )
        self._ds = self._cache_downsampling_factor
        (
            self._downsampling_cache_x,
            self._downsampling_cache_y,
            self._downsampling_cache_connect,
        ) = self._downsample(x, y, connect)

    def _downsample(self, x, y, connect):
        if self._use_downsampling_cache():
            return self._clipToView(
                self._downsampling_cache_x,
                self._downsampling_cache_y,
                self._downsampling_cache_connect,
                x[0],
                x[-1],
            )
        if self.opts["downsampleMethod"] == "subsample":
            return self._subsample(x, y, connect)
        if self.opts["downsampleMethod"] == "mean":
            return self._mean_downsample(x, y, connect)
        if self.opts["downsampleMethod"] == "peak":
            return self._peak_downsample(x, y, connect)
        raise ValueError(
            "Unknown downsample method: %s" % self.opts["downsampleMethod"]
        )

    def _use_downsampling_cache(self):
        has_cache = self._downsampling_cache_x is not None
        return has_cache and (self._ds == self._cache_downsampling_factor)

    def _subsample(self, x, y, connect):
        x = x[:: self._ds]
        y = y[:: self._ds]
        if connect is not None:
            connect = connect[:: self._ds]
        return x, y, connect

    def _mean_downsample(self, x, y, connect):
        n = len(x) // self._ds
        # start of x-values try to select a somewhat centered point
        stx = self._ds // 2
        x = x[stx : stx + n * self._ds : self._ds]
        y = y[: n * self._ds].reshape(n, self._ds).mean(axis=1)
        if connect is not None:
            connect = connect[: n * self._ds].reshape(n, self._ds).all(axis=1)
        return x, y, connect

    def _peak_downsample(self, x, y, connect):
        n = len(x) // self._ds
        x1 = np.empty((n, 2))
        # start of x-values; try to select a somewhat centered point
        stx = self._ds // 2
        x1[:] = x[stx : stx + n * self._ds : self._ds, np.newaxis]
        x = x1.reshape(n * 2)
        y1 = np.empty((n, 2))
        y2 = y[: n * self._ds].reshape((n, self._ds))
        y1[:, 0] = y2.max(axis=1)
        y1[:, 1] = y2.min(axis=1)
        y = y1.reshape(n * 2)
        if connect is not None:
            c = np.ones((n * 2), dtype=bool)
            c[1::2] = connect[: n * self._ds].reshape(n, self._ds).all(axis=1)
            connect = c
        return x, y, connect

    def _getAutoDownsampleFactor(self, finite_x, view_range) -> int:
        ds = 1
        if view_range is not None and len(finite_x) > 1:
            dx = float(finite_x[-1] - finite_x[0]) / (len(finite_x) - 1)
            if dx != 0.0:
                width = self.getViewBox().width()
                if (
                    self._downsampling_cache_x is not None
                    and ((view_range.width() / dx) // self._cache_downsampling_factor)
                    > width * self.opts["minSampPerPxForCache"]
                ):
                    # Keep the maximum value, which allows for caching
                    ds = self._cache_downsampling_factor
                else:
                    if width != 0.0:  # autoDownsampleFactor _should_ be > 1.0
                        ds_float = max(
                            1.0,
                            abs(
                                view_range.width()
                                / dx
                                / (width * self.opts["autoDownsampleFactor"])
                            ),
                        )
                        if math.isfinite(ds_float):
                            ds = int(ds_float)

        # use the last computed value if our new value is not too different.
        # this guards against an infinite cycle where the plot never stabilizes.
        if math.isclose(ds, self._adsLastValue, rel_tol=0.01):
            ds = self._adsLastValue
        self._adsLastValue = ds
        return ds

    def _clipToView(self, x, y, connect, startx, endx, shift=0):
        # find first in-view value (left edge) and first out-of-view value
        # (right edge) since we want the curve to go to the edge of the
        # screen, we need to preserve one down-sampled point on the left and
        # one of the right, so we extend the interval

        # np.searchsorted performs poorly when the array.dtype does not
        # match the type of the value (float) being searched.
        # see: https://github.com/pyqtgraph/pyqtgraph/pull/2719
        # x0 = np.searchsorted(x, view_range.left()) - ds
        x0 = bisect.bisect_left(x, startx) - shift
        # x0 = np.clip(x0, 0, len(x))
        x0 = fn.clip_scalar(x0, 0, len(x))  # workaround

        # x1 = np.searchsorted(x, view_range.right()) + ds
        x1 = bisect.bisect_left(x, endx) + shift
        # x1 = np.clip(x1, 0, len(x))
        x1 = fn.clip_scalar(x1, x0, len(x))
        x = x[x0:x1]
        y = y[x0:x1]
        if connect is not None:
            connect = connect[x0:x1]
        return x, y, connect

    def getData(self) -> tuple[None, None] | tuple[np.ndarray, np.ndarray]:
        """
        Get a representation of the data displayed on screen.

        Returns
        -------
        xData : np.ndarray or None
            The x-axis data, after mapping and data reduction if present or ``None``.
        yData : np.ndarray or None
            The y-axis data, after mapping and data reduction if present or ``None``.

        See Also
        --------
        :meth:`getOriginalDataset`
            This method returns the original data provided to PlotDataItem instead.
        """
        dataset = self._getDisplayDataset()
        return (None, None) if dataset is None else (dataset.x, dataset.y)

    # compatibility method for access to dataRect for full dataset:
    def dataRect(self) -> QtCore.QRectF | None:
        """
        The bounding rectangle for the full set of data.

        Returns
        -------
        :class:`QRectF` or None
            Will return ``None`` if there is no data or if all values (x or y) are
            ``NaN``.
        """
        return None if self._dataset is None else self._dataset.dataRect()

    def dataBounds(
        self,
        ax: int,
        frac: float = 1.0,
        orthoRange: tuple[float, float] | None = None
    ) -> tuple[float, float] | tuple[None, None]:
        """
        Get the range occupied by the data (along a specific axis) for this item.

        This method is called by :class:`~pyqtgraph.ViewBox` when auto-scrolling.

        Parameters
        ----------
        ax : { 0, 1 }
            The axis for which to return this items data range.
            * 0 - x-axis
            * 1 - y-axis
        frac : float, default 1.0
            Specifies the fraction of the total data range to return. By default, the
            entire range is returned.  This allows the :class:`~pyqtgraph.ViewBox` to
            ignore large spikes in the data when auto-scrolling.
        orthoRange : tuple of float, float or None, optional, default None
            Specify that only the data within the given range (orthogonal to `ax`),
            should be measured when returning the data range.  For example, a
            :class:`~pyqtgraph.ViewBox` might ask what is the y-range of all data with
            x-values between the specifies (min, max) range.

        Returns
        -------
        min : float or None
            The minimum end of the range that the data occupies along the specified
            axis. ``None`` if there is no data.
        max : float or None
            The maximum end of the range that the data occupies along the specified
            axis. ``None`` if there is no data.
        """
        bounds: tuple[None, None] | tuple[float, float] = (None, None)
        if self.curve.isVisible():
            bounds = self.curve.dataBounds(ax, frac, orthoRange)
        if self.scatter.isVisible():
            bounds2 = self.scatter.dataBounds(ax, frac, orthoRange)
            bounds = (
                min(
                    (i for i in [bounds2[0], bounds[0]] if i is not None), default=None
                ),
                min(
                    (i for i in [bounds2[1], bounds[1]] if i is not None), default=None
                )
            )
        return bounds

    def pixelPadding(self) -> int:
        """
        Get the size (in pixels) that this item might draw beyond the data.
        
        The size of scatter plot symbols or width of the line plot make the
        displayed image extend further than the extend of the raw data. 

        Returns
        -------
        int
            The padding size in pixels that this item may draw beyond the values
            returned by :meth:`dataBounds`. This method is called by :class:`ViewBox`
            when auto-scaling.
        """
        pad = 0
        if self.curve.isVisible():
            pad = max(pad, self.curve.pixelPadding())
        elif self.scatter.isVisible():
            pad = max(pad, self.scatter.pixelPadding())
        return pad

    def clear(self):
        self._dataset = self._datasetMapped = self._datasetDisplay = None
        self.curve.clear()
        self.scatter.clear()

    def appendData(self, *args, **kwargs):
        pass

    @QtCore.Slot(object, object)
    def curveClicked(self, _: PlotCurveItem, ev):
        warnings.warn(
            (
                "PlotCurveItem.curveClicked is deprecated, and will be removed in a "
                "future version of pyqtgraph."
            ), DeprecationWarning, stacklevel=3
        )
        self.sigClicked.emit(self, ev)

    @QtCore.Slot(object, object, object)
    def scatterClicked(self, _, points, ev):
        self.sigClicked.emit(self, ev)
        self.sigPointsClicked.emit(self, points, ev)

    @QtCore.Slot(object, object, object)
    def scatterHovered(self, _, points, ev):
        warnings.warn(
            (
                "PlotCurveItem.scatterHovered is deprecated, and will be removed in a "
                "future version of pyqtgraph."
            ), DeprecationWarning, stacklevel=3
        )
        self.sigPointsHovered.emit(self, points, ev)

    # def viewTransformChanged(self):
    #   """ view transform (and thus range) has changed, replot if needed """
    # viewTransformChanged is only called when the cached viewRect of GraphicsItem
    # has already been invalidated. However, responding here will make PlotDataItem
    # update curve and scatter later than intended.
    #   super().viewTransformChanged() # this invalidates the viewRect() cache!
        
    @QtCore.Slot(object, object)
    @QtCore.Slot(object, object, object)
    def viewRangeChanged(self, vb=None, ranges=None, changed=None):
        # view range has changed; re-plot if needed 
        update_needed = False
        if changed is None or changed[0]: 
            # if ranges is not None:
            #     print('hor:', ranges[0])
            self.setProperty('xViewRangeWasChanged', True)
            if (
                self.opts['clipToView']
                or self.opts['autoDownsample']
            ):
                self._datasetDisplay = None
                update_needed = True
        if changed is None or changed[1]:
            # if ranges is not None:
            #     print('ver:', ranges[1])
            self.setProperty('yViewRangeWasChanged', True)
            if self.opts['dynamicRangeLimit'] is not None:
                # update, but do not discard cached display data
                update_needed = True
        if update_needed:
            self.updateItems(styleUpdate=False)

    @staticmethod
    def _fourierTransform(x, y):
        # Perform Fourier transform. If x values are not sampled uniformly,
        # then use np.interp to resample before taking fft.
        if len(x) == 1: 
            return np.array([0]), abs(y)
        dx = np.diff(x)
        uniform = not np.any(np.abs(dx - dx[0]) > (abs(dx[0]) / 1000.))
        if not uniform:
            x2 = np.linspace(x[0], x[-1], len(x))
            y = np.interp(x2, x, y)
            x = x2
        n = y.size
        f = np.fft.rfft(y) / n
        d = float(x[-1] - x[0]) / (len(x) - 1)
        x = np.fft.rfftfreq(n, d)
        y = np.abs(f)
        return x, y


def dataType(obj) -> str:
    type_: str
    if hasattr(obj, '__len__') and len(obj) == 0:
        type_ = 'empty'
    elif isinstance(obj, dict):
        type_ = 'dictOfLists'
    elif np.iterable(obj):
        first = obj[0]
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                type_ = 'listOfValues' if obj.dtype.names is None else 'recarray'
            elif obj.ndim == 2 and obj.dtype.names is None and obj.shape[1] == 2:
                type_ = 'Nx2array'
            else:
                raise ValueError(
                    f'array shape must be (N,) or (N,2); got {str(obj.shape)} instead'
                )
        elif isinstance(first, dict):
            type_ = 'listOfDicts'
        else:
            type_ = 'listOfValues'
    else:
        raise ValueError("Cannot identify data-structure.")
    return type_


def isSequence(obj):
    warnings.warn(
        (
            "isSequence is deprecated and will be removed in a future version of"
            "pyqtgraph, use np.iterable(obj) instead."
        ), DeprecationWarning, stacklevel=2
    )
    return (
        hasattr(obj, '__iter__') or
        isinstance(obj, np.ndarray) or
        (
            hasattr(obj, 'implements') and
            obj.implements('MetaArray')
        )
    )