import math
import warnings

import numpy as np

from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore
from .GraphicsObject import GraphicsObject
from .PlotCurveItem import PlotCurveItem
from .ScatterPlotItem import ScatterPlotItem

__all__ = ['PlotDataItem']

class PlotDataset(object):
    """
    :orphan:
    .. warning:: This class is intended for internal use. The interface may change without warning.

    Holds collected information for a plotable dataset. 
    Numpy arrays containing x and y coordinates are available as ``dataset.x`` and ``dataset.y``.
    
    After a search has been performed, typically during a call to :func:`dataRect() <pyqtgraph.PlotDataset.dataRect>`, 
    ``dataset.containsNonfinite`` is `True` if any coordinate values are nonfinite (e.g. NaN or inf) or `False` if all 
    values are finite. If no search has been performed yet, ``dataset.containsNonfinite`` is `None`.

    For internal use in :class:`PlotDataItem <pyqtgraph.PlotDataItem>`, this class should not be instantiated when no data is available. 
    """
    def __init__(self, x, y):
        """ 
        Parameters
        ----------
        x: array
            x coordinates of data points. 
        y: array
            y coordinates of data points. 
        """
        super().__init__()
        self.x = x
        self.y = y
        self._dataRect = None
        self.containsNonfinite = None
        
    def _updateDataRect(self):
        """ 
        Finds bounds of plotable data and stores them as ``dataset._dataRect``, 
        stores information about the presence of nonfinite data points.
            """
        if self.y is None or self.x is None:
            return None
        if self.containsNonfinite is False: # all points are directly usable.
            ymin = np.min( self.y ) # find minimum of all values
            ymax = np.max( self.y ) # find maximum of all values
            xmin = np.min( self.x ) # find minimum of all values
            xmax = np.max( self.x ) # find maximum of all values
        else: # This may contain NaN values and infinites.
            selection = np.isfinite(self.y)    # We are looking for the bounds of the plottable data set. Infinite and Nan are ignored. 
            all_y_are_finite = selection.all() # False if all values are finite, True if there are any non-finites
            try:
                ymin = np.min( self.y[selection] ) # find minimum of all finite values
                ymax = np.max( self.y[selection] ) # find maximum of all finite values
            except ValueError: # is raised when there are no finite values
                ymin = np.nan
                ymax = np.nan
            selection = np.isfinite(self.x) # We are looking for the bounds of the plottable data set. Infinite and Nan are ignored. 
            all_x_are_finite = selection.all() # False if all values are finite, True if there are any non-finites
            try:
                xmin = np.min( self.x[selection] ) # find minimum of all finite values
                xmax = np.max( self.x[selection] ) # find maximum of all finite values
            except ValueError: # is raised when there are no finite values
                xmin = np.nan
                xmax = np.nan
            self.containsNonfinite = not (all_x_are_finite and all_y_are_finite) # This always yields a definite True/False answer
        self._dataRect = QtCore.QRectF( QtCore.QPointF(xmin,ymin), QtCore.QPointF(xmax,ymax) )
        
    def dataRect(self):
        """
        Returns a bounding rectangle (as :class:`QtCore.QRectF`) for the finite subset of data.
        If there is an active mapping function, such as logarithmic scaling, then bounds represent the mapped data. 
        Will return `None` if there is no data or if all values (`x` or `y`) are NaN.
        """
        if self._dataRect is None: 
            self._updateDataRect()
        return self._dataRect

    def applyLogMapping(self, logMode):
        """
        Applies a logarithmic mapping transformation (base 10) if requested for the respective axis.
        This replaces the internal data. Values of ``-inf`` resulting from zeros in the original dataset are
        replaced by ``np.NaN``.
        
        Parameters
        ----------
        logmode: tuple or list of two bool
            A `True` value requests log-scale mapping for the x and y axis (in this order).
        """
        all_x_finite = False
        if logMode[0]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.x = np.log10(self.x)
            nonfinites = ~np.isfinite( self.x )
            if nonfinites.any():
                self.x[nonfinites] = np.nan # set all non-finite values to NaN
                self.containsNonfinite = True
            else:
                all_x_finite = True
        all_y_finite = False
        if logMode[1]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.y = np.log10(self.y)
            nonfinites = ~np.isfinite( self.y )
            if nonfinites.any():
                self.y[nonfinites] = np.nan # set all non-finite values to NaN
                self.containsNonfinite = True
            else:
                all_y_finite = True
        if all_x_finite and all_y_finite: 
            self.containsNonfinite = False # mark as False only if both axes were checked.
        
class PlotDataItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    :class:`PlotDataItem` provides a unified interface for displaying plot curves, scatter plots, or both.
    It also contains methods to transform or decimate the original data before it is displayed. 

    As pyqtgraph's standard plotting object, ``plot()`` methods such as :func:`pyqtgraph.plot` and
    :func:`PlotItem.plot() <pyqtgraph.PlotItem.plot>` create instances of :class:`PlotDataItem`.

    While it is possible to use :class:`PlotCurveItem <pyqtgraph.PlotCurveItem>` or
    :class:`ScatterPlotItem <pyqtgraph.ScatterPlotItem>` individually, this is recommended only
    where performance is critical and the limited functionality of these classes is sufficient.

    ==================================  ==============================================
    **Signals:**
    sigPlotChanged(self)                Emitted when the data in this item is updated.
    sigClicked(self, ev)                Emitted when the item is clicked.
    sigPointsClicked(self, points, ev)  Emitted when a plot point is clicked
                                        Sends the list of points under the mouse.
    sigPointsHovered(self, points, ev)  Emitted when a plot point is hovered over.
                                        Sends the list of points under the mouse.
    ==================================  ==============================================
    """

    sigPlotChanged = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object)
    sigPointsClicked = QtCore.Signal(object, object, object)
    sigPointsHovered = QtCore.Signal(object, object, object)

#        **(x,y data only)**

    def __init__(self, *args, **kargs):
        """
        There are many different ways to create a PlotDataItem.

        **Data initialization arguments:** (x,y data only)

            ========================== =========================================
            PlotDataItem(x, y)         x, y: array_like coordinate values
            PlotDataItem(y)            y values only -- x will be
                                       automatically set to ``range(len(y))``
            PlotDataItem(x=x, y=y)     x and y given by keyword arguments
            PlotDataItem(ndarray(N,2)) single numpy array with shape (N, 2),
                                       where ``x=data[:,0]`` and ``y=data[:,1]``
            ========================== =========================================

        **Data initialization arguments:** (x,y data AND may include spot style)

            ============================ ===============================================
            PlotDataItem(recarray)       numpy record array with ``dtype=[('x', float),
                                         ('y', float), ...]``
            PlotDataItem(list-of-dicts)  ``[{'x': x, 'y': y, ...},   ...]``
            PlotDataItem(dict-of-lists)  ``{'x': [...], 'y': [...],  ...}``
            ============================ ===============================================
        
        **Line style keyword arguments:**

            ============ ==============================================================================
            connect      Specifies how / whether vertexes should be connected. See below for details.
            pen          Pen to use for drawing the lines between points.
                         Default is solid grey, 1px width. Use None to disable line drawing.
                         May be a ``QPen`` or any single argument accepted by 
                         :func:`mkPen() <pyqtgraph.mkPen>`
            shadowPen    Pen for secondary line to draw behind the primary line. Disabled by default.
                         May be a ``QPen`` or any single argument accepted by 
                         :func:`mkPen() <pyqtgraph.mkPen>`
            fillLevel    If specified, the area between the curve and fillLevel is filled.
            fillOutline  (bool) If True, an outline surrounding the *fillLevel* area is drawn.
            fillBrush    Fill to use in the *fillLevel* area. May be any single argument accepted by 
                         :func:`mkBrush() <pyqtgraph.mkBrush>`
            stepMode     (str or None) If specified and not None, a stepped curve is drawn.
                         For 'left' the specified points each describe the left edge of a step.
                         For 'right', they describe the right edge. 
                         For 'center', the x coordinates specify the location of the step boundaries.
                         This mode is commonly used for histograms. Note that it requires an additional
                         x value, such that len(x) = len(y) + 1 .

            ============ ==============================================================================
        
        ``connect`` supports the following arguments:
        
        - 'all' connects all points.  
        - 'pairs' generates lines between every other point.
        - 'finite' creates a break when a nonfinite points is encountered. 
        - If an ndarray is passed, it should contain `N` int32 values of 0 or 1.
          Values of 1 indicate that the respective point will be connected to the next.
        - In the default 'auto' mode, PlotDataItem will normally use 'all', but if any
          nonfinite data points are detected, it will automatically switch to 'finite'.
          
        See :func:`arrayToQPath() <pyqtgraph.arrayToQPath>` for more details.
        
        **Point style keyword arguments:**  (see :func:`ScatterPlotItem.setData() <pyqtgraph.ScatterPlotItem.setData>` for more information)

            ============ ======================================================
            symbol       Symbol to use for drawing points, or a list of symbols
                         for each. The default is no symbol.
            symbolPen    Outline pen for drawing points, or a list of pens, one
                         per point. May be any single argument accepted by
                         :func:`mkPen() <pyqtgraph.mkPen>`.
            symbolBrush  Brush for filling points, or a list of brushes, one 
                         per point. May be any single argument accepted by
                         :func:`mkBrush() <pyqtgraph.mkBrush>`.
            symbolSize   Diameter of symbols, or list of diameters.
            pxMode       (bool) If True, then symbolSize is specified in
                         pixels. If False, then symbolSize is
                         specified in data coordinates.
            ============ ======================================================
            
        Any symbol recognized by :class:`ScatterPlotItem <pyqtgraph.ScatterPlotItem>` can be specified,
        including 'o' (circle), 's' (square), 't', 't1', 't2', 't3' (triangles of different orientation),
        'd' (diamond), '+' (plus sign), 'x' (x mark), 'p' (pentagon), 'h' (hexagon) and 'star'.
        
        Symbols can also be directly given in the form of a :class:`QtGui.QPainterPath` instance.

        **Optimization keyword arguments:**

            ================= =======================================================================
            useCache          (bool) By default, generated point graphics items are cached to
                              improve performance. Setting this to False can improve image quality
                              in certain situations.
            antialias         (bool) By default, antialiasing is disabled to improve performance.
                              Note that in some cases (in particular, when ``pxMode=True``), points
                              will be rendered antialiased even if this is set to `False`.
            downsample        (int) Reduce the number of samples displayed by the given factor.
            downsampleMethod  'subsample': Downsample by taking the first of N samples.
                              This method is fastest and least accurate.
                              'mean': Downsample by taking the mean of N samples.
                              'peak': Downsample by drawing a saw wave that follows the min
                              and max of the original data. This method produces the best
                              visual representation of the data but is slower.
            autoDownsample    (bool) If `True`, resample the data before plotting to avoid plotting
                              multiple line segments per pixel. This can improve performance when
                              viewing very high-density data, but increases the initial overhead
                              and memory usage.
            clipToView        (bool) If `True`, only data visible within the X range of the containing
                              :class:`ViewBox` is plotted. This can improve performance when plotting
                              very large data sets where only a fraction of the data is visible
                              at any time.
            dynamicRangeLimit (float or `None`) Limit off-screen y positions of data points. 
                              `None` disables the limiting. This can increase performance but may
                              cause plots to disappear at high levels of magnification.
                              The default of 1e6 limits data to approximately 1,000,000 times the 
                              :class:`ViewBox` height.
            dynamicRangeHyst  (float) Permits changes in vertical zoom up to the given hysteresis
                              factor (the default is 3.0) before the limit calculation is repeated.
            skipFiniteCheck   (bool, default `False`) Optimization flag that can speed up plotting by not 
                              checking and compensating for NaN values.  If set to `True`, and NaN 
                              values exist, unpredictable behavior will occur. The data may not be
                              displayed or the plot may take a significant performance hit.
                              
                              In the default 'auto' connect mode, `PlotDataItem` will apply this 
                              setting automatically.
            ================= =======================================================================

        **Meta-info keyword arguments:**

            ==========   ================================================
            name         (string) Name of item for use in the plot legend
            ==========   ================================================

        **Notes on performance:**
        
        Plotting lines with the default single-pixel width is the fastest available option. For such lines,
        translucent colors (`alpha` < 1) do not result in a significant slowdown.
        
        Wider lines increase the complexity due to the overlap of individual line segments. Translucent colors
        require merging the entire plot into a single entity before the alpha value can be applied. For plots with more
        than a few hundred points, this can result in excessive slowdown.

        Since version 0.12.4, this slowdown is automatically avoided by an algorithm that draws line segments
        separately for fully opaque lines. Setting `alpha` < 1 reverts to the previous, slower drawing method.
        
        For lines with a width of more than 4 pixels, :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>` will automatically
        create a ``QPen`` with `Qt.PenCapStyle.RoundCap` to ensure a smooth connection of line segments. This incurs a
        small performance penalty.

        """
        GraphicsObject.__init__(self)
        self.setFlag(self.GraphicsItemFlag.ItemHasNoContents)
        # Original data, mapped data, and data processed for display is now all held in PlotDataset objects.
        # The convention throughout PlotDataItem is that a PlotDataset is only instantiated if valid data is available.
        self._dataset        = None # will hold a PlotDataset for the original data, accessed by getOriginalData()
        self._datasetMapped  = None # will hold a PlotDataset for data after mapping transforms (e.g. log scale)
        self._datasetDisplay = None # will hold a PlotDataset for data downsampled and limited for display, accessed by getData()
        self.curve = PlotCurveItem()
        self.scatter = ScatterPlotItem()
        self.curve.setParentItem(self)
        self.scatter.setParentItem(self)

        self.curve.sigClicked.connect(self.curveClicked)
        self.scatter.sigClicked.connect(self.scatterClicked)
        self.scatter.sigHovered.connect(self.scatterHovered)
        
        # self._xViewRangeWasChanged = False
        # self._yViewRangeWasChanged = False
        # self._styleWasChanged = True # force initial update
        
        # update-required notifications are handled through properties to allow future management through
        # the QDynamicPropertyChangeEvent sent on any change.
        self.setProperty('xViewRangeWasChanged', False)
        self.setProperty('yViewRangeWasChanged', False)
        self.setProperty('styleWasChanged', True) # force initial update

        self._drlLastClip = (0.0, 0.0) # holds last clipping points of dynamic range limiter
        #self.clear()
        self.opts = {
            'connect': 'auto', # defaults to 'all', unless overridden to 'finite' for log-scaling
            'skipFiniteCheck': False, 
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
            'symbolPen': (200,200,200),
            'symbolBrush': (50, 50, 150),
            'pxMode': True,

            'antialias': getConfigOption('antialias'),
            'pointMode': None,

            'useCache': True,
            'downsample': 1,
            'autoDownsample': False,
            'downsampleMethod': 'peak',
            'autoDownsampleFactor': 5.,  # draw ~5 samples per pixel
            'clipToView': False,
            'dynamicRangeLimit': 1e6,
            'dynamicRangeHyst': 3.0,
            'data': None,
        }
        self.setCurveClickable(kargs.get('clickable', False))
        self.setData(*args, **kargs)
    
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

    def name(self):
        """ Returns the name that represents this item in the legend. """
        return self.opts.get('name', None)

    def setCurveClickable(self, state, width=None):
        """ ``state=True`` sets the curve to be clickable, with a tolerance margin represented by `width`. """
        self.curve.setClickable(state, width)

    def curveClickable(self):
        """ Returns `True` if the curve is set to be clickable. """
        return self.curve.clickable

    def boundingRect(self):
        return QtCore.QRectF()  ## let child items handle this

    def setPos(self, x, y):
        GraphicsObject.setPos(self, x, y)
        # to update viewRect:
        self.viewTransformChanged()
        # to update displayed point sets, e.g. when clipping (which uses viewRect):
        self.viewRangeChanged()

    def setAlpha(self, alpha, auto):
        if self.opts['alphaHint'] == alpha and self.opts['alphaMode'] == auto:
            return
        self.opts['alphaHint'] = alpha
        self.opts['alphaMode'] = auto
        self.setOpacity(alpha)
        #self.update()

    def setFftMode(self, state):
        """
        ``state = True`` enables mapping the data by a fast Fourier transform.
        If the `x` values are not equidistant, the data set is resampled at
        equal intervals. 
        """
        if self.opts['fftMode'] == state:
            return
        self.opts['fftMode'] = state
        self._datasetMapped  = None
        self._datasetDisplay = None
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setLogMode(self, xState, yState):
        """
        When log mode is enabled for the respective axis by setting ``xState`` or 
        ``yState`` to `True`, a mapping according to ``mapped = np.log10( value )``
        is applied to the data. For negative or zero values, this results in a 
        `NaN` value.
        """
        if self.opts['logMode'] == [xState, yState]:
            return
        self.opts['logMode'] = [xState, yState]
        self._datasetMapped  = None  # invalidate mapped data
        self._datasetDisplay = None  # invalidate display data
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setDerivativeMode(self, state):
        """
        ``state = True`` enables derivative mode, where a mapping according to
        ``y_mapped = dy / dx`` is applied, with `dx` and `dy` representing the 
        differences between adjacent `x` and `y` values.
        """
        if self.opts['derivativeMode'] == state:
            return
        self.opts['derivativeMode'] = state
        self._datasetMapped  = None  # invalidate mapped data
        self._datasetDisplay = None  # invalidate display data
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setPhasemapMode(self, state):
        """
        ``state = True`` enables phase map mode, where a mapping 
        according to ``x_mappped = y`` and ``y_mapped = dy / dx``
        is applied, plotting the numerical derivative of the data over the 
        original `y` values.
        """
        if self.opts['phasemapMode'] == state:
            return
        self.opts['phasemapMode'] = state
        self._datasetMapped  = None  # invalidate mapped data
        self._datasetDisplay = None  # invalidate display data
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setPen(self, *args, **kargs):
        """
        Sets the pen used to draw lines between points.
        The argument can be a :class:`QtGui.QPen` or any combination of arguments accepted by 
        :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>`.
        """
        pen = fn.mkPen(*args, **kargs)
        self.opts['pen'] = pen
        #self.curve.setPen(pen)
        #for c in self.curves:
            #c.setPen(pen)
        #self.update()
        self.updateItems(styleUpdate=True)

    def setShadowPen(self, *args, **kargs):
        """
        Sets the shadow pen used to draw lines between points (this is for enhancing contrast or
        emphasizing data). This line is drawn behind the primary pen and should generally be assigned 
        greater width than the primary pen.
        The argument can be a :class:`QtGui.QPen` or any combination of arguments accepted by 
        :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>`.
        """
        if args[0] is None:
            pen = None
        else:
            pen = fn.mkPen(*args, **kargs)
        self.opts['shadowPen'] = pen
        #for c in self.curves:
            #c.setPen(pen)
        #self.update()
        self.updateItems(styleUpdate=True)

    def setFillBrush(self, *args, **kargs):
        """ 
        Sets the :class:`QtGui.QBrush` used to fill the area under the curve.
        See :func:`mkBrush() <pyqtgraph.mkBrush>`) for arguments.
        """
        if args[0] is None:
            brush = None
        else:
            brush = fn.mkBrush(*args, **kargs)
        if self.opts['fillBrush'] == brush:
            return
        self.opts['fillBrush'] = brush
        self.updateItems(styleUpdate=True)

    def setBrush(self, *args, **kargs):
        """
        See :func:`~pyqtgraph.PlotDataItem.setFillBrush`
        """
        return self.setFillBrush(*args, **kargs)

    def setFillLevel(self, level):
        """
        Enables filling the area under the curve towards the value specified by 
        `level`. `None` disables the filling. 
        """
        if self.opts['fillLevel'] == level:
            return
        self.opts['fillLevel'] = level
        self.updateItems(styleUpdate=True)

    def setSymbol(self, symbol):
        """ `symbol` can be any string recognized by 
        :class:`ScatterPlotItem <pyqtgraph.ScatterPlotItem>` or a list that
        specifies a symbol for each point.
        """
        if self.opts['symbol'] == symbol:
            return
        self.opts['symbol'] = symbol
        #self.scatter.setSymbol(symbol)
        self.updateItems(styleUpdate=True)

    def setSymbolPen(self, *args, **kargs):
        """ 
        Sets the :class:`QtGui.QPen` used to draw symbol outlines.
        See :func:`mkPen() <pyqtgraph.mkPen>`) for arguments.
        """
        pen = fn.mkPen(*args, **kargs)
        if self.opts['symbolPen'] == pen:
            return
        self.opts['symbolPen'] = pen
        #self.scatter.setSymbolPen(pen)
        self.updateItems(styleUpdate=True)

    def setSymbolBrush(self, *args, **kargs):
        """
        Sets the :class:`QtGui.QBrush` used to fill symbols.
        See :func:`mkBrush() <pyqtgraph.mkBrush>`) for arguments.
        """
        brush = fn.mkBrush(*args, **kargs)
        if self.opts['symbolBrush'] == brush:
            return
        self.opts['symbolBrush'] = brush
        #self.scatter.setSymbolBrush(brush)
        self.updateItems(styleUpdate=True)

    def setSymbolSize(self, size):
        """
        Sets the symbol size.
        """
        if self.opts['symbolSize'] == size:
            return
        self.opts['symbolSize'] = size
        #self.scatter.setSymbolSize(symbolSize)
        self.updateItems(styleUpdate=True)

    def setDownsampling(self, ds=None, auto=None, method=None):
        """
        Sets the downsampling mode of this item. Downsampling reduces the number
        of samples drawn to increase performance.

        ==============  =================================================================
        **Arguments:**
        ds              (int) Reduce visible plot samples by this factor. To disable,
                        set ds=1.
        auto            (bool) If True, automatically pick *ds* based on visible range
        mode            'subsample': Downsample by taking the first of N samples.
                        This method is fastest and least accurate.
                        'mean': Downsample by taking the mean of N samples.
                        'peak': Downsample by drawing a saw wave that follows the min
                        and max of the original data. This method produces the best
                        visual representation of the data but is slower.
        ==============  =================================================================
        """
        changed = False
        if ds is not None:
            if self.opts['downsample'] != ds:
                changed = True
                self.opts['downsample'] = ds

        if auto is not None and self.opts['autoDownsample'] != auto:
            self.opts['autoDownsample'] = auto
            changed = True

        if method is not None:
            if self.opts['downsampleMethod'] != method:
                changed = True
                self.opts['downsampleMethod'] = method

        if changed:
            self._datasetMapped  = None  # invalidata mapped data
            self._datasetDisplay = None  # invalidate display data
            self.updateItems(styleUpdate=False)

    def setClipToView(self, state):
        """
        ``state=True`` enables clipping the displayed data set to the
        visible x-axis range.
        """
        if self.opts['clipToView'] == state:
            return
        self.opts['clipToView'] = state
        self._datasetDisplay = None  # invalidate display data
        self.updateItems(styleUpdate=False)

    def setDynamicRangeLimit(self, limit=1e06, hysteresis=3.):
        """
        Limit the off-screen positions of data points at large magnification
        This avoids errors with plots not displaying because their visibility is incorrectly determined. 
        The default setting repositions far-off points to be within ±10^6 times the viewport height.

        =============== ================================================================
        **Arguments:**
        limit           (float or None) Any data outside the range of limit * hysteresis
                        will be constrained to the limit value limit.
                        All values are relative to the viewport height.
                        'None' disables the check for a minimal increase in performance.
                        Default is 1E+06.
                        
        hysteresis      (float) Hysteresis factor that controls how much change
                        in zoom level (vertical height) is allowed before recalculating
                        Default is 3.0
        =============== ================================================================
        """
        if hysteresis < 1.0: 
            hysteresis = 1.0
        self.opts['dynamicRangeHyst']  = hysteresis

        if limit == self.opts['dynamicRangeLimit']:
            return # avoid update if there is no change
        self.opts['dynamicRangeLimit'] = limit # can be None
        self._datasetDisplay = None  # invalidate display data

        self.updateItems(styleUpdate=False)
        
    def setSkipFiniteCheck(self, skipFiniteCheck):
        """
        When it is known that the plot data passed to ``PlotDataItem`` contains only finite numerical values,
        the ``skipFiniteCheck`` property can help speed up plotting. If this flag is set and the data contains 
        any non-finite values (such as `NaN` or `Inf`), unpredictable behavior will occur. The data might not
        be plotted, or there migth be significant performance impact.
        
        In the default 'auto' connect mode, ``PlotDataItem`` will apply this setting automatically.
        """
        self.opts['skipFiniteCheck']  = bool(skipFiniteCheck)

    def setData(self, *args, **kargs):
        """
        Clear any data displayed by this item and display new data.
        See :func:`__init__() <pyqtgraph.PlotDataItem.__init__>` for details; it accepts the same arguments.
        """
        #self.clear()
        if kargs.get("stepMode", None) is True:
            warnings.warn(
                'stepMode=True is deprecated and will result in an error after October 2022. Use stepMode="center" instead.',
                DeprecationWarning, stacklevel=3
            )
        if 'decimate' in kargs.keys():
            warnings.warn(
                'The decimate keyword has been deprecated. It has no effect and may result in an error in releases after October 2022. ',
                DeprecationWarning, stacklevel=2
            )
        
        if 'identical' in kargs.keys():
            warnings.warn(
                'The identical keyword has been deprecated. It has no effect may result in an error in releases after October 2022. ',
                DeprecationWarning, stacklevel=2
            )
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
                x = data[:,0]
                y = data[:,1]
            elif dt == 'recarray' or dt == 'dictOfLists':
                if 'x' in data:
                    x = np.array(data['x'])
                if 'y' in data:
                    y = np.array(data['y'])
            elif dt ==  'listOfDicts':
                if 'x' in data[0]:
                    x = np.array([d.get('x',None) for d in data])
                if 'y' in data[0]:
                    y = np.array([d.get('y',None) for d in data])
                for k in ['data', 'symbolSize', 'symbolPen', 'symbolBrush', 'symbolShape']:
                    if k in data:
                        kargs[k] = [d.get(k, None) for d in data]
            elif dt == 'MetaArray':
                y = data.view(np.ndarray)
                x = data.xvals(0).view(np.ndarray)
            else:
                raise Exception('Invalid data type %s' % type(data))

        elif len(args) == 2:
            seq = ('listOfValues', 'MetaArray', 'empty')
            dtyp = dataType(args[0]), dataType(args[1])
            if dtyp[0] not in seq or dtyp[1] not in seq:
                raise Exception('When passing two unnamed arguments, both must be a list or array of values. (got %s, %s)' % (str(type(args[0])), str(type(args[1]))))
            if not isinstance(args[0], np.ndarray):
                #x = np.array(args[0])
                if dtyp[0] == 'MetaArray':
                    x = args[0].asarray()
                else:
                    x = np.array(args[0])
            else:
                x = args[0].view(np.ndarray)
            if not isinstance(args[1], np.ndarray):
                #y = np.array(args[1])
                if dtyp[1] == 'MetaArray':
                    y = args[1].asarray()
                else:
                    y = np.array(args[1])
            else:
                y = args[1].view(np.ndarray)

        if 'x' in kargs:
            x = kargs['x']
            if dataType(x) == 'MetaArray':
                x = x.asarray()
        if 'y' in kargs:
            y = kargs['y']
            if dataType(y) == 'MetaArray':
                y = y.asarray()

        profiler('interpret data')
        ## pull in all style arguments.
        ## Use self.opts to fill in anything not present in kargs.

        if 'name' in kargs:
            self.opts['name'] = kargs['name']
            self.setProperty('styleWasChanged', True)

        if 'connect' in kargs:
            self.opts['connect'] = kargs['connect']
            self.setProperty('styleWasChanged', True)
            
        if 'skipFiniteCheck' in kargs:
            self.opts['skipFiniteCheck'] = kargs['skipFiniteCheck']

        ## if symbol pen/brush are given with no previously set symbol, then assume symbol is 'o'
        if 'symbol' not in kargs and ('symbolPen' in kargs or 'symbolBrush' in kargs or 'symbolSize' in kargs):
            if self.opts['symbol'] is None: 
                kargs['symbol'] = 'o'

        if 'brush' in kargs:
            kargs['fillBrush'] = kargs['brush']

        for k in list(self.opts.keys()):
            if k in kargs:
                self.opts[k] = kargs[k]
                self.setProperty('styleWasChanged', True)
        #curveArgs = {}
        #for k in ['pen', 'shadowPen', 'fillLevel', 'brush']:
            #if k in kargs:
                #self.opts[k] = kargs[k]
            #curveArgs[k] = self.opts[k]

        #scatterArgs = {}
        #for k,v in [('symbolPen','pen'), ('symbolBrush','brush'), ('symbol','symbol')]:
            #if k in kargs:
                #self.opts[k] = kargs[k]
            #scatterArgs[v] = self.opts[k]

        if y is None or len(y) == 0: # empty data is represented as None
            yData = None
        else: # actual data is represented by ndarray
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            yData = y.view(np.ndarray)
            if x is None:
                x = np.arange(len(y))
                
        if x is None or len(x)==0: # empty data is represented as None
            xData = None
        else: # actual data is represented by ndarray
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            xData = x.view(np.ndarray)  # one last check to make sure there are no MetaArrays getting by

        if xData is None or yData is None:
            self._dataset = None
        else:
            self._dataset = PlotDataset( xData, yData )
        self._datasetMapped  = None  # invalidata mapped data , will be generated in getData() / _getDisplayDataset()
        self._datasetDisplay = None  # invalidate display data, will be generated in getData() / _getDisplayDataset()

        profiler('set data')

        self.updateItems( styleUpdate = self.property('styleWasChanged') )
        self.setProperty('styleWasChanged', False) # items have been updated
        profiler('update items')

        self.informViewBoundsChanged()

        self.sigPlotChanged.emit(self)
        profiler('emit')

    def updateItems(self, styleUpdate=True):
        # override styleUpdate request and always enforce update until we have a better solution for
        # - ScatterPlotItem losing per-point style information
        # - PlotDataItem performing multiple unnecessary setData calls on initialization
        styleUpdate=True
        
        curveArgs = {}
        scatterArgs = {}

        if styleUpdate: # repeat style arguments only when changed
            for k, v in [
                ('pen','pen'),
                ('shadowPen','shadowPen'),
                ('fillLevel','fillLevel'),
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
                ('symbolPen','pen'),
                ('symbolBrush','brush'),
                ('symbol','symbol'),
                ('symbolSize', 'size'),
                ('data', 'data'),
                ('pxMode', 'pxMode'),
                ('antialias', 'antialias'),
                ('useCache', 'useCache')
            ]:
                if k in self.opts:
                    scatterArgs[v] = self.opts[k]

        dataset = self._getDisplayDataset()
        if dataset is None: # then we have nothing to show
            self.curve.hide()
            self.scatter.hide()
            return

        x = dataset.x
        y = dataset.y
        #scatterArgs['mask'] = self.dataMask
        if(
            self.opts['pen'] is not None 
            or (self.opts['fillBrush'] is not None and self.opts['fillLevel'] is not None)
            ): # draw if visible...
            # auto-switch to indicate non-finite values as interruptions in the curve:
            if isinstance(curveArgs['connect'], str) and curveArgs['connect'] == 'auto': # connect can also take a boolean array
                if dataset.containsNonfinite is None:
                    curveArgs['connect'] = 'all' # this is faster, but silently connects the curve across any non-finite values
                else:
                    if dataset.containsNonfinite:
                        curveArgs['connect'] = 'finite'
                    else:
                        curveArgs['connect'] = 'all' # all points can be connected, and no further check is needed.
                        curveArgs['skipFiniteCheck'] = True
            self.curve.setData(x=x, y=y, **curveArgs)
            self.curve.show()
        else: # ...hide if not.
            self.curve.hide()

        if self.opts['symbol'] is not None: # draw if visible...
            ## check against `True` too for backwards compatibility
            if self.opts.get('stepMode', False) in ("center", True):
                x = 0.5 * (x[:-1] + x[1:])                
            self.scatter.setData(x=x, y=y, **scatterArgs)
            self.scatter.show()
        else: # ...hide if not.
            self.scatter.hide()

    def getOriginalDataset(self):
            """
            Returns the original, unmapped data as the tuple (`xData`, `yData`).
            """
            dataset = self._dataset
            if dataset is None:
                return (None, None)
            return dataset.x, dataset.y

    def _getDisplayDataset(self):
        """
        Returns a :class:`~.PlotDataset` object that contains data suitable for display 
        (after mapping and data reduction) as ``dataset.x`` and ``dataset.y``.
        Intended for internal use.
        """
        if self._dataset is None:
            return None
        # Return cached processed dataset if available and still valid:
        if( self._datasetDisplay is not None and
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

            if self.opts['fftMode']:
                x,y = self._fourierTransform(x, y)
                # Ignore the first bin for fft data if we have a logx scale
                if self.opts['logMode'][0]:
                    x=x[1:]
                    y=y[1:]

            if self.opts['derivativeMode']:  # plot dV/dt
                y = np.diff(self._dataset.y)/np.diff(self._dataset.x)
                x = x[:-1]
            if self.opts['phasemapMode']:  # plot dV/dt vs V
                x = self._dataset.y[:-1]
                y = np.diff(self._dataset.y)/np.diff(self._dataset.x)

            dataset = PlotDataset(x,y)
            dataset.containsNonfinite = self._dataset.containsNonfinite
            
            if True in self.opts['logMode']:
                dataset.applyLogMapping( self.opts['logMode'] ) # Apply log scaling for x and/or y axis

            self._datasetMapped = dataset
        
        # apply processing that affects the on-screen display of data:
        x = self._datasetMapped.x
        y = self._datasetMapped.y
        containsNonfinite = self._datasetMapped.containsNonfinite

        view = self.getViewBox()
        if view is None:
            view_range = None
        else:
            view_range = view.viewRect() # this is always up-to-date
        if view_range is None:
            view_range = self.viewRect()

        ds = self.opts['downsample']
        if not isinstance(ds, int):
            ds = 1

        if self.opts['autoDownsample']:
            # this option presumes that x-values have uniform spacing

            finite_x = x[np.isfinite(x)]  # ignore infinite and nan values
            if view_range is not None and len(finite_x) > 1:
                dx = float(finite_x[-1]-finite_x[0]) / (len(finite_x)-1)
                if dx != 0.0:
                    width = self.getViewBox().width()
                    if width != 0.0:  # autoDownsampleFactor _should_ be > 1.0
                        ds_float = max(1.0, abs(view_range.width() / dx / (width * self.opts['autoDownsampleFactor'])))
                        if math.isfinite(ds_float):
                            ds = int(ds_float)
                    ## downsampling is expensive; delay until after clipping.

        if self.opts['clipToView']:
            if view is None or view.autoRangeEnabled()[0]:
                pass # no ViewBox to clip to, or view will autoscale to data range.
            else:
                # clip-to-view always presumes that x-values are in increasing order
                if view_range is not None and len(x) > 1:
                    # find first in-view value (left edge) and first out-of-view value (right edge)
                    # since we want the curve to go to the edge of the screen, we need to preserve
                    # one down-sampled point on the left and one of the right, so we extend the interval
                    x0 = np.searchsorted(x, view_range.left()) - ds
                    x0 = fn.clip_scalar(x0, 0, len(x)) # workaround
                    # x0 = np.clip(x0, 0, len(x))

                    x1 = np.searchsorted(x, view_range.right()) + ds
                    x1 = fn.clip_scalar(x1, x0, len(x))
                    # x1 = np.clip(x1, 0, len(x))
                    x = x[x0:x1]
                    y = y[x0:x1]

        if ds > 1:
            if self.opts['downsampleMethod'] == 'subsample':
                x = x[::ds]
                y = y[::ds]
            elif self.opts['downsampleMethod'] == 'mean':
                n = len(x) // ds
                stx = ds//2 # start of x-values; try to select a somewhat centered point
                x = x[stx:stx+n*ds:ds] 
                y = y[:n*ds].reshape(n,ds).mean(axis=1)
            elif self.opts['downsampleMethod'] == 'peak':
                n = len(x) // ds
                x1 = np.empty((n,2))
                stx = ds//2 # start of x-values; try to select a somewhat centered point
                x1[:] = x[stx:stx+n*ds:ds,np.newaxis]
                x = x1.reshape(n*2)
                y1 = np.empty((n,2))
                y2 = y[:n*ds].reshape((n, ds))
                y1[:,0] = y2.max(axis=1)
                y1[:,1] = y2.min(axis=1)
                y = y1.reshape(n*2)

        if self.opts['dynamicRangeLimit'] is not None:
            if view_range is not None:
                data_range = self._datasetMapped.dataRect()
                if data_range is not None:
                    view_height = view_range.height()
                    limit = self.opts['dynamicRangeLimit']
                    hyst  = self.opts['dynamicRangeHyst']
                    # never clip data if it fits into +/- (extended) limit * view height
                    if ( # note that "bottom" is the larger number, and "top" is the smaller one.
                        view_height > 0                                # never clip if the view does not show anything and would cause division by zero
                        and not data_range.bottom() < view_range.top() # never clip if all data is too small to see
                        and not data_range.top() > view_range.bottom() # never clip if all data is too large to see
                        and data_range.height() > 2 * hyst * limit * view_height
                    ):
                        cache_is_good = False
                        # check if cached display data can be reused:
                        if self._datasetDisplay is not None: # top is minimum value, bottom is maximum value
                            # how many multiples of the current view height does the clipped plot extend to the top and bottom?
                            top_exc =-(self._drlLastClip[0]-view_range.bottom()) / view_height
                            bot_exc = (self._drlLastClip[1]-view_range.top()   ) / view_height
                            # print(top_exc, bot_exc, hyst)
                            if (    top_exc >= limit / hyst and top_exc <= limit * hyst
                                and bot_exc >= limit / hyst and bot_exc <= limit * hyst ):
                                # restore cached values
                                x = self._datasetDisplay.x
                                y = self._datasetDisplay.y
                                cache_is_good = True
                        if not cache_is_good:
                            min_val = view_range.bottom() - limit * view_height
                            max_val = view_range.top()    + limit * view_height
                            # print('alloc:', end='')
                            # workaround for slowdown from numpy deprecation issues in 1.17 to 1.20+ :
                            # y = np.clip(y, a_min=min_val, a_max=max_val)
                            y = fn.clip_array(y, min_val, max_val)
                            self._drlLastClip = (min_val, max_val)
        self._datasetDisplay = PlotDataset( x,y )
        self._datasetDisplay.containsNonfinite = containsNonfinite
        self.setProperty('xViewRangeWasChanged', False)
        self.setProperty('yViewRangeWasChanged', False)

        return self._datasetDisplay

    def getData(self):
        """
        Returns the displayed data as the tuple (`xData`, `yData`) after mapping and data reduction.
        """
        dataset = self._getDisplayDataset()
        if dataset is None:
            return (None, None)
        return dataset.x, dataset.y

    # compatbility method for access to dataRect for full dataset:
    def dataRect(self):
        """
        Returns a bounding rectangle (as :class:`QtCore.QRectF`) for the full set of data.
        Will return `None` if there is no data or if all values (x or y) are NaN.
        """
        if self._dataset is None:
            return None
        return self._dataset.dataRect()

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        """
        Returns the range occupied by the data (along a specific axis) in this item.
        This method is called by :class:`ViewBox` when auto-scaling.

        =============== ====================================================================
        **Arguments:**
        ax              (0 or 1) the axis for which to return this item's data range
        frac            (float 0.0-1.0) Specifies what fraction of the total data
                        range to return. By default, the entire range is returned.
                        This allows the :class:`ViewBox` to ignore large spikes in the data
                        when auto-scaling.
        orthoRange      ([min,max] or None) Specifies that only the data within the
                        given range (orthogonal to *ax*) should me measured when
                        returning the data range. (For example, a ViewBox might ask
                        what is the y-range of all data with x-values between min
                        and max)
        =============== ====================================================================
        """
        range = [None, None]
        if self.curve.isVisible():
            range = self.curve.dataBounds(ax, frac, orthoRange)
        elif self.scatter.isVisible():
            r2 = self.scatter.dataBounds(ax, frac, orthoRange)
            range = [
                r2[0] if range[0] is None else (range[0] if r2[0] is None else min(r2[0], range[0])),
                r2[1] if range[1] is None else (range[1] if r2[1] is None else min(r2[1], range[1]))
                ]
        return range

    def pixelPadding(self):
        """
        Returns the size in pixels that this item may draw beyond the values returned by dataBounds().
        This method is called by :class:`ViewBox` when auto-scaling.
        """
        pad = 0
        if self.curve.isVisible():
            pad = max(pad, self.curve.pixelPadding())
        elif self.scatter.isVisible():
            pad = max(pad, self.scatter.pixelPadding())
        return pad

    def clear(self):
        self._dataset        = None
        self._datasetMapped  = None
        self._datasetDisplay = None
        self.curve.clear()
        self.scatter.clear()

    def appendData(self, *args, **kargs):
        pass

    def curveClicked(self, curve, ev):
        self.sigClicked.emit(self, ev)

    def scatterClicked(self, plt, points, ev):
        self.sigClicked.emit(self, ev)
        self.sigPointsClicked.emit(self, points, ev)

    def scatterHovered(self, plt, points, ev):
        self.sigPointsHovered.emit(self, points, ev)

    # def viewTransformChanged(self):
    #   """ view transform (and thus range) has changed, replot if needed """
    # viewTransformChanged is only called when the cached viewRect of GraphicsItem
    # has already been invalidated. However, responding here will make PlotDataItem
    # update curve and scatter later than intended.
    #   super().viewTransformChanged() # this invalidates the viewRect() cache!
        
    def viewRangeChanged(self, vb=None, ranges=None, changed=None):
        # view range has changed; re-plot if needed 
        update_needed = False
        if changed is None or changed[0]: 
            # if ranges is not None:
            #     print('hor:', ranges[0])
            self.setProperty('xViewRangeWasChanged', True)
            if( self.opts['clipToView']
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

    def _fourierTransform(self, x, y):
        ## Perform Fourier transform. If x values are not sampled uniformly,
        ## then use np.interp to resample before taking fft.
        dx = np.diff(x)
        uniform = not np.any(np.abs(dx-dx[0]) > (abs(dx[0]) / 1000.))
        if not uniform:
            x2 = np.linspace(x[0], x[-1], len(x))
            y = np.interp(x2, x, y)
            x = x2
        n = y.size
        f = np.fft.rfft(y) / n
        d = float(x[-1]-x[0]) / (len(x)-1)
        x = np.fft.rfftfreq(n, d)
        y = np.abs(f)
        return x, y
        
# helper functions:
def dataType(obj):
    if hasattr(obj, '__len__') and len(obj) == 0:
        return 'empty'
    if isinstance(obj, dict):
        return 'dictOfLists'
    elif isSequence(obj):
        first = obj[0]

        if (hasattr(obj, 'implements') and obj.implements('MetaArray')):
            return 'MetaArray'
        elif isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                if obj.dtype.names is None:
                    return 'listOfValues'
                else:
                    return 'recarray'
            elif obj.ndim == 2 and obj.dtype.names is None and obj.shape[1] == 2:
                return 'Nx2array'
            else:
                raise Exception('array shape must be (N,) or (N,2); got %s instead' % str(obj.shape))
        elif isinstance(first, dict):
            return 'listOfDicts'
        else:
            return 'listOfValues'


def isSequence(obj):
    return hasattr(obj, '__iter__') or isinstance(obj, np.ndarray) or (hasattr(obj, 'implements') and obj.implements('MetaArray'))
