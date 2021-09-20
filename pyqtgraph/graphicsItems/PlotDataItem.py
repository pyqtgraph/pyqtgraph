# -*- coding: utf-8 -*-
import warnings
import math
import numpy as np
from ..Qt import QtCore
from .GraphicsObject import GraphicsObject
from .PlotCurveItem import PlotCurveItem
from .ScatterPlotItem import ScatterPlotItem
from .. import functions as fn
from .. import debug as debug
from .. import getConfigOption

__all__ = ['PlotDataItem']

class PlotDataItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    GraphicsItem for displaying plot curves, scatter plots, or both.
    While it is possible to use :class:`PlotCurveItem <pyqtgraph.PlotCurveItem>` or
    :class:`ScatterPlotItem <pyqtgraph.ScatterPlotItem>` individually, this class
    provides a unified interface to both. Instances of :class:`PlotDataItem` are
    usually created by plot() methods such as :func:`pyqtgraph.plot` and
    :func:`PlotItem.plot() <pyqtgraph.PlotItem.plot>`.

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
    
    class PlotDataset(object):
        """ Holds collected information for a plotable dataset """
        def __init__(self, x, y):
            super().__init__()
            assert x is not None
            assert y is not None
            self.x = x
            self.y = y
            self._dataRect = None
            self.containsNonfinite = None
            
        # def copy(self):
        #     """ Provides a shallow copy of this PlotDataset """
        #     dataset = self.PlotDataSet( self.x, self.y )
        #     dataset._dataRect = self._dataRect
        #     dataset.containsNonfinite = self.containsNonfinite
        #     return dataset
            
        def _updateDataRect(self):
            """ Find bounds of plotable data """
            if self.y is None or self.x is None:
                return None
            selection = np.isfinite(self.y) # We are looking for the bounds of the plottable data set. Infinite and Nan are ignored. 
            all_y_are_finite = selection.all() # False if all values are finite, True if there are any non-finites
            try:
                ymin = np.min( self.y[selection] ) # find minimum of all finite values
                ymax = np.max( self.y[selection] ) # find maximum of all finite values
            except ValueError: # is raised then there are no finite values
                ymin = np.nan
                ymax = np.nan
                
            selection = np.isfinite(self.x) # We are looking for the bounds of the plottable data set. Infinite and Nan are ignored. 
            all_x_are_finite = selection.all() # False if all values are finite, True if there are any non-finites
            try:
                xmin = np.min( self.x[selection] ) # find minimum of all finite values
                xmax = np.max( self.x[selection] ) # find maximum of all finite values
            except ValueError: # is reaised then there are no finite values
                xmin = np.nan
                xmax = np.nan
                
            self._dataRect = QtCore.QRectF( QtCore.QPointF(xmin,ymin), QtCore.QPointF(xmax,ymax) )
            self.containsNonfinite = not (all_x_are_finite and all_y_are_finite) # This always yields a definite True/False answer

        def dataRect(self):
            """
            Returns a bounding rectangle (as QRectF) for the finite subset of data.
            If there is an active mapping function, such as logarithmic scaling, then bounds represent the mapped data. 
            Will return None if there is no data or if all values (x or y) are NaN.
            """
            if self._dataRect is None: 
                self._updateDataRect()
            return self._dataRect
            
        def applyLogMapping(self, logMode):
            """
            Applies log10 as a mapping transformation according to logmode.
            logmode is a two-element list of Booleans requesting log mode for x and y axis.
            """
            all_x_finite = False
            if logMode[0]:
                self.x = np.log10(self.x)
                nonfinites = ~np.isfinite( self.x )
                if nonfinites.any():
                    self.x[nonfinites] = np.nan # set all non-finite values to NaN
                    self.containsNonfinite = True
                else:
                    all_x_finite = True
            all_y_finite = False
            if logMode[1]:
                self.y = np.log10(self.y)
                nonfinites = ~np.isfinite( self.y )
                if nonfinites.any():
                    self.y[nonfinites] = np.nan # set all non-finite values to NaN
                    self.containsNonfinite = True
                else:
                    all_y_finite = True
            if all_x_finite and all_y_finite: 
                self.containsNonfinite = False # mark as False only if both axes were checked.

    def __init__(self, *args, **kargs):
        """
        There are many different ways to create a PlotDataItem:

        **Data initialization arguments:** (x,y data only)

            =================================== ======================================
            PlotDataItem(xValues, yValues)      x and y values may be any sequence
                                                (including ndarray) of real numbers
            PlotDataItem(yValues)               y values only -- x will be
                                                automatically set to range(len(y))
            PlotDataItem(x=xValues, y=yValues)  x and y given by keyword arguments
            PlotDataItem(ndarray(Nx2))          numpy array with shape (N, 2) where
                                                ``x=data[:,0]`` and ``y=data[:,1]``
            =================================== ======================================

        **Data initialization arguments:** (x,y data AND may include spot style)

        
            ============================ =========================================
            PlotDataItem(recarray)       numpy array with ``dtype=[('x', float),
                                         ('y', float), ...]``
            PlotDataItem(list-of-dicts)  ``[{'x': x, 'y': y, ...},   ...]``
            PlotDataItem(dict-of-lists)  ``{'x': [...], 'y': [...],  ...}``
            PlotDataItem(MetaArray)      1D array of Y values with X sepecified as
                                         axis values OR 2D array with a column 'y'
                                         and extra columns as needed.
            ============================ =========================================
        
        **Line style keyword arguments:**

            ============ ==============================================================================
            connect      Specifies how / whether vertexes should be connected. See
                         :func:`arrayToQPath() <pyqtgraph.arrayToQPath>`
            pen          Pen to use for drawing line between points.
                         Default is solid grey, 1px width. Use None to disable line drawing.
                         May be any single argument accepted by :func:`mkPen() <pyqtgraph.mkPen>`
            shadowPen    Pen for secondary line to draw behind the primary line. disabled by default.
                         May be any single argument accepted by :func:`mkPen() <pyqtgraph.mkPen>`
            fillLevel    Fill the area between the curve and fillLevel

            fillOutline  (bool) If True, an outline surrounding the *fillLevel* area is drawn.
            fillBrush    Fill to use when fillLevel is specified.
                         May be any single argument accepted by :func:`mkBrush() <pyqtgraph.mkBrush>`
            stepMode     (str or None) If "center", a step is drawn using the x
                         values as boundaries and the given y values are
                         associated to the mid-points between the boundaries of
                         each step. This is commonly used when drawing
                         histograms. Note that in this case, len(x) == len(y) + 1
                         If "left" or "right", the step is drawn assuming that
                         the y value is associated to the left or right boundary,
                         respectively. In this case len(x) == len(y)
                         If not passed or an empty string or None is passed, the
                         step mode is not enabled.
                         Passing True is a deprecated equivalent to "center".
                         (added in version 0.9.9)

            ============ ==============================================================================
        
        **Point style keyword arguments:**  (see :func:`ScatterPlotItem.setData() <pyqtgraph.ScatterPlotItem.setData>` for more information)

            ============   =====================================================
            symbol         Symbol to use for drawing points OR list of symbols,
                           one per point. Default is no symbol.
                           Options are o, s, t, d, +, or any QPainterPath
            symbolPen      Outline pen for drawing points OR list of pens, one
                           per point. May be any single argument accepted by
                           :func:`mkPen() <pyqtgraph.mkPen>`
            symbolBrush    Brush for filling points OR list of brushes, one per
                           point. May be any single argument accepted by
                           :func:`mkBrush() <pyqtgraph.mkBrush>`
            symbolSize     Diameter of symbols OR list of diameters.
            pxMode         (bool) If True, then symbolSize is specified in
                           pixels. If False, then symbolSize is
                           specified in data coordinates.
            ============   =====================================================

        **Optimization keyword arguments:**

            ================= =====================================================================
            antialias         (bool) By default, antialiasing is disabled to improve performance.
                              Note that in some cases (in particluar, when pxMode=True), points
                              will be rendered antialiased even if this is set to False.
            decimate          deprecated.
            downsample        (int) Reduce the number of samples displayed by this value
            downsampleMethod  'subsample': Downsample by taking the first of N samples.
                              This method is fastest and least accurate.
                              'mean': Downsample by taking the mean of N samples.
                              'peak': Downsample by drawing a saw wave that follows the min
                              and max of the original data. This method produces the best
                              visual representation of the data but is slower.
            autoDownsample    (bool) If True, resample the data before plotting to avoid plotting
                              multiple line segments per pixel. This can improve performance when
                              viewing very high-density data, but increases the initial overhead
                              and memory usage.
            clipToView        (bool) If True, only plot data that is visible within the X range of
                              the containing ViewBox. This can improve performance when plotting
                              very large data sets where only a fraction of the data is visible
                              at any time.
            dynamicRangeLimit (float or None) Limit off-screen positions of data points at large
                              magnification to avoids display errors. Disabled if None.
            skipFiniteCheck   (bool) Optimization parameter that can speed up plot time by
                              telling the painter to not check and compensate for NaN
                              values.  If set to True, and NaN values exist, the data
                              may not be displayed or your plot will take a
                              significant performance hit.  Defaults to False.
            identical         *deprecated*
            ================= =====================================================================

        **Meta-info keyword arguments:**

            ==========   ================================================
            name         name of dataset. This would appear in a legend
            ==========   ================================================
        """
        GraphicsObject.__init__(self)
        self.setFlag(self.GraphicsItemFlag.ItemHasNoContents)
        # self.xData = None # original data
        # self.yData = None
        self.dataset       = None # will hold a PlotDataset for the original data
        self.datasetMapped = None # will hold a PlotDataset for data after mapping transforms (e.g. log scale)
        self.datasetDisp   = None # will hold a PlotDataset for data downsampled and limited for display
        # self.xMapped = None # data after applying a possible mapping, e.g. log-scale.
        # self.yMapped = None
        # self.xDisp = None # data in the form to be displayed
        # self.yDisp = None
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

        # self._dataRect = None
        # self._dataRectMapped = None
        self._drlLastClip = (0.0, 0.0) # holds last clipping points of dynamic range limiter
        #self.clear()
        self.opts = {
            'connect': 'default', # defaults to 'all', unless overridden to 'finite' for log-scaling

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

            'downsample': 1,
            'autoDownsample': False,
            'downsampleMethod': 'peak',
            'autoDownsampleFactor': 5.,  # draw ~5 samples per pixel
            'clipToView': False,
            'dynamicRangeLimit': 1e6,
            'dynamicRangeHyst': 3.0,
            'skipFiniteCheck': False,
            'data': None,
        }
        self.setCurveClickable(kargs.get('clickable', False))
        self.setData(*args, **kargs)
    
    # Compatibility with direct property access to previous xData and yData structures:
    @property
    def xData(self):
        if self.dataset is None: return None
        return self.dataset.x
        
    @property
    def yData(self):
        if self.dataset is None: return None
        return self.dataset.y


    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        return self.opts.get('name', None)

    def setCurveClickable(self, s, width=None):
        self.curve.setClickable(s, width)

    def curveClickable(self):
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

    def setFftMode(self, mode):
        if self.opts['fftMode'] == mode:
            return
        self.opts['fftMode'] = mode
        self.datasetMapped = None
        self.datasetDisp   = None
        # self.xDisp   = self.yDisp   = None
        # self.xMapped = self.yMapped = None
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setLogMode(self, xMode, yMode):
        """
        To enable log scaling for y<0 and y>0, the following formula is used:
        
            scaled = sign(y) * log10(abs(y) + eps)

        where eps is the smallest unit of y.dtype.
        This allows for handling of 0. values, scaling of large values,
        as well as the typical log scaling of values in the range -1 < x < 1.
        Note that for values within this range, the signs are inverted.
        """
        if self.opts['logMode'] == [xMode, yMode]:
            return
        self.opts['logMode'] = [xMode, yMode]
        # self.xMapped = self.yMapped = None # invalidata mapped data
        # self.xDisp   = self.yDisp   = None # invalidate display data
        self.datasetMapped = None  # invalidata mapped data
        self.datasetDisp   = None  # invalidate display data
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()


    def setDerivativeMode(self, mode):
        if self.opts['derivativeMode'] == mode:
            return
        self.opts['derivativeMode'] = mode
        # self.xDisp = self.yDisp = None
        self.datasetMapped = None  # invalidata mapped data
        self.datasetDisp   = None  # invalidate display data
        
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setPhasemapMode(self, mode):
        if self.opts['phasemapMode'] == mode:
            return
        self.opts['phasemapMode'] = mode
        # self.xDisp = self.yDisp = None
        self.datasetMapped = None  # invalidata mapped data
        self.datasetDisp   = None  # invalidate display data
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setPointMode(self, mode):
        if self.opts['pointMode'] == mode:
            return
        self.opts['pointMode'] = mode
        self.update()

    def setPen(self, *args, **kargs):
        """
        | Sets the pen used to draw lines between points.
        | *pen* can be a QPen or any argument accepted by :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>`
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
        | Sets the shadow pen used to draw lines between points (this is for enhancing contrast or
          emphasizing data).
        | This line is drawn behind the primary pen (see :func:`setPen() <pyqtgraph.PlotDataItem.setPen>`)
          and should generally be assigned greater width than the primary pen.
        | *pen* can be a QPen or any argument accepted by :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>`
        """
        pen = fn.mkPen(*args, **kargs)
        self.opts['shadowPen'] = pen
        #for c in self.curves:
            #c.setPen(pen)
        #self.update()
        self.updateItems(styleUpdate=True)

    def setFillBrush(self, *args, **kargs):
        brush = fn.mkBrush(*args, **kargs)
        if self.opts['fillBrush'] == brush:
            return
        self.opts['fillBrush'] = brush
        self.updateItems(styleUpdate=True)

    def setBrush(self, *args, **kargs):
        return self.setFillBrush(*args, **kargs)

    def setFillLevel(self, level):
        if self.opts['fillLevel'] == level:
            return
        self.opts['fillLevel'] = level
        self.updateItems(styleUpdate=True)

    def setSymbol(self, symbol):
        if self.opts['symbol'] == symbol:
            return
        self.opts['symbol'] = symbol
        #self.scatter.setSymbol(symbol)
        self.updateItems(styleUpdate=True)

    def setSymbolPen(self, *args, **kargs):
        pen = fn.mkPen(*args, **kargs)
        if self.opts['symbolPen'] == pen:
            return
        self.opts['symbolPen'] = pen
        #self.scatter.setSymbolPen(pen)
        self.updateItems(styleUpdate=True)

    def setSymbolBrush(self, *args, **kargs):
        brush = fn.mkBrush(*args, **kargs)
        if self.opts['symbolBrush'] == brush:
            return
        self.opts['symbolBrush'] = brush
        #self.scatter.setSymbolBrush(brush)
        self.updateItems(styleUpdate=True)


    def setSymbolSize(self, size):
        if self.opts['symbolSize'] == size:
            return
        self.opts['symbolSize'] = size
        #self.scatter.setSymbolSize(symbolSize)
        self.updateItems(styleUpdate=True)

    def setDownsampling(self, ds=None, auto=None, method=None):
        """
        Set the downsampling mode of this item. Downsampling reduces the number
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
            # self.xDisp = self.yDisp = None
            self.datasetMapped = None  # invalidata mapped data
            self.datasetDisp   = None  # invalidate display data
            self.updateItems(styleUpdate=False)

    def setClipToView(self, clip):
        if self.opts['clipToView'] == clip:
            return
        self.opts['clipToView'] = clip
        # self.xDisp = self.yDisp = None
        self.datasetDisp   = None  # invalidate display data
        self.updateItems(styleUpdate=False)

    def setDynamicRangeLimit(self, limit=1e06, hysteresis=3.):
        """
        Limit the off-screen positions of data points at large magnification
        This avoids errors with plots not displaying because their visibility is incorrectly determined. The default setting repositions far-off points to be within +-1E+06 times the viewport height.

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
        # self.xDisp = self.yDisp = None
        self.datasetDisp   = None  # invalidate display data

        self.updateItems(styleUpdate=False)

    def setData(self, *args, **kargs):
        """
        Clear any data displayed by this item and display new data.
        See :func:`__init__() <pyqtgraph.PlotDataItem.__init__>` for details; it accepts the same arguments.
        """
        #self.clear()
        if kargs.get("stepMode", None) is True:
            warnings.warn(
                'stepMode=True is deprecated, use stepMode="center" instead',
                DeprecationWarning, stacklevel=3
            )
        if 'decimate' in kargs.keys():
            warnings.warn(
                'decimate kwarg has been deprecated, it has no effect',
                DeprecationWarning, stacklevel=2
            )
        
        if 'identical' in kargs.keys():
            warnings.warn(
                'identical kwarg has been deprecated, it has no effect',
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
            self.dataset = None
        else:
            self.dataset = self.PlotDataset( xData, yData )
        self.datasetMapped = None  # invalidata mapped data, will be generated in setData()
        self.datasetDisp   = None  # invalidate display data, will be generated in setData()

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
                ('antialias', 'antialias')
            ]:
                if k in self.opts:
                    scatterArgs[v] = self.opts[k]

        dataset = self.getDisplayDataset()
        if dataset is None: # then we have nothing to show
            self.curve.hide()
            self.scatter.hide()
            return

        x = dataset.x
        y = dataset.y
        # print('xdata:', x)
        # print('ydata:', y)

        #scatterArgs['mask'] = self.dataMask
        if(
            self.opts['pen'] is not None 
            or (self.opts['fillBrush'] is not None and self.opts['fillLevel'] is not None)
            ): # draw if visible...
            # print('connect is', curveArgs['connect'], 'expect nonfinites:', dataset.containsNonfinite)
            if curveArgs['connect'] == 'default':
                if dataset.containsNonfinite:
                    curveArgs['connect'] = 'finite' # auto-switch to indicate non-finite values as interruptions in the curve
                else:
                    curveArgs['connect'] = 'all' # this is faster, but silently connects the curve across any non-finite values
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

    # def _applyLogMapping(self, x, y):
    #     ### Updates the mapped (e.g. log-scale) data if needed ###
    #     ### Mapped data is filtered to avoid infinities in the data set
    #     ### x and y data is combined into a PlotDataset that can also contain metainformation
    #     dataset = self.PlotDataset(x, y)        
    #     all_x_finite = False
    #     if self.opts['logMode'][0]:
    #         with warnings.catch_warnings(record=True) as w:
    #             dataset.x = np.log10(dataset.x)
    #             if len(w) == 0: # Did numpy tell us that there are non-finite values?
    #                 all_x_finite = True
    #             else:
    #                 dataset.containsNonfinite = True
    #                 nonfinites = ~np.isfinite( dataset.x )
    #                 dataset.x[nonfinites] = np.nan # set all non-finite values to NaN
    #     all_y_finite = False
    #     if self.opts['logMode'][1]:
    #         with warnings.catch_warnings(record=True) as w:
    #             dataset.y = np.log10(dataset.y)
    #             if len(w) == 0: # Did numpy tell us that there are non-finite values?
    #                 all_y_finite = True
    #             else:
    #                 dataset.containsNonfinite = True
    #                 nonfinites = ~np.isfinite( self.yMapped )
    #                 dataset.y[nonfinites] = np.nan # set all non-finite values to NaN
    #     if all_x_finite and all_y_finite: 
    #         dataset.containsNonfinite = False # mark as False only if both axes were checked.
    #     return dataset

    def getDisplayDataset(self):
        """
        Returns a `dataset` object that contains the displayed data as `dataset.x` and `dataset.y`, after mapping and data reduction. 
        Intended for internal use.
        """

        if self.dataset is None:
            return None
            
        # Return cached processed dataset if available and still valid:
        if( self.datasetDisp is not None and
            not (self.property('xViewRangeWasChanged') and self.opts['clipToView']) and
            not (self.property('xViewRangeWasChanged') and self.opts['autoDownsample']) and
            not (self.property('yViewRangeWasChanged') and self.opts['dynamicRangeLimit'] is not None)
        ):
            return self.datasetDisp
        
        # Apply data mapping functions if mapped dataset is not yet available: 
        if self.datasetMapped is None:
            # print( self.dataset, self.dataset.y )
            x = self.dataset.x
            y = self.dataset.y
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
                y = np.diff(self.dataset.y)/np.diff(self.dataset.x)
                x = x[:-1]
            if self.opts['phasemapMode']:  # plot dV/dt vs V
                x = self.dataset.y[:-1]
                y = np.diff(self.dataset.y)/np.diff(self.dataset.x)
            
            dataset = self.PlotDataset(x,y)
            dataset.containsNonfinite = self.dataset.containsNonfinite
            
            if True in self.opts['logMode']:
                dataset.applyLogMapping( self.opts['logMode'] ) # Apply log scaling for x and/or y axis

            self.datasetMapped = dataset
        
        # apply processing that affects the on-screen display of data:
        x = self.datasetMapped.x
        y = self.datasetMapped.y
        containsNonfinite = self.datasetMapped.containsNonfinite

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
            if view_range is not None and len(x) > 1:
                dx = float(x[-1]-x[0]) / (len(x)-1)
                if dx != 0.0:
                    x0 = (view_range.left()-x[0]) / dx
                    x1 = (view_range.right()-x[0]) / dx
                    width = self.getViewBox().width()
                    if width != 0.0:
                        ds = int(max(1, int((x1-x0) / (width*self.opts['autoDownsampleFactor']))))
                    ## downsampling is expensive; delay until after clipping.

        if self.opts['clipToView']:
            if view is None or view.autoRangeEnabled()[0]:
                pass # no ViewBox to clip to, or view will autoscale to data range.
            else:
                # clip-to-view always presumes that x-values are in increasing order
                if view_range is not None and len(x) > 1:
                    # print('search:', view_range.left(),'-',view_range.right() )
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
                # x = x[:n*ds:ds]
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
            # print('mapped:', self.yMapped )
            if view_range is not None:
                data_range = self.datasetMapped.dataRect()
                if data_range is not None:
                    # print('drange:', data_range)
                    # print('vrange:', view_range)
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
                        if self.datasetDisp is not None: # top is minimum value, bottom is maximum value
                            # how many multiples of the current view height does the clipped plot extend to the top and bottom?
                            top_exc =-(self._drlLastClip[0]-view_range.bottom()) / view_height
                            bot_exc = (self._drlLastClip[1]-view_range.top()   ) / view_height
                            # print(top_exc, bot_exc, hyst)
                            if (    top_exc >= limit / hyst and top_exc <= limit * hyst
                                and bot_exc >= limit / hyst and bot_exc <= limit * hyst ):
                                # restore cached values
                                x = self.datasetDisp.x
                                y = self.datasetDisp.y
                                cache_is_good = True
                        if not cache_is_good:
                            min_val = view_range.bottom() - limit * view_height
                            max_val = view_range.top()    + limit * view_height
                            # print('alloc:', end='')
                            # workaround for slowdown from numpy deprecation issues in 1.17 to 1.20+ :
                            # y = np.clip(y, a_min=min_val, a_max=max_val)
                            y = fn.clip_array(y, min_val, max_val)
                            self._drlLastClip = (min_val, max_val)
                            # print('{:.1e}<->{:.1e}'.format( min_val, max_val ))
        self.datasetDisp = self.PlotDataset( x,y )
        self.datasetDisp.containsNonfinite = containsNonfinite
        self.setProperty('xViewRangeWasChanged', False)
        self.setProperty('yViewRangeWasChanged', False)

        # print('Data x/y   :', self.dataset.x,'/',self.dataset.y )
        # print('Mapped x/y :', self.datasetMapped.x,'/', self.datasetMapped.y)
        # print('Display x/y:', self.datasetDisp.x,'/', self.datasetDisp.y)
        return self.datasetDisp

    def getData(self):
        """
        Legacy method that returns the displayed data as the tuple (`xData`, `yData`) after mapping and data reduction. 
        Intended for internal use.
        """
        dataset = self.getDisplayDataset()
        if dataset is None:
            return (None, None)
        return dataset.x, dataset.y
    
    # def _prepare_dataRect(self, x, y):
    #     ### find bounds of plotable data ###
    #     if y is None or x is None:
    #         return None
    #     selection = np.isfinite(y) # We are looking for the bounds of the plottable data set. Infinite and Nan are ignored. 
    #     all_y_are_finite = selection.all() # False if all values are finite, True if there are any non-finites
    #     try:
    #         ymin = np.min( y[selection] ) # find minimum of all finite values
    #         ymax = np.max( y[selection] ) # find maximum of all finite values
    #     except ValueError: # is raised then there are no finite values
    #         ymin = np.nan
    #         ymax = np.nan
            
    #     selection = np.isfinite(x) # We are looking for the bounds of the plottable data set. Infinite and Nan are ignored. 
    #     all_x_are_finite = selection.all() # False if all values are finite, True if there are any non-finites
    #     try:
    #         xmin = np.min( x[selection] ) # find minimum of all finite values
    #         xmax = np.max( x[selection] ) # find maximum of all finite values
    #     except ValueError: # is reaised then there are no finite values
    #         xmin = np.nan
    #         xmax = np.nan
            
    #     rect = QtCore.QRectF( QtCore.QPointF(xmin,ymin), QtCore.QPointF(xmax,ymax) )
    #     containsNonfinite = not (all_x_are_finite and all_y_are_finite)
    #     return rect, containsNonfinite

    # def _update_dataRectMapped(self):
    #     ### Determines bounds of the plotable mapped (e.g. log-scale) dataset represented by x and y values ###
    #     # At this time, dataRectMapped is only used in getData, which ensures xMapped and yMapped are updated in time.
    #     self._dataRectMapped = self._prepare_dataRect(self.xMapped, self.yMapped)

    # def _update_dataRect(self):
    #     ### Determines bounds of the original plotable dataset represented by x and y values ###
    #     self._dataRect = self._prepare_dataRect(self.xData, self.yData)


    # def dataRectMapped(self):
    #     """
    #     Returns a bounding rectangle (as QRectF) for the plottable data.
    #     If there is an active mapping function, such as logarithmic scaling, then bounds represent the mapped data. 
    #     Will return None if there is no data or if all values (x or y) are NaN.
    #     """
    #     if not (self.opts['logMode'][0] or self.opts['logMode'][1]): # no log scaling is active.
    #         return self.dataRect() # return bounds of unmmapped data
    #     if self._dataRectMapped is None:
    #         self._dataRectMapped = self._prepare_dataRect(self.xMapped, self.yMapped)
    #     return self._dataRectMapped

    # compatbility method for access to dataRect for full dataset:
    def dataRect(self):
        """
        Returns a bounding rectangle (as QRectF) for the full set of data.
        Will return None if there is no data or if all values (x or y) are NaN.
        """
        if self.dataset is None:
            return None
        return self.dataset.dataRect()
        # if self._dataRect is None:
        #     self._dataRect = self._prepare_dataRect(self.xData, self.yData)
        # return self._dataRect
        
        # if self._dataRect is not None:
        #     return self._dataRect
        # if self.xData is None or self.yData is None:
        #     return None
        # # This is now handled when adding data:
        # # if len(self.xData) == 0: # avoid crash if empty data is represented by [] instead of None
        # #     return None
        # with warnings.catch_warnings(): 
        #     # All-NaN data is handled by returning None; Explicit numpy warning is not needed.
        #     warnings.simplefilter("ignore")
        #     ymin = np.nanmin(self.yData)
        #     if math.isnan( ymin ):
        #         return None # most likely case for all-NaN data
        #     xmin = np.nanmin(self.xData)
        #     if math.isnan( xmin ):
        #         return None # less likely case for all-NaN data
        #     ymax = np.nanmax(self.yData)
        #     xmax = np.nanmax(self.xData)
        # self._dataRect = QtCore.QRectF(
        #     QtCore.QPointF(xmin,ymin),
        #     QtCore.QPointF(xmax,ymax) )
        # return self._dataRect

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        """
        Returns the range occupied by the data (along a specific axis) in this item.
        This method is called by ViewBox when auto-scaling.

        =============== =============================================================
        **Arguments:**
        ax              (0 or 1) the axis for which to return this item's data range
        frac            (float 0.0-1.0) Specifies what fraction of the total data
                        range to return. By default, the entire range is returned.
                        This allows the ViewBox to ignore large spikes in the data
                        when auto-scaling.
        orthoRange      ([min,max] or None) Specifies that only the data within the
                        given range (orthogonal to *ax*) should me measured when
                        returning the data range. (For example, a ViewBox might ask
                        what is the y-range of all data with x-values between min
                        and max)
        =============== =============================================================
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
        Return the size in pixels that this item may draw beyond the values returned by dataBounds().
        This method is called by ViewBox when auto-scaling.
        """
        pad = 0
        if self.curve.isVisible():
            pad = max(pad, self.curve.pixelPadding())
        elif self.scatter.isVisible():
            pad = max(pad, self.scatter.pixelPadding())
        return pad


    def clear(self):
        self.dataset       = None
        self.datasetMapped = None
        self.datasetDisp   = None
        # self.xData = None
        # self.yData = None
        # self.xMapped = None
        # self.yMapped = None
        # self.xDisp = None
        # self.yDisp = None
        # self._dataRect = None
        # self._dataRectMapped = None
        # self._containsNonfinite = None
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
        """ view range has changed; re-plot if needed """
        update_needed = False
        if changed is None or changed[0]: 
            # if ranges is not None:
            #     print('hor:', ranges[0])
            self.setProperty('xViewRangeWasChanged', True)
            if( self.opts['clipToView']
                or self.opts['autoDownsample']
            ):
                self.datasetDisp = None
                # self.xDisp = self.yDisp = None # only the display data needs to be invalidated, not the mapped data
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



#class TableData:
    #"""
    #Class for presenting multiple forms of tabular data through a consistent interface.
    #May contain:
        #- numpy record array
        #- list-of-dicts (all dicts are _not_ required to have the same keys)
        #- dict-of-lists
        #- dict (single record)
               #Note: if all the values in this record are lists, it will be interpreted as multiple records

    #Data can be accessed and modified by column, by row, or by value
        #data[columnName]
        #data[rowId]
        #data[columnName, rowId] = value
        #data[columnName] = [value, value, ...]
        #data[rowId] = {columnName: value, ...}
    #"""

    #def __init__(self, data):
        #self.data = data
        #if isinstance(data, np.ndarray):
            #self.mode = 'array'
        #elif isinstance(data, list):
            #self.mode = 'list'
        #elif isinstance(data, dict):
            #types = set(map(type, data.values()))
            ### dict may be a dict-of-lists or a single record
            #types -= set([list, np.ndarray]) ## if dict contains any non-sequence values, it is probably a single record.
            #if len(types) != 0:
                #self.data = [self.data]
                #self.mode = 'list'
            #else:
                #self.mode = 'dict'
        #elif isinstance(data, TableData):
            #self.data = data.data
            #self.mode = data.mode
        #else:
            #raise TypeError(type(data))

        #for fn in ['__getitem__', '__setitem__']:
            #setattr(self, fn, getattr(self, '_TableData'+fn+self.mode))

    #def originalData(self):
        #return self.data

    #def toArray(self):
        #if self.mode == 'array':
            #return self.data
        #if len(self) < 1:
            ##return np.array([])  ## need to return empty array *with correct columns*, but this is very difficult, so just return None
            #return None
        #rec1 = self[0]
        #dtype = functions.suggestRecordDType(rec1)
        ##print rec1, dtype
        #arr = np.empty(len(self), dtype=dtype)
        #arr[0] = tuple(rec1.values())
        #for i in range(1, len(self)):
            #arr[i] = tuple(self[i].values())
        #return arr

    #def __getitem__array(self, arg):
        #if isinstance(arg, tuple):
            #return self.data[arg[0]][arg[1]]
        #else:
            #return self.data[arg]

    #def __getitem__list(self, arg):
        #if isinstance(arg, str):
            #return [d.get(arg, None) for d in self.data]
        #elif isinstance(arg, int):
            #return self.data[arg]
        #elif isinstance(arg, tuple):
            #arg = self._orderArgs(arg)
            #return self.data[arg[0]][arg[1]]
        #else:
            #raise TypeError(type(arg))

    #def __getitem__dict(self, arg):
        #if isinstance(arg, str):
            #return self.data[arg]
        #elif isinstance(arg, int):
            #return dict([(k, v[arg]) for k, v in self.data.items()])
        #elif isinstance(arg, tuple):
            #arg = self._orderArgs(arg)
            #return self.data[arg[1]][arg[0]]
        #else:
            #raise TypeError(type(arg))

    #def __setitem__array(self, arg, val):
        #if isinstance(arg, tuple):
            #self.data[arg[0]][arg[1]] = val
        #else:
            #self.data[arg] = val

    #def __setitem__list(self, arg, val):
        #if isinstance(arg, str):
            #if len(val) != len(self.data):
                #raise Exception("Values (%d) and data set (%d) are not the same length." % (len(val), len(self.data)))
            #for i, rec in enumerate(self.data):
                #rec[arg] = val[i]
        #elif isinstance(arg, int):
            #self.data[arg] = val
        #elif isinstance(arg, tuple):
            #arg = self._orderArgs(arg)
            #self.data[arg[0]][arg[1]] = val
        #else:
            #raise TypeError(type(arg))

    #def __setitem__dict(self, arg, val):
        #if isinstance(arg, str):
            #if len(val) != len(self.data[arg]):
                #raise Exception("Values (%d) and data set (%d) are not the same length." % (len(val), len(self.data[arg])))
            #self.data[arg] = val
        #elif isinstance(arg, int):
            #for k in self.data:
                #self.data[k][arg] = val[k]
        #elif isinstance(arg, tuple):
            #arg = self._orderArgs(arg)
            #self.data[arg[1]][arg[0]] = val
        #else:
            #raise TypeError(type(arg))

    #def _orderArgs(self, args):
        ### return args in (int, str) order
        #if isinstance(args[0], str):
            #return (args[1], args[0])
        #else:
            #return args

    #def __iter__(self):
        #for i in range(len(self)):
            #yield self[i]

    #def __len__(self):
        #if self.mode == 'array' or self.mode == 'list':
            #return len(self.data)
        #else:
            #return max(map(len, self.data.values()))

    #def columnNames(self):
        #"""returns column names in no particular order"""
        #if self.mode == 'array':
            #return self.data.dtype.names
        #elif self.mode == 'list':
            #names = set()
            #for row in self.data:
                #names.update(row.keys())
            #return list(names)
        #elif self.mode == 'dict':
            #return self.data.keys()

    #def keys(self):
        #return self.columnNames()
