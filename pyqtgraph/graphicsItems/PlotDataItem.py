import pyqtgraph.metaarray as metaarray
from pyqtgraph.Qt import QtCore
from .GraphicsObject import GraphicsObject
from .PlotCurveItem import PlotCurveItem
from .ScatterPlotItem import ScatterPlotItem
import numpy as np
import scipy
import pyqtgraph.functions as fn
import pyqtgraph.debug as debug
import pyqtgraph as pg

class PlotDataItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    
    GraphicsItem for displaying plot curves, scatter plots, or both. 
    While it is possible to use :class:`PlotCurveItem <pyqtgraph.PlotCurveItem>` or
    :class:`ScatterPlotItem <pyqtgraph.ScatterPlotItem>` individually, this class
    provides a unified interface to both. Inspances of :class:`PlotDataItem` are 
    usually created by plot() methods such as :func:`pyqtgraph.plot` and
    :func:`PlotItem.plot() <pyqtgraph.PlotItem.plot>`.
    
    ============================== ==============================================
    **Signals:**
    sigPlotChanged(self)           Emitted when the data in this item is updated.  
    sigClicked(self)               Emitted when the item is clicked.
    sigPointsClicked(self, points) Emitted when a plot point is clicked
                                   Sends the list of points under the mouse.
    ============================== ==============================================
    """
    
    sigPlotChanged = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object)
    sigPointsClicked = QtCore.Signal(object, object)
    
    def __init__(self, *args, **kargs):
        """
        There are many different ways to create a PlotDataItem:
        
        **Data initialization arguments:** (x,y data only)
        
            =================================== ======================================
            PlotDataItem(xValues, yValues)      x and y values may be any sequence (including ndarray) of real numbers
            PlotDataItem(yValues)               y values only -- x will be automatically set to range(len(y))
            PlotDataItem(x=xValues, y=yValues)  x and y given by keyword arguments
            PlotDataItem(ndarray(Nx2))          numpy array with shape (N, 2) where x=data[:,0] and y=data[:,1]
            =================================== ======================================
        
        **Data initialization arguments:** (x,y data AND may include spot style)
        
            ===========================   =========================================
            PlotDataItem(recarray)        numpy array with dtype=[('x', float), ('y', float), ...]
            PlotDataItem(list-of-dicts)   [{'x': x, 'y': y, ...},   ...] 
            PlotDataItem(dict-of-lists)   {'x': [...], 'y': [...],  ...}           
            PlotDataItem(MetaArray)       1D array of Y values with X sepecified as axis values 
                                          OR 2D array with a column 'y' and extra columns as needed.
            ===========================   =========================================
        
        **Line style keyword arguments:**
            ==========   ================================================
            pen          Pen to use for drawing line between points. 
                         Default is solid grey, 1px width. Use None to disable line drawing.
                         May be any single argument accepted by :func:`mkPen() <pyqtgraph.mkPen>`
            shadowPen    Pen for secondary line to draw behind the primary line. disabled by default.
                         May be any single argument accepted by :func:`mkPen() <pyqtgraph.mkPen>`
            fillLevel    Fill the area between the curve and fillLevel
            fillBrush    Fill to use when fillLevel is specified. 
                         May be any single argument accepted by :func:`mkBrush() <pyqtgraph.mkBrush>`
            ==========   ================================================
        
        **Point style keyword arguments:**  (see :func:`ScatterPlotItem.setData() <pyqtgraph.ScatterPlotItem.setData>` for more information)
        
            ============   ================================================
            symbol         Symbol to use for drawing points OR list of symbols, one per point. Default is no symbol.
                           Options are o, s, t, d, +, or any QPainterPath
            symbolPen      Outline pen for drawing points OR list of pens, one per point.
                           May be any single argument accepted by :func:`mkPen() <pyqtgraph.mkPen>`
            symbolBrush    Brush for filling points OR list of brushes, one per point.
                           May be any single argument accepted by :func:`mkBrush() <pyqtgraph.mkBrush>`
            symbolSize     Diameter of symbols OR list of diameters.
            pxMode         (bool) If True, then symbolSize is specified in pixels. If False, then symbolSize is 
                           specified in data coordinates.
            ============   ================================================
        
        **Optimization keyword arguments:**
        
            ==========   =====================================================================
            antialias    (bool) By default, antialiasing is disabled to improve performance.
                         Note that in some cases (in particluar, when pxMode=True), points 
                         will be rendered antialiased even if this is set to False.
            identical    *deprecated*
            decimate     (int) sub-sample data by selecting every nth sample before plotting
            ==========   =====================================================================
        
        **Meta-info keyword arguments:**
        
            ==========   ================================================
            name         name of dataset. This would appear in a legend
            ==========   ================================================
        """
        GraphicsObject.__init__(self)
        self.setFlag(self.ItemHasNoContents)
        self.xData = None
        self.yData = None
        self.xDisp = None
        self.yDisp = None
        self.dataMask = None
        #self.curves = []
        #self.scatters = []
        self.curve = PlotCurveItem()
        self.scatter = ScatterPlotItem()
        self.curve.setParentItem(self)
        self.scatter.setParentItem(self)
        
        self.curve.sigClicked.connect(self.curveClicked)
        self.scatter.sigClicked.connect(self.scatterClicked)
        
        
        #self.clear()
        self.opts = {
            'fftMode': False,
            'logMode': [False, False],
            'downsample': False,
            'alphaHint': 1.0,
            'alphaMode': False,
            
            'pen': (200,200,200),
            'shadowPen': None,
            'fillLevel': None,
            'fillBrush': None,
            
            'symbol': None,
            'symbolSize': 10,
            'symbolPen': (200,200,200),
            'symbolBrush': (50, 50, 150),
            'pxMode': True,
            
            'antialias': pg.getConfigOption('antialias'),
            'pointMode': None,
            
            'data': None,
        }
        self.setData(*args, **kargs)
    
    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints
    
    def boundingRect(self):
        return QtCore.QRectF()  ## let child items handle this

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
        self.xDisp = self.yDisp = None
        self.updateItems()
        self.informViewBoundsChanged()
    
    def setLogMode(self, xMode, yMode):
        if self.opts['logMode'] == [xMode, yMode]:
            return
        self.opts['logMode'] = [xMode, yMode]
        self.xDisp = self.yDisp = None
        self.updateItems()
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
        self.updateItems()
        
    def setShadowPen(self, *args, **kargs):
        """
        | Sets the shadow pen used to draw lines between points (this is for enhancing contrast or 
          emphacizing data). 
        | This line is drawn behind the primary pen (see :func:`setPen() <pyqtgraph.PlotDataItem.setPen>`)
          and should generally be assigned greater width than the primary pen.
        | *pen* can be a QPen or any argument accepted by :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>`
        """
        pen = fn.mkPen(*args, **kargs)
        self.opts['shadowPen'] = pen
        #for c in self.curves:
            #c.setPen(pen)
        #self.update()
        self.updateItems()
        
    def setFillBrush(self, *args, **kargs):
        brush = fn.mkBrush(*args, **kargs)
        if self.opts['fillBrush'] == brush:
            return
        self.opts['fillBrush'] = brush
        self.updateItems()
        
    def setBrush(self, *args, **kargs):
        return self.setFillBrush(*args, **kargs)
    
    def setFillLevel(self, level):
        if self.opts['fillLevel'] == level:
            return
        self.opts['fillLevel'] = level
        self.updateItems()

    def setSymbol(self, symbol):
        if self.opts['symbol'] == symbol:
            return
        self.opts['symbol'] = symbol
        #self.scatter.setSymbol(symbol)
        self.updateItems()
        
    def setSymbolPen(self, *args, **kargs):
        pen = fn.mkPen(*args, **kargs)
        if self.opts['symbolPen'] == pen:
            return
        self.opts['symbolPen'] = pen
        #self.scatter.setSymbolPen(pen)
        self.updateItems()
        
    
    
    def setSymbolBrush(self, *args, **kargs):
        brush = fn.mkBrush(*args, **kargs)
        if self.opts['symbolBrush'] == brush:
            return
        self.opts['symbolBrush'] = brush
        #self.scatter.setSymbolBrush(brush)
        self.updateItems()
    
    
    def setSymbolSize(self, size):
        if self.opts['symbolSize'] == size:
            return
        self.opts['symbolSize'] = size
        #self.scatter.setSymbolSize(symbolSize)
        self.updateItems()

    def setDownsampling(self, ds):
        if self.opts['downsample'] == ds:
            return
        self.opts['downsample'] = ds
        self.xDisp = self.yDisp = None
        self.updateItems()
        
    def setData(self, *args, **kargs):
        """
        Clear any data displayed by this item and display new data.
        See :func:`__init__() <pyqtgraph.PlotDataItem.__init__>` for details; it accepts the same arguments.
        """
        #self.clear()
        prof = debug.Profiler('PlotDataItem.setData (0x%x)' % id(self), disabled=True)
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
            seq = ('listOfValues', 'MetaArray')
            if dataType(args[0]) not in seq or  dataType(args[1]) not in seq:
                raise Exception('When passing two unnamed arguments, both must be a list or array of values. (got %s, %s)' % (str(type(args[0])), str(type(args[1]))))
            if not isinstance(args[0], np.ndarray):
                x = np.array(args[0])
            else:
                x = args[0].view(np.ndarray)
            if not isinstance(args[1], np.ndarray):
                y = np.array(args[1])
            else:
                y = args[1].view(np.ndarray)
            
        if 'x' in kargs:
            x = kargs['x']
        if 'y' in kargs:
            y = kargs['y']

        prof.mark('interpret data')
        ## pull in all style arguments. 
        ## Use self.opts to fill in anything not present in kargs.
        
        if 'name' in kargs:
            self.opts['name'] = kargs['name']

        ## if symbol pen/brush are given with no symbol, then assume symbol is 'o'
        
        if 'symbol' not in kargs and ('symbolPen' in kargs or 'symbolBrush' in kargs or 'symbolSize' in kargs):
            kargs['symbol'] = 'o'
            
        if 'brush' in kargs:
            kargs['fillBrush'] = kargs['brush']
            
        for k in list(self.opts.keys()):
            if k in kargs:
                self.opts[k] = kargs[k]
                
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
        

        if y is None:
            return
        if y is not None and x is None:
            x = np.arange(len(y))
        
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        
        self.xData = x.view(np.ndarray)  ## one last check to make sure there are no MetaArrays getting by
        self.yData = y.view(np.ndarray)
        self.xDisp = None
        self.yDisp = None
        prof.mark('set data')
        
        self.updateItems()
        prof.mark('update items')
        
        self.informViewBoundsChanged()
        #view = self.getViewBox()
        #if view is not None:
            #view.itemBoundsChanged(self)  ## inform view so it can update its range if it wants
        
        self.sigPlotChanged.emit(self)
        prof.mark('emit')
        prof.finish()


    def updateItems(self):
        
        curveArgs = {}
        for k,v in [('pen','pen'), ('shadowPen','shadowPen'), ('fillLevel','fillLevel'), ('fillBrush', 'brush'), ('antialias', 'antialias')]:
            curveArgs[v] = self.opts[k]
        
        scatterArgs = {}
        for k,v in [('symbolPen','pen'), ('symbolBrush','brush'), ('symbol','symbol'), ('symbolSize', 'size'), ('data', 'data'), ('pxMode', 'pxMode'), ('antialias', 'antialias')]:
            if k in self.opts:
                scatterArgs[v] = self.opts[k]
        
        x,y = self.getData()
        scatterArgs['mask'] = self.dataMask
        
        if curveArgs['pen'] is not None or (curveArgs['brush'] is not None and curveArgs['fillLevel'] is not None):
            self.curve.setData(x=x, y=y, **curveArgs)
            self.curve.show()
        else:
            self.curve.hide()
        
        if scatterArgs['symbol'] is not None:
            self.scatter.setData(x=x, y=y, **scatterArgs)
            self.scatter.show()
        else:
            self.scatter.hide()


    def getData(self):
        if self.xData is None:
            return (None, None)
        if self.xDisp is None:
            nanMask = np.isnan(self.xData) | np.isnan(self.yData) | np.isinf(self.xData) | np.isinf(self.yData)
            if any(nanMask):
                self.dataMask = ~nanMask
                x = self.xData[self.dataMask]
                y = self.yData[self.dataMask]
            else:
                self.dataMask = None
                x = self.xData
                y = self.yData
                
            
            ds = self.opts['downsample']
            if ds > 1:
                x = x[::ds]
                #y = resample(y[:len(x)*ds], len(x))  ## scipy.signal.resample causes nasty ringing
                y = y[::ds]
            if self.opts['fftMode']:
                f = np.fft.fft(y) / len(y)
                y = abs(f[1:len(f)/2])
                dt = x[-1] - x[0]
                x = np.linspace(0, 0.5*len(x)/dt, len(y))
            if self.opts['logMode'][0]:
                x = np.log10(x)
            if self.opts['logMode'][1]:
                y = np.log10(y)
            if any(self.opts['logMode']):  ## re-check for NANs after log
                nanMask = np.isinf(x) | np.isinf(y) | np.isnan(x) | np.isnan(y)
                if any(nanMask):
                    self.dataMask = ~nanMask
                    x = x[self.dataMask]
                    y = y[self.dataMask]
                else:
                    self.dataMask = None
            self.xDisp = x
            self.yDisp = y
        #print self.yDisp.shape, self.yDisp.min(), self.yDisp.max()
        #print self.xDisp.shape, self.xDisp.min(), self.xDisp.max()
        return self.xDisp, self.yDisp

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
        
        #if frac <= 0.0:
            #raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        
        #(x, y) = self.getData()
        #if x is None or len(x) == 0:
            #return None
            
        #if ax == 0:
            #d = x
            #d2 = y
        #elif ax == 1:
            #d = y
            #d2 = x
            
        #if orthoRange is not None:
            #mask = (d2 >= orthoRange[0]) * (d2 <= orthoRange[1])
            #d = d[mask]
            ##d2 = d2[mask]
            
        #if len(d) > 0:
            #if frac >= 1.0:
                #return (np.min(d), np.max(d))
            #else:
                #return (scipy.stats.scoreatpercentile(d, 50 - (frac * 50)), scipy.stats.scoreatpercentile(d, 50 + (frac * 50)))
        #else:
            #return None
    
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
        #for i in self.curves+self.scatters:
            #if i.scene() is not None:
                #i.scene().removeItem(i)
        #self.curves = []
        #self.scatters = []
        self.xData = None
        self.yData = None
        self.xDisp = None
        self.yDisp = None
        self.curve.setData([])
        self.scatter.setData([])
            
    def appendData(self, *args, **kargs):
        pass
    
    def curveClicked(self):
        self.sigClicked.emit(self)
        
    def scatterClicked(self, plt, points):
        self.sigClicked.emit(self)
        self.sigPointsClicked.emit(self, points)
    
    
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
        #for i in xrange(1, len(self)):
            #arr[i] = tuple(self[i].values())
        #return arr
            
    #def __getitem__array(self, arg):
        #if isinstance(arg, tuple):
            #return self.data[arg[0]][arg[1]]
        #else:
            #return self.data[arg]
            
    #def __getitem__list(self, arg):
        #if isinstance(arg, basestring):
            #return [d.get(arg, None) for d in self.data]
        #elif isinstance(arg, int):
            #return self.data[arg]
        #elif isinstance(arg, tuple):
            #arg = self._orderArgs(arg)
            #return self.data[arg[0]][arg[1]]
        #else:
            #raise TypeError(type(arg))
        
    #def __getitem__dict(self, arg):
        #if isinstance(arg, basestring):
            #return self.data[arg]
        #elif isinstance(arg, int):
            #return dict([(k, v[arg]) for k, v in self.data.iteritems()])
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
        #if isinstance(arg, basestring):
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
        #if isinstance(arg, basestring):
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
        #if isinstance(args[0], basestring):
            #return (args[1], args[0])
        #else:
            #return args
        
    #def __iter__(self):
        #for i in xrange(len(self)):
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
