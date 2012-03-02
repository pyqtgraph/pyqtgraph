try: 
    import metaarray
    HAVE_METAARRAY = True
except:
    HAVE_METAARRAY = False

from pyqtgraph.Qt import QtCore
from GraphicsObject import GraphicsObject
from PlotCurveItem import PlotCurveItem
from ScatterPlotItem import ScatterPlotItem
import numpy as np
import scipy
import pyqtgraph.functions as fn

class PlotDataItem(GraphicsObject):
    """GraphicsItem for displaying plot curves, scatter plots, or both."""
    
    sigPlotChanged = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object)
    
    def __init__(self, *args, **kargs):
        """
        There are many different ways to create a PlotDataItem:
        
        Data initialization: (x,y data only)
        
            =================================== ======================================
            PlotDataItem(xValues, yValues)      x and y values may be any sequence (including ndarray) of real numbers
            PlotDataItem(yValues)               y values only -- x will be automatically set to range(len(y))
            PlotDataItem(x=xValues, y=yValues)  x and y given by keyword arguments
            PlotDataItem(ndarray(Nx2))          numpy array with shape (N, 2) where x=data[:,0] and y=data[:,1]
            =================================== ======================================
        
        Data initialization: (x,y data AND may include spot style)
        
            ===========================   =========================================
            PlotDataItem(recarray)        numpy array with dtype=[('x', float), ('y', float), ...]
            PlotDataItem(list-of-dicts)   [{'x': x, 'y': y, ...},   ...] 
            PlotDataItem(dict-of-lists)   {'x': [...], 'y': [...],  ...}           
            PlotDataItem(MetaArray)       1D array of Y values with X sepecified as axis values 
                                          OR 2D array with a column 'y' and extra columns as needed.
            ===========================   =========================================
        
        Line style keyword         
            ==========   ================================================
            pen          pen to use for drawing line between points. Default is solid grey, 1px width. Use None to disable line drawing.
            shadowPen    pen for secondary line to draw behind the primary line. disabled by default.
            fillLevel    fill the area between the curve and fillLevel
            fillBrush    fill to use when fillLevel is specified
            ==========   ================================================
        
        Point style keyword arguments:
        
            ============   ================================================
            symbol         symbol to use for drawing points OR list of symbols, one per point. Default is no symbol.
                           options are o, s, t, d, +
            symbolPen      outline pen for drawing points OR list of pens, one per point
            symbolBrush    brush for filling points OR list of brushes, one per point
            symbolSize     diameter of symbols OR list of diameters
            pxMode         (bool) If True, then symbolSize is specified in pixels. If False, then symbolSize is 
                           specified in data coordinates.
            ============   ================================================
        
        Optimization keyword arguments:
        
            ==========   ================================================
            identical    spots are all identical. The spot image will be rendered only once and repeated for every point
            decimate     (int) decimate data
            ==========   ================================================
        
        Meta-info keyword arguments:
        
            ==========   ================================================
            name         name of dataset. This would appear in a legend
            ==========   ================================================
        """
        GraphicsObject.__init__(self)
        self.setFlag(self.ItemHasNoContents)
        self.xData = None
        self.yData = None
        self.curves = []
        self.scatters = []
        self.clear()
        self.opts = {
            'fftMode': False,
            'logMode': [False, False],
            'downsample': False,
            'alphaHint': 1.0,
            'alphaMode': False,
            
            'pen': (200,200,200),
            'shadowPen': None,
            'fillLevel': None,
            'brush': None,
            
            'symbol': None,
            'symbolSize': 10,
            'symbolPen': (200,200,200),
            'symbolBrush': (50, 50, 150),
            'identical': False,
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
        self.opts['alphaHint'] = alpha
        self.opts['alphaMode'] = auto
        self.setOpacity(alpha)
        #self.update()
        
    def setFftMode(self, mode):
        self.opts['fftMode'] = mode
        self.xDisp = self.yDisp = None
        self.updateItems()
    
    def setLogMode(self, mode):
        self.opts['logMode'] = mode
        self.xDisp = self.yDisp = None
        self.updateItems()
    
    def setPointMode(self, mode):
        self.opts['pointMode'] = mode
        self.update()
        
    def setPen(self, pen):
        """
        | Sets the pen used to draw lines between points.
        | *pen* can be a QPen or any argument accepted by :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>`
        """
        self.opts['pen'] = fn.mkPen(pen)
        for c in self.curves:
            c.setPen(pen)
        self.update()
        
    def setShadowPen(self, pen):
        """
        | Sets the shadow pen used to draw lines between points (this is for enhancing contrast or 
          emphacizing data). 
        | This line is drawn behind the primary pen (see :func:`setPen() <pyqtgraph.PlotDataItem.setPen>`)
          and should generally be assigned greater width than the primary pen.
        | *pen* can be a QPen or any argument accepted by :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>`
        """
        self.opts['shadowPen'] = pen
        for c in self.curves:
            c.setPen(pen)
        self.update()

    def setDownsampling(self, ds):
        if self.opts['downsample'] != ds:
            self.opts['downsample'] = ds
            self.xDisp = self.yDisp = None
            self.updateItems()
        
    def setData(self, *args, **kargs):
        """
        Clear any data displayed by this item and display new data.
        See :func:`__init__() <pyqtgraph.PlotDataItem.__init__>` for details; it accepts the same arguments.
        """
        
        self.clear()
        
        y = None
        x = None
        if len(args) == 1:
            data = args[0]
            dt = dataType(data)
            if dt == 'empty':
                return
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


        ## pull in all style arguments. 
        ## Use self.opts to fill in anything not present in kargs.
        

        ## if symbol pen/brush are given with no symbol, then assume symbol is 'o'
        if 'symbol' not in kargs and ('symbolPen' in kargs or 'symbolBrush' in kargs):
            kargs['symbol'] = 'o'
            
        for k in self.opts.keys():
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
        
        self.updateItems()
        view = self.getViewBox()
        if view is not None:
            view.itemBoundsChanged(self)  ## inform view so it can update its range if it wants
        self.sigPlotChanged.emit(self)


    def updateItems(self):
        for c in self.curves+self.scatters:
            if c.scene() is not None:
                c.scene().removeItem(c)
            
        curveArgs = {}
        for k in ['pen', 'shadowPen', 'fillLevel', 'brush']:
            curveArgs[k] = self.opts[k]
        
        scatterArgs = {}
        for k,v in [('symbolPen','pen'), ('symbolBrush','brush'), ('symbol','symbol')]:
            scatterArgs[v] = self.opts[k]
        
        x,y = self.getData()
        
        if curveArgs['pen'] is not None or curveArgs['brush'] is not None:
            curve = PlotCurveItem(x=x, y=y, **curveArgs)
            curve.setParentItem(self)
            self.curves.append(curve)
        
        if scatterArgs['symbol'] is not None:
            sp = ScatterPlotItem(x=x, y=y, **scatterArgs)
            sp.setParentItem(self)
            self.scatters.append(sp)


    def getData(self):
        if self.xData is None:
            return (None, None)
        if self.xDisp is None:
            nanMask = np.isnan(self.xData) | np.isnan(self.yData)
            if any(nanMask):
                x = self.xData[~nanMask]
                y = self.yData[~nanMask]
            else:
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
            self.xDisp = x
            self.yDisp = y
        #print self.yDisp.shape, self.yDisp.min(), self.yDisp.max()
        #print self.xDisp.shape, self.xDisp.min(), self.xDisp.max()
        return self.xDisp, self.yDisp

    def dataBounds(self, ax, frac=1.0):
        (x, y) = self.getData()
        if x is None or len(x) == 0:
            return (0, 0)
            
        if ax == 0:
            d = x
        elif ax == 1:
            d = y
            
        if frac >= 1.0:
            return (np.min(d), np.max(d))
        elif frac <= 0.0:
            raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        else:
            return (scipy.stats.scoreatpercentile(d, 50 - (frac * 50)), scipy.stats.scoreatpercentile(d, 50 + (frac * 50)))


    def clear(self):
        for i in self.curves+self.scatters:
            if i.scene() is not None:
                i.scene().removeItem(i)
        self.curves = []
        self.scatters = []
        self.xData = None
        self.yData = None
        self.xDisp = None
        self.yDisp = None
            
    def appendData(self, *args, **kargs):
        pass
    
    
def dataType(obj):
    if hasattr(obj, '__len__') and len(obj) == 0:
        return 'empty'
    if isSequence(obj):
        first = obj[0]
        if isinstance(obj, np.ndarray):
            if HAVE_METAARRAY and isinstance(obj, metaarray.MetaArray):
                return 'MetaArray'
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
            return 'listOfDict'
        else:
            return 'listOfValues'
    elif isinstance(obj, dict):
        return 'dict'
        
        
def isSequence(obj):
    return isinstance(obj, list) or isinstance(obj, np.ndarray)
    
            
            
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