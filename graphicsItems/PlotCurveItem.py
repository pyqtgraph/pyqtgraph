from pyqtgraph.Qt import QtGui, QtCore
from scipy.fftpack import fft
import numpy as np
import scipy.stats
from GraphicsObject import GraphicsObject
import pyqtgraph.functions as fn
from pyqtgraph import debug
from pyqtgraph.Point import Point
import struct

__all__ = ['PlotCurveItem']
class PlotCurveItem(GraphicsObject):
    
    
    """
    Class representing a single plot curve. Instances of this class are created
    automatically as part of PlotDataItem; these rarely need to be instantiated
    directly.
    
    Features:
    
    - Fast data update
    - FFT display mode (accessed via PlotItem context menu)
    - Fill under curve
    - Mouse interaction
    
    ====================  ===============================================
    **Signals:**
    sigPlotChanged(self)  Emitted when the data being plotted has changed
    sigClicked(self)      Emitted when the curve is clicked
    ====================  ===============================================
    """
    
    sigPlotChanged = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object)
    
    def __init__(self, y=None, x=None, fillLevel=None, copy=False, pen=None, shadowPen=None, brush=None, parent=None, clickable=False):
        """
        ==============  =======================================================
        **Arguments:**
        x, y            (numpy arrays) Data to show 
        pen             Pen to use when drawing. Any single argument accepted by
                        :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        shadowPen       Pen for drawing behind the primary pen. Usually this
                        is used to emphasize the curve by providing a 
                        high-contrast border. Any single argument accepted by
                        :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        fillLevel       (float or None) Fill the area 'under' the curve to
                        *fillLevel*
        brush           QBrush to use when filling. Any single argument accepted
                        by :func:`mkBrush <pyqtgraph.mkBrush>` is allowed.
        clickable       If True, the item will emit sigClicked when it is 
                        clicked on.
        ==============  =======================================================
        
        
        
        """
        GraphicsObject.__init__(self, parent)
        self.clear()
        self.path = None
        self.fillPath = None
        self.exportOpts = False
        self.antialias = False
        
        if y is not None:
            self.updateData(y, x)
            
        ## this is disastrous for performance.
        #self.setCacheMode(QtGui.QGraphicsItem.DeviceCoordinateCache)
        
        self.metaData = {}
        self.opts = {
            #'spectrumMode': False,
            #'logMode': [False, False],
            #'downsample': False,
            #'alphaHint': 1.0,
            #'alphaMode': False,
            'pen': 'w',
            'shadowPen': None,
            'fillLevel': fillLevel,
            'brush': brush,
        }
        self.setPen(pen)
        self.setShadowPen(shadowPen)
        self.setFillLevel(fillLevel)
        self.setBrush(brush)
        self.setClickable(clickable)
        #self.fps = None
        
    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints
    
    def setClickable(self, s):
        """Sets whether the item responds to mouse clicks."""
        self.clickable = s
        
        
    def getData(self):
        return self.xData, self.yData
        #if self.xData is None:
            #return (None, None)
        #if self.xDisp is None:
            #nanMask = np.isnan(self.xData) | np.isnan(self.yData)
            #if any(nanMask):
                #x = self.xData[~nanMask]
                #y = self.yData[~nanMask]
            #else:
                #x = self.xData
                #y = self.yData
            #ds = self.opts['downsample']
            #if ds > 1:
                #x = x[::ds]
                ##y = resample(y[:len(x)*ds], len(x))  ## scipy.signal.resample causes nasty ringing
                #y = y[::ds]
            #if self.opts['spectrumMode']:
                #f = fft(y) / len(y)
                #y = abs(f[1:len(f)/2])
                #dt = x[-1] - x[0]
                #x = np.linspace(0, 0.5*len(x)/dt, len(y))
            #if self.opts['logMode'][0]:
                #x = np.log10(x)
            #if self.opts['logMode'][1]:
                #y = np.log10(y)
            #self.xDisp = x
            #self.yDisp = y
        ##print self.yDisp.shape, self.yDisp.min(), self.yDisp.max()
        ##print self.xDisp.shape, self.xDisp.min(), self.xDisp.max()
        #return self.xDisp, self.yDisp
            
    #def generateSpecData(self):
        #f = fft(self.yData) / len(self.yData)
        #self.ySpec = abs(f[1:len(f)/2])
        #dt = self.xData[-1] - self.xData[0]
        #self.xSpec = linspace(0, 0.5*len(self.xData)/dt, len(self.ySpec))
        
    def dataBounds(self, ax, frac=1.0):
        (x, y) = self.getData()
        if x is None or len(x) == 0:
            return (0, 0)
            
        if ax == 0:
            d = x
        elif ax == 1:
            d = y
            
        if frac >= 1.0:
            return (d.min(), d.max())
        elif frac <= 0.0:
            raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        else:
            return (scipy.stats.scoreatpercentile(d, 50 - (frac * 50)), scipy.stats.scoreatpercentile(d, 50 + (frac * 50)))
            
    #def setMeta(self, data):
        #self.metaData = data
        
    #def meta(self):
        #return self.metaData
        
    def setPen(self, *args, **kargs):
        """Set the pen used to draw the curve."""
        self.opts['pen'] = fn.mkPen(*args, **kargs)
        self.update()
        
    def setShadowPen(self, *args, **kargs):
        """Set the shadow pen used to draw behind tyhe primary pen.
        This pen must have a larger width than the primary 
        pen to be visible.
        """
        self.opts['shadowPen'] = fn.mkPen(*args, **kargs)
        self.update()

    def setBrush(self, *args, **kargs):
        """Set the brush used when filling the area under the curve"""
        self.opts['brush'] = fn.mkBrush(*args, **kargs)
        self.update()
        
    def setFillLevel(self, level):
        """Set the level filled to when filling under the curve"""
        self.opts['fillLevel'] = level
        self.fillPath = None
        self.update()

    #def setColor(self, color):
        #self.pen.setColor(color)
        #self.update()
        
    #def setAlpha(self, alpha, auto):
        #self.opts['alphaHint'] = alpha
        #self.opts['alphaMode'] = auto
        #self.update()
        
    #def setSpectrumMode(self, mode):
        #self.opts['spectrumMode'] = mode
        #self.xDisp = self.yDisp = None
        #self.path = None
        #self.update()
    
    #def setLogMode(self, mode):
        #self.opts['logMode'] = mode
        #self.xDisp = self.yDisp = None
        #self.path = None
        #self.update()
    
    #def setPointMode(self, mode):
        #self.opts['pointMode'] = mode
        #self.update()
        

    #def setDownsampling(self, ds):
        #if self.opts['downsample'] != ds:
            #self.opts['downsample'] = ds
            #self.xDisp = self.yDisp = None
            #self.path = None
            #self.update()

    def setData(self, *args, **kargs):
        """
        Accepts most of the same arguments as __init__.
        """
        self.updateData(*args, **kargs)
        
    def updateData(self, *args, **kargs):
        prof = debug.Profiler('PlotCurveItem.updateData', disabled=True)

        if len(args) == 1:
            kargs['y'] = args[0]
        elif len(args) == 2:
            kargs['x'] = args[0]
            kargs['y'] = args[1]
        
        if 'y' not in kargs or kargs['y'] is None:
            kargs['y'] = np.array([])
        if 'x' not in kargs or kargs['x'] is None:
            kargs['x'] = np.arange(len(kargs['y']))
            
        for k in ['x', 'y']:
            data = kargs[k]
            if isinstance(data, list):
                data = np.array(data)
                kargs[k] = data
            if not isinstance(data, np.ndarray) or data.ndim > 1:
                raise Exception("Plot data must be 1D ndarray.")
            if 'complex' in str(data.dtype):
                raise Exception("Can not plot complex data types.")
            
        prof.mark("data checks")
        
        #self.setCacheMode(QtGui.QGraphicsItem.NoCache)  ## Disabling and re-enabling the cache works around a bug in Qt 4.6 causing the cached results to display incorrectly
                                                        ##    Test this bug with test_PlotWidget and zoom in on the animated plot
        self.prepareGeometryChange()
        self.yData = kargs['y'].view(np.ndarray)
        self.xData = kargs['x'].view(np.ndarray)
        
        prof.mark('copy')
        
        if self.xData.shape != self.yData.shape:
            raise Exception("X and Y arrays must be the same shape--got %s and %s." % (str(x.shape), str(y.shape)))
        
        self.path = None
        self.fillPath = None
        #self.xDisp = self.yDisp = None
        
        if 'pen' in kargs:
            self.setPen(kargs['pen'])
        if 'shadowPen' in kargs:
            self.setShadowPen(kargs['shadowPen'])
        if 'fillLevel' in kargs:
            self.setFillLevel(kargs['fillLevel'])
        if 'brush' in kargs:
            self.setBrush(kargs['brush'])
        
        
        prof.mark('set')
        self.update()
        prof.mark('update')
        self.sigPlotChanged.emit(self)
        prof.mark('emit')
        prof.finish()
        
    def generatePath(self, x, y):
        prof = debug.Profiler('PlotCurveItem.generatePath', disabled=True)
        path = QtGui.QPainterPath()
        
        ## Create all vertices in path. The method used below creates a binary format so that all 
        ## vertices can be read in at once. This binary format may change in future versions of Qt, 
        ## so the original (slower) method is left here for emergencies:
        #path.moveTo(x[0], y[0])
        #for i in range(1, y.shape[0]):
        #    path.lineTo(x[i], y[i])
            
        ## Speed this up using >> operator
        ## Format is:
        ##    numVerts(i4)   0(i4)
        ##    x(f8)   y(f8)   0(i4)    <-- 0 means this vertex does not connect
        ##    x(f8)   y(f8)   1(i4)    <-- 1 means this vertex connects to the previous vertex
        ##    ...
        ##    0(i4)
        ##
        ## All values are big endian--pack using struct.pack('>d') or struct.pack('>i')
        
        n = x.shape[0]
        # create empty array, pad with extra space on either end
        arr = np.empty(n+2, dtype=[('x', '>f8'), ('y', '>f8'), ('c', '>i4')])
        # write first two integers
        prof.mark('allocate empty')
        arr.data[12:20] = struct.pack('>ii', n, 0)
        prof.mark('pack header')
        # Fill array with vertex values
        arr[1:-1]['x'] = x
        arr[1:-1]['y'] = y
        arr[1:-1]['c'] = 1
        prof.mark('fill array')
        # write last 0
        lastInd = 20*(n+1) 
        arr.data[lastInd:lastInd+4] = struct.pack('>i', 0)
        prof.mark('footer')
        # create datastream object and stream into path
        buf = QtCore.QByteArray(arr.data[12:lastInd+4])  # I think one unnecessary copy happens here
        prof.mark('create buffer')
        ds = QtCore.QDataStream(buf)
        prof.mark('create datastream')
        ds >> path
        prof.mark('load')
        
        prof.finish()
        return path


    def shape(self):
        if self.path is None:
            try:
                self.path = self.generatePath(*self.getData())
            except:
                return QtGui.QPainterPath()
        return self.path

    def boundingRect(self):
        (x, y) = self.getData()
        if x is None or y is None or len(x) == 0 or len(y) == 0:
            return QtCore.QRectF()
            
            
        if self.opts['shadowPen'] is not None:
            lineWidth = (max(self.opts['pen'].width(), self.opts['shadowPen'].width()) + 1)
        else:
            lineWidth = (self.opts['pen'].width()+1)
            
        
        pixels = self.pixelVectors()
        if pixels is None:
            pixels = [Point(0,0), Point(0,0)]
        xmin = x.min() - pixels[0].x() * lineWidth
        xmax = x.max() + pixels[0].x() * lineWidth
        ymin = y.min() - abs(pixels[1].y()) * lineWidth
        ymax = y.max() + abs(pixels[1].y()) * lineWidth
        
            
        return QtCore.QRectF(xmin, ymin, xmax-xmin, ymax-ymin)

    def paint(self, p, opt, widget):
        prof = debug.Profiler('PlotCurveItem.paint '+str(id(self)), disabled=True)
        if self.xData is None:
            return
        #if self.opts['spectrumMode']:
            #if self.specPath is None:
                
                #self.specPath = self.generatePath(*self.getData())
            #path = self.specPath
        #else:
        x = None
        y = None
        if self.path is None:
            x,y = self.getData()
            if x is None or len(x) == 0 or y is None or len(y) == 0:
                return
            self.path = self.generatePath(x,y)
            self.fillPath = None
            
            
        path = self.path
        prof.mark('generate path')
            
        if self.opts['brush'] is not None and self.opts['fillLevel'] is not None:
            if self.fillPath is None:
                if x is None:
                    x,y = self.getData()
                p2 = QtGui.QPainterPath(self.path)
                p2.lineTo(x[-1], self.opts['fillLevel'])
                p2.lineTo(x[0], self.opts['fillLevel'])
                p2.lineTo(x[0], y[0])
                p2.closeSubpath()
                self.fillPath = p2
                
            prof.mark('generate fill path')
            p.fillPath(self.fillPath, self.opts['brush'])
            prof.mark('draw fill path')
            

        ## Copy pens and apply alpha adjustment
        sp = QtGui.QPen(self.opts['shadowPen'])
        cp = QtGui.QPen(self.opts['pen'])
        #for pen in [sp, cp]:
            #if pen is None:
                #continue
            #c = pen.color()
            #c.setAlpha(c.alpha() * self.opts['alphaHint'])
            #pen.setColor(c)
            ##pen.setCosmetic(True)
            
        if self.exportOpts is not False:
            aa = self.exportOpts['antialias']
        else:
            aa = self.antialias
        
        p.setRenderHint(p.Antialiasing, aa)
            
            
        if sp is not None:
            p.setPen(sp)
            p.drawPath(path)
        p.setPen(cp)
        p.drawPath(path)
        prof.mark('drawPath')
        
        #print "Render hints:", int(p.renderHints())
        prof.finish()
        #p.setPen(QtGui.QPen(QtGui.QColor(255,0,0)))
        #p.drawRect(self.boundingRect())
        
        
    def clear(self):
        self.xData = None  ## raw values
        self.yData = None
        self.xDisp = None  ## display values (after log / fft)
        self.yDisp = None
        self.path = None
        #del self.xData, self.yData, self.xDisp, self.yDisp, self.path
        
    #def mousePressEvent(self, ev):
        ##GraphicsObject.mousePressEvent(self, ev)
        #if not self.clickable:
            #ev.ignore()
        #if ev.button() != QtCore.Qt.LeftButton:
            #ev.ignore()
        #self.mousePressPos = ev.pos()
        #self.mouseMoved = False
        
    #def mouseMoveEvent(self, ev):
        ##GraphicsObject.mouseMoveEvent(self, ev)
        #self.mouseMoved = True
        ##print "move"
        
    #def mouseReleaseEvent(self, ev):
        ##GraphicsObject.mouseReleaseEvent(self, ev)
        #if not self.mouseMoved:
            #self.sigClicked.emit(self)

    def mouseClickEvent(self, ev):
        if not self.clickable or ev.button() != QtCore.Qt.LeftButton:
            return
        ev.accept()
        self.sigClicked.emit(self)

    def setExportMode(self, export, opts):
        if export:
            self.exportOpts = opts
            if 'antialias' not in opts:
                self.exportOpts['antialias'] = True
        else:
            self.exportOpts = False

class ROIPlotItem(PlotCurveItem):
    """Plot curve that monitors an ROI and image for changes to automatically replot."""
    def __init__(self, roi, data, img, axes=(0,1), xVals=None, color=None):
        self.roi = roi
        self.roiData = data
        self.roiImg = img
        self.axes = axes
        self.xVals = xVals
        PlotCurveItem.__init__(self, self.getRoiData(), x=self.xVals, color=color)
        #roi.connect(roi, QtCore.SIGNAL('regionChanged'), self.roiChangedEvent)
        roi.sigRegionChanged.connect(self.roiChangedEvent)
        #self.roiChangedEvent()
        
    def getRoiData(self):
        d = self.roi.getArrayRegion(self.roiData, self.roiImg, axes=self.axes)
        if d is None:
            return
        while d.ndim > 1:
            d = d.mean(axis=1)
        return d
        
    def roiChangedEvent(self):
        d = self.getRoiData()
        self.updateData(d, self.xVals)

