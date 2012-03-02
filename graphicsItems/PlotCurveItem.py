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
    
    
    """Class representing a single plot curve. Provides:
        - Fast data update
        - FFT display mode
        - shadow pen
        - mouse interaction
    """
    
    sigPlotChanged = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object)
    
    def __init__(self, y=None, x=None, fillLevel=None, copy=False, pen=None, shadowPen=None, brush=None, parent=None, color=None, clickable=False):
        GraphicsObject.__init__(self, parent)
        self.clear()
        self.path = None
        self.fillPath = None
        if pen is None:
            if color is None:
                self.setPen((200,200,200))
            else:
                self.setPen(color)
        else:
            self.setPen(pen)
        
        self.shadowPen = shadowPen
        if y is not None:
            self.updateData(y, x, copy)
            
        ## this is disastrous for performance.
        #self.setCacheMode(QtGui.QGraphicsItem.DeviceCoordinateCache)
        
        self.fillLevel = fillLevel
        self.brush = brush
        
        self.metaData = {}
        self.opts = {
            'spectrumMode': False,
            'logMode': [False, False],
            'pointMode': False,
            'pointStyle': None,
            'downsample': False,
            'alphaHint': 1.0,
            'alphaMode': False
        }
            
        self.setClickable(clickable)
        #self.fps = None
        
    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints
    
    def setClickable(self, s):
        self.clickable = s
        
        
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
            if self.opts['spectrumMode']:
                f = fft(y) / len(y)
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
            
    def setMeta(self, data):
        self.metaData = data
        
    def meta(self):
        return self.metaData
        
    def setPen(self, pen):
        self.pen = fn.mkPen(pen)
        self.update()
        
    def setColor(self, color):
        self.pen.setColor(color)
        self.update()
        
    def setAlpha(self, alpha, auto):
        self.opts['alphaHint'] = alpha
        self.opts['alphaMode'] = auto
        self.update()
        
    def setSpectrumMode(self, mode):
        self.opts['spectrumMode'] = mode
        self.xDisp = self.yDisp = None
        self.path = None
        self.update()
    
    def setLogMode(self, mode):
        self.opts['logMode'] = mode
        self.xDisp = self.yDisp = None
        self.path = None
        self.update()
    
    def setPointMode(self, mode):
        self.opts['pointMode'] = mode
        self.update()
        
    def setShadowPen(self, pen):
        self.shadowPen = pen
        self.update()

    def setDownsampling(self, ds):
        if self.opts['downsample'] != ds:
            self.opts['downsample'] = ds
            self.xDisp = self.yDisp = None
            self.path = None
            self.update()

    def setData(self, x, y, copy=False):
        """For Qwt compatibility"""
        self.updateData(y, x, copy)
        
    def updateData(self, data, x=None, copy=False):
        prof = debug.Profiler('PlotCurveItem.updateData', disabled=True)
        if isinstance(data, list):
            data = np.array(data)
        if isinstance(x, list):
            x = np.array(x)
        if not isinstance(data, np.ndarray) or data.ndim > 2:
            raise Exception("Plot data must be 1 or 2D ndarray (data shape is %s)" % str(data.shape))
        if x == None:
            if 'complex' in str(data.dtype):
                raise Exception("Can not plot complex data types.")
        else:
            if 'complex' in str(data.dtype)+str(x.dtype):
                raise Exception("Can not plot complex data types.")
        
        if data.ndim == 2:  ### If data is 2D array, then assume x and y values are in first two columns or rows.
            if x is not None:
                raise Exception("Plot data may be 2D only if no x argument is supplied.")
            ax = 0
            if data.shape[0] > 2 and data.shape[1] == 2:
                ax = 1
            ind = [slice(None), slice(None)]
            ind[ax] = 0
            y = data[tuple(ind)]
            ind[ax] = 1
            x = data[tuple(ind)]
        elif data.ndim == 1:
            y = data
        prof.mark("data checks")
        
        self.setCacheMode(QtGui.QGraphicsItem.NoCache)  ## Disabling and re-enabling the cache works around a bug in Qt 4.6 causing the cached results to display incorrectly
                                                        ##    Test this bug with test_PlotWidget and zoom in on the animated plot
        
        self.prepareGeometryChange()
        if copy:
            self.yData = y.view(np.ndarray).copy()
        else:
            self.yData = y.view(np.ndarray)
            
        if x is None:
            self.xData = np.arange(0, self.yData.shape[0])
        else:
            if copy:
                self.xData = x.view(np.ndarray).copy()
            else:
                self.xData = x.view(np.ndarray)
        prof.mark('copy')
        

        if self.xData.shape != self.yData.shape:
            raise Exception("X and Y arrays must be the same shape--got %s and %s." % (str(x.shape), str(y.shape)))
        
        self.path = None
        self.xDisp = self.yDisp = None
        
        prof.mark('set')
        self.update()
        prof.mark('update')
        #self.emit(QtCore.SIGNAL('plotChanged'), self)
        self.sigPlotChanged.emit(self)
        prof.mark('emit')
        #prof.finish()
        #self.setCacheMode(QtGui.QGraphicsItem.DeviceCoordinateCache)
        prof.mark('set cache mode')
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
            
            
        if self.shadowPen is not None:
            lineWidth = (max(self.pen.width(), self.shadowPen.width()) + 1)
        else:
            lineWidth = (self.pen.width()+1)
            
        
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
            
        if self.brush is not None:
            if self.fillPath is None:
                if x is None:
                    x,y = self.getData()
                p2 = QtGui.QPainterPath(self.path)
                p2.lineTo(x[-1], self.fillLevel)
                p2.lineTo(x[0], self.fillLevel)
                p2.closeSubpath()
                self.fillPath = p2
                
            p.fillPath(self.fillPath, fn.mkBrush(self.brush))
            
        if self.shadowPen is not None:
            sp = QtGui.QPen(self.shadowPen)
        else:
            sp = None

        ## Copy pens and apply alpha adjustment
        cp = QtGui.QPen(self.pen)
        for pen in [sp, cp]:
            if pen is None:
                continue
            c = pen.color()
            c.setAlpha(c.alpha() * self.opts['alphaHint'])
            pen.setColor(c)
            #pen.setCosmetic(True)
            
        if self.shadowPen is not None:
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

