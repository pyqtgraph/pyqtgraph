# -*- coding: utf-8 -*-
"""
graphicsItems.py -  Defines several graphics item classes for use in Qt graphics/view framework
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.

Provides ImageItem, PlotCurveItem, and ViewBox, amongst others.
"""


from PyQt4 import QtGui, QtCore
from ObjectWorkaround import *
#tryWorkaround(QtCore, QtGui)
from numpy import *
try:
    import scipy.weave as weave
    from scipy.weave import converters
except:
    pass
from scipy.fftpack import fft
from scipy.signal import resample
import scipy.stats
#from metaarray import MetaArray
from Point import *
from functions import *
import types, sys, struct
import weakref
#from debug import *


## Should probably just use QGraphicsGroupItem and instruct it to pass events on to children..
class ItemGroup(QtGui.QGraphicsItem):
    def __init__(self, *args):
        QtGui.QGraphicsItem.__init__(self, *args)
        if hasattr(self, "ItemHasNoContents"):
            self.setFlag(self.ItemHasNoContents)
    
    def boundingRect(self):
        return QtCore.QRectF()
        
    def paint(self, *args):
        pass
    
    def addItem(self, item):
        item.setParentItem(self)


#if hasattr(QtGui, "QGraphicsObject"):
    #QGraphicsObject = QtGui.QGraphicsObject
#else:
    #class QObjectWorkaround:
        #def __init__(self):
            #self._qObj_ = QtCore.QObject()
        #def connect(self, *args):
            #return QtCore.QObject.connect(self._qObj_, *args)
        #def disconnect(self, *args):
            #return QtCore.QObject.disconnect(self._qObj_, *args)
        #def emit(self, *args):
            #return QtCore.QObject.emit(self._qObj_, *args)
            
    #class QGraphicsObject(QtGui.QGraphicsItem, QObjectWorkaround):
        #def __init__(self, *args):
            #QtGui.QGraphicsItem.__init__(self, *args)
            #QObjectWorkaround.__init__(self)

    
    
class GraphicsObject(QGraphicsObject):
    """Extends QGraphicsObject with a few important functions. 
    (Most of these assume that the object is in a scene with a single view)"""
    
    def __init__(self, *args):
        QGraphicsObject.__init__(self, *args)
        self._view = None
    
    def getViewWidget(self):
        """Return the view widget for this item. If the scene has multiple views, only the first view is returned.
        the view is remembered for the lifetime of the object, so expect trouble if the object is moved to another view."""
        if self._view is None:
            scene = self.scene()
            if scene is None:
                return None
            views = scene.views()
            if len(views) < 1:
                return None
            self._view = weakref.ref(self.scene().views()[0])
        return self._view()
    
    def getBoundingParents(self):
        """Return a list of parents to this item that have child clipping enabled."""
        p = self
        parents = []
        while True:
            p = p.parentItem()
            if p is None:
                break
            if p.flags() & self.ItemClipsChildrenToShape:
                parents.append(p)
        return parents
    
    def viewBounds(self):
        """Return the allowed visible boundaries for this item. Takes into account the viewport as well as any parents that clip."""
        bounds = QtCore.QRectF(0, 0, 1, 1)
        view = self.getViewWidget()
        if view is None:
            return None
        bounds = self.mapRectFromScene(view.visibleRange())
        
        for p in self.getBoundingParents():
            bounds &= self.mapRectFromScene(p.sceneBoundingRect())
            
        return bounds
        
    def viewTransform(self):
        """Return the transform that maps from local coordinates to the item's view coordinates"""
        view = self.getViewWidget()
        if view is None:
            return None
        return self.deviceTransform(view.viewportTransform())

    def pixelVectors(self):
        """Return vectors in local coordinates representing the width and height of a view pixel."""
        vt = self.viewTransform()
        if vt is None:
            return None
        vt = vt.inverted()[0]
        orig = vt.map(QtCore.QPointF(0, 0))
        return vt.map(QtCore.QPointF(1, 0))-orig, vt.map(QtCore.QPointF(0, 1))-orig

    def pixelWidth(self):
        vt = self.viewTransform()
        if vt is None:
            return 0
        vt = vt.inverted()[0]
        return abs((vt.map(QtCore.QPointF(1, 0))-vt.map(QtCore.QPointF(0, 0))).x())
        
    def pixelHeight(self):
        vt = self.viewTransform()
        if vt is None:
            return 0
        vt = vt.inverted()[0]
        return abs((vt.map(QtCore.QPointF(0, 1))-vt.map(QtCore.QPointF(0, 0))).y())

    def mapToView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        return vt.map(obj)
        
    def mapRectToView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        return vt.mapRect(obj)
        
    def mapFromView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        vt = vt.inverted()[0]
        return vt.map(obj)

    def mapRectFromView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        vt = vt.inverted()[0]
        return vt.mapRect(obj)
        
        
        
        

class ImageItem(QtGui.QGraphicsPixmapItem):
    useWeave = True
    
    def __init__(self, image=None, copy=True, parent=None, *args):
        self.qimage = QtGui.QImage()
        self.pixmap = None
        #self.useWeave = True
        self.blackLevel = None
        self.whiteLevel = None
        self.alpha = 1.0
        self.image = None
        self.clipLevel = None
        QtGui.QGraphicsPixmapItem.__init__(self, parent, *args)
        #self.pixmapItem = QtGui.QGraphicsPixmapItem(self)
        if image is not None:
            self.updateImage(image, copy, autoRange=True)
        #self.setCacheMode(QtGui.QGraphicsItem.DeviceCoordinateCache)
        
    def setAlpha(self, alpha):
        self.alpha = alpha
        self.updateImage()
        
    #def boundingRect(self):
        #return self.pixmapItem.boundingRect()
        #return QtCore.QRectF(0, 0, self.qimage.width(), self.qimage.height())
        
    def width(self):
        if self.pixmap is None:
            return None
        return self.pixmap.width()
        
    def height(self):
        if self.pixmap is None:
            return None
        return self.pixmap.height()
        
    def setClipLevel(self, level=None):
        self.clipLevel = level
        
    #def paint(self, p, opt, widget):
        #pass
        #if self.pixmap is not None:
            #p.drawPixmap(0, 0, self.pixmap)
            #print "paint"

    def setLevels(self, white=None, black=None):
        if white is not None:
            self.whiteLevel = white
        if black is not None:
            self.blackLevel = black  
        self.updateImage()

    def updateImage(self, image=None, copy=True, autoRange=False, clipMask=None, white=None, black=None):
        axh = {'x': 0, 'y': 1, 'c': 2}
        #print "Update image", black, white
        if white is not None:
            self.whiteLevel = white
        if black is not None:
            self.blackLevel = black  
        
        
        if image is None:
            if self.image is None:
                return
        else:
            if copy:
                self.image = image.copy()
            else:
                self.image = image
        #print "  image max:", self.image.max(), "min:", self.image.min()
        
        # Determine scale factors
        if autoRange or self.blackLevel is None:
            self.blackLevel = self.image.min()
            self.whiteLevel = self.image.max()
        #print "Image item using", self.blackLevel, self.whiteLevel
        
        if self.blackLevel != self.whiteLevel:
            scale = 255. / (self.whiteLevel - self.blackLevel)
        else:
            scale = 0.
        
        
        ## Recolor and convert to 8 bit per channel
        # Try using weave, then fall back to python
        shape = self.image.shape
        black = float(self.blackLevel)
        try:
            if not ImageItem.useWeave:
                raise Exception('Skipping weave compile')
            sim = ascontiguousarray(self.image)
            sim.shape = sim.size
            im = zeros(sim.shape, dtype=ubyte)
            n = im.size
            
            code = """
            for( int i=0; i<n; i++ ) {
                float a = (sim(i)-black) * (float)scale;
                if( a > 255.0 )
                    a = 255.0;
                else if( a < 0.0 )
                    a = 0.0;
                im(i) = a;
            }
            """
            
            weave.inline(code, ['sim', 'im', 'n', 'black', 'scale'], type_converters=converters.blitz, compiler = 'gcc')
            sim.shape = shape
            im.shape = shape
        except:
            if ImageItem.useWeave:
                ImageItem.useWeave = False
                #sys.excepthook(*sys.exc_info())
                #print "=============================================================================="
                print "Weave compile failed, falling back to slower version."
            self.image.shape = shape
            im = ((self.image - black) * scale).clip(0.,255.).astype(ubyte)
                

        try:
            im1 = empty((im.shape[axh['y']], im.shape[axh['x']], 4), dtype=ubyte)
        except:
            print im.shape, axh
            raise
        alpha = clip(int(255 * self.alpha), 0, 255)
        # Fill image 
        if im.ndim == 2:
            im2 = im.transpose(axh['y'], axh['x'])
            im1[..., 0] = im2
            im1[..., 1] = im2
            im1[..., 2] = im2
            im1[..., 3] = alpha
        elif im.ndim == 3:
            im2 = im.transpose(axh['y'], axh['x'], axh['c'])
            
            for i in range(0, im.shape[axh['c']]):
                im1[..., i] = im2[..., i]
            
            for i in range(im.shape[axh['c']], 3):
                im1[..., i] = 0
            if im.shape[axh['c']] < 4:
                im1[..., 3] = alpha
                
        else:
            raise Exception("Image must be 2 or 3 dimensions")
        #self.im1 = im1
        # Display image
        
        if self.clipLevel is not None or clipMask is not None:
                if clipMask is not None:
                        mask = clipMask.transpose()
                else:
                        mask = (self.image < self.clipLevel).transpose()
                im1[..., 0][mask] *= 0.5
                im1[..., 1][mask] *= 0.5
                im1[..., 2][mask] = 255
        #print "Final image:", im1.dtype, im1.min(), im1.max(), im1.shape
        self.ims = im1.tostring()  ## Must be held in memory here because qImage won't do it for us :(
        qimage = QtGui.QImage(self.ims, im1.shape[1], im1.shape[0], QtGui.QImage.Format_ARGB32)
        self.pixmap = QtGui.QPixmap.fromImage(qimage)
        ##del self.ims
        self.setPixmap(self.pixmap)
        self.update()
        
    def getPixmap(self):
        return self.pixmap.copy()

        

class PlotCurveItem(GraphicsObject):
    """Class representing a single plot curve."""
    def __init__(self, y=None, x=None, copy=False, pen=None, shadow=None, parent=None, color=None):
        GraphicsObject.__init__(self, parent)
        self.free()
        #self.dispPath = None
        
        if pen is None:
            if color is None:
                pen = QtGui.QPen(QtGui.QColor(200, 200, 200))
            else:
                pen = QtGui.QPen(color)
        self.pen = pen
        
        self.shadow = shadow
        if y is not None:
            self.updateData(y, x, copy)
        #self.setCacheMode(QtGui.QGraphicsItem.DeviceCoordinateCache)
        
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
            
        #self.fps = None
        
    def getData(self):
        if self.xData is None:
            return (None, None)
        if self.xDisp is None:
            nanMask = isnan(self.xData) | isnan(self.yData)
            x = self.xData[~nanMask]
            y = self.yData[~nanMask]
            ds = self.opts['downsample']
            if ds > 1:
                x = x[::ds]
                y = resample(y[:len(x)*ds], len(x))
            if self.opts['spectrumMode']:
                f = fft(y) / len(y)
                y = abs(f[1:len(f)/2])
                dt = x[-1] - x[0]
                x = linspace(0, 0.5*len(x)/dt, len(y))
            if self.opts['logMode'][0]:
                x = log10(x)
            if self.opts['logMode'][1]:
                y = log10(y)
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
        
    def getRange(self, ax, frac=1.0):
        #print "getRange", ax, frac
        (x, y) = self.getData()
        if x is None or len(x) == 0:
            return (0, 1)
            
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
            #bins = 1000
            #h = histogram(d, bins)
            #s = len(d) * (1.0-frac)
            #mnTot = mxTot = 0
            #mnInd = mxInd = 0
            #for i in range(bins):
                #mnTot += h[0][i]
                #if mnTot > s:
                    #mnInd = i
                    #break
            #for i in range(bins):
                #mxTot += h[0][-i-1]
                #if mxTot > s:
                    #mxInd = -i-1
                    #break
            ##print mnInd, mxInd, h[1][mnInd], h[1][mxInd]
            #return(h[1][mnInd], h[1][mxInd])
                
            
            
        
    def setMeta(self, data):
        self.metaData = data
        
    def meta(self):
        return self.metaData
        
    def setPen(self, pen):
        self.pen = pen
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
        self.shadow = pen
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
        if isinstance(data, list):
            data = array(data)
        if isinstance(x, list):
            x = array(x)
        if not isinstance(data, ndarray) or data.ndim > 2:
            raise Exception("Plot data must be 1 or 2D ndarray (data shape is %s)" % str(data.shape))
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
            
        self.prepareGeometryChange()
        if copy:
            self.yData = y.copy()
        else:
            self.yData = y
            
        if copy and x is not None:
            self.xData = x.copy()
        else:
            self.xData = x
        
        if x is None:
            self.xData = arange(0, self.yData.shape[0])

        if self.xData.shape != self.yData.shape:
            raise Exception("X and Y arrays must be the same shape--got %s and %s." % (str(x.shape), str(y.shape)))
        
        self.path = None
        #self.specPath = None
        self.xDisp = self.yDisp = None
        self.update()
        self.emit(QtCore.SIGNAL('plotChanged'), self)
        
    def generatePath(self, x, y):
        path = QtGui.QPainterPath()
        
        ## Create all vertices in path. The method used below creates a binary format so that all 
        ## vertices can be read in at once. This binary format may change in future versions of Qt, 
        ## so the original (slower) method is left here for emergencies:
        #self.path.moveTo(x[0], y[0])
        #for i in range(1, y.shape[0]):
            #self.path.lineTo(x[i], y[i])
            
        ## Speed this up using >> operator
        ## Format is:
        ##    numVerts(i4)   0(i4)
        ##    x(f8)   y(f8)   0(i4)    <-- 0 means this vertex does not connect
        ##    x(f8)   y(f8)   1(i4)    <-- 1 means this vertex connects to the previous vertex
        ##    ...
        ##    0(i4)
        ##
        ## All values are big endian--pack using struct.pack('>d') or struct.pack('>i')
        #
        n = x.shape[0]
        # create empty array, pad with extra space on either end
        arr = empty(n+2, dtype=[('x', '>f8'), ('y', '>f8'), ('c', '>i4')])
        # write first two integers
        arr.data[12:20] = struct.pack('>ii', n, 0)
        # Fill array with vertex values
        arr[1:-1]['x'] = x
        arr[1:-1]['y'] = y
        arr[1:-1]['c'] = 1
        # write last 0
        lastInd = 20*(n+1) 
        arr.data[lastInd:lastInd+4] = struct.pack('>i', 0)
        
        # create datastream object and stream into path
        buf = QtCore.QByteArray(arr.data[12:lastInd+4])  # I think one unnecessary copy happens here
        ds = QtCore.QDataStream(buf)
        ds >> path
        
        return path
        
    def boundingRect(self):
        (x, y) = self.getData()
        if x is None or y is None or len(x) == 0 or len(y) == 0:
            return QtCore.QRectF()
            
            
        if self.shadow is not None:
            lineWidth = (max(self.pen.width(), self.shadow.width()) + 1)
        else:
            lineWidth = (self.pen.width()+1)
            
        
        pixels = self.pixelVectors()
        xmin = x.min() - pixels[0].x() * lineWidth
        xmax = x.max() + pixels[0].x() * lineWidth
        ymin = y.min() - abs(pixels[1].y()) * lineWidth
        ymax = y.max() + abs(pixels[1].y()) * lineWidth
        
            
        return QtCore.QRectF(xmin, ymin, xmax-xmin, ymax-ymin)

    def paint(self, p, opt, widget):
        if self.xData is None:
            return
        #if self.opts['spectrumMode']:
            #if self.specPath is None:
                
                #self.specPath = self.generatePath(*self.getData())
            #path = self.specPath
        #else:
        if self.path is None:
            self.path = self.generatePath(*self.getData())
        path = self.path
            
        if self.shadow is not None:
            sp = QtGui.QPen(self.shadow)
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
            
        if self.shadow is not None:
            p.setPen(sp)
            p.drawPath(path)
        p.setPen(cp)
        p.drawPath(path)
        
        #p.setPen(QtGui.QPen(QtGui.QColor(255,0,0)))
        #p.drawRect(self.boundingRect())
        
        
    def free(self):
        self.xData = None  ## raw values
        self.yData = None
        self.xDisp = None  ## display values (after log / fft)
        self.yDisp = None
        self.path = None
        #del self.xData, self.yData, self.xDisp, self.yDisp, self.path
        
        
class ROIPlotItem(PlotCurveItem):
    def __init__(self, roi, data, img, axes=(0,1), xVals=None, color=None):
        self.roi = roi
        self.roiData = data
        self.roiImg = img
        self.axes = axes
        self.xVals = xVals
        PlotCurveItem.__init__(self, self.getRoiData(), x=self.xVals, color=color)
        roi.connect(QtCore.SIGNAL('regionChanged'), self.roiChangedEvent)
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




class UIGraphicsItem(GraphicsObject):
    """Base class for graphics items with boundaries relative to a GraphicsView widget"""
    def __init__(self, view, bounds=None):
        GraphicsObject.__init__(self)
        self._view = weakref.ref(view)
        if bounds is None:
            self._bounds = QtCore.QRectF(0, 0, 1, 1)
        else:
            self._bounds = bounds
        self._viewRect = self._view().rect()
        self._viewTransform = self.viewTransform()
        self.setNewBounds()
        QtCore.QObject.connect(view, QtCore.SIGNAL('viewChanged'), self.viewChangedEvent)
        
    def viewRect(self):
        """Return the viewport widget rect"""
        return self._view().rect()
    
    def viewTransform(self):
        """Returns a matrix that maps viewport coordinates onto scene coordinates"""
        if self._view() is None:
            return QtGui.QTransform()
        else:
            return self._view().viewportTransform()
        
    def boundingRect(self):
        if self._view() is None:
            self.bounds = self._bounds
        else:
            vr = self._view().rect()
            tr = self.viewTransform()
            if vr != self._viewRect or tr != self._viewTransform:
                #self.viewChangedEvent(vr, self._viewRect)
                self._viewRect = vr
                self._viewTransform = tr
                self.setNewBounds()
        #print "viewRect", self._viewRect.x(), self._viewRect.y(), self._viewRect.width(), self._viewRect.height()
        #print "bounds", self.bounds.x(), self.bounds.y(), self.bounds.width(), self.bounds.height()
        return self.bounds

    def setNewBounds(self):
        bounds = QtCore.QRectF(
            QtCore.QPointF(self._bounds.left()*self._viewRect.width(), self._bounds.top()*self._viewRect.height()),
            QtCore.QPointF(self._bounds.right()*self._viewRect.width(), self._bounds.bottom()*self._viewRect.height())
        )
        bounds.adjust(0.5, 0.5, 0.5, 0.5)
        self.bounds = self.viewTransform().inverted()[0].mapRect(bounds)
        self.prepareGeometryChange()

    def viewChangedEvent(self):
        """Called when the view widget is resized"""
        self.boundingRect()
        self.update()
        
    def unitRect(self):
        return self.viewTransform().inverted()[0].mapRect(QtCore.QRectF(0, 0, 1, 1))

    def paint(self, *args):
        pass




class LabelItem(QtGui.QGraphicsWidget):
    def __init__(self, text, parent=None, **args):
        QtGui.QGraphicsWidget.__init__(self, parent)
        self.item = QtGui.QGraphicsTextItem(self)
        self.opts = args
        if 'color' not in args:
            self.opts['color'] = 'CCC'
        else:
            if isinstance(args['color'], QtGui.QColor):
                self.opts['color'] = colorStr(args['color'])[:6]
        self.sizeHint = {}
        self.setText(text)
        
            
    def setAttr(self, attr, value):
        """Set default text properties. See setText() for accepted parameters."""
        self.opts[attr] = value
        
    def setText(self, text, **args):
        """Set the text and text properties in the label. Accepts optional arguments for auto-generating
        a CSS style string:
           color:   string (example: 'CCFF00')
           size:    string (example: '8pt')
           bold:    boolean
           italic:  boolean
           """
        self.text = text
        opts = self.opts.copy()
        for k in args:
            opts[k] = args[k]
        
        optlist = []
        if 'color' in opts:
            optlist.append('color: #' + opts['color'])
        if 'size' in opts:
            optlist.append('font-size: ' + opts['size'])
        if 'bold' in opts and opts['bold'] in [True, False]:
            optlist.append('font-weight: ' + {True:'bold', False:'normal'}[opts['bold']])
        if 'italic' in opts and opts['italic'] in [True, False]:
            optlist.append('font-style: ' + {True:'italic', False:'normal'}[opts['italic']])
        full = "<span style='%s'>%s</span>" % ('; '.join(optlist), text)
        #print full
        self.item.setHtml(full)
        self.updateMin()
        
    def resizeEvent(self, ev):
        c1 = self.boundingRect().center()
        c2 = self.item.mapToParent(self.item.boundingRect().center()) # + self.item.pos()
        dif = c1 - c2
        self.item.moveBy(dif.x(), dif.y())
        #print c1, c2, dif, self.item.pos()
        
    def setAngle(self, angle):
        self.angle = angle
        self.item.resetMatrix()
        self.item.rotate(angle)
        self.updateMin()
        
    def updateMin(self):
        bounds = self.item.mapRectToParent(self.item.boundingRect())
        self.setMinimumWidth(bounds.width())
        self.setMinimumHeight(bounds.height())
        #print self.text, bounds.width(), bounds.height()
        
        #self.sizeHint = {
            #QtCore.Qt.MinimumSize: (bounds.width(), bounds.height()),
            #QtCore.Qt.PreferredSize: (bounds.width(), bounds.height()),
            #QtCore.Qt.MaximumSize: (bounds.width()*2, bounds.height()*2),
            #QtCore.Qt.MinimumDescent: (0, 0)  ##?? what is this?
        #}
            
        
    #def sizeHint(self, hint, constraint):
        #return self.sizeHint[hint]
        




class ScaleItem(QtGui.QGraphicsWidget):
    def __init__(self, orientation, pen=None, linkView=None, parent=None):
        """GraphicsItem showing a single plot axis with ticks, values, and label.
        Can be configured to fit on any side of a plot, and can automatically synchronize its displayed scale with ViewBox items.
        Ticks can be extended to make a grid."""
        QtGui.QGraphicsWidget.__init__(self, parent)
        self.label = QtGui.QGraphicsTextItem(self)
        self.orientation = orientation
        if orientation not in ['left', 'right', 'top', 'bottom']:
            raise Exception("Orientation argument must be one of 'left', 'right', 'top', or 'bottom'.")
        if orientation in ['left', 'right']:
            #self.setMinimumWidth(25)
            #self.setSizePolicy(QtGui.QSizePolicy(
                #QtGui.QSizePolicy.Minimum,
                #QtGui.QSizePolicy.Expanding
            #))
            self.label.rotate(-90)
        #else:
            #self.setMinimumHeight(50)
            #self.setSizePolicy(QtGui.QSizePolicy(
                #QtGui.QSizePolicy.Expanding,
                #QtGui.QSizePolicy.Minimum
            #))
        #self.drawLabel = False
        
        self.labelText = ''
        self.labelUnits = ''
        self.labelUnitPrefix=''
        self.labelStyle = {'color': '#CCC'}
        
        self.textHeight = 18
        self.tickLength = 10
        self.scale = 1.0
        self.autoScale = True
            
        self.setRange(0, 1)
        
        if pen is None:
            pen = QtGui.QPen(QtGui.QColor(100, 100, 100))
        self.setPen(pen)
        
        self.linkedView = None
        if linkView is not None:
            self.linkToView(linkView)
            
        self.showLabel(False)
        
        self.grid = False
            
        
    def setGrid(self, grid):
        """Set the alpha value for the grid, or False to disable."""
        self.grid = grid
        self.update()
        
        
    def resizeEvent(self, ev=None):
        #s = self.size()
        
        ## Set the position of the label
        nudge = 5
        br = self.label.boundingRect()
        p = QtCore.QPointF(0, 0)
        if self.orientation == 'left':
            p.setY(int(self.size().height()/2 + br.width()/2))
            p.setX(-nudge)
            #s.setWidth(10)
        elif self.orientation == 'right':
            #s.setWidth(10)
            p.setY(int(self.size().height()/2 + br.width()/2))
            p.setX(int(self.size().width()-br.height()+nudge))
        elif self.orientation == 'top':
            #s.setHeight(10)
            p.setY(-nudge)
            p.setX(int(self.size().width()/2. - br.width()/2.))
        elif self.orientation == 'bottom':
            p.setX(int(self.size().width()/2. - br.width()/2.))
            #s.setHeight(10)
            p.setY(int(self.size().height()-br.height()+nudge))
        #self.label.resize(s)
        self.label.setPos(p)
        
    def showLabel(self, show=True):
        #self.drawLabel = show
        self.label.setVisible(show)
        if self.orientation in ['left', 'right']:
            self.setWidth()
        else:
            self.setHeight()
        if self.autoScale:
            self.setScale()
        
    def setLabel(self, text=None, units=None, unitPrefix=None, **args):
        if text is not None:
            self.labelText = text
            self.showLabel()
        if units is not None:
            self.labelUnits = units
            self.showLabel()
        if unitPrefix is not None:
            self.labelUnitPrefix = unitPrefix
        if len(args) > 0:
            self.labelStyle = args
        self.label.setHtml(self.labelString())
        self.resizeEvent()
        self.update()
            
    def labelString(self):
        if self.labelUnits == '':
            if self.scale == 1.0:
                units = ''
            else:
                units = u'(x%g)' % (1.0/self.scale)
        else:
            #print repr(self.labelUnitPrefix), repr(self.labelUnits)
            units = u'(%s%s)' % (self.labelUnitPrefix, self.labelUnits)
            
        s = u'%s %s' % (self.labelText, units)
        
        style = ';'.join(['%s: "%s"' % (k, self.labelStyle[k]) for k in self.labelStyle])
        
        return u"<span style='%s'>%s</span>" % (style, s)
        
    def setHeight(self, h=None):
        if h is None:
            h = self.textHeight + self.tickLength
            if self.label.isVisible():
                h += self.textHeight
        self.setMaximumHeight(h)
        self.setMinimumHeight(h)
        
        
    def setWidth(self, w=None):
        if w is None:
            w = self.tickLength + 40
            if self.label.isVisible():
                w += self.textHeight
        self.setMaximumWidth(w)
        self.setMinimumWidth(w)
        
    def setPen(self, pen):
        self.pen = pen
        self.update()
        
    def setScale(self, scale=None):
        if scale is None:
            #if self.drawLabel:  ## If there is a label, then we are free to rescale the values 
            if self.label.isVisible():
                d = self.range[1] - self.range[0]
                #pl = 1-int(log10(d))
                #scale = 10 ** pl
                (scale, prefix) = siScale(d / 2.)
                if self.labelUnits == '' and prefix in ['k', 'm']:  ## If we are not showing units, wait until 1e6 before scaling.
                    scale = 1.0
                    prefix = ''
                self.setLabel(unitPrefix=prefix)
            else:
                scale = 1.0
        
        
        if scale != self.scale:
            self.scale = scale
            self.setLabel()
            self.update()
        
    def setRange(self, mn, mx):
        if mn in [nan, inf, -inf] or mx in [nan, inf, -inf]:
            raise Exception("Not setting range to [%s, %s]" % (str(mn), str(mx)))
        self.range = [mn, mx]
        if self.autoScale:
            self.setScale()
        self.update()
        
    def linkToView(self, view):
        if self.orientation in ['right', 'left']:
            signal = QtCore.SIGNAL('yRangeChanged')
        else:
            signal = QtCore.SIGNAL('xRangeChanged')
            
        if self.linkedView is not None:
            QtCore.QObject.disconnect(view, signal, self.linkedViewChanged)
        self.linkedView = view
        QtCore.QObject.connect(view, signal, self.linkedViewChanged)
        
    def linkedViewChanged(self, _, newRange):
        self.setRange(*newRange)
        
    def boundingRect(self):
        if self.linkedView is None or self.grid is False:
            return self.mapRectFromParent(self.geometry())
        else:
            return self.mapRectFromParent(self.geometry()) | self.mapRectFromScene(self.linkedView.mapRectToScene(self.linkedView.boundingRect()))
        
    def paint(self, p, opt, widget):
        p.setPen(self.pen)
        
        #bounds = self.boundingRect()
        bounds = self.mapRectFromParent(self.geometry())
        
        if self.linkedView is None or self.grid is False:
            tbounds = bounds
        else:
            tbounds = self.mapRectFromScene(self.linkedView.mapRectToScene(self.linkedView.boundingRect()))
        
        if self.orientation == 'left':
            p.drawLine(bounds.topRight(), bounds.bottomRight())
            tickStart = tbounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == 'right':
            p.drawLine(bounds.topLeft(), bounds.bottomLeft())
            tickStart = tbounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == 'top':
            p.drawLine(bounds.bottomLeft(), bounds.bottomRight())
            tickStart = tbounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == 'bottom':
            p.drawLine(bounds.topLeft(), bounds.topRight())
            tickStart = tbounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1
        
        ## Determine optimal tick spacing
        #intervals = [1., 2., 5., 10., 20., 50.]
        #intervals = [1., 2.5, 5., 10., 25., 50.]
        intervals = [1., 2., 10., 20., 100.]
        dif = abs(self.range[1] - self.range[0])
        if dif == 0.0:
            return
        #print "dif:", dif
        pw = 10 ** (floor(log10(dif))-1)
        for i in range(len(intervals)):
            i1 = i
            if dif / (pw*intervals[i]) < 10:
                break
        
        textLevel = i1  ## draw text at this scale level
        
        #print "range: %s   dif: %f   power: %f  interval: %f   spacing: %f" % (str(self.range), dif, pw, intervals[i1], sp)
        
        #print "  start at %f,  %d ticks" % (start, num)
        
        
        if axis == 0:
            xs = -bounds.height() / dif
        else:
            xs = bounds.width() / dif
            
        ## draw ticks and text
        for i in [i1, i1+1, i1+2]:  ## draw three different intervals
            if i > len(intervals):
                continue
            ## spacing for this interval
            sp = pw*intervals[i]
            
            ## determine starting tick
            start = ceil(self.range[0] / sp) * sp
            
            ## determine number of ticks
            num = int(dif / sp) + 1
            
            ## last tick value
            last = start + sp * num
            
            ## Number of decimal places to print
            maxVal = max(abs(start), abs(last))
            places = max(0, 1-int(log10(sp*self.scale)))
        
            ## length of tick
            h = min(self.tickLength, (self.tickLength*3 / num) - 1.)
            
            ## alpha
            a = min(255, (765. / num) - 1.)
            
            if axis == 0:
                offset = self.range[0] * xs - bounds.height()
            else:
                offset = self.range[0] * xs
            
            for j in range(num):
                v = start + sp * j
                x = (v * xs) - offset
                p1 = [0, 0]
                p2 = [0, 0]
                p1[axis] = tickStart
                p2[axis] = tickStop + h*tickDir
                p1[1-axis] = p2[1-axis] = x
                
                if p1[1-axis] > [bounds.width(), bounds.height()][1-axis]:
                    continue
                if p1[1-axis] < 0:
                    continue
                p.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100, a)))
                p.drawLine(Point(p1), Point(p2))
                if i == textLevel:
                    if abs(v) < .001 or abs(v) >= 10000:
                        vstr = "%g" % (v * self.scale)
                    else:
                        vstr = ("%%0.%df" % places) % (v * self.scale)
                        
                    textRect = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignCenter, vstr)
                    height = textRect.height()
                    self.textHeight = height
                    if self.orientation == 'left':
                        textFlags = QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter
                        rect = QtCore.QRectF(tickStop-100, x-(height/2), 100-self.tickLength, height)
                    elif self.orientation == 'right':
                        textFlags = QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter
                        rect = QtCore.QRectF(tickStop+self.tickLength, x-(height/2), 100-self.tickLength, height)
                    elif self.orientation == 'top':
                        textFlags = QtCore.Qt.AlignCenter|QtCore.Qt.AlignBottom
                        rect = QtCore.QRectF(x-100, tickStop-self.tickLength-height, 200, height)
                    elif self.orientation == 'bottom':
                        textFlags = QtCore.Qt.AlignCenter|QtCore.Qt.AlignTop
                        rect = QtCore.QRectF(x-100, tickStop+self.tickLength, 200, height)
                    
                    p.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100)))
                    p.drawText(rect, textFlags, vstr)
                    #p.drawRect(rect)
        
        ## Draw label
        #if self.drawLabel:
            #height = self.size().height()
            #width = self.size().width()
            #if self.orientation == 'left':
                #p.translate(0, height)
                #p.rotate(-90)
                #rect = QtCore.QRectF(0, 0, height, self.textHeight)
                #textFlags = QtCore.Qt.AlignCenter|QtCore.Qt.AlignTop
            #elif self.orientation == 'right':
                #p.rotate(10)
                #rect = QtCore.QRectF(0, 0, height, width)
                #textFlags = QtCore.Qt.AlignCenter|QtCore.Qt.AlignBottom
                ##rect = QtCore.QRectF(tickStart+self.tickLength, x-(height/2), 100-self.tickLength, height)
            #elif self.orientation == 'top':
                #rect = QtCore.QRectF(0, 0, width, height)
                #textFlags = QtCore.Qt.AlignCenter|QtCore.Qt.AlignTop
                ##rect = QtCore.QRectF(x-100, tickStart-self.tickLength-height, 200, height)
            #elif self.orientation == 'bottom':
                #rect = QtCore.QRectF(0, 0, width, height)
                #textFlags = QtCore.Qt.AlignCenter|QtCore.Qt.AlignBottom
                ##rect = QtCore.QRectF(x-100, tickStart+self.tickLength, 200, height)
            #p.drawText(rect, textFlags, self.labelString())
            ##p.drawRect(rect)
        
    def show(self):
        
        if self.orientation in ['left', 'right']:
            self.setWidth()
        else:
            self.setHeight()
        QtGui.QGraphicsWidget.show(self)
        
    def hide(self):
        if self.orientation in ['left', 'right']:
            self.setWidth(0)
        else:
            self.setHeight(0)
        QtGui.QGraphicsWidget.hide(self)
        
    
        
        
        


class ViewBox(QtGui.QGraphicsWidget):
    """Box that allows internal scaling/panning of children by mouse drag. Not compatible with GraphicsView having the same functionality."""
    def __init__(self, parent=None):
        QtGui.QGraphicsWidget.__init__(self, parent)
        #self.gView = view
        #self.showGrid = showGrid
        self.range = [[0,1], [0,1]]   ## child coord. range visible [[xmin, xmax], [ymin, ymax]]
        
        self.aspectLocked = False
        self.setFlag(QtGui.QGraphicsItem.ItemClipsChildrenToShape)
        #self.setFlag(QtGui.QGraphicsItem.ItemClipsToShape)
        
        #self.childGroup = QtGui.QGraphicsItemGroup(self)
        self.childGroup = ItemGroup(self)
        self.currentScale = Point(1, 1)
        
        self.yInverted = False
        #self.invertY()
        self.setZValue(-100)
        #self.picture = None
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))
        
        self.drawFrame = True
        
        self.mouseEnabled = [True, True]
    
    def setMouseEnabled(self, x, y):
        self.mouseEnabled = [x, y]
    
    def addItem(self, item):
        if item.zValue() < self.zValue():
            item.setZValue(self.zValue()+1)
        item.setParentItem(self.childGroup)
        #print "addItem:", item, item.boundingRect()
        
    def removeItem(self, item):
        self.scene().removeItem(item)
        
    def resizeEvent(self, ev):
        #self.setRange(self.range, padding=0)
        self.updateMatrix()
        

    def viewRect(self):
        try:
            return QtCore.QRectF(self.range[0][0], self.range[1][0], self.range[0][1]-self.range[0][0], self.range[1][1] - self.range[1][0])
        except:
            print "make qrectf failed:", self.range
            raise
    
    def updateMatrix(self):
        #print "udpateMatrix:"
        #print "  range:", self.range
        vr = self.viewRect()
        translate = Point(vr.center())
        bounds = self.boundingRect()
        #print "  bounds:", bounds
        if vr.height() == 0 or vr.width() == 0:
            return
        scale = Point(bounds.width()/vr.width(), bounds.height()/vr.height())
        #print "  scale:", scale
        m = QtGui.QMatrix()
        
        ## First center the viewport at 0
        self.childGroup.resetMatrix()
        center = self.transform().inverted()[0].map(bounds.center())
        #print "  transform to center:", center
        if self.yInverted:
            m.translate(center.x(), -center.y())
            #print "  inverted; translate", center.x(), center.y()
        else:
            m.translate(center.x(), center.y())
            #print "  not inverted; translate", center.x(), -center.y()
            
        ## Now scale and translate properly
        if self.aspectLocked:
            scale = Point(scale.min())
        if not self.yInverted:
            scale = scale * Point(1, -1)
        m.scale(scale[0], scale[1])
        #print "  scale:", scale
        st = translate
        m.translate(-st[0], -st[1])
        #print "  translate:", st
        self.childGroup.setMatrix(m)
        self.currentScale = scale
        
    def invertY(self, b=True):
        self.yInverted = b
        self.updateMatrix()
        
    def childTransform(self):
        m = self.childGroup.transform()
        m1 = QtGui.QTransform()
        m1.translate(self.childGroup.pos().x(), self.childGroup.pos().y())
        return m*m1
    
    def setAspectLocked(self, s):
        self.aspectLocked = s

    def viewScale(self):
        pr = self.range
        #print "viewScale:", self.range
        xd = pr[0][1] - pr[0][0]
        yd = pr[1][1] - pr[1][0]
        if xd == 0 or yd == 0:
            print "Warning: 0 range in view:", xd, yd
            return array([1,1])
        
        #cs = self.canvas().size()
        cs = self.boundingRect()
        scale = array([cs.width() / xd, cs.height() / yd])
        #print "view scale:", scale
        return scale

    def scaleBy(self, s, center=None):
        #print "scaleBy", s, center
        xr, yr = self.range
        if center is None:
            xc = (xr[1] + xr[0]) * 0.5
            yc = (yr[1] + yr[0]) * 0.5
        else:
            (xc, yc) = center
        
        x1 = xc + (xr[0]-xc) * s[0]
        x2 = xc + (xr[1]-xc) * s[0]
        y1 = yc + (yr[0]-yc) * s[1]
        y2 = yc + (yr[1]-yc) * s[1]
        
        #print xr, xc, s, (xr[0]-xc) * s[0], (xr[1]-xc) * s[0]
        #print [[x1, x2], [y1, y2]]
        
        
        
        self.setXRange(x1, x2, update=False, padding=0)
        self.setYRange(y1, y2, padding=0)
        #print self.range
        
    def translateBy(self, t, viewCoords=False):
        t = t.astype(float)
        #print "translate:", t, self.viewScale()
        if viewCoords:  ## scale from pixels
            t /= self.viewScale()
        xr, yr = self.range
        #self.setAxisScale(self.xBottom, xr[0] + t[0], xr[1] + t[0])
        #self.setAxisScale(self.yLeft, yr[0] + t[1], yr[1] + t[1])
        #print xr, yr, t
        self.setXRange(xr[0] + t[0], xr[1] + t[0], update=False, padding=0)
        self.setYRange(yr[0] + t[1], yr[1] + t[1], padding=0)
        #self.replot(autoRange=False)
        #self.updateMatrix()
        
        
    def mouseMoveEvent(self, ev):
        pos = array([ev.pos().x(), ev.pos().y()])
        dif = pos - self.mousePos
        dif *= -1
        self.mousePos = pos
        
        ## Ignore axes if mouse is disabled
        mask = array(self.mouseEnabled, dtype=float)
        
        ## Scale or translate based on mouse button
        if ev.buttons() & QtCore.Qt.LeftButton:
            if not self.yInverted:
                mask *= array([1, -1])
            tr = dif*mask
            self.translateBy(tr, viewCoords=True)
            self.emit(QtCore.SIGNAL('rangeChangedManually'), self.mouseEnabled)
            ev.accept()
        elif ev.buttons() & QtCore.Qt.RightButton:
            dif = ev.screenPos() - ev.lastScreenPos()
            dif = array([dif.x(), dif.y()])
            dif[0] *= -1
            s = ((mask * 0.02) + 1) ** dif
            #print mask, dif, s
            center = Point(self.childGroup.transform().inverted()[0].map(ev.buttonDownPos(QtCore.Qt.RightButton)))
            self.scaleBy(s, center)
            self.emit(QtCore.SIGNAL('rangeChangedManually'), self.mouseEnabled)
            ev.accept()
        else:
            ev.ignore()
        
    def mousePressEvent(self, ev):
        self.mousePos = array([ev.pos().x(), ev.pos().y()])
        self.pressPos = self.mousePos.copy()
        ev.accept()
        
    def mouseReleaseEvent(self, ev):
        pos = array([ev.pos().x(), ev.pos().y()])
        #if sum(abs(self.pressPos - pos)) < 3:  ## Detect click
            #if ev.button() == QtCore.Qt.RightButton:
                #self.ctrlMenu.popup(self.mapToGlobal(ev.pos()))
        self.mousePos = pos
        ev.accept()
        
    def setRange(self, ax, min, max, padding=0.02, update=True):
        if ax == 0:
            self.setXRange(min, max, update=update, padding=padding)
        else:
            self.setYRange(min, max, update=update, padding=padding)
            
    def setYRange(self, min, max, update=True, padding=0.02):
        #print "setYRange:", min, max
        if min == max:   ## If we requested no range, try to preserve previous scale. Otherwise just pick an arbitrary scale.
            dy = self.range[1][1] - self.range[1][0]
            if dy == 0:
                dy = 1
            min -= dy*0.5
            max += dy*0.5
            #raise Exception("Tried to set range with 0 width.")
        if any(isnan([min, max])) or any(isinf([min, max])):
            raise Exception("Not setting range [%s, %s]" % (str(min), str(max)))
            
        padding = (max-min) * padding
        min -= padding
        max += padding
        if self.range[1] != [min, max]:
            #self.setAxisScale(self.yLeft, min, max)
            self.range[1] = [min, max]
            #self.ctrl.yMinText.setText('%g' % min)
            #self.ctrl.yMaxText.setText('%g' % max)
            self.emit(QtCore.SIGNAL('yRangeChanged'), self, (min, max))
            self.emit(QtCore.SIGNAL('viewChanged'), self)
        if update:
            self.updateMatrix()
        
    def setXRange(self, min, max, update=True, padding=0.02):
        #print "setXRange:", min, max
        if min == max:
            dx = self.range[0][1] - self.range[0][0]
            if dx == 0:
                dx = 1
            min -= dx*0.5
            max += dx*0.5
            #print "Warning: Tried to set range with 0 width."
            #raise Exception("Tried to set range with 0 width.")
        if any(isnan([min, max])) or any(isinf([min, max])):
            raise Exception("Not setting range [%s, %s]" % (str(min), str(max)))
        padding = (max-min) * padding
        min -= padding
        max += padding
        if self.range[0] != [min, max]:
            #self.setAxisScale(self.xBottom, min, max)
            self.range[0] = [min, max]
            #self.ctrl.xMinText.setText('%g' % min)
            #self.ctrl.xMaxText.setText('%g' % max)
            self.emit(QtCore.SIGNAL('xRangeChanged'), self, (min, max))
            self.emit(QtCore.SIGNAL('viewChanged'), self)
        if update:
            self.updateMatrix()

    def autoRange(self, padding=0.02):
        br = self.childGroup.childrenBoundingRect()
        #print br
        #px = br.width() * padding
        #py = br.height() * padding
        self.setXRange(br.left(), br.right(), padding=padding, update=False)
        self.setYRange(br.top(), br.bottom(), padding=padding)
        
    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.size().width(), self.size().height())
        
    def paint(self, p, opt, widget):
        if self.drawFrame:
            bounds = self.boundingRect()
            p.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100)))
            #p.fillRect(bounds, QtGui.QColor(0, 0, 0))
            p.drawRect(bounds)


class InfiniteLine(GraphicsObject):
    def __init__(self, view, pos=0, angle=90, pen=None, movable=False, bounds=None):
        GraphicsObject.__init__(self)
        self.bounds = QtCore.QRectF()   ## graphicsitem boundary
        
        if bounds is None:              ## allowed value boundaries for orthogonal lines
            self.maxRange = [None, None]
        else:
            self.maxRange = bounds
        self.movable = movable
        self.view = weakref.ref(view)
        self.p = [0, 0]
        self.setAngle(angle)
        self.setPos(pos)
            
        if movable:
            self.setAcceptHoverEvents(True)
            

        
        if pen is None:
            pen = QtGui.QPen(QtGui.QColor(200, 200, 100))
        self.setPen(pen)
        self.currentPen = self.pen
        #self.setFlag(self.ItemSendsScenePositionChanges)
        #for p in self.getBoundingParents():
            #QtCore.QObject.connect(p, QtCore.SIGNAL('viewChanged'), self.updateLine)
        QtCore.QObject.connect(self.view(), QtCore.SIGNAL('viewChanged'), self.updateLine)
        
    def setBounds(self, bounds):
        self.maxRange = bounds
        self.setValue(self.value())
        
    def hoverEnterEvent(self, ev):
        self.currentPen = QtGui.QPen(QtGui.QColor(255, 0,0))
        self.update()
        ev.ignore()

    def hoverLeaveEvent(self, ev):
        self.currentPen = self.pen
        self.update()
        ev.ignore()
        
    def setPen(self, pen):
        self.pen = pen
        self.currentPen = self.pen
        
    def setAngle(self, angle):
        self.angle = ((angle+45) % 180) - 45   ##  -45 <= angle < 135
        self.updateLine()
        
    def setPos(self, pos):
        if type(pos) in [list, tuple]:
            newPos = pos
        elif isinstance(pos, QtCore.QPointF):
            newPos = [pos.x(), pos.y()]
        else:
            if self.angle == 90:
                newPos = [pos, 0]
            elif self.angle == 0:
                newPos = [0, pos]
            else:
                raise Exception("Must specify 2D coordinate for non-orthogonal lines.")
            
        ## check bounds (only works for orthogonal lines)
        if self.angle == 90:
            if self.maxRange[0] is not None:    
                newPos[0] = max(newPos[0], self.maxRange[0])
            if self.maxRange[1] is not None:
                newPos[0] = min(newPos[0], self.maxRange[1])
        elif self.angle == 0:
            if self.maxRange[0] is not None:
                newPos[1] = max(newPos[1], self.maxRange[0])
            if self.maxRange[1] is not None:
                newPos[1] = min(newPos[1], self.maxRange[1])
            
            
        if self.p != newPos:
            self.p = newPos
            self.updateLine()
            self.emit(QtCore.SIGNAL('positionChanged'), self)

    def getXPos(self):
        return self.p[0]
        
    def getYPos(self):
        return self.p[1]
        
    def getPos(self):
        return self.p

    def value(self):
        if self.angle%180 == 0:
            return self.getYPos()
        elif self.angle%180 == 90:
            return self.getXPos()
        else:
            return self.getPos()
                
    def setValue(self, v):
        self.setPos(v)

    ## broken in 4.7
    #def itemChange(self, change, val):
        #if change in [self.ItemScenePositionHasChanged, self.ItemSceneHasChanged]:
            #self.updateLine()
            #print "update", change
            #print self.getBoundingParents()
        #else:
            #print "ignore", change
        #return GraphicsObject.itemChange(self, change, val)
                
    def updateLine(self):

        #unit = QtCore.QRect(0, 0, 10, 10)
        #if self.scene() is not None:
            #gv = self.scene().views()[0]
            #unit = gv.mapToScene(unit).boundingRect()
            ##print unit
            #unit = self.mapRectFromScene(unit)
            ##print unit
        
        vr = self.view().viewRect()
        #vr = self.viewBounds()
        if vr is None:
            return
        #print 'before', self.bounds
        
        if self.angle > 45:
            m = tan((90-self.angle) * pi / 180.)
            y2 = vr.bottom()
            y1 = vr.top()
            x1 = self.p[0] + (y1 - self.p[1]) * m
            x2 = self.p[0] + (y2 - self.p[1]) * m
        else:
            m = tan(self.angle * pi / 180.)
            x1 = vr.left()
            x2 = vr.right()
            y2 = self.p[1] + (x1 - self.p[0]) * m
            y1 = self.p[1] + (x2 - self.p[0]) * m
        #print vr, x1, y1, x2, y2
        self.prepareGeometryChange()
        self.line = (QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
        self.bounds = QtCore.QRectF(self.line[0], self.line[1])
        ## Stupid bug causes lines to disappear:
        if self.angle % 180 == 90:
            px = self.pixelWidth()
            #self.bounds.setWidth(1e-9)
            self.bounds.setX(x1 + px*-5)
            self.bounds.setWidth(px*10)
        if self.angle % 180 == 0:
            px = self.pixelHeight()
            #self.bounds.setHeight(1e-9)
            self.bounds.setY(y1 + px*-5)
            self.bounds.setHeight(px*10)

        #QtGui.QGraphicsLineItem.setLine(self, x1, y1, x2, y2)
        #self.update()
        
    def boundingRect(self):
        #self.updateLine()
        #return QtGui.QGraphicsLineItem.boundingRect(self)
        #print "bounds", self.bounds
        return self.bounds
    
    def paint(self, p, *args):
        w,h  = self.pixelWidth()*5, self.pixelHeight()*5*1.1547
        #self.updateLine()
        l = self.line
        
        p.setPen(self.currentPen)
        #print "paint", self.line
        p.drawLine(l[0], l[1])
        
        p.setBrush(QtGui.QBrush(self.currentPen.color()))
        p.drawConvexPolygon(QtGui.QPolygonF([
            l[0] + QtCore.QPointF(-w, 0),
            l[0] + QtCore.QPointF(0, h),
            l[0] + QtCore.QPointF(w, 0),
        ]))
        
        #p.setPen(QtGui.QPen(QtGui.QColor(255,0,0)))
        #p.drawRect(self.boundingRect())
        
    def mousePressEvent(self, ev):
        if self.movable and ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            self.pressDelta = self.mapToParent(ev.pos()) - QtCore.QPointF(*self.p)
        else:
            ev.ignore()
            
    def mouseMoveEvent(self, ev):
        self.setPos(self.mapToParent(ev.pos()) - self.pressDelta)
        self.emit(QtCore.SIGNAL('dragged'), self)
 


class LinearRegionItem(GraphicsObject):
    """Used for marking a horizontal or vertical region in plots."""
    def __init__(self, view, orientation="horizontal", vals=[0,1], brush=None, movable=True, bounds=None):
        GraphicsObject.__init__(self)
        self.orientation = orientation
        if hasattr(self, "ItemHasNoContents"):  
            self.setFlag(self.ItemHasNoContents)
        self.rect = QtGui.QGraphicsRectItem(self)
        self.rect.setParentItem(self)
        self.bounds = QtCore.QRectF()
        self.view = weakref.ref(view)
        
        self.setBrush = self.rect.setBrush
        self.brush = self.rect.brush
        
        if orientation[0] == 'h':
            self.lines = [
                InfiniteLine(view, QtCore.QPointF(0, vals[0]), 0, movable=movable, bounds=bounds), 
                InfiniteLine(view, QtCore.QPointF(0, vals[1]), 0, movable=movable, bounds=bounds)]
        else:
            self.lines = [
                InfiniteLine(view, QtCore.QPointF(vals[0], 0), 90, movable=movable, bounds=bounds), 
                InfiniteLine(view, QtCore.QPointF(vals[1], 0), 90, movable=movable, bounds=bounds)]
        QtCore.QObject.connect(self.view(), QtCore.SIGNAL('viewChanged'), self.updateBounds)
        
        for l in self.lines:
            l.setParentItem(self)
            l.connect(QtCore.SIGNAL('positionChanged'), self.lineMoved)
            
        if brush is None:
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, 50))
        self.setBrush(brush)
            
    def setBounds(self, bounds):
        for l in self.lines:
            l.setBounds(bounds)
        
        
    def boundingRect(self):
        return self.rect.boundingRect()
            
    def lineMoved(self):
        self.updateBounds()
        self.emit(QtCore.SIGNAL('regionChanged'), self)
            
    def updateBounds(self):
        vb = self.view().viewRect()
        vals = [self.lines[0].value(), self.lines[1].value()]
        if self.orientation[0] == 'h':
            vb.setTop(max(vals))
            vb.setBottom(min(vals))
        else:
            vb.setLeft(min(vals))
            vb.setRight(max(vals))
        if vb != self.bounds:
            self.bounds = vb
            self.rect.setRect(vb)
        
    def mousePressEvent(self, ev):
        for l in self.lines:
            l.mousePressEvent(ev)
        #if self.movable and ev.button() == QtCore.Qt.LeftButton:
            #ev.accept()
            #self.pressDelta = self.mapToParent(ev.pos()) - QtCore.QPointF(*self.p)
        #else:
            #ev.ignore()
            
    def mouseMoveEvent(self, ev):
        self.lines[0].blockSignals(True)  # only want to update once
        for l in self.lines:
            l.mouseMoveEvent(ev)
        self.lines[0].blockSignals(False)
        #self.setPos(self.mapToParent(ev.pos()) - self.pressDelta)
        #self.emit(QtCore.SIGNAL('dragged'), self)

    def getRegion(self):
        if self.orientation[0] == 'h':
            r = (self.bounds.top(), self.bounds.bottom())
        else:
            r = (self.bounds.left(), self.bounds.right())
        return (min(r), max(r))

    def setRegion(self, rgn):
        self.lines[0].setValue(rgn[0])
        self.lines[1].setValue(rgn[1])


class VTickGroup(QtGui.QGraphicsPathItem):
    def __init__(self, xvals=None, yrange=None, pen=None, relative=False, view=None):
        QtGui.QGraphicsPathItem.__init__(self)
        if yrange is None:
            yrange = [0, 1]
        if xvals is None:
            xvals = []
        if pen is None:
            pen = QtGui.QPen(QtGui.QColor(200, 200, 200))
        self.ticks = []
        self.xvals = []
        if view is None:
            self.view = None
        else:
            self.view = weakref.ref(view)
        self.yrange = [0,1]
        self.setPen(pen)
        self.setYRange(yrange, relative)
        self.setXVals(xvals)
        self.valid = False
        
        
    #def setPen(self, pen=None):
        #if pen is None:
            #pen = self.pen
        #self.pen = pen
        #for t in self.ticks:
            #t.setPen(pen)
        ##self.update()

    def setXVals(self, vals):
        self.xvals = vals
        self.rebuildTicks()
        self.valid = False
        
    def setYRange(self, vals, relative=False):
        self.yrange = vals
        self.relative = relative
        if self.view is not None:
            if relative:
                #QtCore.QObject.connect(self.view, QtCore.SIGNAL('viewChanged'), self.rebuildTicks)
                QtCore.QObject.connect(self.view(), QtCore.SIGNAL('viewChanged'), self.rescale)
            else:
                try:
                    #QtCore.QObject.disconnect(self.view, QtCore.SIGNAL('viewChanged'), self.rebuildTicks)
                    QtCore.QObject.disconnect(self.view(), QtCore.SIGNAL('viewChanged'), self.rescale)
                except:
                    pass
        self.rebuildTicks()
        self.valid = False
            
    def rescale(self):
        #print "RESCALE:"
        self.resetTransform()
        #height = self.view.size().height()
        #p1 = self.mapFromScene(self.view.mapToScene(QtCore.QPoint(0, height * (1.0-self.yrange[0]))))
        #p2 = self.mapFromScene(self.view.mapToScene(QtCore.QPoint(0, height * (1.0-self.yrange[1]))))
        #yr = [p1.y(), p2.y()]
        vb = self.view().viewRect()
        p1 = vb.bottom() - vb.height() * self.yrange[0]
        p2 = vb.bottom() - vb.height() * self.yrange[1]
        yr = [p1, p2]
        
        #print "  ", vb, yr
        self.translate(0.0, yr[0])
        self.scale(1.0, (yr[1]-yr[0]))
        #print "  ", self.mapRectToScene(self.boundingRect())
        self.boundingRect()
        self.update()
            
    def boundingRect(self):
        #print "--request bounds:"
        b = QtGui.QGraphicsPathItem.boundingRect(self)
        #print "  ", self.mapRectToScene(b)
        return b
            
    def yRange(self):
        #if self.relative:
            #height = self.view.size().height()
            #p1 = self.mapFromScene(self.view.mapToScene(QtCore.QPoint(0, height * (1.0-self.yrange[0]))))
            #p2 = self.mapFromScene(self.view.mapToScene(QtCore.QPoint(0, height * (1.0-self.yrange[1]))))
            #return [p1.y(), p2.y()]
        #else:
            #return self.yrange
            
        return self.yrange
            
    def rebuildTicks(self):
        self.path = QtGui.QPainterPath()
        yrange = self.yRange()
        #print "rebuild ticks:", yrange
        for x in self.xvals:
            #path.moveTo(x, yrange[0])
            #path.lineTo(x, yrange[1])
            self.path.moveTo(x, 0.)
            self.path.lineTo(x, 1.)
        self.setPath(self.path)
        self.valid = True
        self.rescale()
        #print "  done..", self.boundingRect()
        
    def paint(self, *args):
        if not self.valid:
            self.rebuildTicks()
        #print "Paint", self.boundingRect()
        QtGui.QGraphicsPathItem.paint(self, *args)
        

class GridItem(UIGraphicsItem):
    def __init__(self, view, bounds=None, *args):
        UIGraphicsItem.__init__(self, view, bounds)
        #QtGui.QGraphicsItem.__init__(self, *args)
        self.setFlag(QtGui.QGraphicsItem.ItemClipsToShape)
        #self.setCacheMode(QtGui.QGraphicsItem.DeviceCoordinateCache)
        
        self.picture = None
        
        
    def viewChangedEvent(self, newRect, oldRect):
        self.picture = None
        
    def paint(self, p, opt, widget):
        #p.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100)))
        #p.drawRect(self.boundingRect())
        
        ## draw picture
        if self.picture is None:
            #print "no pic, draw.."
            self.generatePicture()
        p.drawPicture(0, 0, self.picture)
        #print "draw"
        
        
    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter()
        p.begin(self.picture)
        
        dt = self.viewTransform().inverted()[0]
        vr = self.viewRect()
        unit = self.unitRect()
        dim = [vr.width(), vr.height()]
        lvr = self.boundingRect()
        ul = array([lvr.left(), lvr.top()])
        br = array([lvr.right(), lvr.bottom()])
        
        texts = []
        
        if ul[1] > br[1]:
            x = ul[1]
            ul[1] = br[1]
            br[1] = x
        
        for i in range(2, -1, -1):   ## Draw three different scales of grid
            
            dist = br-ul
            nlTarget = 10.**i
            d = 10. ** floor(log10(abs(dist/nlTarget))+0.5)
            ul1 = floor(ul / d) * d
            br1 = ceil(br / d) * d
            dist = br1-ul1
            nl = (dist / d) + 0.5
            for ax in range(0,2):  ## Draw grid for both axes
                ppl = dim[ax] / nl[ax]
                c = clip(3.*(ppl-3), 0., 30.)
                linePen = QtGui.QPen(QtGui.QColor(255, 255, 255, c)) 
                textPen = QtGui.QPen(QtGui.QColor(255, 255, 255, c*2)) 
                
                bx = (ax+1) % 2
                for x in range(0, int(nl[ax])):
                    p.setPen(linePen)
                    p1 = array([0.,0.])
                    p2 = array([0.,0.])
                    p1[ax] = ul1[ax] + x * d[ax]
                    p2[ax] = p1[ax]
                    p1[bx] = ul[bx]
                    p2[bx] = br[bx]
                    p.drawLine(QtCore.QPointF(p1[0], p1[1]), QtCore.QPointF(p2[0], p2[1]))
                    if i < 2:
                        p.setPen(textPen)
                        if ax == 0:
                            x = p1[0] + unit.width()
                            y = ul[1] + unit.height() * 8.
                        else:
                            x = ul[0] + unit.width()*3
                            y = p1[1] + unit.height()
                        texts.append((QtCore.QPointF(x, y), "%g"%p1[ax]))
        tr = self.viewTransform()
        tr.scale(1.5, 1.5)
        p.setWorldTransform(tr.inverted()[0])
        for t in texts:
            x = tr.map(t[0])
            p.drawText(x, t[1])
        p.end()

class ScaleBar(UIGraphicsItem):
    def __init__(self, view, size, width=5, color=(100, 100, 255)):
        self.size = size
        UIGraphicsItem.__init__(self, view)
        self.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        #self.pen = QtGui.QPen(QtGui.QColor(*color))
        #self.pen.setWidth(width)
        #self.pen.setCosmetic(True)
        #self.pen2 = QtGui.QPen(QtGui.QColor(0,0,0))
        #self.pen2.setWidth(width+2)
        #self.pen2.setCosmetic(True)
        self.brush = QtGui.QBrush(QtGui.QColor(*color))
        self.pen = QtGui.QPen(QtGui.QColor(0,0,0))
        self.width = width
        
    def paint(self, p, opt, widget):
        rect = self.boundingRect()
        unit = self.unitRect()
        y = rect.bottom() + (rect.top()-rect.bottom()) * 0.02
        y1 = y + unit.height()*self.width
        x = rect.right() + (rect.left()-rect.right()) * 0.02
        x1 = x - self.size
        
        
        p.setPen(self.pen)
        p.setBrush(self.brush)
        rect = QtCore.QRectF(
            QtCore.QPointF(x1, y1), 
            QtCore.QPointF(x, y)
        )
        p.translate(x1, y1)
        p.scale(rect.width(), rect.height())
        p.drawRect(0, 0, 1, 1)
        
        alpha = clip(((self.size/unit.width()) - 40.) * 255. / 80., 0, 255)
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, alpha)))
        for i in range(1, 10):
            #x2 = x + (x1-x) * 0.1 * i
            x2 = 0.1 * i
            p.drawLine(QtCore.QPointF(x2, 0), QtCore.QPointF(x2, 1))
        

    def setSize(self, s):
        self.size = s
        
class ColorScaleBar(UIGraphicsItem):
    def __init__(self, view, size, offset):
        self.size = size
        self.offset = offset
        UIGraphicsItem.__init__(self, view)
        self.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self.brush = QtGui.QBrush(QtGui.QColor(200,0,0))
        self.pen = QtGui.QPen(QtGui.QColor(0,0,0))
        self.labels = {'max': 1, 'min': 0}
        self.gradient = QtGui.QLinearGradient()
        self.gradient.setColorAt(0, QtGui.QColor(0,0,0))
        self.gradient.setColorAt(1, QtGui.QColor(255,0,0))
        
    def setGradient(self, g):
        self.gradient = g
        self.update()
        
    def setLabels(self, l):
        self.labels = l
        self.update()
        
    def paint(self, p, opt, widget):
        rect = self.boundingRect()   ## Boundaries of visible area in scene coords.
        unit = self.unitRect()       ## Size of one view pixel in scene coords.
        
        ## determine max width of all labels
        labelWidth = 0
        labelHeight = 0
        for k in self.labels:
            b = p.boundingRect(QtCore.QRectF(0, 0, 0, 0), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, k)
            labelWidth = max(labelWidth, b.width())
            labelHeight = max(labelHeight, b.height())
            
        labelWidth *= unit.width()
        labelHeight *= unit.height()
        
        textPadding = 2  # in px
        
        if self.offset[0] < 0:
            x3 = rect.right() + unit.width() * self.offset[0]
            x2 = x3 - labelWidth - unit.width()*textPadding*2
            x1 = x2 - unit.width() * self.size[0]
        else:
            x1 = rect.left() + unit.width() * self.offset[0]
            x2 = x1 + unit.width() * self.size[0]
            x3 = x2 + labelWidth + unit.width()*textPadding*2
        if self.offset[1] < 0:
            y2 = rect.top() - unit.height() * self.offset[1]
            y1 = y2 + unit.height() * self.size[1]
        else:
            y1 = rect.bottom() - unit.height() * self.offset[1]
            y2 = y1 - unit.height() * self.size[1]
        self.b = [x1,x2,x3,y1,y2,labelWidth]
            
        ## Draw background
        p.setPen(self.pen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(255,255,255,100)))
        rect = QtCore.QRectF(
            QtCore.QPointF(x1 - unit.width()*textPadding, y1 + labelHeight/2 + unit.height()*textPadding), 
            QtCore.QPointF(x3, y2 - labelHeight/2 - unit.height()*textPadding)
        )
        p.drawRect(rect)
        
        
        ## Have to scale painter so that text and gradients are correct size. Bleh.
        p.scale(unit.width(), unit.height())
        
        ## Draw color bar
        self.gradient.setStart(0, y1/unit.height())
        self.gradient.setFinalStop(0, y2/unit.height())
        p.setBrush(self.gradient)
        rect = QtCore.QRectF(
            QtCore.QPointF(x1/unit.width(), y1/unit.height()), 
            QtCore.QPointF(x2/unit.width(), y2/unit.height())
        )
        p.drawRect(rect)
        
        
        ## draw labels
        p.setPen(QtGui.QPen(QtGui.QColor(0,0,0)))
        tx = x2 + unit.width()*textPadding
        lh = labelHeight/unit.height()
        for k in self.labels:
            y = y1 + self.labels[k] * (y2-y1)
            p.drawText(QtCore.QRectF(tx/unit.width(), y/unit.height() - lh/2.0, 1000, lh), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, k)
        
        
