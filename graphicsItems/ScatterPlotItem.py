from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.Point import Point
import pyqtgraph.functions as fn
from GraphicsObject import GraphicsObject
import numpy as np
import scipy.stats

__all__ = ['ScatterPlotItem', 'SpotItem']
class ScatterPlotItem(GraphicsObject):
    """
    Displays a set of x/y points. Instances of this class are created
    automatically as part of PlotDataItem; these rarely need to be instantiated
    directly.
    
    The size, shape, pen, and fill brush may be set for each point individually 
    or for all points. 
    
    
    ========================  ===============================================
    **Signals:**
    sigPlotChanged(self)      Emitted when the data being plotted has changed
    sigClicked(self, points)  Emitted when the curve is clicked. Sends a list
                              of all the points under the mouse pointer.
    ========================  ===============================================
    
    """
    #sigPointClicked = QtCore.Signal(object, object)
    sigClicked = QtCore.Signal(object, object)  ## self, points
    sigPlotChanged = QtCore.Signal(object)
    
    def __init__(self, *args, **kargs):
        """
        Accepts the same arguments as setData()
        """
        
        GraphicsObject.__init__(self)
        self.data = None
        self.spots = []
        self.bounds = [None, None]
        self.opts = {}
        self.spotsValid = False
        self._spotPixmap = None
        
        self.setPen(200,200,200)
        self.setBrush(100,100,150)
        self.setSymbol('o')
        self.setSize(7)
        self.setPxMode(True)
        self.setIdentical(False)
        
        self.setData(*args, **kargs)
        
        
    def setData(self, *args, **kargs):
        """
        **Ordered Arguments:**
        
        * If there is only one unnamed argument, it will be interpreted like the 'spots' argument.
        * If there are two unnamed arguments, they will be interpreted as sequences of x and y values.
        
        ====================== =================================================================================================
        **Keyword Arguments:**
        *spots*                Optional list of dicts. Each dict specifies parameters for a single spot:
                               {'pos': (x,y), 'size', 'pen', 'brush', 'symbol'}. This is just an alternate method
                               of passing in data for the corresponding arguments.
        *x*,*y*                1D arrays of x,y values.
        *pos*                  2D structure of x,y pairs (such as Nx2 array or list of tuples)
        *pxMode*               If True, spots are always the same size regardless of scaling, and size is given in px.
                               Otherwise, size is in scene coordinates and the spots scale with the view.
                               Default is True
        *identical*            If True, all spots are forced to look identical. 
                               This can result in performance enhancement.
                               Default is False
        *symbol*               can be one (or a list) of:
                               
                               * 'o'  circle (default)
                               * 's'  square
                               * 't'  triangle
                               * 'd'  diamond
                               * '+'  plus
        *pen*                  The pen (or list of pens) to use for drawing spot outlines.
        *brush*                The brush (or list of brushes) to use for filling spots.
        *size*                 The size (or list of sizes) of spots. If *pxMode* is True, this value is in pixels. Otherwise,
                               it is in the item's local coordinate system.
        *data*                 a list of python objects used to uniquely identify each spot.
        ====================== =================================================================================================
        """
        
        self.clear()

        
        ## deal with non-keyword arguments
        if len(args) == 1:
            kargs['spots'] = args[0]
        elif len(args) == 2:
            kargs['x'] = args[0]
            kargs['y'] = args[1]
        elif len(args) > 2:
            raise Exception('Only accepts up to two non-keyword arguments.')
        
        ## convert 'pos' argument to 'x' and 'y'
        if 'pos' in kargs:
            pos = kargs['pos']
            if isinstance(pos, np.ndarray):
                kargs['x'] = pos[:,0]
                kargs['y'] = pos[:,1]
            else:
                x = []
                y = []
                for p in pos:
                    if isinstance(p, QtCore.QPointF):
                        x.append(p.x())
                        y.append(p.y())
                    else:
                        x.append(p[0])
                        y.append(p[1])
                kargs['x'] = x
                kargs['y'] = y
        
        ## determine how many spots we have
        if 'spots' in kargs:
            numPts = len(kargs['spots'])
        elif 'y' in kargs and kargs['y'] is not None:
            numPts = len(kargs['y'])
        else:
            kargs['x'] = []
            kargs['y'] = []
            numPts = 0
        
        ## create empty record array
        self.data = np.empty(numPts, dtype=[('x', float), ('y', float), ('size', float), ('symbol', 'S1'), ('pen', object), ('brush', object), ('spot', object)])
        self.data['size'] = -1  ## indicates use default size
        self.data['symbol'] = ''
        self.data['pen'] = None
        self.data['brush'] = None
        self.pointData = np.empty(numPts, dtype=object)
        self.pointData[:] = None
        
        if 'spots' in kargs:
            spots = kargs['spots']
            for i in xrange(len(spots)):
                spot = spots[i]
                for k in spot:
                    if k == 'pen':
                        self.data[i][k] = fn.mkPen(spot[k])
                    elif k == 'brush':
                        self.data[i][k] = fn.mkBrush(spot[k])
                    elif k == 'pos':
                        pos = spot[k]
                        if isinstance(pos, QtCore.QPointF):
                            x,y = pos.x(), pos.y()
                        else:
                            x,y = pos[0], pos[1]
                        self.data[i]['x'] = x
                        self.data[i]['y'] = y
                    elif k in ['x', 'y', 'size', 'symbol']:
                        self.data[i][k] = spot[k]
                    elif k == 'data':
                        self.pointData[i] = spot[k]
                    else:
                        raise Exception("Unknown spot parameter: %s" % k)
        elif 'y' in kargs:
            self.data['x'] = kargs['x']
            self.data['y'] = kargs['y']
        
        
        ## Set any extra parameters provided in keyword arguments
        for k in ['pxMode', 'identical', 'pen', 'brush', 'symbol', 'size']:
            if k in kargs:
                setMethod = getattr(self, 'set' + k[0].upper() + k[1:])
                setMethod(kargs[k])
                
        if 'data' in kargs:
            self.setPointData(kargs['data'])
            
        self.updateSpots()
        
        
        
        
        
        
        
        
        #pen = kargs.get('pen', (200,200,200))
        #brush = kargs.get('pen', (100,100,150))
        
        #if hasattr(pen, '__len__'):
            #pen = map(pg.mkPen(pen))
        #self.data['pen'] = pen
        
        #if hasattr(pen, '__len__'):
            #brush = map(pg.mkPen(pen))
        #self.data['brush'] = pen
        
        #self.data['size'] = kargs.get('size', 7)
        #self.data['symbol'] = kargs.get('symbol', 'o')
        
        
        
        #if spots is not None and len(spots) > 0:
            #spot = spots[0]
            #for k in spot:
                #self.data[k] = []
            #for spot in spots:
                #for k,v in spot.iteritems():
                    #self.data[k].append(v)
        
    def setPoints(self, *args, **kargs):
        ##Deprecated; use setData
        return self.setData(*args, **kargs)
        
    #def setPoints(self, spots=None, x=None, y=None, data=None):
        #"""
        #Remove all existing points in the scatter plot and add a new set.
        #Arguments:
            #spots - list of dicts specifying parameters for each spot
                    #[ {'pos': (x,y), 'pen': 'r', ...}, ...]
            #x, y -  arrays specifying location of spots to add. 
                    #all other parameters (pen, symbol, etc.) will be set to the default
                    #values for this scatter plot.
                    #these arguments are IGNORED if 'spots' is specified
            #data -  list of arbitrary objects to be assigned to spot.data for each spot
                    #(this is useful for identifying spots that are clicked on)
        #"""
        #self.clear()
        #self.bounds = [[0,0],[0,0]]
        #self.addPoints(spots, x, y, data)
        
    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints
    
    def setPen(self, *args, **kargs):
        if len(args) == 1 and (isinstance(args[0], np.ndarray) or isinstance(args[0], list)):
            pens = args[0]
            if self.data is None:
                raise Exception("Must set data before setting multiple pens.")
            if len(pens) != len(self.data):
                raise Exception("Number of pens does not match number of points (%d != %d)" % (len(pens), len(self.data)))
            for i in xrange(len(pens)):
                self.data[i]['pen'] = fn.mkPen(pens[i])
        else:
            self.opts['pen'] = fn.mkPen(*args, **kargs)
        self.updateSpots()
        
    def setBrush(self, *args, **kargs):
        if len(args) == 1 and (isinstance(args[0], np.ndarray) or isinstance(args[0], list)):
            brushes = args[0]
            if self.data is None:
                raise Exception("Must set data before setting multiple brushes.")
            if len(brushes) != len(self.data):
                raise Exception("Number of brushes does not match number of points (%d != %d)" % (len(brushes), len(self.data)))
            for i in xrange(len(brushes)):
                self.data[i]['brush'] = fn.mkBrush(brushes[i], **kargs)
        else:
            self.opts['brush'] = fn.mkBrush(*args, **kargs)
        self.updateSpots()

    def setSymbol(self, symbol):
        if isinstance(symbol, np.ndarray) or isinstance(symbol, list):
            symbols = symbol
            if self.data is None:
                raise Exception("Must set data before setting multiple symbols.")
            if len(symbols) != len(self.data):
                raise Exception("Number of symbols does not match number of points (%d != %d)" % (len(symbols), len(self.data)))
            self.data['symbol'] = symbols
        else:
            self.opts['symbol'] = symbol
        self.updateSpots()
        
    def setSize(self, size):
        if isinstance(size, np.ndarray) or isinstance(size, list):
            sizes = size
            if self.data is None:
                raise Exception("Must set data before setting multiple sizes.")
            if len(sizes) != len(self.data):
                raise Exception("Number of sizes does not match number of points (%d != %d)" % (len(sizes), len(self.data)))
            self.data['size'] = sizes
        else:
            self.opts['size'] = size
        self.updateSpots()
        
    def setPointData(self, data):
        if isinstance(data, np.ndarray) or isinstance(data, list):
            if self.data is None:
                raise Exception("Must set xy data before setting meta data.")
            if len(data) != len(self.data):
                raise Exception("Length of meta data does not match number of points (%d != %d)" % (len(data), len(self.data)))
        self.pointData = data
        self.updateSpots()
        
        
    def setIdentical(self, ident):
        self.opts['identical'] = ident
        self.updateSpots()
        
    def setPxMode(self, mode):
        self.opts['pxMode'] = mode
        self.updateSpots()
        
    def updateSpots(self):
        self.spotsValid = False
        self.update()
        
    def clear(self):
        for i in self.spots:
            i.setParentItem(None)
            s = i.scene()
            if s is not None:
                s.removeItem(i)
        self.spots = []
        self.data = None
        self.spotsValid = False
        self.bounds = [None, None]
        

    def dataBounds(self, ax, frac=1.0):
        if frac >= 1.0 and self.bounds[ax] is not None:
            return self.bounds[ax]
        
        if self.data is None or len(self.data) == 0:
            return (None, None)
        
        if ax == 0:
            d = self.data['x']
        elif ax == 1:
            d = self.data['y']
            
        if frac >= 1.0:
            minIndex = np.argmin(d)
            maxIndex = np.argmax(d)
            minVal = d[minIndex]
            maxVal = d[maxIndex]
            if not self.opts['pxMode']:
                minVal -= self.data[minIndex]['size']
                maxVal += self.data[maxIndex]['size']
            self.bounds[ax] = (minVal, maxVal)
            return self.bounds[ax]
        elif frac <= 0.0:
            raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        else:
            return (scipy.stats.scoreatpercentile(d, 50 - (frac * 50)), scipy.stats.scoreatpercentile(d, 50 + (frac * 50)))
            
            

    
    def addPoints(self, *args, **kargs):
        """
        Add new points to the scatter plot. 
        Arguments are the same as setData()
        Note: this is expensive; plenty of room for optimization here.
        """
        if self.data is None:
            self.setData(*args, **kargs)
            return
            
        
        data1 = self.data[:]
        #range1 = [self.bounds[0][:], self.bounds[1][:]]
        self.setData(*args, **kargs)
        newData = np.empty(len(self.data) + len(data1), dtype=self.data.dtype)
        newData[:len(data1)] = data1
        newData[len(data1):] = self.data
        #self.bounds = [
            #[min(self.bounds[0][0], range1[0][0]), max(self.bounds[0][1], range1[0][1])],
            #[min(self.bounds[1][0], range1[1][0]), max(self.bounds[1][1], range1[1][1])],
        #]
        self.data = newData
        self.sigPlotChanged.emit(self)
    
    
    def generateSpots(self):
        xmn = ymn = xmx = ymx = None
        
        ## apply defaults
        size = self.data['size'].copy()
        size[size<0] = self.opts['size']
        
        pen = self.data['pen'].copy()
        pen[pen<0] = self.opts['pen']  ## note pen<0 checks for pen==None
        
        brush = self.data['brush'].copy()
        brush[brush<0] = self.opts['brush']
        
        symbol = self.data['symbol'].copy()
        symbol[symbol==''] = self.opts['symbol']

        
        for i in xrange(len(self.data)):
            s = self.data[i]
            pos = Point(s['x'], s['y'])
            if self.opts['pxMode']:
                psize = 0
            else:
                psize = size[i]
                
            if self.pointData is None or self.pointData[i] is None:
                data = self.opts.get('data', None)
            else:
                data = self.pointData[i]
                
            #if xmn is None:
                #xmn = pos[0]-psize
                #xmx = pos[0]+psize
                #ymn = pos[1]-psize
                #ymx = pos[1]+psize
            #else:
                #xmn = min(xmn, pos[0]-psize)
                #xmx = max(xmx, pos[0]+psize)
                #ymn = min(ymn, pos[1]-psize)
                #ymx = max(ymx, pos[1]+psize)
                
            item = self.mkSpot(pos, size[i], self.opts['pxMode'], brush[i], pen[i], data, symbol=symbol[i], index=len(self.spots))
            self.spots.append(item)
            self.data[i]['spot'] = item
            #if self.optimize:
                #item.hide()
                #frag = QtGui.QPainter.PixmapFragment.create(pos, QtCore.QRectF(0, 0, size, size))
                #self.optimizeFragments.append(frag)
                
        #self.bounds = [[xmn, xmx], [ymn, ymx]]
        self.spotsValid = True
        self.sigPlotChanged.emit(self)
        
        
    #def setPointSize(self, size):
        #for s in self.spots:
            #s.size = size
        ##self.setPoints([{'size':s.size, 'pos':s.pos(), 'data':s.data} for s in self.spots])
        #self.setPoints()
                
    #def paint(self, p, *args):
        #if not self.optimize:
            #return
        ##p.setClipRegion(self.boundingRect())
        #p.drawPixmapFragments(self.optimizeFragments, self.optimizePixmap)

    def paint(self, *args):
        if not self.spotsValid:
            self.generateSpots()

    def spotPixmap(self):
        ## If all spots are identical, return the pixmap to use for all spots
        ## Otherwise return None
        
        if not self.opts['identical']:
            return None
        if self._spotPixmap is None:
            spot = SpotItem(size=self.opts['size'], pxMode=True, brush=self.opts['brush'], pen=self.opts['pen'], symbol=self.opts['symbol'])
            self._spotPixmap = spot.pixmap
        return self._spotPixmap

    def mkSpot(self, pos, size, pxMode, brush, pen, data, symbol=None, index=None):
        ## Make and return a SpotItem (or PixmapSpotItem if in pxMode)
        brush = fn.mkBrush(brush)
        pen = fn.mkPen(pen)
        if pxMode:
            img = self.spotPixmap()  ## returns None if not using identical mode
            #item = PixmapSpotItem(size, brush, pen, data, image=img, symbol=symbol, index=index)
            item = SpotItem(size, pxMode, brush, pen, data, symbol=symbol, image=img, index=index)
        else:
            item = SpotItem(size, pxMode, brush, pen, data, symbol=symbol, index=index)
        item.setParentItem(self)
        item.setPos(pos)
        #item.sigClicked.connect(self.pointClicked)
        return item
        
    def boundingRect(self):
        (xmn, xmx) = self.dataBounds(ax=0)
        (ymn, ymx) = self.dataBounds(ax=1)
        if xmn is None or xmx is None:
            xmn = 0
            xmx = 0
        if ymn is None or ymx is None:
            ymn = 0
            ymx = 0
        return QtCore.QRectF(xmn, ymn, xmx-xmn, ymx-ymn)
        
        #if xmn is None or xmx is None or ymn is None or ymx is None:
            #return QtCore.QRectF()
        #return QtCore.QRectF(xmn, ymn, xmx-xmn, ymx-ymn)
        #return QtCore.QRectF(xmn-1, ymn-1, xmx-xmn+2, ymx-ymn+2)
        
    #def pointClicked(self, point):
        #self.sigPointClicked.emit(self, point)

    def points(self):
        if not self.spotsValid:
            self.generateSpots()
        return self.spots[:]

    def pointsAt(self, pos):
        if not self.spotsValid:
            self.generateSpots()
        x = pos.x()
        y = pos.y()
        pw = self.pixelWidth()
        ph = self.pixelHeight()
        pts = []
        for s in self.spots:
            sp = s.pos()
            ss = s.size
            sx = sp.x()
            sy = sp.y()
            s2x = s2y = ss * 0.5
            if self.opts['pxMode']:
                s2x *= pw
                s2y *= ph
            if x > sx-s2x and x < sx+s2x and y > sy-s2y and y < sy+s2y:
                pts.append(s)
                #print "HIT:", x, y, sx, sy, s2x, s2y
            #else:
                #print "No hit:", (x, y), (sx, sy)
                #print "       ", (sx-s2x, sy-s2y), (sx+s2x, sy+s2y)
        pts.sort(lambda a,b: cmp(b.zValue(), a.zValue()))
        return pts
            

    #def mousePressEvent(self, ev):
        #QtGui.QGraphicsItem.mousePressEvent(self, ev)
        #if ev.button() == QtCore.Qt.LeftButton:
            #pts = self.pointsAt(ev.pos())
            #if len(pts) > 0:
                #self.mouseMoved = False
                #self.ptsClicked = pts
                #ev.accept()
            #else:
                ##print "no spots"
                #ev.ignore()
        #else:
            #ev.ignore()
        
    #def mouseMoveEvent(self, ev):
        #QtGui.QGraphicsItem.mouseMoveEvent(self, ev)
        #self.mouseMoved = True
        #pass
    
    #def mouseReleaseEvent(self, ev):
        #QtGui.QGraphicsItem.mouseReleaseEvent(self, ev)
        #if not self.mouseMoved:
            #self.sigClicked.emit(self, self.ptsClicked)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            pts = self.pointsAt(ev.pos())
            if len(pts) > 0:
                self.ptsClicked = pts
                self.sigClicked.emit(self, self.ptsClicked)
                ev.accept()
            else:
                #print "no spots"
                ev.ignore()
        else:
            ev.ignore()



class SpotItem(GraphicsObject):
    #sigClicked = QtCore.Signal(object)
    
    def __init__(self, size, pxMode, brush, pen, data=None, symbol=None, image=None, index=None):
        GraphicsObject.__init__(self)
        self.pxMode = pxMode

        try:
            symbol = int(symbol)
        except: 
            pass
        
        if symbol is None:
            symbol = 'o'    ## circle by default
        elif isinstance(symbol, int):  ## allow symbols specified by integer for easy iteration
            symbol = ['o', 's', 't', 'd', '+'][symbol]
            
            
            
        ####print 'SpotItem symbol: ', symbol
        self.data = data
        self.pen = pen
        self.brush = brush
        self.size = size
        self.index = index
        self.symbol = symbol
        #s2 = size/2.
        self.path = QtGui.QPainterPath()
        
        if symbol == 'o':
            self.path.addEllipse(QtCore.QRectF(-0.5, -0.5, 1, 1))
        elif symbol == 's':
            self.path.addRect(QtCore.QRectF(-0.5, -0.5, 1, 1))
        elif symbol == 't' or symbol == '^':
            self.path.moveTo(-0.5, -0.5)
            self.path.lineTo(0, 0.5)
            self.path.lineTo(0.5, -0.5)
            self.path.closeSubpath()
            #self.path.connectPath(self.path)
        elif symbol == 'd':
            self.path.moveTo(0., -0.5)
            self.path.lineTo(-0.4, 0.)
            self.path.lineTo(0, 0.5)
            self.path.lineTo(0.4, 0)
            self.path.closeSubpath()
            #self.path.connectPath(self.path)
        elif symbol == '+':
            self.path.moveTo(-0.5, -0.01)
            self.path.lineTo(-0.5, 0.01)
            self.path.lineTo(-0.01, 0.01)
            self.path.lineTo(-0.01, 0.5)
            self.path.lineTo(0.01, 0.5)
            self.path.lineTo(0.01, 0.01)
            self.path.lineTo(0.5, 0.01)
            self.path.lineTo(0.5, -0.01)
            self.path.lineTo(0.01, -0.01)
            self.path.lineTo(0.01, -0.5)
            self.path.lineTo(-0.01, -0.5)
            self.path.lineTo(-0.01, -0.01)
            self.path.closeSubpath()
            #self.path.connectPath(self.path)
        #elif symbol == 'x':
        else:
            raise Exception("Unknown spot symbol '%s' (type=%s)" % (str(symbol), str(type(symbol))))
            #self.path.addEllipse(QtCore.QRectF(-0.5, -0.5, 1, 1))
        
        if pxMode:
            ## pre-render an image of the spot and display this rather than redrawing every time.
            if image is None:
                self.pixmap = self.makeSpotImage(size, pen, brush, symbol)
            else:
                self.pixmap = image ## image is already provided (probably shared with other spots)
            self.setFlags(self.flags() | self.ItemIgnoresTransformations | self.ItemHasNoContents)
            self.pi = QtGui.QGraphicsPixmapItem(self.pixmap, self)
            self.pi.setPos(-0.5*size, -0.5*size)
        else:
            self.scale(size, size)


    def makeSpotImage(self, size, pen, brush, symbol=None):
        self.spotImage = QtGui.QImage(size+2, size+2, QtGui.QImage.Format_ARGB32_Premultiplied)
        self.spotImage.fill(0)
        p = QtGui.QPainter(self.spotImage)
        p.setRenderHint(p.Antialiasing)
        p.translate(size*0.5+1, size*0.5+1)
        p.scale(size, size)
        self.paint(p, None, None)
        p.end()
        return QtGui.QPixmap(self.spotImage)


    def setBrush(self, brush):
        self.brush = fn.mkBrush(brush)
        self.update()
        
    def setPen(self, pen):
        self.pen = fn.mkPen(pen)
        self.update()
        
    def boundingRect(self):
        return self.path.boundingRect()
        
    def shape(self):
        return self.path
        
    def paint(self, p, *opts):
        p.setPen(self.pen)
        p.setBrush(self.brush)
        p.drawPath(self.path)
        
