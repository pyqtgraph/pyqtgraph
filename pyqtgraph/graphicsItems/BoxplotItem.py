import numpy as np

from ..Qt import QtGui
from ..Qt.QtCore import QPointF, QRectF
from .. import functions as fn
from .GraphicsObject import GraphicsObject
from .ScatterPlotItem import Symbols


__all__ = ['BoxplotItem']

DEFAULT_BOX_WIDTH = 0.8
DEFAULT_SYM_SIZE  = 10

def IQR_1p5(data):
    '''
    use 1.5IQR to get whisker boundaries
    returns (lower whisker, upper whisker)
    '''
    data = np.asarray(data)
    p75, p25 = np.percentile(data, [75, 25])
    upper_theory = p75 + 1.5 * (p75 - p25)
    lower_theory = p25 - 1.5 * (p75 - p25)
    upper = np.max(data[data<=upper_theory])
    lower = np.min(data[data>=lower_theory])
    return lower, upper


def validateWhiskerFunc(func):
    valid = False
    isNumber = lambda n: isinstance(n, (int, float, np.number))
    try:
        l, h = func([1, 2, 3])
        if isNumber(l) and isNumber(h):
            valid = True
    except (TypeError, ValueError):
        # when func is not callable or 
        # returned value is not a tuple of two objects
        valid = False
    
    return valid


class BoxplotItem(GraphicsObject):
    def __init__(self, **opts):
        GraphicsObject.__init__(self)
        self.opts = dict(
            loc=None,
            data=None,
            locAsX=True,
            width=None,
            pen='y',
            brush=None,
            medianPen='r',
            outlier=True,
            symbol=None,
            symbolSize=None,
            symbolPen=None,
            symbolBrush=None
        )
        self.setWhiskerFunc(IQR_1p5)
        self.setData(**opts)
    
    def setData(self, **opts):
        '''
        Keyword Arguments
        -----------------

        `loc`:          (Optional) array-like. Coordinates for placing boxes. Its length must be the same as that of `data`.

        `data`:         Array-like of array-like. Numpy 2D array or list of arrays. User should ensure all values are not NaN.

        `locAsX`:       If True, `loc` is regarded as x-coordinates, otherwise y. Default is True.

        `width`:        Width of boxes. Default is 0.8.

        `pen`:          Pen for drawing box outlines and whiskers. Default is yellow. Hidden if None.

        `brush`:        Brush for filling boxes. Default is None.

        `medianPen`:    Pen for drawing median line. Default is red. Hidden if None.

        `outlier`:      If True, outlier points will be drawn on the plot. Default is True.

        `symbol`:       Symbol for outlier points, can be any supported symbol used in `ScatterPlotItem`, 
                        or a custom `QPainterPath`. Default is `'o'`.

        `symbolSize`:   Size of outlier symbols. Default is 10.

        `symbolPen`:    Pen for drawing outlines of outlier symbols.

        `symbolBrush`:  Brush for filling outlier symbols.
        '''
        self.opts.update(opts)

        if self.opts["width"] is None:
            self.opts["width"] = DEFAULT_BOX_WIDTH

        if self.opts["pen"] is None and \
           self.opts["brush"] is None and \
           self.opts["medianPen"] is None:
            # set width to 0 when box is not drew
            self.opts["width"] = 0
        
        # prepare pen and brush object
        self._pen = fn.mkPen(self.opts["pen"])
        self._brush = fn.mkBrush(self.opts["brush"])
        self._medianPen = fn.mkPen(self.opts["medianPen"])
        self._symbolPen = fn.mkPen(self.opts["symbolPen"])
        self._symbolBrush = fn.mkBrush(self.opts["symbolBrush"])
        
        self._dataBoundRect = None
        self._penWidth = self._pen.widthF() if self._pen.isCosmetic() else 0
        self._symbolSize = self.opts["symbolSize"] or DEFAULT_SYM_SIZE

        self.picture = None
        self.outlierData = {}
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
    
    def setWhiskerFunc(self, func):
        '''
        Use a custome function to get whisker boundaries.
        `func` must accept 1d arraylike (np.array, list, set, etc...) as argument,
        and returns a tuple of (lower whisker, upper whisker)
        '''
        if validateWhiskerFunc(func):
            self.whiskerFunc = func
            self.picture = None
            self.outlierData = {}
        else:
            print(f"{func} is not a valid whisker function")
    
    def generatePicture(self):
        self.picture = QtGui.QPicture()
        
        loc, data = self.opts["loc"], self.opts["data"]
        # data should be a 2d numpy array or a list of array-like
        if data is None or \
           not (isinstance(data, np.ndarray) or isinstance(data, list)):
            return
        # loc decides where to draw boxes, it should be the same size as data
        if isinstance(loc, list) or isinstance(loc, np.ndarray):
            if len(loc) != len(data):
                raise ValueError(f"len of `loc` ({len(loc)}) and `data` ({len(data)}) should be the same")
        else:
            loc = np.arange(len(data))
        
        locAsX = self.opts["locAsX"]
        width = self.opts["width"]
        
        p = QtGui.QPainter(self.picture)
        for pos, dataset in zip(loc, data):
            dataset = np.asarray(dataset)
            p75, median, p25 = np.percentile(dataset, [75, 50, 25])
            lower, upper = self.whiskerFunc(dataset)
            # get outlier data points if enabled
            if self.opts["outlier"]:
                mask = np.logical_or(dataset<lower, dataset>upper)
                self.outlierData[pos] = dataset[mask]
            
            # box width to 0 means hide box lines
            if width == 0:
                continue
            
            p.setPen(self._pen)
            # whiskers
            if locAsX:
                p.drawLine(QPointF(pos-width/4, upper), QPointF(pos+width/4, upper))
                p.drawLine(QPointF(pos-width/4, lower), QPointF(pos+width/4, lower))
                p.drawLine(QPointF(pos, upper), QPointF(pos, p75))
                p.drawLine(QPointF(pos, lower), QPointF(pos, p25))
            else:
                p.drawLine(QPointF(upper, pos-width/4), QPointF(upper, pos+width/4))
                p.drawLine(QPointF(lower, pos-width/4), QPointF(lower, pos+width/4))
                p.drawLine(QPointF(upper, pos), QPointF(p75, pos))
                p.drawLine(QPointF(lower, pos), QPointF(p25, pos))
            # box
            p.setBrush(self._brush)
            if locAsX:
                p.drawRect(QRectF(pos-width/2, p25, width, p75-p25))
            else:
                p.drawRect(QRectF(p25, pos-width/2, p75-p25, width))
            # median
            p.setPen(self._medianPen)
            if locAsX:
                p.drawLine(QPointF(pos-width/2, median), QPointF(pos+width/2, median))
            else:
                p.drawLine(QPointF(median, pos-width/2), QPointF(median, pos+width/2))
        p.end()
            
    def paint(self, p, *args):
        if self.picture is None:
            self.generatePicture()
        p.drawPicture(0, 0, self.picture)

        if not self.opts["outlier"]:
            return

        # outlier related style
        s = self.opts["symbol"]
        if isinstance(s, str) and s in Symbols:
            symbol = Symbols[s]
        elif isinstance(s, QtGui.QPainterPath):
            symbol = s
        else:
            symbol = Symbols["o"]
        
        p.setPen(self._symbolPen)
        p.setBrush(self._symbolBrush)
        tr = p.transform()
        for pos, outliers in self.outlierData.items():
            for o in outliers:
                x, y = (pos, o) if self.opts["locAsX"] else (o, pos)
                p.resetTransform()
                p.translate(*tr.map(x, y))
                p.scale(self._symbolSize, self._symbolSize)                
                p.drawPath(symbol)
                    
    def boundingRect(self):
        xmn, xmx = self.dataBounds(ax=0)
        ymn, ymx = self.dataBounds(ax=1)
        rect = QRectF(xmn, ymn, xmx-xmn, ymx-ymn)
        
        px = py = 0
        pxPad = self.pixelPadding()
        if pxPad > 0:
            # determine length of pixel in local x, y directions
            px, py = self.pixelVectors()
            px = 0 if px is None else px.length()
            py = 0 if py is None else py.length()
            # return bounds expanded by pixel size
            px *= pxPad
            py *= pxPad
        
        # bounding rect of boxes
        rect = rect.adjusted(-px, -py, px, py)
        return rect

    def calculateDataBounds(self):
        loc, data = self.opts["loc"], self.opts["data"]
        if data is None:
            return QRectF()

        lst_lower = []
        lst_upper = []
        for dataset in data:
            dataset = np.asarray(dataset)
            if self.opts["outlier"]:
                lower, upper = np.min(dataset), np.max(dataset)
            else:
                lower, upper = self.whiskerFunc(dataset)
            lst_lower.append(lower)
            lst_upper.append(upper)
        miny = np.min(lst_lower)
        maxy = np.max(lst_upper)

        if loc is None:
            loc = np.arange(len(data))
        loc = np.array(loc)
        minx, maxx = np.min(loc), np.max(loc)
        width = self.opts["width"]
        minx -= width/2
        maxx += width/2

        if not self.opts["locAsX"]:
            minx, maxx, miny, maxy = miny, maxy, minx, maxx

        return QRectF(QPointF(minx, miny), QPointF(maxx, maxy))

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if self._dataBoundRect is None:
            self._dataBoundRect = self.calculateDataBounds()
            
        if ax == 0:
            return [self._dataBoundRect.left(), self._dataBoundRect.right()]
        else:
            return [self._dataBoundRect.top(), self._dataBoundRect.bottom()]

    def pixelPadding(self):
        symPadding = 0.7072 * self._symbolSize if self.opts["outlier"] else 0
        penPadding = 0.5 * self._penWidth
        return max(symPadding, penPadding)
