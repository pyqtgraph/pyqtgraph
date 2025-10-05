import numpy as np

from ..Qt import QtGui
from ..Qt.QtCore import QPointF, QRectF
from .. import functions as fn
from .GraphicsObject import GraphicsObject
from .ScatterPlotItem import Symbols


__all__ = ['BoxplotItem']


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
            pen=None,
            brush=None,
            medianPen=None,
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
        
        `loc`:          Optional, used as coordinates for placing boxes, its length
                        must be the same as that of `data`.
        `data`:         Numpy 2D array or a list of arraylike object, user
                        should ensure all value is not NAN.
        `locAsX`:       If True, `loc` is regarded as x coordinates, 
                        otherwise as y, default to True.
        `width`:        Width of boxes, default to 0.8.
        `pen`:          The pen for drawing box outlines and whiskers, 
                        default to yellow.
        `brush`:        The brush for filling boxes.
        `medianPen`:    The pen for drawing median line, default to red.
        `outlier`:      If True, outlier points will be drew on the plot, 
                        default to True.
        `symbol`:       Symbol of outlier points, can be one of supported symbol
                        used in `ScatterPlotItem` or a custom `QPainterPath` symbol,
                        default to 'o'.
        `symbolSize`:   The size of outlier symbols, default to 10.
        `symbolPen`:    The pen for drawing outlines of outlier symbols.
        `symbolBrush`:  The brush for filling outlier symbols.
        '''
        self.opts.update(opts)
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
        if self.picture is None:
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
        width = 0.8 if self.opts["width"] is None else self.opts["width"]
        pen = fn.mkPen("y" if self.opts["pen"] is None else self.opts["pen"])
        brush = fn.mkBrush(self.opts["brush"])
        medianPen = fn.mkPen("r" if self.opts["medianPen"] is None else self.opts["medianPen"])
        
        p = QtGui.QPainter(self.picture)
        # for calculating bounding rect
        pw = pen.widthF() * 0.7072
        boxBounds = QRectF()
        pixelPadding = 0
        for pos, dataset in zip(loc, data):
            dataset = np.asarray(dataset)
            p75, median, p25 = np.percentile(dataset, [75, 50, 25])
            lower, upper = self.whiskerFunc(dataset)
            # get outlier data points if enabled
            if self.opts["outlier"]:
                mask = np.logical_or(dataset<lower, dataset>upper)
                self.outlierData[pos] = dataset[mask]
            
            p.setPen(pen)
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
            p.setBrush(brush)
            if locAsX:
                p.drawRect(QRectF(pos-width/2, p25, width, p75-p25))
            else:
                p.drawRect(QRectF(p25, pos-width/2, p75-p25, width))
            # median
            p.setPen(medianPen)
            if locAsX:
                p.drawLine(QPointF(pos-width/2, median), QPointF(pos+width/2, median))
            else:
                p.drawLine(QPointF(median, pos-width/2), QPointF(median, pos+width/2))
            # bounding rect
            if locAsX:
                rect = QRectF(pos-width/2, lower, width, upper-lower)
            else:
                rect = QRectF(lower, pos-width/2, upper-lower, width)
            if pen.isCosmetic():
                boxBounds |= rect
                pixelPadding = max(pixelPadding, pw)
            else:
                boxBounds |= rect.adjusted(-pw, -pw, pw, pw)
            
        p.end()
        self._boxBounds = boxBounds
        self._pixelPadding = pixelPadding
            
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
        symbolPen = fn.mkPen(self.opts["symbolPen"])
        symbolSize = 10 if self.opts["symbolSize"] is None else self.opts["symbolSize"]
        symbolBrush = fn.mkBrush(self.opts["symbolBrush"])

        p.setPen(symbolPen)
        p.setBrush(symbolBrush)
        tr = p.transform()
        for pos, outliers in self.outlierData.items():
            for o in outliers:
                x, y = (pos, o) if self.opts["locAsX"] else (o, pos)
                p.resetTransform()
                p.translate(*tr.map(x, y))
                p.scale(symbolSize, symbolSize)                
                p.drawPath(symbol)
                    
    def boundingRect(self):
        if self.picture is None:
            self.generatePicture()
        
        bpx = bpy = 0.0
        spx = spy = 0.0
        pxPad = self._pixelPadding
        symbolPad = 0.7072 * (10 if self.opts["symbolSize"] is None else self.opts["symbolSize"])
        if pxPad > 0 or symbolPad > 0:
            # determine length of pixel in local x, y directions
            px, py = self.pixelVectors()
            try:
                px = 0 if px is None else px.length()
            except OverflowError:
                px = 0
            try:
                py = 0 if py is None else py.length()
            except OverflowError:
                py = 0
            # return bounds expanded by pixel size
            if pxPad > 0:
                bpx = px * pxPad
                bpy = py * pxPad
            if symbolPad > 0:
                spx = px * symbolPad
                spy = py * symbolPad
        
        # bounding rect of boxes
        rect = self._boxBounds.adjusted(-bpx, -bpy, bpx, bpy)
        # bounding rect of outliers
        if self.opts["outlier"]:
            pos_min = min(self.outlierData.keys())
            pos_max = max(self.outlierData.keys())
            out_min = None
            out_max = None
            for v in self.outlierData.values():
                if len(v) == 0:
                    continue
                out_min = v.min() if out_min is None else min(out_min, v.min())
                out_max = v.max() if out_max is None else max(out_max, v.max())
            
            if out_min is not None and out_max is not None:
                if self.opts["locAsX"]:
                    rect |= QRectF(pos_min-spx, out_min-spy, pos_max-pos_min+2*spx, out_max-out_min+2*spy)
                else:
                    rect |= QRectF(out_min-spx, pos_min-spy, out_max-out_min+2*spx, pos_max-pos_min+2*spy)
        return rect

