from ..Qt import QtGui, QtCore
from .GraphicsObject import GraphicsObject
from .. import getConfigOption
from .. import functions as fn
import numpy as np


__all__ = ['BarGraphItem']

class BarGraphItem(GraphicsObject):
    def __init__(self, **opts):
        """
        Valid keyword options are:
        x, x0, x1, y, y0, y1, width, height, pen, brush
        
        x specifies the x-position of the center of the bar.
        x0, x1 specify left and right edges of the bar, respectively.
        width specifies distance from x0 to x1.
        You may specify any combination:
            
            x, width
            x0, width
            x1, width
            x0, x1
            
        Likewise y, y0, y1, and height. 
        If only height is specified, then y0 will be set to 0
        
        Example uses:
        
            BarGraphItem(x=range(5), height=[1,5,2,4,3], width=0.5)
            
        
        """
        GraphicsObject.__init__(self)
        self.opts = dict(
            x=None,
            y=None,
            x0=None,
            y0=None,
            x1=None,
            y1=None,
            height=None,
            width=None,
            pen=None,
            brush=None,
            pens=None,
            brushes=None,
        )
        self._shape = None
        self.picture = None
        if opts:
            self.setOpts(**opts)
        
    def setOpts(self, **opts):
        self.opts.update(opts)
        self.checkOpts()
        self.picture = None
        self._shape = None
        self.update()
        self.informViewBoundsChanged()
        
    def checkOpts(self):
        def get_styles(scalar, multiple, default, func):
            mult = self.opts[multiple]
            if mult is None:
                sclr = self.opts.get(scalar, default)
                mult = [sclr, ] * len(self.x0)
            return [func(m) for m in mult]
        
        def array_from_opts(label, label0, label1, dimension):
        
            def asarray(x):
                if x is None or np.isscalar(x) or isinstance(x, np.ndarray):
                    return x
                return np.array(x)

            l = asarray(self.opts.get(label))
            l0 = asarray(self.opts.get(label0))
            l1 = asarray(self.opts.get(label1))
            dim = asarray(self.opts.get(dimension))
        
            if l0 is None:
                if dim is None:
                    raise TypeError('must specify either'
                                    ' %s or %s' % (label0, dimension))
                if l1 is not None:
                    l0 = l1 - dim
                elif l is not None:
                    l0 = l - dim/2.
                else:
                    l0 = 0
            if dim is None:
                if l1 is None:
                    raise TypeError('must specify either '
                                    '%s or %s' % (label1, dimension))
                dim = l1 - l0
            if np.isscalar(l0) and np.isscalar(dim):
                raise TypeError('At listone parameter must be iterable')
            return (np.array([l0, ] * len(dim)) if np.isscalar(l0) else l0,
                    np.array([dim, ] * len(l0)) if np.isscalar(dim) else dim)

        # check and save the values
        self.x0, self.widths = array_from_opts('x', 'x0', 'x1', 'width')
        self.y0, self.heights = array_from_opts('y', 'y0', 'y1', 'height')
        self.pens = get_styles('pen', 'pens',
                               getConfigOption('foreground'), fn.mkPen)
        self.brushes = get_styles('brush', 'brushes',
                                  (128, 128, 128), fn.mkBrush)

    def drawPicture(self):
        self.picture = QtGui.QPicture()
        self._shape = QtGui.QPainterPath()
        pict = QtGui.QPainter(self.picture)
        for x, y, width, height, pen, brush in zip(self.x0, self.y0,
                                                   self.widths, self.heights,
                                                   self.pens, self.brushes):
            pict.setPen(pen)
            pict.setBrush(brush)
            rect = QtCore.QRectF(x, y, width, height)
            pict.drawRect(rect)
            self._shape.addRect(rect)
        pict.end()
        self.prepareGeometryChange()
        
        
    def paint(self, p, *args):
        if self.picture is None:
            self.drawPicture()
        self.picture.play(p)
            
    def boundingRect(self):
        if self.picture is None:
            self.drawPicture()
        return QtCore.QRectF(self.picture.boundingRect())
    
    def shape(self):
        if self.picture is None:
            self.drawPicture()
        return self._shape
