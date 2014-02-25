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
        self.setOpts(**opts)
        
    def setOpts(self, **opts):
        self.setData(**opts)
        self.opts.update(opts)
        self.picture = None
        self._shape = None
        self.update()
        self.informViewBoundsChanged()
        
    def setPen(self, *args, **kargs):
        """Set the pen used to draw the curve."""
        self.setOpts(pen = fn.mkPen(*args, **kargs))


    def setBrush(self, *args, **kargs):
        """Set the brush used when filling the area under the curve"""
        self.setOpts(brush = fn.mkBrush(*args, **kargs))


    def setData(self, **kargs):
        
        def asarray(x):
            if x is None or np.isscalar(x) or isinstance(x, np.ndarray):
                return x
            return np.array(x)

        x = asarray(kargs.pop('x', None))
        self.x0 = asarray(kargs.pop('x0', None))
        x1 = asarray(kargs.pop('x1', None))
        self.width = asarray(kargs.pop('width', None))
        
        if self.x0 is None:
            if self.width is None:
                raise Exception('must specify either x0 or width')
            if x1 is not None:
                self.x0 = x1 - self.width
            elif x is not None:
                self.x0 = x - self.width/2.
            else:
                raise Exception('must specify at least one of x, x0, or x1')
        if self.width is None:
            if x1 is None:
                raise Exception('must specify either x1 or width')
            self.width = x1 - self.x0
            
        y = asarray(kargs.pop('y', None))
        self.y0 = asarray(kargs.pop('y0', None))
        y1 = asarray(kargs.pop('y1', None))
        self.height = asarray(kargs.pop('height', None))

        if self.y0 is None:
            if self.height is None:
                self.y0 = 0
            elif y1 is not None:
                self.y0 = y1 - self.height
            elif y is not None:
                self.y0 = y - self.height/2.
            else:
                self.y0 = 0
        if self.height is None:
            if y1 is None:
                raise Exception('must specify either y1 or height')
            self.height = y1 - self.y0


    def drawPicture(self):
        self.picture = QtGui.QPicture()
        self._shape = QtGui.QPainterPath()
        p = QtGui.QPainter(self.picture)

        pen = self.opts['pen']
        pens = self.opts['pens']

        if pen is None and pens is None:
            pen = getConfigOption('foreground')

        brush = self.opts['brush']
        brushes = self.opts['brushes']
        if brush is None and brushes is None:
            brush = (128, 128, 128)
        
        p.setPen(fn.mkPen(pen))
        p.setBrush(fn.mkBrush(brush))
        for i in range(len(self.x0)):
            if pens is not None:
                p.setPen(fn.mkPen(pens[i]))
            if brushes is not None:
                p.setBrush(fn.mkBrush(brushes[i]))
                
            if np.isscalar(self.x0):
                x = self.x0
            else:
                x = self.x0[i]
            if np.isscalar(self.y0):
                y = self.y0
            else:
                y = self.y0[i]
            if np.isscalar(self.width):
                w = self.width
            else:
                w = self.width[i]
            if np.isscalar(self.height):
                h = self.height
            else:
                h = self.height[i]


            rect = QtCore.QRectF(x, y, w, h)
            p.drawRect(rect)
            self._shape.addRect(rect)
            
        p.end()
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
