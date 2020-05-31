from ..Qt import QtGui, QtCore
from .GraphicsObject import GraphicsObject
from .. import getConfigOption
from .. import functions as fn

__all__ = ['ErrorBarItem']

class ErrorBarItem(GraphicsObject):
    def __init__(self, **opts):
        """
        All keyword arguments are passed to setData().
        """
        GraphicsObject.__init__(self)
        self.opts = dict(
            x=None,
            y=None,
            height=None,
            width=None,
            top=None,
            bottom=None,
            left=None,
            right=None,
            beam=None,
            pen=None
        )
        self.setVisible(False)
        self.setData(**opts)

    def setData(self, **opts):
        """
        Update the data in the item. All arguments are optional.
        
        Valid keyword options are:
        x, y, height, width, top, bottom, left, right, beam, pen
        
        * x and y must be numpy arrays specifying the coordinates of data points.
        * height, width, top, bottom, left, right, and beam may be numpy arrays,
          single values, or None to disable. All values should be positive.
        * top, bottom, left, and right specify the lengths of bars extending
          in each direction.
        * If height is specified, it overrides top and bottom.
        * If width is specified, it overrides left and right.
        * beam specifies the width of the beam at the end of each bar.
        * pen may be any single argument accepted by pg.mkPen().
        
        This method was added in version 0.9.9. For prior versions, use setOpts.
        """
        self.opts.update(opts)
        self.setVisible(all(self.opts[ax] is not None for ax in ['x', 'y']))
        self.path = None
        self.update()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        
    def setOpts(self, **opts):
        # for backward compatibility
        self.setData(**opts)
        
    def drawPath(self):
        p = QtGui.QPainterPath()
        
        x, y = self.opts['x'], self.opts['y']
        if x is None or y is None:
            self.path = p
            return
        
        beam = self.opts['beam']
        
        
        height, top, bottom = self.opts['height'], self.opts['top'], self.opts['bottom']
        if height is not None or top is not None or bottom is not None:
            ## draw vertical error bars
            if height is not None:
                y1 = y - height/2.
                y2 = y + height/2.
            else:
                if bottom is None:
                    y1 = y
                else:
                    y1 = y - bottom
                if top is None:
                    y2 = y
                else:
                    y2 = y + top
            
            for i in range(len(x)):
                p.moveTo(x[i], y1[i])
                p.lineTo(x[i], y2[i])
                
            if beam is not None and beam > 0:
                x1 = x - beam/2.
                x2 = x + beam/2.
                if height is not None or top is not None:
                    for i in range(len(x)):
                        p.moveTo(x1[i], y2[i])
                        p.lineTo(x2[i], y2[i])
                if height is not None or bottom is not None:
                    for i in range(len(x)):
                        p.moveTo(x1[i], y1[i])
                        p.lineTo(x2[i], y1[i])
        
        width, right, left = self.opts['width'], self.opts['right'], self.opts['left']
        if width is not None or right is not None or left is not None:
            ## draw vertical error bars
            if width is not None:
                x1 = x - width/2.
                x2 = x + width/2.
            else:
                if left is None:
                    x1 = x
                else:
                    x1 = x - left
                if right is None:
                    x2 = x
                else:
                    x2 = x + right
            
            for i in range(len(x)):
                p.moveTo(x1[i], y[i])
                p.lineTo(x2[i], y[i])
                
            if beam is not None and beam > 0:
                y1 = y - beam/2.
                y2 = y + beam/2.
                if width is not None or right is not None:
                    for i in range(len(x)):
                        p.moveTo(x2[i], y1[i])
                        p.lineTo(x2[i], y2[i])
                if width is not None or left is not None:
                    for i in range(len(x)):
                        p.moveTo(x1[i], y1[i])
                        p.lineTo(x1[i], y2[i])
                    
        self.path = p
        self.prepareGeometryChange()
        
        
    def paint(self, p, *args):
        if self.path is None:
            self.drawPath()
        pen = self.opts['pen']
        if pen is None:
            pen = getConfigOption('foreground')
        p.setPen(fn.mkPen(pen))
        p.drawPath(self.path)
            
    def boundingRect(self):
        if self.path is None:
            self.drawPath()
        return self.path.boundingRect()
    
        
