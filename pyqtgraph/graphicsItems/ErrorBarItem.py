import copy

import numpy as np

from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtGui
from .GraphicsObject import GraphicsObject

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
        self._orig_opts = {}
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
        # save the original data in case of the log mode set
        _orig_opts = opts.copy()
        if _orig_opts.get('height', None) is not None:
            _orig_opts['top'] = _orig_opts['bottom'] = _orig_opts['height'] / 2
        if _orig_opts.get('width', None) is not None:
            _orig_opts['left'] = _orig_opts['right'] = _orig_opts['width'] / 2
        self._orig_opts.update(_orig_opts)

        self.opts.update(opts)
        self.setVisible(all(self.opts[ax] is not None for ax in ['x', 'y']))
        self.path = None
        self.update()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        
    def setOpts(self, **opts):
        # for backward compatibility
        self.setData(**opts)
        
    def setLogMode(self, x=None, y=None):
        if x is not None:
            self.opts['left'] = copy.copy(self._orig_opts.get('left', None))
            self.opts['right'] = copy.copy(self._orig_opts.get('right', None))
            self.opts['x'] = copy.copy(self._orig_opts.get('x', None))
            self.opts['width'] = copy.copy(self._orig_opts.get('width', None))
        if x is True and self.opts['x'] is not None:
            # to use 'left' and 'right'
            self.opts['width'] = None
            _x = self.opts['x']
            if self.opts['left'] is not None:
                x_err = (np.full(_x.shape, self.opts['left'])
                         if np.isscalar(self.opts['left'])
                         else self.opts['left'].copy())
                valid = (_x > 0) & (_x - x_err > 0)
                left = np.full(_x.shape, np.inf, dtype=_x.dtype)
                left[valid] = np.log10(_x[valid]) - np.log10(_x[valid] - x_err[valid])
                self.opts['left'] = left
            if self.opts['right'] is not None:
                x_err = (np.full(_x.shape, self.opts['right'])
                         if np.isscalar(self.opts['right'])
                         else self.opts['right'].copy())
                valid = (_x > 0) & (_x + x_err > 0)
                right = np.full(_x.shape, np.inf, dtype=_x.dtype)
                right[valid] = np.log10(_x[valid] + x_err[valid]) - np.log10(_x[valid])
                self.opts['right'] = right
            valid = _x > 0
            _x[valid] = np.log10(_x[valid])
            _x[~valid] = np.nan
            self.opts['x'] = _x

        if y is not None:
            self.opts['bottom'] = copy.copy(self._orig_opts.get('bottom', None))
            self.opts['top'] = copy.copy(self._orig_opts.get('top', None))
            self.opts['y'] = copy.copy(self._orig_opts.get('y', None))
            self.opts['height'] = copy.copy(self._orig_opts.get('height', None))
        if y is True and self.opts['y'] is not None:
            # to use 'top' and 'bottom'
            self.opts['height'] = None
            _y = self.opts['y']
            if self.opts['bottom'] is not None:
                y_err = (np.full(_y.shape, self.opts['bottom'])
                         if np.isscalar(self.opts['bottom'])
                         else self.opts['bottom'].copy())
                valid = (_y > 0) & (_y - y_err > 0)
                bottom = np.full(_y.shape, np.inf, dtype=_y.dtype)
                bottom[valid] = np.log10(_y[valid]) - np.log10(_y[valid] - y_err[valid])
                self.opts['bottom'] = bottom
            if self.opts['top'] is not None:
                y_err = (np.full(_y.shape, self.opts['top'])
                         if np.isscalar(self.opts['top'])
                         else self.opts['top'].copy())
                valid = (_y > 0) & (_y + y_err > 0)
                top = np.full(_y.shape, np.inf, dtype=_y.dtype)
                top[valid] = np.log10(_y[valid] + y_err[valid]) - np.log10(_y[valid])
                self.opts['top'] = top
            valid = _y > 0
            _y[valid] = np.log10(_y[valid])
            _y[~valid] = np.nan
            self.opts['y'] = _y

        if x is not None or y is not None:
            self.setData(**self.opts)

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

            xs = fn.interweaveArrays(x, x)
            y1_y2 = fn.interweaveArrays(y1, y2)
            verticalLines = fn.arrayToQPath(xs, y1_y2, connect="pairs")
            p.addPath(verticalLines)

            if beam is not None and beam > 0:
                x1 = x - beam/2.
                x2 = x + beam/2.

                x1_x2 = fn.interweaveArrays(x1, x2)
                if height is not None or top is not None:
                    y2s = fn.interweaveArrays(y2, y2)
                    topEnds = fn.arrayToQPath(x1_x2, y2s, connect="pairs")
                    p.addPath(topEnds)

                if height is not None or bottom is not None:
                    y1s = fn.interweaveArrays(y1, y1)
                    bottomEnds = fn.arrayToQPath(x1_x2, y1s, connect="pairs")
                    p.addPath(bottomEnds)

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
            
            ys = fn.interweaveArrays(y, y)
            x1_x2 = fn.interweaveArrays(x1, x2)
            ends = fn.arrayToQPath(x1_x2, ys, connect='pairs')
            p.addPath(ends)

            if beam is not None and beam > 0:
                y1 = y - beam/2.
                y2 = y + beam/2.
                y1_y2 = fn.interweaveArrays(y1, y2)
                if width is not None or right is not None:
                    x2s = fn.interweaveArrays(x2, x2)
                    rightEnds = fn.arrayToQPath(x2s, y1_y2, connect="pairs")
                    p.addPath(rightEnds)

                if width is not None or left is not None:
                    x1s = fn.interweaveArrays(x1, x1)
                    leftEnds = fn.arrayToQPath(x1s, y1_y2, connect="pairs")
                    p.addPath(leftEnds)
                    
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
    
        
