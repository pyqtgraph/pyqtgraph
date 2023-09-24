import numpy as np

from .. import functions as fn
from .. import getConfigOption
from .. import Qt
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject

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
            name=None,
            height=None,
            width=None,
            pen=None,
            brush=None,
            pens=None,
            brushes=None,
        )

        if 'pen' not in opts:
            opts['pen'] = getConfigOption('foreground')
        if 'brush' not in opts:
            opts['brush'] = (128, 128, 128)
        # the first call to _updateColors() will thus always be an update

        self._rectarray = Qt.internals.PrimitiveArray(QtCore.QRectF, 4)
        self._shape = None
        self.picture = None
        self.setOpts(**opts)
        
    def setOpts(self, **opts):
        self.opts.update(opts)
        self.picture = None
        self._shape = None
        self._prepareData()
        self._updateColors(opts)
        self.prepareGeometryChange()
        self.update()
        self.informViewBoundsChanged()

    def _updatePenWidth(self, pen):
        no_pen = pen is None or pen.style() == QtCore.Qt.PenStyle.NoPen
        if no_pen:
            return

        idx = pen.isCosmetic()
        self._penWidth[idx] = max(self._penWidth[idx], pen.widthF())

    def _updateColors(self, opts):
        # the logic here is to permit the user to update only data
        # without updating pens/brushes

        # update only if fresh pen/pens supplied
        if 'pen' in opts or 'pens' in opts:
            self._penWidth = [0, 0]

            if self.opts['pens'] is None:
                # pens not configured, use single pen
                pen = self.opts['pen']
                pen = fn.mkPen(pen)
                self._updatePenWidth(pen)
                self._sharedPen = pen
                self._pens = None
            else:
                # pens configured, ignore single pen (if any)
                pens = []
                for pen in self.opts['pens']:
                    if not isinstance(pen, QtGui.QPen):
                        pen = fn.mkPen(pen)
                    pens.append(pen)
                self._updatePenWidth(pen)
                self._sharedPen = None
                self._pens = pens

        # update only if fresh brush/brushes supplied
        if 'brush' in opts or 'brushes' in opts:
            if self.opts['brushes'] is None:
                # brushes not configured, use single brush
                brush = self.opts['brush']
                self._sharedBrush = fn.mkBrush(brush)
                self._brushes = None
            else:
                # brushes configured, ignore single brush (if any)
                brushes = []
                for brush in self.opts['brushes']:
                    if not isinstance(brush, QtGui.QBrush):
                        brush = fn.mkBrush(brush)
                    brushes.append(brush)
                self._sharedBrush = None
                self._brushes = brushes

        self._singleColor = (
            self._sharedPen is not None and
            self._sharedBrush is not None
        )

    def _getNormalizedCoords(self):
        def asarray(x):
            if x is None or np.isscalar(x) or isinstance(x, np.ndarray):
                return x
            return np.array(x)

        x = asarray(self.opts.get('x'))
        x0 = asarray(self.opts.get('x0'))
        x1 = asarray(self.opts.get('x1'))
        width = asarray(self.opts.get('width'))
        
        if x0 is None:
            if width is None:
                raise Exception('must specify either x0 or width')
            if x1 is not None:
                x0 = x1 - width
            elif x is not None:
                x0 = x - width/2.
            else:
                raise Exception('must specify at least one of x, x0, or x1')
        if width is None:
            if x1 is None:
                raise Exception('must specify either x1 or width')
            width = x1 - x0
            
        y = asarray(self.opts.get('y'))
        y0 = asarray(self.opts.get('y0'))
        y1 = asarray(self.opts.get('y1'))
        height = asarray(self.opts.get('height'))

        if y0 is None:
            if height is None:
                y0 = 0
            elif y1 is not None:
                y0 = y1 - height
            elif y is not None:
                y0 = y - height/2.
            else:
                y0 = 0
        if height is None:
            if y1 is None:
                raise Exception('must specify either y1 or height')
            height = y1 - y0

        # ensure x0 < x1 and y0 < y1
        t0, t1 = x0, x0 + width
        x0 = np.minimum(t0, t1, dtype=np.float64)
        x1 = np.maximum(t0, t1, dtype=np.float64)
        t0, t1 = y0, y0 + height
        y0 = np.minimum(t0, t1, dtype=np.float64)
        y1 = np.maximum(t0, t1, dtype=np.float64)

        # here, all of x0, y0, x1, y1 are numpy objects,
        # BUT could possibly be numpy scalars
        return x0, y0, x1, y1

    def _prepareData(self):
        x0, y0, x1, y1 = self._getNormalizedCoords()
        if x0.size == 0 or y0.size == 0:
            self._dataBounds = (None, None), (None, None)
            self._rectarray.resize(0)
            return

        xmn, xmx = np.min(x0), np.max(x1)
        ymn, ymx = np.min(y0), np.max(y1)
        self._dataBounds = (xmn, xmx), (ymn, ymx)

        self._rectarray.resize(max(x0.size, y0.size))
        memory = self._rectarray.ndarray()
        memory[:, 0] = x0
        memory[:, 1] = y0
        memory[:, 2] = x1 - x0
        memory[:, 3] = y1 - y0

    def _render(self, painter):
        multi_pen = self._pens is not None
        multi_brush = self._brushes is not None
        no_pen = (
            not multi_pen
            and self._sharedPen.style() == QtCore.Qt.PenStyle.NoPen
        )

        rects = self._rectarray.instances()

        if no_pen and multi_brush:
            for idx, rect in enumerate(rects):
                painter.fillRect(rect, self._brushes[idx])
        else:
            if not multi_pen:
                painter.setPen(self._sharedPen)
            if not multi_brush:
                painter.setBrush(self._sharedBrush)

            for idx, rect in enumerate(rects):
                if multi_pen:
                    painter.setPen(self._pens[idx])
                if multi_brush:
                    painter.setBrush(self._brushes[idx])

                painter.drawRect(rect)

    def drawPicture(self):
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        self._render(painter)
        painter.end()

    def paint(self, p, *args):
        if self._singleColor:
            p.setPen(self._sharedPen)
            p.setBrush(self._sharedBrush)
            drawargs = self._rectarray.drawargs()
            p.drawRects(*drawargs)
        else:
            if self.picture is None:
                self.drawPicture()
            self.picture.play(p)
            
    def shape(self):
        if self._shape is None:
            shape = QtGui.QPainterPath()
            rects = self._rectarray.instances()
            for rect in rects:
                shape.addRect(rect)
            self._shape = shape
        return self._shape

    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        return self.opts.get('name', None)

    def getData(self):
        return self.opts.get('x'),  self.opts.get('height')

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        # _penWidth is available after _updateColors()
        pw = self._penWidth[0] * 0.5
        # _dataBounds is available after _prepareData()
        bounds = self._dataBounds[ax]
        if bounds[0] is None or bounds[1] is None:
            return None, None

        return (bounds[0] - pw, bounds[1] + pw)

    def pixelPadding(self):
        # _penWidth is available after _updateColors()
        pw = (self._penWidth[1] or 1) * 0.5
        return pw

    def boundingRect(self):
        xmn, xmx = self.dataBounds(ax=0)
        if xmn is None or xmx is None:
            return QtCore.QRectF()
        ymn, ymx = self.dataBounds(ax=1)
        if ymn is None or ymx is None:
            return QtCore.QRectF()

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

        return QtCore.QRectF(xmn-px, ymn-py, (2*px)+xmx-xmn, (2*py)+ymx-ymn)
