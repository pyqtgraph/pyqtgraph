import numpy as np

from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtGui
from .ArrowItem import ArrowItem
from .GraphicsObject import GraphicsObject

__all__ = ['ErrorBarItem']

# ArrowItem angles for an arrow tip pointing in each direction.
# ArrowItem with angle=0 has its tip on the left; rotation is counter-clockwise
# in screen space (y increases downward), so angle=+90 puts the tip at the top.
_LIMIT_ARROW_ANGLE = {
    'top': 90,      # tip up
    'bottom': -90,  # tip down
    'left': 0,      # tip left
    'right': 180,   # tip right
}


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
            pen=None,
            topLimit=None,
            bottomLimit=None,
            leftLimit=None,
            rightLimit=None,
            arrowHeadLen=10,
            arrowHeadWidth=None,
            arrowTipAngle=25,
        )
        self._arrowItems = []
        self.setVisible(False)
        self.setData(**opts)

    def setData(self, **opts):
        """
        Update the data in the item. All arguments are optional.

        Valid keyword options are:
        x, y, height, width, top, bottom, left, right, beam, pen,
        topLimit, bottomLimit, leftLimit, rightLimit,
        arrowHeadLen, arrowHeadWidth, arrowTipAngle

          * x and y must be numpy arrays specifying the coordinates of data points.
          * height, width, top, bottom, left, right, and beam may be numpy arrays,
            single values, or None to disable. All values should be positive.
          * top, bottom, left, and right specify the lengths of bars extending
            in each direction.
          * If height is specified, it overrides top and bottom.
          * If width is specified, it overrides left and right.
          * beam specifies the width of the beam at the end of each bar.
          * pen may be any single argument accepted by pg.mkPen().
          * topLimit, bottomLimit, leftLimit, rightLimit may be boolean arrays
            (one entry per point) or a single bool. Where True, the corresponding
            bar end is drawn as an arrow head (pointing away from the data point)
            instead of a perpendicular beam, indicating that the value is a limit
            rather than a measurement. The beam on that side is suppressed for
            those points.
          * arrowHeadLen is the length of the arrow head in pixels. default=10
          * arrowHeadWidth, if given, overrides arrowTipAngle and sets the half
            width of the arrow base in pixels.
          * arrowTipAngle is the full angle (degrees) of the arrow tip when
            arrowHeadWidth is not specified. default=25

        This method was added in version 0.9.9. For prior versions, use setOpts.
        """
        self.opts.update(opts)
        self.setVisible(all(self.opts[ax] is not None for ax in ['x', 'y']))
        self.path = None
        self._rebuildArrows()
        self.update()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

    def setOpts(self, **opts):
        # for backward compatibility
        self.setData(**opts)

    def _limitMask(self, key, n):
        """Return a boolean ndarray of length n for the named limit option, or None if no limits."""
        val = self.opts.get(key)
        if val is None or val is False:
            return None
        if val is True:
            return np.ones(n, dtype=bool)
        arr = np.asarray(val, dtype=bool)
        if arr.ndim == 0:
            return np.full(n, bool(arr), dtype=bool)
        if not arr.any():
            return None
        return arr

    def _clearArrows(self):
        for a in self._arrowItems:
            a.setParentItem(None)
            scene = a.scene()
            if scene is not None:
                scene.removeItem(a)
        self._arrowItems = []

    def _arrowStyle(self):
        style = {
            'headLen': self.opts.get('arrowHeadLen') or 10,
            'pxMode': True,
            'tailLen': None,
        }
        headWidth = self.opts.get('arrowHeadWidth')
        if headWidth is not None:
            style['headWidth'] = headWidth
        else:
            style['tipAngle'] = self.opts.get('arrowTipAngle') or 25
        pen = self.opts.get('pen')
        if pen is None:
            pen = getConfigOption('foreground')
        style['pen'] = fn.mkPen(pen)
        style['brush'] = fn.mkBrush(pen)
        return style

    def _rebuildArrows(self):
        self._clearArrows()

        x, y = self.opts['x'], self.opts['y']
        if x is None or y is None:
            return
        x = np.asarray(x)
        y = np.asarray(y)
        n = len(x)
        if n == 0:
            return

        height, top, bottom = self.opts['height'], self.opts['top'], self.opts['bottom']
        width, right, left = self.opts['width'], self.opts['right'], self.opts['left']

        # Per-direction endpoint coordinates, broadcast to length n.
        directions = []
        if height is not None or top is not None:
            if height is not None:
                yEnd = y + np.broadcast_to(np.asarray(height) / 2., (n,))
            else:
                yEnd = y + np.broadcast_to(np.asarray(top), (n,))
            directions.append(('top', x, yEnd))
        if height is not None or bottom is not None:
            if height is not None:
                yEnd = y - np.broadcast_to(np.asarray(height) / 2., (n,))
            else:
                yEnd = y - np.broadcast_to(np.asarray(bottom), (n,))
            directions.append(('bottom', x, yEnd))
        if width is not None or right is not None:
            if width is not None:
                xEnd = x + np.broadcast_to(np.asarray(width) / 2., (n,))
            else:
                xEnd = x + np.broadcast_to(np.asarray(right), (n,))
            directions.append(('right', xEnd, y))
        if width is not None or left is not None:
            if width is not None:
                xEnd = x - np.broadcast_to(np.asarray(width) / 2., (n,))
            else:
                xEnd = x - np.broadcast_to(np.asarray(left), (n,))
            directions.append(('left', xEnd, y))

        style = self._arrowStyle()
        for side, xEnd, yEnd in directions:
            mask = self._limitMask(f'{side}Limit', n)
            if mask is None:
                continue
            angle = _LIMIT_ARROW_ANGLE[side]
            for i in np.flatnonzero(mask):
                arrow = ArrowItem(angle=angle, **style)
                arrow.setParentItem(self)
                arrow.setPos(float(xEnd[i]), float(yEnd[i]))
                self._arrowItems.append(arrow)

    def drawPath(self):
        p = QtGui.QPainterPath()

        x, y = self.opts['x'], self.opts['y']
        if x is None or y is None:
            self.path = p
            return

        beam = self.opts['beam']
        n = len(x)

        topLim = self._limitMask('topLimit', n)
        bottomLim = self._limitMask('bottomLimit', n)
        leftLim = self._limitMask('leftLimit', n)
        rightLim = self._limitMask('rightLimit', n)

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
                    sel = ~topLim if topLim is not None else None
                    if sel is None or sel.any():
                        y2s = fn.interweaveArrays(y2, y2)
                        if sel is not None:
                            mask = np.repeat(sel, 2)
                            topEnds = fn.arrayToQPath(x1_x2[mask], y2s[mask], connect="pairs")
                        else:
                            topEnds = fn.arrayToQPath(x1_x2, y2s, connect="pairs")
                        p.addPath(topEnds)

                if height is not None or bottom is not None:
                    sel = ~bottomLim if bottomLim is not None else None
                    if sel is None or sel.any():
                        y1s = fn.interweaveArrays(y1, y1)
                        if sel is not None:
                            mask = np.repeat(sel, 2)
                            bottomEnds = fn.arrayToQPath(x1_x2[mask], y1s[mask], connect="pairs")
                        else:
                            bottomEnds = fn.arrayToQPath(x1_x2, y1s, connect="pairs")
                        p.addPath(bottomEnds)

        width, right, left = self.opts['width'], self.opts['right'], self.opts['left']
        if width is not None or right is not None or left is not None:
            ## draw horizontal error bars
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
                    sel = ~rightLim if rightLim is not None else None
                    if sel is None or sel.any():
                        x2s = fn.interweaveArrays(x2, x2)
                        if sel is not None:
                            mask = np.repeat(sel, 2)
                            rightEnds = fn.arrayToQPath(x2s[mask], y1_y2[mask], connect="pairs")
                        else:
                            rightEnds = fn.arrayToQPath(x2s, y1_y2, connect="pairs")
                        p.addPath(rightEnds)

                if width is not None or left is not None:
                    sel = ~leftLim if leftLim is not None else None
                    if sel is None or sel.any():
                        x1s = fn.interweaveArrays(x1, x1)
                        if sel is not None:
                            mask = np.repeat(sel, 2)
                            leftEnds = fn.arrayToQPath(x1s[mask], y1_y2[mask], connect="pairs")
                        else:
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
