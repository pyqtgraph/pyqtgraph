import numpy as np
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from .. import functions as fn
from .. import configStyle
from ..style.core import (
    ConfigColorHint,
    ConfigKeyHint,
    ConfigValueHint,
    initItemStyle)
from ..Qt import QtCore, QtWidgets, QtGui
from ..Point import Point
from ..Qt import QtCore, QtGui
from .UIGraphicsItem import UIGraphicsItem

__all__ = ['GridItem']


optsHint = TypedDict('optsHint',
                     {'lineColor' : ConfigColorHint,
                      'textColor' : ConfigColorHint,
                      'tickSpacingX' : List[Optional[float]],
                      'tickSpacingY' : List[Optional[float]],

                      # "style" option to stay compatible with previous version
                      'pen' : Optional[QtGui.QPen],
                      'textPen' : Optional[QtGui.QPen],
                      'tickSpacing': Tuple[List[Optional[float]], List[Optional[float]]]},
                     total=False)
# kwargs are not typed because mypy has not ye included Unpack[Typeddict]

class GridItem(UIGraphicsItem):
    """
    **Bases:** :class:`UIGraphicsItem <pyqtgraph.UIGraphicsItem>`

    Displays a rectangular grid of lines indicating major divisions within a coordinate system.
    Automatically determines what divisions to use.
    """

    def __init__(self, pen: str='default',
                       textPen: str='default',
                       **kwargs) -> None:
        UIGraphicsItem.__init__(self)
        #QtWidgets.QGraphicsItem.__init__(self, *args)
        #self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemClipsToShape)
        #self.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)

        # Store style options in opts dict
        self.opts: optsHint = {}
        # Get default stylesheet
        initItemStyle(self, 'GridItem', configStyle)
        # Update style if needed
        if len(kwargs)>0:
            self.setStyle(**kwargs)


        self.setPen(pen)
        self.setTextPen(textPen)
        self.setTickSpacing(x=[None, None, None], y=[None, None, None])

    ##############################################################
    #
    #                   Style methods
    #
    ##############################################################

    def setLineColor(self, lineColor: ConfigColorHint) -> None:
        """
        Set the lineColor
        """
        self.opts['lineColor'] = lineColor

    def lineColor(self) -> ConfigColorHint:
        """
        Get the current lineColor
        """
        return self.opts['lineColor']

    def setTextColor(self, textColor: ConfigColorHint) -> None:
        """
        Set the textColor
        """
        self.opts['textColor'] = textColor

    def textColor(self) -> ConfigColorHint:
        """
        Get the current textColor
        """
        return self.opts['textColor']

    def setTickSpacingX(self, tickSpacingX: List[Optional[float]]) -> None:
        """
        Set the tickSpacing along the x axis
        """
        self.opts['tickSpacingX'] = tickSpacingX

    def tickSpacingX(self) -> List[Optional[float]]:
        """
        Get the current tickSpacing along the x axis
        """
        return self.opts['tickSpacingX']

    def setTickSpacingY(self, tickSpacingY: List[Optional[float]]) -> None:
        """
        Set the tickSpacing along the y axis
        """
        self.opts['tickSpacingY'] = tickSpacingY

    def tickSpacingY(self) -> List[Optional[float]]:
        """
        Get the current tickSpacing along the y axis
        """
        return self.opts['tickSpacingY']

    def setPen(self, *args, **kwargs) -> None:
        """Set the pen used to draw the grid."""
        if kwargs == {} and (args == () or args == ('default',)):
            self.opts['pen'] = fn.mkPen(self.opts['lineColor'])
        else:
            self.opts['pen'] = fn.mkPen(*args, **kwargs)

        self.picture = None
        self.update()

    def setTextPen(self, *args, **kwargs) -> None:
        """Set the pen used to draw the texts."""
        if kwargs == {} and (args == () or args == ('default',)):
            self.opts['textPen'] = fn.mkPen(self.opts['textColor'])
        else:
            if args == (None,):
                self.opts['textPen'] = None
            else:
                self.opts['textPen'] = fn.mkPen(*args, **kwargs)

        self.picture = None
        self.update()

    def setTickSpacing(self, x: List[Optional[float]]=None,
                             y: List[Optional[float]]=None) -> None:
        """
        Set the grid tick spacing to use.

        Tick spacing for each axis shall be specified as an array of
        descending values, one for each tick scale. When the value
        is set to None, grid line distance is chosen automatically
        for this particular level.

        Example:
            Default setting of 3 scales for each axis:
            setTickSpacing(x=[None, None, None], y=[None, None, None])

            Single scale with distance of 1.0 for X axis, Two automatic
            scales for Y axis:
            setTickSpacing(x=[1.0], y=[None, None])

            Single scale with distance of 1.0 for X axis, Two scales
            for Y axis, one with spacing of 1.0, other one automatic:
            setTickSpacing(x=[1.0], y=[1.0, None])
        """
        self.opts['tickSpacing'] = (x or self.opts['tickSpacing'][0],
                                    y or self.opts['tickSpacing'][1])

        self.grid_depth = max([len(s) for s in self.opts['tickSpacing']])

        self.picture = None
        self.update()

    def setStyle(self, **kwargs) -> None:
        """
        Set the style of the GridItem.

        Parameters
        ----------
        lineColor (ConfigColorHint):
            Color of the GridItem line
        textColor (ConfigColorHint):
            Color of the GridItem text
        tickSpacingX (List[Optional[float]]):
            Set the grid tick spacing to use
        tickSpacingY (List[Optional[float]]):
            Set the grid tick spacing to use
        """
        for k, v in kwargs.items():
            # If the attr is a valid entry of the stylesheet
            if k in configStyle['GridItem'].keys():
                fun = getattr(self, 'set{}{}'.format(k[:1].upper(), k[1:]))
                fun(v)
            else:
                raise ValueError('Your argument: "{}" is not a valid style argument.'.format(k))

    ##############################################################
    #
    #                   Item method
    #
    ##############################################################

    def viewRangeChanged(self) -> None:
        UIGraphicsItem.viewRangeChanged(self)
        self.picture = None
        #UIGraphicsItem.viewRangeChanged(self)
        #self.update()

    def paint(self, p: QtGui.QPainter,
                    opt: QtWidgets.QStyleOptionGraphicsItem,
                    widget: QtWidgets.QWidget) -> None:
        #p.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100)))
        #p.drawRect(self.boundingRect())
        #UIGraphicsItem.paint(self, p, opt, widget)
        ### draw picture
        if self.picture is None:
            #print "no pic, draw.."
            self.generatePicture()
        p.drawPicture(QtCore.QPointF(0, 0), self.picture)
        #p.setPen(QtGui.QPen(QtGui.QColor(255,0,0)))
        #p.drawLine(0, -100, 0, 100)
        #p.drawLine(-100, 0, 100, 0)
        #print "drawing Grid."

    def generatePicture(self) -> None:
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter()
        p.begin(self.picture)

        vr = self.getViewWidget().rect()
        unit = self.pixelWidth(), self.pixelHeight()
        dim = [vr.width(), vr.height()]
        lvr = self.boundingRect()
        ul = np.array([lvr.left(), lvr.top()])
        br = np.array([lvr.right(), lvr.bottom()])

        texts = []

        if ul[1] > br[1]:
            x = ul[1]
            ul[1] = br[1]
            br[1] = x

        lastd = [None, None]
        for i in range(self.grid_depth - 1, -1, -1):
            dist = br-ul
            nlTarget = 10.**i
            d = 10. ** np.floor(np.log10(np.abs(dist/nlTarget))+0.5)
            for ax in range(0,2):
                ts = self.opts['tickSpacing'][ax]
                try:
                    if ts[i] is not None:
                        d[ax] = ts[i]
                except IndexError:
                    pass
                lastd[ax] = d[ax]

            ul1 = np.floor(ul / d) * d
            br1 = np.ceil(br / d) * d
            dist = br1-ul1
            nl = (dist / d) + 0.5
            for ax in range(0,2):  ## Draw grid for both axes
                if i >= len(self.opts['tickSpacing'][ax]):
                    continue
                if d[ax] < lastd[ax]:
                    continue

                ppl = dim[ax] / nl[ax]
                c = int(fn.clip_scalar(5 * (ppl-3), 0, 50))

                assert isinstance(self.opts['pen'], QtGui.QPen)
                linePen = self.opts['pen']
                lineColor = self.opts['pen'].color()
                lineColor.setAlpha(c)
                linePen.setColor(lineColor)

                textPen = self.opts['textPen']
                if isinstance(textPen, QtGui.QPen):
                    textColor = textPen.color()
                    textColor.setAlpha(c * 2)
                    textPen.setColor(textColor)

                bx = (ax+1) % 2
                for x in range(0, int(nl[ax])):
                    linePen.setCosmetic(True)
                    p.setPen(linePen)
                    p1 = np.array([0.,0.])
                    p2 = np.array([0.,0.])
                    p1[ax] = ul1[ax] + x * d[ax]
                    p2[ax] = p1[ax]
                    p1[bx] = ul[bx]
                    p2[bx] = br[bx]
                    ## don't draw lines that are out of bounds.
                    if p1[ax] < min(ul[ax], br[ax]) or p1[ax] > max(ul[ax], br[ax]):
                        continue
                    p.drawLine(QtCore.QPointF(p1[0], p1[1]), QtCore.QPointF(p2[0], p2[1]))
                    if i < 2 and textPen is not None:
                        if ax == 0:
                            x = p1[0] + unit[0]
                            y = ul[1] + unit[1] * 8.
                        else:
                            x = ul[0] + unit[0]*3
                            y = p1[1] + unit[1]
                        texts.append((QtCore.QPointF(x, y), "%g"%p1[ax]))
        tr = self.deviceTransform()
        p.setWorldTransform(fn.invertQTransform(tr))

        if textPen is not None and len(texts) > 0:
            # if there is at least one text, then c is set
            textColor.setAlpha(c * 2)
            p.setPen(QtGui.QPen(textColor))
            for t in texts:
                x = tr.map(t[0]) + Point(0.5, 0.5)
                p.drawText(x, t[1])

        p.end()
