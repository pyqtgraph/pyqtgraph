from typing import Any, Optional, Union, TypedDict
import numpy as np
import warnings

from .. import functions as fn
from .. import configStyle
from ..style.core import (
    ConfigColorHint,
    ConfigKeyHint,
    ConfigLinestyleHint,
    ConfigValueHint,
    initItemStyle,
    parseLineStyle)
from ..Qt import QtGui, QtCore
from .GraphicsObject import GraphicsObject

__all__ = ['ErrorBarItem']


optsHint = TypedDict('optsHint',
                     {'color'     : ConfigColorHint,
                      'linestyle' : Union[str, int],
                      'linewidth' : float,
                      'beam'      : float},
                     total=False)
# kwargs are not typed because mypy has not ye included Unpack[Typeddict]


class ErrorBarItem(GraphicsObject):
    def __init__(self, x: Optional[np.ndarray],
                       y: Optional[np.ndarray],
                       height: Optional[Union[float, np.ndarray]]=None,
                       width: Optional[Union[float, np.ndarray]]=None,
                       top: Optional[Union[float, np.ndarray]]=None,
                       bottom: Optional[Union[float, np.ndarray]]=None,
                       left: Optional[Union[float, np.ndarray]]=None,
                       right: Optional[Union[float, np.ndarray]]=None,
                       **kwargs) -> None:
        """
        Draw error bar around a given dataset.

        Parameters
        ----------
        x
            coordinates of data points.
        y
            coordinates of data points.
        height
            If specified, it overrides top and bottom, specify the lengths of
            bars extending verticaly.
            All values should be positive.
            By default None
        width
            If specified, it overrides left and right, specify the lengths of
            bars extending in horizontaly.
            All values should be positive.
            By default None
        top
            Specify the lengths of bars extending in each direction.
            All values should be positive.
            By default None
        bottom
            Specify the lengths of bars extending in each direction.
            All values should be positive.
            By default None
        left
            Specify the lengths of bars extending in each direction.
            All values should be positive.
            By default None
        right
            Specify the lengths of bars extending in each direction.
            All values should be positive.
            By default None
        **kwargs:
            Style options , see setStyle() for accepted style parameters.
        """
        GraphicsObject.__init__(self)

        # Store errorBar data as internal attributes
        self._x      = x
        self._y      = y
        self._height = height
        self._width  = width
        self._top    = top
        self._bottom = bottom
        self._left   = left
        self._right  = right

        # Store style options in opts dict
        self.opts: optsHint = {}
        # Get default stylesheet
        initItemStyle(self, 'ErrorBarItem', configStyle)
        # Update style if needed
        if len(kwargs)>0:
            self.setStyle(**kwargs)

        # Since all arguments are optional, the default visibility is False.
        # The visibility is set in setData()
        self.setVisible(False)

        self.setData(x,
                     y,
                     height,
                     width,
                     top,
                     bottom,
                     left,
                     right)

    ##############################################################
    #
    #                   Style methods
    #
    ##############################################################

    ## Color
    def setColor(self, color: ConfigColorHint) -> None:
        """
        Set the color.

        Parameters
        ----------
        color : ConfigColorHint
            The color to set.

        Returns
        -------
        None
        """
        self.opts['color'] = color

    def color(self) -> ConfigColorHint:
        """
        Get the current color.

        Returns
        -------
        ConfigColorHint
            The current color.
        """
        return self.opts['color']

    ## LineStyle
    def setLinestyle(self, linestyle: ConfigLinestyleHint) -> None:
        """
        Set the linestyle.
        """
        self.opts['linestyle'] = linestyle

    def linestyle(self) -> ConfigLinestyleHint:
        """
        Get the current linestyle.
        """
        return self.opts['linestyle']

    ## LineStyle
    def setLinewidth(self, linewidth: float) -> None:
        """
        Set the linewidth.
        """
        self.opts['linewidth'] = linewidth

    def linewidth(self) -> float:
        """
        Get the current linewidth.
        """
        return self.opts['linewidth']

    ## beam
    def setBeam(self, beam: float) -> None:
        """
        Set the beam.
        """
        self.opts['beam'] = beam

    def beam(self) -> float:
        """
        Get the current beam.
        """
        return self.opts['beam']

    def __pen(self) -> QtGui.QPen:
        """
        Return a pen to draw the error bar using the following style options:
            * color
            * linewidth
            * linestyle

        If a pen has been specified, it overrides other style arguments
            * pen
        """

        # If a pen has been specified, it overrides other style arguments
        if hasattr(self, '_pen'):
            if self._pen is not None:
                return fn.mkPen(self._pen)
            else:
                return fn.mkPen(color=self.opts['color'],
                                width=self.opts['linewidth'],
                                style=parseLineStyle(self.opts['linestyle']))
        else:
            return fn.mkPen(color=self.opts['color'],
                            width=self.opts['linewidth'],
                            style=parseLineStyle(self.opts['linestyle']))

    def setStyle(self, **kwargs) -> None:
        """
        Set the style of the ErrorBarItem.

        Parameters
        ----------
        color: ConfigColorHint or None, optional
            Color to use when drawing the error bar.
        linestyle : ConfigLinestyleHint or None, optional
            LineStyle to use when drawing the error bar.
            Possible values are:
                * "-", "1"   -> A plain line.
                * "--, "2"   -> Dashes separated by a few pixels.
                * ":", "3"   -> Dots separated by a few pixels.
                * "-.", "4"  -> Alternate dots and dashes.
                * "-..", "5" -> One dash, two dots, one dash, two dots.
        linewidth : float or None, optional
            LineWidth to use when drawing the error bar.
        pen : Any or None, optional
            Pen to use when drawing the error bar.
            If specified, it overrides all previous parameters.
        beam : float or None, optional
            Specifies the width of the beam at the end of each bar.

        Notes
        -----
        The parameters that are not provided will not be modified.

        Examples
        --------
        >>> setStyle(linewidth=1.2, linestyle='--')
        """

        for k, v in kwargs.items():
            # If the attr is a valid entry of the stylesheet
            if k in configStyle['ErrorBarItem'].keys():
                fun = getattr(self, 'set{}{}'.format(k[:1].upper(), k[1:]))
                fun(v)
            # If a pen has been specified, it overrides other style arguments
            elif k=='pen':
                self._setPen(v)
            else:
                raise ValueError('Your argument: "{}" is not a valid style argument.'.format(k))

    ##############################################################
    #
    #                   Other style methods
    #
    ##############################################################

    def _setPen(self, pen: Any) -> None:
        """
        Set a given pen
        """
        self._pen = pen

    ##############################################################
    #
    #                   Item
    #
    ##############################################################

    def setData(self, x: Optional[np.ndarray],
                      y: Optional[np.ndarray],
                      height: Optional[Union[float, np.ndarray]]=None,
                      width: Optional[Union[float, np.ndarray]]=None,
                      top: Optional[Union[float, np.ndarray]]=None,
                      bottom: Optional[Union[float, np.ndarray]]=None,
                      left: Optional[Union[float, np.ndarray]]=None,
                      right: Optional[Union[float, np.ndarray]]=None,
                      **kwargs):
        """
        Update the data in the item. All arguments are optional.

        Parameters
        ----------
        x
            coordinates of data points.
        y
            coordinates of data points.
        height
            If specified, it overrides top and bottom, specify the lengths of
            bars extending verticaly.
            All values should be positive.
            By default None
        width
            If specified, it overrides left and right, specify the lengths of
            bars extending in horizontaly.
            All values should be positive.
            By default None
        top
            Specify the lengths of bars extending in each direction.
            All values should be positive.
            By default None
        bottom
            Specify the lengths of bars extending in each direction.
            All values should be positive.
            By default None
        left
            Specify the lengths of bars extending in each direction.
            All values should be positive.
            By default None
        right
            Specify the lengths of bars extending in each direction.
            All values should be positive.
            By default None
        **kwargs:
            Style options , see setStyle() for accepted style parameters.
        """

        # Update style if needed
        if len(kwargs)>0:
            self.setStyle(**kwargs)

        # Update errorBar data as internal attributes
        if x is not None:
            self._x      = x
        if y is not None:
            self._y      = y
        if height is not None:
            self._height = height
        if width is not None:
            self._width  = width
        if top is not None:
            self._top    = top
        if bottom is not None:
            self._bottom = bottom
        if left is not None:
            self._left   = left
        if right is not None:
            self._right  = right

        self.setVisible(all(ax is not None for ax in (self._x, self._y)))
        self.path = None
        self.update()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

    def setOpts(self, **opts):
        # for backward compatibility

        warnings.warn('Method "setOpts" is deprecated, "setData" should be used instead.',
                       DeprecationWarning,
                       stacklevel=2)
        self.setData(**opts)

    def drawPath(self) -> None:
        """
        Draw the errorbar.
        """
        p = QtGui.QPainterPath()

        if self._x is None or self._y is None:
            self.path = p
            return

        if self._height is not None or self._top is not None or self._bottom is not None:
            ## draw vertical error bars
            if self._height is not None:
                y1 = self._y - self._height/2.
                y2 = self._y + self._height/2.
            else:
                if self._bottom is None:
                    y1 = self._y
                else:
                    y1 = self._y - self._bottom
                if self._top is None:
                    y2 = self._y
                else:
                    y2 = self._y + self._top

            xs = fn.interweaveArrays(self._x, self._x)
            y1_y2 = fn.interweaveArrays(y1, y2)
            verticalLines = fn.arrayToQPath(xs, y1_y2, connect="pairs")
            p.addPath(verticalLines)

            if self.beam() is not None and self.beam() > 0:
                x1 = self._x - self.beam()/2.
                x2 = self._x + self.beam()/2.

                x1_x2 = fn.interweaveArrays(x1, x2)
                if self._height is not None or self._top is not None:
                    y2s = fn.interweaveArrays(y2, y2)
                    topEnds = fn.arrayToQPath(x1_x2, y2s, connect="pairs")
                    p.addPath(topEnds)

                if self._height is not None or self._bottom is not None:
                    y1s = fn.interweaveArrays(y1, y1)
                    bottomEnds = fn.arrayToQPath(x1_x2, y1s, connect="pairs")
                    p.addPath(bottomEnds)

        if self._width is not None or self._right is not None or self._left is not None:
            ## draw vertical error bars
            if self._width is not None:
                x1 = self._x - self._width/2.
                x2 = self._x + self._width/2.
            else:
                if self._left is None:
                    x1 = self._x
                else:
                    x1 = self._x - self._left
                if self._right is None:
                    x2 = self._x
                else:
                    x2 = self._x + self._right

            ys = fn.interweaveArrays(self._y, self._y)
            x1_x2 = fn.interweaveArrays(x1, x2)
            ends = fn.arrayToQPath(x1_x2, ys, connect='pairs')
            p.addPath(ends)

            if self.beam() is not None and self.beam() > 0:
                y1 = self._y - self.beam()/2.
                y2 = self._y + self.beam()/2.
                y1_y2 = fn.interweaveArrays(y1, y2)
                if self._width is not None or self._right is not None:
                    x2s = fn.interweaveArrays(x2, x2)
                    rightEnds = fn.arrayToQPath(x2s, y1_y2, connect="pairs")
                    p.addPath(rightEnds)

                if self._width is not None or self._left is not None:
                    x1s = fn.interweaveArrays(x1, x1)
                    leftEnds = fn.arrayToQPath(x1s, y1_y2, connect="pairs")
                    p.addPath(leftEnds)

        self.path = p
        self.prepareGeometryChange()

    def paint(self, p: QtGui.QPainter, *args) -> None:
        if self.path is None:
            self.drawPath()
        p.setPen(self.__pen())
        p.drawPath(self.path)

    def boundingRect(self) -> QtCore.QRectF:
        """
        Return the item rectangle coordinates
        """
        if self.path is None:
            self.drawPath()
        assert self.path is not None
        return self.path.boundingRect()
