from math import ceil, floor, isfinite, log, log10
from typing import Any, List, Optional, Tuple, TypedDict, Union
import warnings
import weakref

import numpy as np

from .. import debug as debug
from .. import functions as fn
from .. import configStyle
from ..style.core import (
    ConfigColorHint,
    ConfigKeyHint,
    ConfigValueHint,
    initItemStyle)
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
from ..graphicsItems.ViewBox import ViewBox
from ..GraphicsScene.mouseEvents import MouseDragEvent, MouseClickEvent

__all__ = ['AxisItem']


optsHint = TypedDict('optsHint',
                     {'lineColor' : ConfigColorHint,

                      'tickAlpha' : float,
                      'tickColor' : ConfigColorHint,
                      'tickFont' : Optional[str],
                      'tickFontsize' : int,
                      'tickLabelColor' : ConfigColorHint,
                      'tickLength' : int,
                      'tickTextHeight' : int,
                      'tickTextOffset' : List[int],
                      'tickTextWidth' : int,

                      'labelColor' : ConfigColorHint,
                      'labelFontsize' : float,
                      'labelFontweight' : str,
                      'labelFontstyle' : str,

                      'autoExpandTextSpace' : bool,
                      'autoReduceTextSpace' : bool,
                      'hideOverlappingLabels' : bool,
                      'stopAxisAtTick' : List[bool],
                      'textFillLimits' : List[Tuple[float, float]],
                      'showValues' : bool,
                      'maxTickLength' : int,
                      'maxTickLevel' : int,
                      'maxTextLevel' : int},
                     total=False)
# kwargs are not typed because mypy has not ye included Unpack[Typeddict]

class AxisItem(GraphicsWidget):
    """
    GraphicsItem showing a single plot axis with ticks, values, and label.
    Can be configured to fit on any side of a plot,
    Can automatically synchronize its displayed scale with ViewBox items.
    Ticks can be extended to draw a grid.
    If maxTickLength is negative, ticks point into the plot.
    """

    def __init__(self, orientation: str,
                       pen: QtGui.QPen=None,
                       textPen: QtGui.QPen=None,
                       tickPen: QtGui.QPen=None,
                       linkView: ViewBox=None,
                       text: str='',
                       units: str='',
                       unitPrefix: str='',
                       parent=None,
                       **kwargs) -> None:
        """
        =============== ===============================================================
        **Arguments:**
        orientation     one of 'left', 'right', 'top', or 'bottom'
        maxTickLength   (px) maximum length of ticks to draw. Negative values draw
                        into the plot, positive values draw outward.
        linkView        (ViewBox) causes the range of values displayed in the axis
                        to be linked to the visible range of a ViewBox.
        showValues      (bool) Whether to display values adjacent to ticks
        pen             (QPen) Pen used when drawing axis and (by default) ticks
        textPen         (QPen) Pen used when drawing tick labels.
        tickPen         (QPen) Pen used when drawing ticks.
        text            The text (excluding units) to display on the label for this
                        axis.
        units           The units for this axis. Units should generally be given
                        without any scaling prefix (eg, 'V' instead of 'mV'). The
                        scaling prefix will be automatically prepended based on the
                        range of data displayed.
        kwargs            All extra keyword arguments become CSS style options for
                        the <span> tag which will surround the axis label and units.
        =============== ===============================================================
        """

        GraphicsWidget.__init__(self, parent)

        # Set the axis label
        self.label = QtWidgets.QGraphicsTextItem(self)
        self.picture = None
        self.orientation = orientation
        if orientation not in ['left', 'right', 'top', 'bottom']:
            raise Exception("Orientation argument must be one of 'left', 'right', 'top', or 'bottom'.")
        if orientation in ['left', 'right']:
            self.label.setRotation(-90)
        self.labelText = text
        self.labelUnits = units
        self.labelUnitPrefix = unitPrefix

        # Keeps track of maximum width / height of tick text in px
        self.textWidth: int = 30
        self.textHeight: int = 18

        # If the user specifies a width / height, remember that setting
        # indefinitely.
        self.fixedWidth: Optional[int] = None
        self.fixedHeight: Optional[int] = None

        self.logMode: bool = False
        self._tickLevels: Optional[list] = None  ## used to override the automatic ticking system with explicit ticks
        self._tickSpacing: Optional[List[Tuple[float, float]]] = None  # used to override default tickSpacing method
        self.scale: float = 1.0
        self.autoSIPrefix: bool = True
        self.autoSIPrefixScale: float = 1.0

        # Store style options in opts dict
        self.style: optsHint = {}
        # Get default stylesheet
        initItemStyle(self, 'AxisItem', configStyle)
        # Update style if needed
        if len(kwargs)>0:
            self.setStyle(**kwargs)

        self.showLabel(False)
        self.setRange(0, 1)

        if pen is None:
            self.setPen()
        else:
            self.setPen(pen)

        if textPen is None:
            self.setTextPen()
        else:
            self.setTextPen(textPen)

        if tickPen is None:
            self.setTickPen()
        else:
            self.setTickPen(tickPen)

        self._linkedView = None
        if linkView is not None:
            self._linkToView_internal(linkView)

        self.grid: Union[bool, int] = False

    ##############################################################
    #
    #                   Style methods
    #
    ##############################################################

    def setLineColor(self, lineColor: ConfigColorHint) -> None:
        """
        Set the axis lineColor.
        """
        self.style['lineColor'] = lineColor

    def lineColor(self) -> ConfigColorHint:
        """
        Get the current axis lineColor.
        """
        return self.style['lineColor']

    def setTickAlpha(self, tickAlpha: Optional[Union[float, int]]) -> None:
        """
        Set the tick alpha channel.
        """
        if not isinstance(tickAlpha, int) and not isinstance(tickAlpha, float) and not None:
            raise ValueError('The given tickAlpha type is "{}", must be "str"'.format(type(tickAlpha).__name__))
        self.style['tickAlpha'] = tickAlpha

    def tickAlpha(self) -> Optional[Union[float, int]]:
        """
        Get the current tick alpha channel.
        """
        return self.style['tickAlpha']

    def setTickColor(self, tickColor: ConfigColorHint) -> None:
        """
        Set the tick color.
        """
        self.style['tickColor'] = tickColor

    def tickColor(self) -> ConfigColorHint:
        """
        Get the current tick color.
        """
        return self.style['tickColor']

    def setTickFont(self, tickFont: Optional[Union[str, QtGui.QFont]]) -> None:
        """
        Set the tick font.
        """
        if isinstance(tickFont, str):
            self.style['tickFont'] = tickFont
        elif isinstance(tickFont, QtGui.QFont):
            warnings.warn('Give argument "tickFont" as QFont is deprecated, "str" should be used instead.',
                        DeprecationWarning,
                        stacklevel=2)

            # Store the font as a string
            t = tickFont.toString().split(',')[0]
            if t=='':
                self.style['tickFont'] = None
            else:
                self.style['tickFont'] = t
        elif tickFont is None:
            self.style['tickFont'] = None
        else:
            raise ValueError('The given tickFont type is "{}", must be "str"'.format(type(tickFont).__name__))

    def tickFont(self) -> Optional[str]:
        """
        Get the current tick font.
        """
        return self.style['tickFont']

    def setTickFontsize(self, tickFontsize: int) -> None:
        """
        Set the tickFontsize.
        """
        if not isinstance(tickFontsize, int):
            raise ValueError('The given tickFontsize type is "{}", must be "int"'.format(type(tickFontsize).__name__))
        self.style['tickFontsize'] = tickFontsize

    def tickFontsize(self) -> int:
        """
        Get the current tick font.
        """
        return self.style['tickFontsize']

    def setTickLabelColor(self, tickLabelColor: ConfigColorHint) -> None:
        """
        Set the tick labelcolor.
        """
        self.style['tickLabelColor'] = tickLabelColor

    def tickLabelColor(self) -> ConfigColorHint:
        """
        Get the current tick labelcolor.
        """
        return self.style['tickLabelColor']

    def setTickLength(self, tickLength: int) -> None:
        """
        Set the tick length.
        """
        if not isinstance(tickLength, int):
            raise ValueError('The given tickLength type is "{}", must be "int"'.format(type(tickLength).__name__))
        self.style['tickLength'] = tickLength

    def tickLength(self) -> int:
        """
        Get the current tick length.
        """
        return self.style['tickLength']

    def setTickTextHeight(self, tickTextHeight: int) -> None:
        """
        Set the tick textheight.
        """
        if not isinstance(tickTextHeight, int):
            raise ValueError('The given tickTextHeight type is "{}", must a "int"'.format(type(tickTextHeight).__name__))
        self.style['tickTextHeight'] = tickTextHeight

    def tickTextHeight(self) -> int:
        """
        Get the current tick textheight.
        """
        return self.style['tickTextHeight']

    def setTickTextOffset(self, tickTextOffset: Union[int, List[int]]) -> None:
        """
        Set the tick text offset.
        """
        if isinstance(tickTextOffset, list):
            if isinstance(tickTextOffset[0], int) and isinstance(tickTextOffset[1], int):
                self.style['tickTextOffset'] = tickTextOffset
            else:
                raise ValueError('The given tickTextOffset type is "{}", must a "int"'.format(type(tickTextOffset).__name__))
        elif isinstance(tickTextOffset, int):
            if self.orientation in ('left', 'right'):
                self.style['tickTextOffset'][0] = tickTextOffset
            else:
                self.style['tickTextOffset'][1] = tickTextOffset
        else:
            raise ValueError('The given tickTextOffset type is "{}", must a "int"'.format(type(tickTextOffset).__name__))

    def tickTextOffset(self) -> int:
        """
        Get the current tick text offset.
        """
        if self.orientation in ('left', 'right'):
            return self.style['tickTextOffset'][0]
        else:
            return self.style['tickTextOffset'][1]

    def setTickTextWidth(self, tickTextWidth: int) -> None:
        """
        Set the tick text width.
        """
        if not isinstance(tickTextWidth, int):
            raise ValueError('The given tickTextWidth type is "{}", must a "int"'.format(type(tickTextWidth).__name__))
        self.style['tickTextWidth'] = tickTextWidth

    def tickTextWidth(self) -> int:
        """
        Get the current tick text width.
        """
        return self.style['tickTextWidth']

    def setLabelColor(self, labelColor: ConfigColorHint) -> None:
        """
        Set the tick text width.
        """
        # if not isinstance(labelColor, ConfigColorHint):
        #     raise ValueError('The given labelColor type is "{}", must a "ConfigColorHint"'.format(type(labelColor).__name__))
        self.style['labelColor'] = labelColor

    def labelColor(self) -> ConfigColorHint:
        """
        Get the current tick text width.
        """
        return self.style['labelColor']

    def setLabelFontsize(self, labelFontsize: float) -> None:
        """
        Set the tick text width.
        """
        if not isinstance(labelFontsize, float):
            raise ValueError('The given labelFontsize type is "{}", must a "float"'.format(type(labelFontsize).__name__))
        self.style['labelFontsize'] = labelFontsize

    def labelFontsize(self) -> float:
        """
        Get the current tick text width.
        """
        return self.style['labelFontsize']

    def setLabelFontweight(self, labelFontweight: str) -> None:
        """
        Set the tick text width.
        """
        if not isinstance(labelFontweight, str):
            raise ValueError('The given labelFontweight type is "{}", must a "str"'.format(type(labelFontweight).__name__))
        self.style['labelFontweight'] = labelFontweight

    def labelFontweight(self) -> str:
        """
        Get the current tick text width.
        """
        return self.style['labelFontweight']

    def setLabelFontstyle(self, labelFontstyle: str) -> None:
        """
        Set the tick text width.
        """
        if not isinstance(labelFontstyle, str):
            raise ValueError('The given labelFontstyle type is "{}", must a "str"'.format(type(labelFontstyle).__name__))
        self.style['labelFontstyle'] = labelFontstyle

    def labelFontstyle(self) -> str:
        """
        Get the current tick text width.
        """
        return self.style['labelFontstyle']

    def setAutoExpandTextSpace(self, autoExpandTextSpace: bool) -> None:
        """
        Set the auto expand text space.
        Automatically expand text space if the tick strings become too long
        """
        if not isinstance(autoExpandTextSpace, bool):
            raise ValueError('The given autoExpandTextSpace is {}, must be a "bool"'.format(type(autoExpandTextSpace).__name__))
        self.style['autoExpandTextSpace'] = autoExpandTextSpace

    def autoExpandTextSpace(self) -> bool:
        """
        Get the current auto expand text space.
        Automatically expand text space if the tick strings become too long
        """
        return self.style['autoExpandTextSpace']

    def setAutoReduceTextSpace(self, autoReduceTextSpace: bool) -> None:
        """
        Set the auto reduce text space.
        Automatically shrink the axis if necessary
        """
        if not isinstance(autoReduceTextSpace, bool):
            raise ValueError('The given autoReduceTextSpace is {}, must be a "bool"'.format(type(autoReduceTextSpace).__name__))
        self.style['autoReduceTextSpace'] = autoReduceTextSpace

    def autoReduceTextSpace(self) -> bool:
        """
        Get the current auto reduce text space.
        Automatically shrink the axis if necessary
        """
        return self.style['autoReduceTextSpace']

    def setHideOverlappingLabels(self, hideOverlappingLabels: bool) -> None:
        """
        Set the hide over lapping labels
        Automatically shrink the axis if necessary
        """
        if not isinstance(hideOverlappingLabels, bool):
            raise ValueError('The given hideOverlappingLabels is {}, must be a "bool"'.format(type(hideOverlappingLabels).__name__))
        self.style['hideOverlappingLabels'] = hideOverlappingLabels

    def hideOverlappingLabels(self) -> bool:
        """
        Get the current hide over lapping labels
        Automatically shrink the axis if necessary
        """
        return self.style['hideOverlappingLabels']

    def setStopAxisAtTick(self, stopAxisAtTick: List[bool]) -> None:
        """
        Set the stop axis at tick
        If True, the axis line is drawn only as far as the last tick.
        Otherwise, the line is drawn to the edge of the AxisItem boundary.
        """
        if not isinstance(stopAxisAtTick, list):
            raise ValueError('The given stopAxisAtTick is {}, must be a "list"'.format(type(stopAxisAtTick).__name__))
        elif not isinstance(stopAxisAtTick[0], bool) and not isinstance(stopAxisAtTick[1], bool):
            raise ValueError('The given stopAxisAtTick arguments is {}, must be "bool"'.format(type(stopAxisAtTick).__name__))
        self.style['stopAxisAtTick'] = stopAxisAtTick

    def stopAxisAtTick(self) -> List[bool]:
        """
        Get the current stop axis at tick
        If True, the axis line is drawn only as far as the last tick.
        Otherwise, the line is drawn to the edge of the AxisItem boundary.
        """
        return self.style['stopAxisAtTick']

    def setTextFillLimits(self, textFillLimits: List[Tuple[float, float]]) -> None:
        """
        Set the text fill limits
        """
        if not isinstance(textFillLimits, list):
            raise ValueError('The given textFillLimits type is "{}", must be a "list"'.format(type(textFillLimits).__name__))
        self.style['textFillLimits'] = textFillLimits

    def textFillLimits(self) -> List[Tuple[float, float]]:
        """
        Get the current text fill limits
        """
        return self.style['textFillLimits']

    def setShowValues(self, showValues: bool) -> None:
        """
        Set the show values
        """
        if not isinstance(showValues, bool):
            raise ValueError('The given showValues type is "{}", must be a "bool"'.format(type(showValues).__name__))
        self.style['showValues'] = showValues

    def showValues(self) -> bool:
        """
        Get the current show values
        """
        return self.style['showValues']

    def setMaxTickLength(self, maxTickLength: int) -> None:
        """
        Set the max tick length
        """
        if not isinstance(maxTickLength, int):
            raise ValueError('The given maxTickLength type is "{}", must be a "int"'.format(type(maxTickLength).__name__))
        self.style['maxTickLength'] = maxTickLength

    def maxTickLength(self) -> int:
        """
        Get the current max tick length
        """
        return self.style['maxTickLength']

    def setMaxTickLevel(self, maxTickLevel: int) -> None:
        """
        Set the max tick level
        """
        if not isinstance(maxTickLevel, int):
            raise ValueError('The given maxTickLevel type is "{}", must be a "int"'.format(type(maxTickLevel).__name__))
        self.style['maxTickLevel'] = maxTickLevel

    def maxTickLevel(self) -> int:
        """
        Get the current max tick level
        """
        return self.style['maxTickLevel']

    def setMaxTextLevel(self, maxTextLevel: int) -> None:
        """
        Set the max text level
        """
        if not isinstance(maxTextLevel, int):
            raise ValueError('The given maxTextLevel type is "{}", must be a "int"'.format(type(maxTextLevel).__name__))
        self.style['maxTextLevel'] = maxTextLevel

    def maxTextLevel(self) -> int:
        """
        Get the current max text level
        """
        return self.style['maxTextLevel']

    def setStyle(self, **kwargs) -> None:
        """
        Set various style options.
        Added in version 0.9.9

        Parameters
        ----------
        lineColor :
            color of the line.
        tickAlpha : float or int or None
            If None, pyqtgraph will draw the ticks with the alpha it deems
            appropriate.  Otherwise, the alpha will be fixed at the value passed.
            With int, accepted values are [0..255].  With value of type float,
            accepted values are from [0..1].
        tickColor :
            Color used to draw the ticks
        tickFont : str or None
            Set the font used for tick values.
            Use None for the default font.
        tickFontSize : int
            Set the font size used for tick values.
            Use None for the default font size.
        tickLabelColor :
            Color use for the ticks label.
        tickLength : int
            The maximum length of ticks in pixels.
            Positive values point toward the text; negative values point away.
        tickTextHeight : int
            Vertical space reserved for tick text in px
        tickTextOffset : int
            Reserved spacing between text and axis in px
        tickTextWidth : int
            Horizontal space reserved for tick text in px

        labelColor : ConfigColorHint
            Set the color of the axis label.
        labelFontsize : float
            Set the fontsize of the axis label.
        labelFontweight : str
            Set the fontweight of the axis label.
        labelFontstyle : str
            Set the fontstyle of the axis label.

        autoExpandTextSpace : bool
            Automatically expand text space if the tick strings become too long.
        autoReduceTextSpace : bool
            Automatically shrink the axis if necessary
        hideOverlappingLabels : bool
            Hide tick labels which overlap the AxisItems' geometry rectangle. If False, labels might be drawn overlapping with tick labels from neighboring plots.
        stopAxisAtTick : tuple(bool, bool)
            If True, the axis line is drawn only as far as the last tick.
            Otherwise, the line is drawn to the edge of the AxisItem boundary.
        textFillLimits (list: of tick #, % fill) tuples).
            This structure determines how the AxisItem decides how many ticks
            should have text appear next to them. Each tuple in the list
            specifies what fraction of the axis length may be occupied by text,
            given the number of ticks that already have text displayed.
            For example:
                [(0, 0.8), # Never fill more than 80% of the axis
                (2, 0.6), # If we already have 2 ticks with text,
                            # fill no more than 60% of the axis
                (4, 0.4), # If we already have 4 ticks with text,
                            # fill no more than 40% of the axis
                (6, 0.2)] # If we already have 6 ticks with text,
                            # fill no more than 20% of the axis
        showValues : bool
            Indicates whether text is displayed adjacent to ticks.
        maxTickLevel : int
            Decide whether to include the last level of ticks
        maxTextLevel : int
            Decide whether to include the last level of ticks
        """

        for k, v in kwargs.items():
            # If the attr is a valid entry of the stylesheet
            if k in configStyle['AxisItem'].keys():
                fun = getattr(self, 'set{}{}'.format(k[:1].upper(), k[1:]))
                fun(v)
            else:
                raise ValueError('Your argument: "{}" is not a valid style argument.'.format(k))

        self.picture = None
        self._adjustSize()
        self.update()

    def close(self) -> None:
        self.scene().removeItem(self.label)
        self.label = None
        self.scene().removeItem(self)

    def setGrid(self, grid: Union[bool, int]) -> None:
        """Set the alpha value (0-255) for the grid, or False to disable.

        When grid lines are enabled, the axis tick lines are extended to cover
        the extent of the linked ViewBox, if any.
        """
        self.grid = grid
        self.picture = None
        self.prepareGeometryChange()
        self.update()

    def setLogMode(self, *args, **kwargs) -> None:
        """
        Set log scaling for x and/or y axes.

        If two positional arguments are provided, the first will set log scaling
        for the x axis and the second for the y axis. If a single positional
        argument is provided, it will set the log scaling along the direction of
        the AxisItem. Alternatively, x and y can be passed as keyword arguments.

        If an axis is set to log scale, ticks are displayed on a logarithmic scale
        and values are adjusted accordingly. (This is usually accessed by changing
        the log mode of a :func:`PlotItem <pyqtgraph.PlotItem.setLogMode>`.) The
        linked ViewBox will be informed of the change.
        """
        if len(args) == 1:
            self.logMode = args[0]
        else:
            if len(args) == 2:
                x, y = args
            else:
                x = kwargs.get('x')
                y = kwargs.get('y')

            if x is not None and self.orientation in ('top', 'bottom'):
                self.logMode = x
            if y is not None and self.orientation in ('left', 'right'):
                self.logMode = y

        if self._linkedView is not None:
            if self.orientation in ('top', 'bottom'):
                self._linkedView().setLogMode('x', self.logMode)
            elif self.orientation in ('left', 'right'):
                self._linkedView().setLogMode('y', self.logMode)

        self.picture = None

        self.update()

    def resizeEvent(self, ev: Optional[QtWidgets.QGraphicsSceneResizeEvent]=None) -> None:

        ## Set the position of the label
        nudge = 5
        if self.label is None: # self.label is set to None on close, but resize events can still occur.
            self.picture = None
            return

        br = self.label.boundingRect()
        p = QtCore.QPointF(0, 0)
        if self.orientation == 'left':
            p.setY(int(self.size().height()/2 + br.width()/2))
            p.setX(-nudge)
        elif self.orientation == 'right':
            p.setY(int(self.size().height()/2 + br.width()/2))
            p.setX(int(self.size().width()-br.height()+nudge))
        elif self.orientation == 'top':
            p.setY(-nudge)
            p.setX(int(self.size().width()/2. - br.width()/2.))
        elif self.orientation == 'bottom':
            p.setX(int(self.size().width()/2. - br.width()/2.))
            p.setY(int(self.size().height()-br.height()+nudge))
        self.label.setPos(p)
        self.picture = None

    def showLabel(self, show: bool=True) -> None:
        """Show/hide the label text for this axis."""
        #self.drawLabel = show
        self.label.setVisible(show)
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()
        if self.autoSIPrefix:
            self.updateAutoSIPrefix()

    def setLabel(self, text: Optional[str]=None,
                       units: Optional[str]=None,
                       unitPrefix: Optional[str]=None,
                       **args) -> None:
        """Set the text displayed adjacent to the axis.

        ==============  =============================================================
        **Arguments:**
        text            The text (excluding units) to display on the label for this
                        axis.
        units           The units for this axis. Units should generally be given
                        without any scaling prefix (eg, 'V' instead of 'mV'). The
                        scaling prefix will be automatically prepended based on the
                        range of data displayed.
        args            All extra keyword arguments become CSS style options for
                        the <span> tag which will surround the axis label and units.
        ==============  =============================================================

        The final text generated for the label will look like::

            <span style="...options...">{text} (prefix{units})</span>

        Each extra keyword argument will become a CSS option in the above template.
        For example, you can set the font size and color of the label::

            labelStyle = {'color': '#FFF', 'font-size': '14pt'}
            axis.setLabel('label text', units='V', **labelStyle)

        """
        # `None` input is kept for backward compatibility!
        self.labelText = text or ""
        self.labelUnits = units or ""
        self.labelUnitPrefix = unitPrefix or ""
        # Account empty string and `None` for units and text
        visible = True if (text or units) else False
        self.showLabel(visible)
        self._updateLabel()

    def _updateLabel(self) -> None:
        """Internal method to update the label according to the text"""
        self.label.setHtml(self.labelString())
        self._adjustSize()
        self.picture = None
        self.update()

    def labelString(self) -> str:
        if self.labelUnits == '':
            if not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
                units = ''
            else:
                units = '(x%g)' % (1.0/self.autoSIPrefixScale)
        else:
            units = '(%s%s)' % (self.labelUnitPrefix, self.labelUnits)

        s = '%s %s' % (self.labelText, units)

        # The style is applied via css
        style = []
        style.append('color: {}'.format(fn.mkColor(self.style['labelColor']).name(QtGui.QColor.NameFormat.HexArgb)))
        style.append('font-size: {}pt'.format(self.style['labelFontsize']))
        style.append('font-weight: {}'.format(self.style['labelFontweight']))
        style.append('font-style: {}'.format(self.style['labelFontstyle']))
        return "<span style='%s'>%s</span>" % ('; '.join(style), s)

    def _updateMaxTextSize(self, x: int) -> None:
        ## Informs that the maximum tick size orthogonal to the axis has
        ## changed; we use this to decide whether the item needs to be resized
        ## to accomodate.
        if self.orientation in ['left', 'right']:
            if self.style['autoReduceTextSpace']:
                if x > self.textWidth or x < self.textWidth - 10:
                    self.textWidth = x
            else:
                mx = max(self.textWidth, x)
                if mx > self.textWidth or mx < self.textWidth - 10:
                    self.textWidth = mx
            if self.style['autoExpandTextSpace']:
                self._updateWidth()

        else:
            if self.style['autoReduceTextSpace']:
                if x > self.textHeight or x < self.textHeight - 10:
                    self.textHeight = x
            else:
                mx = max(self.textHeight, x)
                if mx > self.textHeight or mx < self.textHeight - 10:
                    self.textHeight = mx
            if self.style['autoExpandTextSpace']:
                self._updateHeight()

    def _adjustSize(self) -> None:
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()

    def setHeight(self, h: Optional[int]=None) -> None:
        """Set the height of this axis reserved for ticks and tick labels.
        The height of the axis label is automatically added.

        If *height* is None, then the value will be determined automatically
        based on the size of the tick text."""
        self.fixedHeight = h
        self._updateHeight()

    def _updateHeight(self) -> None:
        if not self.isVisible():
            h = 0
        else:
            if self.fixedHeight is None:
                if not self.style['showValues']:
                    h = 0
                elif self.style['autoExpandTextSpace']:
                    h = self.textHeight
                else:
                    h = self.style['tickTextHeight']
                h += self.style['tickTextOffset'][1] if self.style['showValues'] else 0
                h += max(0, self.style['tickLength'])
                if self.label.isVisible():
                    h += self.label.boundingRect().height() * 0.8
            else:
                h = self.fixedHeight

        self.setMaximumHeight(h)
        self.setMinimumHeight(h)
        self.picture = None

    def setWidth(self, w: Optional[int]=None) -> None:
        """Set the width of this axis reserved for ticks and tick labels.
        The width of the axis label is automatically added.

        If *width* is None, then the value will be determined automatically
        based on the size of the tick text."""
        self.fixedWidth = w
        self._updateWidth()

    def _updateWidth(self) -> None:
        if not self.isVisible():
            w = 0
        else:
            if self.fixedWidth is None:
                if not self.style['showValues']:
                    w = 0
                elif self.style['autoExpandTextSpace']:
                    w = self.textWidth
                else:
                    w = self.style['tickTextWidth']
                w += self.style['tickTextOffset'][0] if self.style['showValues'] else 0
                w += max(0, self.style['tickLength'])
                if self.label.isVisible():
                    w += self.label.boundingRect().height() * 0.8  ## bounding rect is usually an overestimate
            else:
                w = self.fixedWidth

        self.setMaximumWidth(w)
        self.setMinimumWidth(w)
        self.picture = None

    def pen(self) -> QtGui.QPen:
        if self._pen is None:
            return fn.mkPen(configStyle['AxisItem']['lineColor'])
        return fn.mkPen(self._pen)

    def setPen(self, *args, **kwargs) -> None:
        """
        Set the pen used for drawing text, axes, ticks, and grid lines.
        (see :func:`setConfigOption <pyqtgraph.setConfigOption>`).
        """
        self.picture = None
        if args or kwargs:
            self._pen = fn.mkPen(*args, **kwargs)
        else:
            self._pen = fn.mkPen(configStyle['AxisItem']['lineColor'])
        self.setTickLabelColor(self._pen.color().name()) #RRGGBB
        self._updateLabel()

    def textPen(self) -> QtGui.QPen:
        if self._textPen is None:
            return fn.mkPen(configStyle['AxisItem']['tickLabelColor'])
        return fn.mkPen(self._textPen)

    def setTextPen(self, *args, **kwargs) -> None:
        """
        Set the pen used for drawing text.
        """
        self.picture = None
        if args or kwargs:
            self._textPen = fn.mkPen(*args, **kwargs)
        else:
            self._textPen = fn.mkPen(configStyle['AxisItem']['tickLabelColor'])
        self.setLabelColor(self._textPen.color().name()) #RRGGBB
        self._updateLabel()

    def tickPen(self) -> QtGui.QPen:
        if self._tickPen is None:
            return fn.mkPen(configStyle['AxisItem']['tickColor'])
        return fn.mkPen(self._tickPen)

    def setTickPen(self, *args, **kwargs) -> None:
        """
        Set the pen used for drawing tick marks.
        If no arguments are given, the default pen will be used.
        """
        self.picture = None
        if args or kwargs:
            self._tickPen = fn.mkPen(*args, **kwargs)
        else:
            self._tickPen = None

        self._updateLabel()

    def setScale(self, scale: float) -> None:
        """
        Set the value scaling for this axis.

        Setting this value causes the axis to draw ticks and tick labels as if
        the view coordinate system were scaled. By default, the axis scaling is
        1.0.
        """
        if scale != self.scale:
            self.scale = scale
            self._updateLabel()

    def enableAutoSIPrefix(self, enable: bool=True) -> None:
        """
        Enable (or disable) automatic SI prefix scaling on this axis.

        When enabled, this feature automatically determines the best SI prefix
        to prepend to the label units, while ensuring that axis values are scaled
        accordingly.

        For example, if the axis spans values from -0.1 to 0.1 and has units set
        to 'V' then the axis would display values -100 to 100
        and the units would appear as 'mV'

        This feature is enabled by default, and is only available when a suffix
        (unit string) is provided to display on the label.
        """
        self.autoSIPrefix = enable
        self.updateAutoSIPrefix()

    def updateAutoSIPrefix(self) -> None:
        if self.label.isVisible():
            if self.logMode:
                _range = 10**np.array(self.range)
            else:
                _range = self.range
            (scale, prefix) = fn.siScale(max(abs(_range[0]*self.scale), abs(_range[1]*self.scale)))
            if self.labelUnits == '' and prefix in ['k', 'm']:  ## If we are not showing units, wait until 1e6 before scaling.
                scale = 1.0
                prefix = ''
            self.autoSIPrefixScale = scale
            self.labelUnitPrefix = prefix
        else:
            self.autoSIPrefixScale = 1.0

        self._updateLabel()

    def setRange(self, mn: float, mx: float) -> None:
        """Set the range of values displayed by the axis.
        Usually this is handled automatically by linking the axis to a ViewBox with :func:`linkToView <pyqtgraph.AxisItem.linkToView>`"""
        if not isfinite(mn) or not isfinite(mx):
            raise Exception("Not setting range to [%s, %s]" % (str(mn), str(mx)))
        self.range: np.ndarray = np.array([mn, mx])
        if self.autoSIPrefix:
            # XXX: Will already update once!
            self.updateAutoSIPrefix()
        else:
            self.picture = None
            self.update()

    def linkedView(self) -> Optional[ViewBox]:
        """Return the ViewBox this axis is linked to"""
        if self._linkedView is None:
            return None
        else:
            return self._linkedView()

    def _linkToView_internal(self, view: ViewBox) -> None:
        # We need this code to be available without override,
        # even though DateAxisItem overrides the user-side linkToView method
        self.unlinkFromView()

        self._linkedView = weakref.ref(view) # type: ignore
        if self.orientation in ['right', 'left']:
            view.sigYRangeChanged.connect(self.linkedViewChanged)
        else:
            view.sigXRangeChanged.connect(self.linkedViewChanged)
        view.sigResized.connect(self.linkedViewChanged)

    def linkToView(self, view: ViewBox) -> None:
        """Link this axis to a ViewBox, causing its displayed range to match the visible range of the view."""
        self._linkToView_internal(view)

    def unlinkFromView(self) -> None:
        """Unlink this axis from a ViewBox."""
        oldView = self.linkedView()
        self._linkedView = None
        if self.orientation in ['right', 'left']:
            if oldView is not None:
                oldView.sigYRangeChanged.disconnect(self.linkedViewChanged)
        else:
            if oldView is not None:
                oldView.sigXRangeChanged.disconnect(self.linkedViewChanged)

        if oldView is not None:
            oldView.sigResized.disconnect(self.linkedViewChanged)

    def linkedViewChanged(self, view: ViewBox, newRange: Optional[tuple]=None) -> None:
        if self.orientation in ['right', 'left']:
            if newRange is None:
                newRange = view.viewRange()[1]
            if view.yInverted():
                self.setRange(*newRange[::-1])
            else:
                self.setRange(*newRange)
        else:
            if newRange is None:
                newRange = view.viewRange()[0]
            if view.xInverted():
                self.setRange(*newRange[::-1])
            else:
                self.setRange(*newRange)

    def boundingRect(self) -> QtCore.QRectF:
        m = 0 if self.style['hideOverlappingLabels'] else 15
        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            rect = self.mapRectFromParent(self.geometry())
            ## extend rect if ticks go in negative direction
            ## also extend to account for text that flows past the edges
            tl = self.style['tickLength']
            if self.orientation == 'left':
                rect = rect.adjusted(0, -m, -min(0,tl), m)
            elif self.orientation == 'right':
                rect = rect.adjusted(min(0,tl), -m, 0, m)
            elif self.orientation == 'top':
                rect = rect.adjusted(-15, 0, 15, -min(0,tl))
            elif self.orientation == 'bottom':
                rect = rect.adjusted(-m, min(0,tl), m, 0)
            return rect
        else:
            return self.mapRectFromParent(self.geometry()) | linkedView.mapRectToItem(self, linkedView.boundingRect())

    def paint(self, p: QtGui.QPainter,
                    opt: QtWidgets.QStyleOptionGraphicsItem,
                    widget: QtWidgets.QWidget) -> None:
        profiler = debug.Profiler()
        if self.picture is None:
            try:
                picture = QtGui.QPicture()
                painter = QtGui.QPainter(picture)
                if self.style['tickFont']:
                    painter.setFont(QtGui.QFont(self.style['tickFont'], self.style['tickFontsize']))
                specs = self.generateDrawSpecs(painter)
                profiler('generate specs')
                if specs is not None:
                    self.drawPicture(painter, *specs)
                    profiler('draw picture')
            finally:
                painter.end()
            self.picture = picture
        #p.setRenderHint(p.RenderHint.Antialiasing, False)   ## Sometimes we get a segfault here ???
        #p.setRenderHint(p.RenderHint.TextAntialiasing, True)
        assert isinstance(self.picture, QtGui.QPicture)
        self.picture.play(p)

    def setTicks(self, ticks: list) -> None:
        """Explicitly determine which ticks to display.
        This overrides the behavior specified by tickSpacing(), tickValues(), and tickStrings()
        The format for *ticks* looks like::

            [
                [ (majorTickValue1, majorTickString1), (majorTickValue2, majorTickString2), ... ],
                [ (minorTickValue1, minorTickString1), (minorTickValue2, minorTickString2), ... ],
                ...
            ]

        If *ticks* is None, then the default tick system will be used instead.
        """
        self._tickLevels = ticks
        self.picture = None
        self.update()

    def setTickSpacing(self, major: Optional[float]=None,
                             minor: Optional[float]=None,
                             levels: Optional[List[Tuple[float, float]]]=None) -> None:
        """
        Explicitly determine the spacing of major and minor ticks. This
        overrides the default behavior of the tickSpacing method, and disables
        the effect of setTicks(). Arguments may be either *major* and *minor*,
        or *levels* which is a list of (spacing, offset) tuples for each
        tick level desired.

        If no arguments are given, then the default behavior of tickSpacing
        is enabled.

        Examples::

            # two levels, all offsets = 0
            axis.setTickSpacing(5, 1)
            # three levels, all offsets = 0
            axis.setTickSpacing([(3, 0), (1, 0), (0.25, 0)])
            # reset to default
            axis.setTickSpacing()
        """

        if levels is None:
            if major is None or minor is None:
                levels = None
            else:
                levels = [(major, 0.), (minor, 0.)]
        self._tickSpacing = levels
        self.picture = None
        self.update()

    def tickSpacing(self, minVal: float,
                          maxVal: float, size: float) -> List[Tuple[float, float]]:
        """Return values describing the desired spacing and offset of ticks.

        This method is called whenever the axis needs to be redrawn and is a
        good method to override in subclasses that require control over tick locations.

        The return value must be a list of tuples, one for each set of ticks::

            [
                (major tick spacing, offset),
                (minor tick spacing, offset),
                (sub-minor tick spacing, offset),
                ...
            ]
        """
        # First check for override tick spacing
        if self._tickSpacing is not None:
            return self._tickSpacing

        dif = abs(maxVal - minVal)
        if dif == 0:
            return []

        ## decide optimal minor tick spacing in pixels (this is just aesthetics)
        optimalTickCount = max(2., log(size))

        ## optimal minor tick spacing
        optimalSpacing = dif / optimalTickCount

        ## the largest power-of-10 spacing which is smaller than optimal
        p10unit = 10 ** floor(log10(optimalSpacing))

        ## Determine major/minor tick spacings which flank the optimal spacing.
        intervals = np.array([1., 2., 10., 20., 100.]) * p10unit
        minorIndex = 0
        while intervals[minorIndex+1] <= optimalSpacing:
            minorIndex += 1

        levels: List[Tuple[float, float]] = [
            (intervals[minorIndex+2], 0.),
            (intervals[minorIndex+1], 0.),
            #(intervals[minorIndex], 0.)    ## Pretty, but eats up CPU
        ]

        if self.style['maxTickLevel'] >= 2:
            ## decide whether to include the last level of ticks
            minSpacing = min(size / 20., 30.)
            maxTickCount = size / minSpacing
            if dif / intervals[minorIndex] <= maxTickCount:
                levels.append((intervals[minorIndex], 0))

        return levels

        ##### This does not work -- switching between 2/5 confuses the automatic text-level-selection
        ### Determine major/minor tick spacings which flank the optimal spacing.
        #intervals = np.array([1., 2., 5., 10., 20., 50., 100.]) * p10unit
        #minorIndex = 0
        #while intervals[minorIndex+1] <= optimalSpacing:
            #minorIndex += 1

        ### make sure we never see 5 and 2 at the same time
        #intIndexes = [
            #[0,1,3],
            #[0,2,3],
            #[2,3,4],
            #[3,4,6],
            #[3,5,6],
        #][minorIndex]

        #return [
            #(intervals[intIndexes[2]], 0),
            #(intervals[intIndexes[1]], 0),
            #(intervals[intIndexes[0]], 0)
        #]

    def tickValues(self, minVal: float,
                         maxVal: float,
                         size: float) -> Union[List[Tuple[float, List[float]]], # linear ticks values
                                               List[Tuple[Optional[float], List[float]]]]: # log tick values
        """
        Return the values and spacing of ticks to draw::

            [
                (spacing, [major ticks]),
                (spacing, [minor ticks]),
                ...
            ]

        By default, this method calls tickSpacing to determine the correct tick locations.
        This is a good method to override in subclasses.
        """
        minVal, maxVal = sorted((minVal, maxVal))


        minVal *= self.scale
        maxVal *= self.scale
        #size *= self.scale

        ticks: List[Tuple[float, List[float]]] = []
        tickLevels = self.tickSpacing(minVal, maxVal, size)
        allValues = np.array([])
        for i in range(len(tickLevels)):
            spacing, offset = tickLevels[i]

            ## determine starting tick
            start = (ceil((minVal-offset) / spacing) * spacing) + offset

            ## determine number of ticks
            num = int((maxVal-start) / spacing) + 1
            values = (np.arange(num) * spacing + start) / self.scale
            ## remove any ticks that were present in higher levels
            ## we assume here that if the difference between a tick value and a previously seen tick value
            ## is less than spacing/100, then they are 'equal' and we can ignore the new tick.
            close = np.any(
                np.isclose(allValues, values[:, np.newaxis], rtol=0, atol=spacing/self.scale*0.01)
                , axis=-1
            )
            values = values[~close]
            allValues = np.concatenate([allValues, values])
            ticks.append((spacing/self.scale, values.tolist()))

        if self.logMode:
            return self.logTickValues(minVal, maxVal, size, ticks)


        #nticks = []
        #for t in ticks:
            #nvals = []
            #for v in t[1]:
                #nvals.append(v/self.scale)
            #nticks.append((t[0]/self.scale,nvals))
        #ticks = nticks

        return ticks

    def logTickValues(self, minVal: float,
                            maxVal: float,
                            size: float,
                            stdTicks: List[Tuple[float, List[float]]]) -> List[Tuple[Optional[float], List[float]]]:
        ## start with the tick spacing given by tickValues().
        ## Any level whose spacing is < 1 needs to be converted to log scale

        ticks: List[Tuple[Optional[float], List[float]]] = []
        for (spacing, t) in stdTicks:
            if spacing >= 1.0:
                ticks.append((spacing, t))

        if len(ticks) < 3:
            v1 = int(floor(minVal))
            v2 = int(ceil(maxVal))
            #major = list(range(v1+1, v2))

            minor = []
            for v in range(v1, v2):
                minor.extend(v + np.log10(np.arange(1, 10)))
            minor = [x for x in minor if x>minVal and x<maxVal]
            ticks.append((None, minor))
        return ticks

    def tickStrings(self, values: List[float],
                          scale: float,
                          spacing: float) -> List[str]:
        """Return the strings that should be placed next to ticks. This method is called
        when redrawing the axis and is a good method to override in subclasses.
        The method is called with a list of tick values, a scaling factor (see below), and the
        spacing between ticks (this is required since, in some instances, there may be only
        one tick and thus no other way to determine the tick spacing)

        The scale argument is used when the axis label is displaying units which may have an SI scaling prefix.
        When determining the text to display, use value*scale to correctly account for this prefix.
        For example, if the axis label's units are set to 'V', then a tick value of 0.001 might
        be accompanied by a scale value of 1000. This indicates that the label is displaying 'mV', and
        thus the tick should display 0.001 * 1000 = 1.
        """
        if self.logMode:
            return self.logTickStrings(values, scale, spacing)

        places = max(0, ceil(-log10(spacing*scale)))
        strings = []
        for v in values:
            vs = v * scale
            if abs(vs) < .001 or abs(vs) >= 10000:
                vstr = "%g" % vs
            else:
                vstr = ("%%0.%df" % places) % vs
            strings.append(vstr)
        return strings

    def logTickStrings(self, values: List[float],
                             scale: float,
                             spacing: float) -> List[str]:
        estrings = ["%0.1g"%x for x in 10 ** np.array(values).astype(float) * np.array(scale)]
        convdict = {"0": "⁰",
                    "1": "¹",
                    "2": "²",
                    "3": "³",
                    "4": "⁴",
                    "5": "⁵",
                    "6": "⁶",
                    "7": "⁷",
                    "8": "⁸",
                    "9": "⁹",
                    }
        dstrings = []
        for e in estrings:
            if e.count("e"):
                v, p = e.split("e")
                sign = "⁻" if p[0] == "-" else ""
                pot = "".join([convdict[pp] for pp in p[1:].lstrip("0")])
                if v == "1":
                    v = ""
                else:
                    v = v + "·"
                dstrings.append(v + "10" + sign + pot)
            else:
                dstrings.append(e)
        return dstrings

    def generateDrawSpecs(self, p: QtGui.QPainter) -> Optional[Tuple[Tuple[QtGui.QPen, QtCore.QPointF, QtCore.QPointF],
                                                                     List[Tuple[QtGui.QPen, Point, Point]],
                                                                     List[Tuple[QtCore.QRectF, QtCore.Qt.AlignmentFlag, str]]]]:
        """
        Calls tickValues() and tickStrings() to determine where and how ticks should
        be drawn, then generates from this a set of drawing commands to be
        interpreted by drawPicture().
        """
        profiler = debug.Profiler()
        if self.style['tickFont'] is not None:
            p.setFont(QtGui.QFont(self.style['tickFont'], self.style['tickFontsize']))
        bounds = self.mapRectFromParent(self.geometry())

        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            tickBounds = bounds
        else:
            tickBounds = linkedView.mapRectToItem(self, linkedView.boundingRect())

        if self.orientation == 'left':
            span = (bounds.topRight(), bounds.bottomRight())
            tickStart = tickBounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == 'right':
            span = (bounds.topLeft(), bounds.bottomLeft())
            tickStart = tickBounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == 'top':
            span = (bounds.bottomLeft(), bounds.bottomRight())
            tickStart = tickBounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == 'bottom':
            span = (bounds.topLeft(), bounds.topRight())
            tickStart = tickBounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1
        else:
            raise ValueError("self.orientation must be in ('left', 'right', 'top', 'bottom')")
        #print tickStart, tickStop, span

        ## determine size of this item in pixels
        points = list(map(self.mapToDevice, span))
        if None in points:
            return None
        lengthInPixels = Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return None

        # Determine major / minor / subminor axis ticks
        if self._tickLevels is None:
            tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
            tickStrings = None
        else:
            ## parse self.tickLevels into the formats returned by tickLevels() and tickStrings()
            tickLevels = [] # type: ignore
            tickStrings = []
            for level in self._tickLevels:
                values: List = []
                strings: List = []
                tickLevels.append((None, values)) # type: ignore
                tickStrings.append(strings)
                for val, strn in level:
                    values.append(val)
                    strings.append(strn)

        ## determine mapping between tick values and local coordinates
        dif = self.range[1] - self.range[0]
        if dif == 0:
            xScale = 1
            offset = 0
        else:
            if axis == 0:
                xScale = -bounds.height() / dif
                offset = self.range[0] * xScale - bounds.height()
            else:
                xScale = bounds.width() / dif
                offset = self.range[0] * xScale

        xRange = [x * xScale - offset for x in self.range]
        xMin = min(xRange)
        xMax = max(xRange)

        profiler('init')

        tickPositions: List = [] # remembers positions of previously drawn ticks

        ## compute coordinates to draw ticks
        ## draw three different intervals, long ticks first
        tickSpecs = []
        for i in range(len(tickLevels)):
            tickPositions.append([])
            ticks = tickLevels[i][1]

            ## length of tick
            tickLength = self.style['tickLength'] / ((i*0.5)+1.0)

            lineAlpha = self.style["tickAlpha"]
            if lineAlpha is None:
                lineAlpha = 255 / (i+1)
                if self.grid is not False:
                    lineAlpha *= self.grid/255. * fn.clip_scalar((0.05  * lengthInPixels / (len(ticks)+1)), 0., 1.)
            elif isinstance(lineAlpha, float):
                lineAlpha *= 255
                lineAlpha = max(0, int(round(lineAlpha)))
                lineAlpha = min(255, int(round(lineAlpha)))
            elif isinstance(lineAlpha, int):
                if (lineAlpha > 255) or (lineAlpha < 0):
                    raise ValueError("lineAlpha should be [0..255]")
            else:
                raise TypeError("Line Alpha should be of type None, float or int")

            for v in ticks:
                ## determine actual position to draw this tick
                x = (v * xScale) - offset
                if x < xMin or x > xMax:  ## last check to make sure no out-of-bounds ticks are drawn
                    tickPositions[i].append(None)
                    continue
                tickPositions[i].append(x)

                p1 = [x, x]
                p2 = [x, x]
                p1[axis] = tickStart
                p2[axis] = tickStop
                if self.grid is False:
                    p2[axis] += tickLength*tickDir
                tickPen = self.tickPen()
                color = tickPen.color()
                color.setAlpha(int(lineAlpha))
                tickPen.setColor(color)
                tickSpecs.append((tickPen, Point(p1), Point(p2)))
        profiler('compute ticks')


        if self.style['stopAxisAtTick'][0] is True:
            minTickPosition = min(map(min, tickPositions))
            if axis == 0:
                stop = max(span[0].y(), minTickPosition)
                span[0].setY(stop)
            else:
                stop = max(span[0].x(), minTickPosition)
                span[0].setX(stop)
        if self.style['stopAxisAtTick'][1] is True:
            maxTickPosition = max(map(max, tickPositions))
            if axis == 0:
                stop = min(span[1].y(), maxTickPosition)
                span[1].setY(stop)
            else:
                stop = min(span[1].x(), maxTickPosition)
                span[1].setX(stop)
        axisSpec = (self.pen(), span[0], span[1])


        textOffset = self.style['tickTextOffset'][axis]  ## spacing between axis and text
        #if self.style['autoExpandTextSpace'] is True:
            #textWidth = self.textWidth
            #textHeight = self.textHeight
        #else:
            #textWidth = self.style['tickTextWidth'] ## space allocated for horizontal text
            #textHeight = self.style['tickTextHeight'] ## space allocated for horizontal text

        textSize2 = 0
        lastTextSize2 = 0
        textRects: List = []
        textSpecs: List = []  ## list of draw

        # If values are hidden, return early
        if not self.style['showValues']:
            return (axisSpec, tickSpecs, textSpecs)

        for i in range(min(len(tickLevels), self.style['maxTextLevel']+1)):
            ## Get the list of strings to display for this level
            if tickStrings is None:
                spacing, values = tickLevels[i]
                strings = self.tickStrings(values, self.autoSIPrefixScale * self.scale, spacing)
            else:
                strings = tickStrings[i]

            if len(strings) == 0:
                continue

            ## ignore strings belonging to ticks that were previously ignored
            for j in range(len(strings)):
                if tickPositions[i][j] is None:
                    strings[j] = None

            ## Measure density of text; decide whether to draw this level
            rects: List = []
            for s in strings:
                if s is None:
                    rects.append(None)
                else:
                    br = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignmentFlag.AlignCenter, s)
                    ## boundingRect is usually just a bit too large
                    ## (but this probably depends on per-font metrics?)
                    br.setHeight(br.height() * 0.8)

                    rects.append(br)
                    textRects.append(rects[-1])

            if len(textRects) > 0:
                ## measure all text, make sure there's enough room
                if axis == 0:
                    textSize = np.sum([r.height() for r in textRects])
                    textSize2 = np.max([r.width() for r in textRects])
                else:
                    textSize = np.sum([r.width() for r in textRects])
                    textSize2 = np.max([r.height() for r in textRects])
            else:
                textSize = 0
                textSize2 = 0

            if i > 0:  ## always draw top level
                ## If the strings are too crowded, stop drawing text now.
                ## We use three different crowding limits based on the number
                ## of texts drawn so far.
                textFillRatio = float(textSize) / lengthInPixels
                finished = False
                for nTexts, limit in self.style['textFillLimits']:
                    if len(textSpecs) >= nTexts and textFillRatio >= limit:
                        finished = True
                        break
                if finished:
                    break

            lastTextSize2 = textSize2

            #spacing, values = tickLevels[best]
            #strings = self.tickStrings(values, self.scale, spacing)
            # Determine exactly where tick text should be drawn
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None: ## this tick was ignored because it is out of bounds
                    continue
                x = tickPositions[i][j]
                #textRect = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignmentFlag.AlignCenter, vstr)
                textRect = rects[j]
                height = textRect.height()
                width = textRect.width()
                #self.textHeight = height
                offset = max(0,self.style['tickLength']) + textOffset

                rect = QtCore.QRectF()
                if self.orientation == 'left':
                    alignFlags = QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignVCenter
                    rect = QtCore.QRectF(tickStop-offset-width, x-(height/2), width, height)
                elif self.orientation == 'right':
                    alignFlags = QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter
                    rect = QtCore.QRectF(tickStop+offset, x-(height/2), width, height)
                elif self.orientation == 'top':
                    alignFlags = QtCore.Qt.AlignmentFlag.AlignHCenter|QtCore.Qt.AlignmentFlag.AlignBottom
                    rect = QtCore.QRectF(x-width/2., tickStop-offset-height, width, height)
                elif self.orientation == 'bottom':
                    alignFlags = QtCore.Qt.AlignmentFlag.AlignHCenter|QtCore.Qt.AlignmentFlag.AlignTop
                    rect = QtCore.QRectF(x-width/2., tickStop+offset, width, height)

                textFlags = alignFlags | QtCore.Qt.TextFlag.TextDontClip
                #p.setPen(self.pen())
                #p.drawText(rect, textFlags, vstr)

                br = self.boundingRect()
                if not br.contains(rect):
                    continue

                textSpecs.append((rect, textFlags, vstr))
        profiler('compute text')

        ## update max text size if needed.
        self._updateMaxTextSize(lastTextSize2)

        return (axisSpec, tickSpecs, textSpecs)

    def drawPicture(self, p: QtGui.QPainter,
                          axisSpec: Tuple[QtGui.QPen, QtCore.QPointF, QtCore.QPointF],
                          tickSpecs: List[Tuple[QtGui.QPen, Point, Point]],
                          textSpecs: List[Tuple[QtCore.QRectF, QtCore.Qt.AlignmentFlag, str]]) -> None:
        profiler = debug.Profiler()

        p.setRenderHint(p.RenderHint.Antialiasing, False)
        p.setRenderHint(p.RenderHint.TextAntialiasing, True)

        ## draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        # p.translate(0.5,0)  ## resolves some damn pixel ambiguity

        ## draw ticks
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)
        profiler('draw ticks')

        # Draw all text
        if self.style['tickFont'] is not None:
            p.setFont(QtGui.QFont(self.style['tickFont'], self.style['tickFontsize']))
        p.setPen(self.textPen())
        bounding = self.boundingRect().toAlignedRect()
        p.setClipRect(bounding)
        for rect, flags, text in textSpecs:
            p.drawText(rect, int(flags), text)

        profiler('draw text')

    def show(self) -> None:
        GraphicsWidget.show(self)
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()

    def hide(self) -> None:
        GraphicsWidget.hide(self)
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()

    def wheelEvent(self, event: QtWidgets.QGraphicsSceneWheelEvent) -> None:
        lv = self.linkedView()
        if lv is None:
            return
        # Did the event occur inside the linked ViewBox (and not over the axis iteself)?
        if lv.sceneBoundingRect().contains(event.scenePos()):
            event.ignore()
            return
        else:
            # pass event to linked viewbox with appropriate single axis zoom parameter
            if self.orientation in ['left', 'right']:
                lv.wheelEvent(event, axis=1)
            else:
                lv.wheelEvent(event, axis=0)
        event.accept()

    def mouseDragEvent(self, event: MouseDragEvent) -> None:
        lv = self.linkedView()
        if lv is None:
            return
        # Did the mouse down event occur inside the linked ViewBox (and not the axis)?
        if lv.sceneBoundingRect().contains(event.buttonDownScenePos()):
            event.ignore()
            return
        # otherwise pass event to linked viewbox with appropriate single axis parameter
        if self.orientation in ['left', 'right']:
            return lv.mouseDragEvent(event, axis=1)
        else:
            return lv.mouseDragEvent(event, axis=0)

    def mouseClickEvent(self, event: MouseClickEvent) -> None:
        lv = self.linkedView()
        if lv is None:
            return
        return lv.mouseClickEvent(event)
