import weakref
from math import ceil, floor, frexp, isfinite, log10, sqrt

import numpy as np

from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget

__all__ = ['AxisItem']


class AxisItem(GraphicsWidget):
    """
    GraphicsItem showing a single plot axis with ticks, values, and label.
    
    Can be configured to fit on any side of a plot, automatically synchronize its
    displayed scale with ViewBox items. Ticks can be extended to draw a grid.
    
    If maxTickLength is negative, ticks point into the plot.

    Parameters
    ----------
    orientation : {'left', 'right', 'top', 'bottom'}
        The side of the plot the axis is attached to.
    pen : QPen or None
        Pen used when drawing axis and (by default) ticks.
    textPen : QPen or None
        Pen used when drawing tick labels.
    tickPen : QPen or None
        Pen used when drawing ticks.
    linkView : ViewBox or None
        Causes the range of values displayed in the axis to be linked to the visible
        range of a ViewBox.
    parent : QtWidgets.QGraphicsItem or None
        Parent Qt object to set to. End users are not expected to set, pyqtgraph should
        set correctly on its own.
    maxTickLength : int
        Maximum length of ticks to draw in pixels. Negative values draw into the
        plot, positive values draw outward.  Default -5.
    showValues : bool
        Whether to display values adjacent to ticks. Default true.
    **args
        All additional keyword arguments are passed to :func:`setLabel`.
    """
    def __init__(
            self,
            orientation: str,
            pen=None,
            textPen=None,
            tickPen = None,
            linkView=None,
            parent=None,
            maxTickLength=-5,
            showValues=True,
            **args,
    ):
        super().__init__(parent)
        self.label = QtWidgets.QGraphicsTextItem(self)
        self.picture = None
        self.orientation = orientation

        if orientation in {'left', 'right'}:
            self.label.setRotation(-90)
            # allow labels on vertical axis to extend above and below the length of the axis
            hide_overlapping_labels = False
        elif orientation in {'top', 'bottom'}:
            # stop labels on horizontal axis from overlapping so vertical axis labels have room
            hide_overlapping_labels = True
        else:
            raise ValueError(
                "Orientation argument must be one of 'left', 'right', 'top', or 'bottom'."
            )
        self.style = {
            'tickTextOffset': [5, 2],  ## (horizontal, vertical) spacing between text and axis
            'tickTextWidth': 30,  ## space reserved for tick text
            'tickTextHeight': 18,
            'autoExpandTextSpace': True,  ## automatically expand text space if needed
            'autoReduceTextSpace': True,
            'hideOverlappingLabels': hide_overlapping_labels,
            'tickFont': None,
            'stopAxisAtTick': (False, False),  ## whether axis is drawn to edge of box or to last tick
            'textFillLimits': [  ## how much of the axis to fill up with tick text, maximally.
                (0, 0.8),    ## never fill more than 80% of the axis
                (2, 0.6),    ## If we already have 2 ticks with text, fill no more than 60% of the axis
                (4, 0.4),    ## If we already have 4 ticks with text, fill no more than 40% of the axis
                (6, 0.2),    ## If we already have 6 ticks with text, fill no more than 20% of the axis
            ],
            'showValues': showValues,
            'tickLength': maxTickLength,
            'maxTickLevel': 2,
            'maxTextLevel': 2,
            'tickAlpha': None,  ## If not none, use this alpha for all ticks.
        }

        self.textWidth = 30  ## Keeps track of maximum width / height of tick text
        self.textHeight = 18

        # If the user specifies a width / height, remember that setting
        # indefinitely.
        self.fixedWidth = None
        self.fixedHeight = None

        self.logMode = False

        self._tickDensity = 1.0   # used to adjust scale the number of automatically generated ticks
        self._tickLevels  = None  # used to override the automatic ticking system with explicit ticks
        self._tickSpacing = None  # used to override default tickSpacing method
        self.scale = 1.0
        self.autoSIPrefix = True
        self.autoSIPrefixScale = 1.0

        self.labelText = ""
        self.labelUnits = ""
        self.labelUnitPrefix = ""
        self.unitPower = 1
        self.labelStyle = {}
        self._siPrefixEnableRanges = None
        self.setRange(0, 1)
        self.setLabel(**args)
        self.showLabel(False)


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

        self.grid = False

        #self.setCacheMode(self.DeviceCoordinateCache)

    def setStyle(self, **kwargs):
        """
        Set various style options.

        Parameters
        ----------
        **kwargs : dict, optional
            Here are a list of supported arguments.

            ===================== ======================================================
            Property              Description
            ===================== ======================================================
            tickLength            ``int``
                                  The maximum length of ticks in pixels. Positive values
                                  point toward the text; negative values point away.

            tickTextOffset        ``int`` 
                                  Reserved spacing between text and axis in pixels.

            tickTextWidth         ``int``
                                  Horizontal space reserved for tick text in pixels.

            tickTextHeight        ``int``
                                  Vertical space reserved for tick text in pixels.

            autoExpandTextSpace   ``bool``
                                  Automatically expand text space if the tick strings
                                  become too long.

            autoReduceTextSpace   ``bool``
                                  Automatically shrink the axis if necessary.

            hideOverlappingLabels ``bool`` or ``int``

                                  - ``True``  (default for horizontal axis): Hide tick
                                    labels which extend beyond the AxisItem's geometry
                                    rectangle.
                                  - ``False`` (default for vertical axis): Labels may be
                                    drawn extending beyond the extent of the axis.
                                  - ``int`` sets the tolerance limit for how many pixels
                                    a label is allowed to extend beyond the axis.
                                    Defaults to 15 for
                                    ``hideOverlappingLabels = False``.

            tickFont              :class:`QFont` or ``None``
                                  Determines the font used for tick values. Use None for
                                  the default font.
            
            stopAxisAtTick        tuple of ``bool, bool`` 
                                  The first element represents the horizontal axis, the
                                  second element represents the vertical axis.

                                  - ``True`` - The axis line is drawn only as far as the
                                    last tick.
                                  - ``False`` - The line is drawn to the edge of the
                                    :class:`~pyqtgraph.AxisItem` boundary.

            textFillLimits        list of ``(int, float)``
                                  This structure determines how the AxisItem decides how
                                  many ticks should have text appear next to them.
                                  The first value corresponds to the tick number.  The
                                  second value corresponds to the fill percentage. Each
                                  tuple in the list specifies what fraction of the axis
                                  length may be occupied by text, given the number of
                                  ticks that already have text displayed.
                                  
                                  For example ::

                                    [
                                        # Never fill more than 80% of the axis
                                        (0, 0.8),
                                        # If we already have 2 ticks with text, fill no
                                        # more than 60% of the axis
                                        (2, 0.6), 
                                        # If we already have 4 ticks with text, fill no
                                        # more than 40% of the axis
                                        (4, 0.4), 
                                        # If we already have 6 ticks with text, fill no
                                        # more than 20% of the axis
                                        (6, 0.2)
                                    ]
                                                
            showValues            ``bool``
                                  indicates whether text is displayed adjacent to ticks.
            
            tickAlpha             ``float``, ``int`` or ``None`` 
                                  If ``None``, pyqtgraph will draw the ticks with the
                                  alpha it deems appropriate. Otherwise, the alpha will
                                  be fixed at the value passed. With ``int``, accepted
                                  values are [0..255]. With value of type ``float``,
                                  accepted values are from [0..1].

            maxTickLevel          ``int``
                                  default: 2

                                  Tick (and grid line) density level.

                                  - 0: Show major ticks only
                                  - 1: Show major ticks and one level of minor ticks
                                  - 2: Show major ticks and two levels of minor ticks
                                    (higher CPU usage)
            ===================== ======================================================

        Raises
        ------
        NameError
            Raised when the name of a keyword argument is not recognized.
        TypeError
            Raised when a value for a keyword argument is of the wrong type.
        """
        for kwd, value in kwargs.items():
            if kwd not in self.style:
                raise NameError(f"{kwd} is not a valid style argument.")

            if (
                kwd in (
                    'tickLength',
                    'tickTextOffset',
                    'tickTextWidth',
                    'tickTextHeight'
                ) and 
                not isinstance(value, int)
            ):
                raise TypeError(f"Argument '{kwd}' must be int")

            if kwd == 'tickTextOffset':
                if self.orientation in ('left', 'right'):
                    self.style['tickTextOffset'][0] = value
                else:
                    self.style['tickTextOffset'][1] = value
            elif kwd == 'stopAxisAtTick':
                if len(value) != 2 or not all(isinstance(val, bool) for val in value):
                    raise TypeError(
                        "Argument 'stopAxisAtTick' must have type (bool, bool)"
                    )
                self.style[kwd] = value
            else:
                self.style[kwd] = value

        self.picture = None
        self._adjustSize()
        self.update()

    def close(self):
        self.scene().removeItem(self.label)
        self.label = None
        self.scene().removeItem(self)

    def setGrid(self, grid: int | float | bool):
        """
        Set the alpha value for the grid, or ``False`` to disable.

        When grid lines are enabled, the axis tick lines are extended to cover the
        extent of the linked ViewBox, if any.

        Parameters
        ----------
        grid : bool or int or float
            Alpha value to apply to :class:`~pyqtgraph.GridItem`.
            
            - ``False`` - Disable the grid.
            - ``int`` - Values between [0, 255] to set the alpha of the grid to.
            - ``float`` - Values between [0..1] to set the alpha of the grid to.
        """
        if isinstance(grid, float):
            grid = int(grid * 255)
            grid = min(grid, 255)
            grid = max(grid, 0)
        self.grid = grid
        self.picture = None
        self.prepareGeometryChange()
        self.update()

    def setLogMode(
        self,
        *args: tuple[bool] | tuple[bool, bool] | None,
        **kwargs: dict[str, bool] | None
    ):
        """
        Set log scaling for x and / or y axes.

        If two positional arguments are provided, the first will set log scaling
        for the x axis and the second for the y axis. If a single positional
        argument is provided, it will set the log scaling along the direction of
        the AxisItem. Alternatively, x and y can be passed as keyword arguments.

        If an axis is set to log scale, ticks are displayed on a logarithmic scale and
        values are adjusted accordingly. The linked ViewBox will be informed of the
        change.

        Parameters
        ----------
        *args : tuple of bool
            If length 1, sets log mode regardless of orientation.  If length 2, the
            first element toggles log mode for x-axis, and the second element toggles
            log mode for the y-axis.
        **kwargs : dict
            Pass a dictionary with keys `x` and `y`, where the values are ``bool`` to
            set the log mode for the respective `x` or `y` axis.  Trying to set the `y`
            axis log mode while this axis item is horizontal (or vice versa) will be
            ignored.

        See Also
        --------
        :meth:`~pyqtgraph.PlotItem.setLogMode`
            The method called to shift the values of the data.
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

        # inform the linked views of the change
        if self._linkedView is not None:
            if self.orientation in ('top', 'bottom'):
                self._linkedView().setLogMode('x', self.logMode)
            elif self.orientation in ('left', 'right'):
                self._linkedView().setLogMode('y', self.logMode)

        self.picture = None
        self.update()

    def setTickFont(self, font: QtGui.QFont | None):
        """
        Set the font used for tick values.
        
        Parameters
        ----------
        font : QtGui.QFont or None
            The font to use for the tick values. Set to ``None`` for the default font.
        """
        self.style['tickFont'] = font
        self.picture = None
        self.prepareGeometryChange()
        # Need to re-allocate space depending on font size?
        self.update()

    def resizeEvent(self, ev=None):
        # Set the position of the label
        nudge = 5
        # self.label is set to None on close, but resize events can still occur.
        if self.label is None:
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

    def showLabel(self, show: bool=True):
        """
        Show or hide the label text for this axis.

        Parameters
        ----------
        show : bool, optional
            Show the label text, by default True.
        """
        self.label.setVisible(show)
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()
        if self.autoSIPrefix:
            self.updateAutoSIPrefix()

    def setLabel(
        self,
        text: str | None=None,
        units: str | None=None,
        unitPrefix: str | None=None,
        siPrefixEnableRanges: tuple[tuple[float, float], ...] | None=None,
        unitPower: int | float=1,
        **kwargs
    ):
        """
        Set the text displayed adjacent to the axis.

        Parameters
        ----------
        text : str
            The text (excluding units) to display on the label for this axis.
        units : str
            The units for this axis. Units should generally be given without any scaling
            prefix (eg, 'V' instead of 'mV'). The scaling prefix will be automatically
            prepended based on the range of data displayed.
        unitPrefix : str
            An extra prefix to prepend to the units.
        siPrefixEnableRanges : tuple of tuple of float, float, Optional
            The ranges in which automatic SI prefix scaling is enabled. Defaults to
            everywhere, unless units is empty, in which case it defaults to
            ``((0., 1.), (1e9, inf))``.
        unitPower : int or float, optional
            The power to which the units are raised. For example, if units='m²', the
            unitPower should be 2. This ensures correct scaling when using SI prefixes.
            Supports positive, negative and non-integral powers.  Default is 1.
            Note: The power only affects the scaling, not the units themselves. For
            example, with units='m' and unitPower=2, the displayed units will still be 'm'.
        **kwargs
            All extra keyword arguments become CSS style options for the ``<span>`` tag
            which will surround the axis label and units. Note that CSS attributes are
            not always valid python arguments. Examples: ``color='#FFF'``,
            ``**{'font-size': '14pt'}``.

        Notes
        -----
        The final text generated for the label will usually take the form::

            <span style="...args...">{text} (prefix{units})</span>
        """
        self.labelText = text or ""
        self.labelUnits = units or ""
        self.labelUnitPrefix = unitPrefix or ""
        self.unitPower = unitPower
        if kwargs:
            self.labelStyle = kwargs
        self.setSIPrefixEnableRanges(siPrefixEnableRanges)
        # Account empty string and `None` for units and text
        visible = bool(text or units)
        self.showLabel(visible)
        self._updateLabel()

    def setSIPrefixEnableRanges(self, ranges=None):
        """
        Set the ranges in which automatic SI prefix scaling is enabled.

        This function allows you to define specific ranges where SI prefixes will be
        used. By default, SI prefix scaling is enabled everywhere, unless units are
        empty, in which case it defaults to ``((0., 1.), (1e9, inf))``.

        Parameters
        ----------
        ranges : tuple of tuple of float, float, optional
            A tuple of ranges where SI prefix scaling is enabled. Each range is a tuple
            containing two floats representing the start and end of the range.
        """
        self._siPrefixEnableRanges = ranges

    def getSIPrefixEnableRanges(self):
        """
        Get the ranges in which automatic SI prefix scaling is enabled.

        Returns
        -------
        tuple of tuple of float, float
            A tuple of ranges where SI prefix scaling is enabled. Each range is a tuple
            containing two floats representing the start and end of the range. If no
            custom ranges are set, then the default ranges are returned. The default
            ranges are ``((0., 1.), (1e9, inf))`` if units are empty, and 
            ``((0., inf))`` otherwise.
        """
        if self._siPrefixEnableRanges is not None:
            return self._siPrefixEnableRanges
        elif self.labelUnits == '':
            return (0., 1.), (1e9, float('inf'))
        else:
            return ((0., float('inf')),)

    def _updateLabel(self):
        self.label.setHtml(self.labelString())
        self._adjustSize()
        self.picture = None
        self.update()

    def labelString(self) -> str:
        """
        Generate the label string based on current label, units, and prefix.

        Returns
        -------
        str
            The complete label string, including units and any prefixes.
        """
        if self.labelUnits == '':
            if not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
                units = ''
            else:
                units = f'(x{1.0 / self.autoSIPrefixScale:g})'
        else:
            units = f'({self.labelUnitPrefix}{self.labelUnits})'

        s = f'{self.labelText} {units}'

        style = ';'.join([f'{k}: {self.labelStyle[k]}' for k in self.labelStyle])

        return f"<span style='{style}'>{s}</span>"

    def _updateMaxTextSize(self, x: int):
        ## Informs that the maximum tick size orthogonal to the axis has
        ## changed; we use this to decide whether the item needs to be resized
        ## to accommodate.
        if self.orientation in ['left', 'right']:
            if self.style["autoReduceTextSpace"]:
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

    def _adjustSize(self):
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()

    def setHeight(self, h: int | None=None):
        """
        Set the height of this axis reserved for ticks and tick labels.

        The height of the axis label is automatically added.

        Parameters
        ----------
        h : int or None, optional
            If ``None``, then the value will be determined automatically based on the
            size of the tick text, by default None.
        """
        self.fixedHeight = h
        self._updateHeight()

    def _updateHeight(self):
        if not self.isVisible():
            h = 0
        elif self.fixedHeight is None:
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

    def setWidth(self, w: int | None=None):
        """
        Set the width of this axis reserved for ticks and tick labels.

        The width of the axis label is automatically added.

        Parameters
        ----------
        w : int or None, optional
            If ``None``, then the value will be determined automatically based on the
            size of the tick text, by default None.
        """        
        self.fixedWidth = w
        self._updateWidth()

    def _updateWidth(self):
        if not self.isVisible():
            w = 0
        elif self.fixedWidth is None:
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
        """
        Get the pen used for drawing text, axes, ticks, and grid lines.

        If no custom pen has been set, this method will return a pen with the
        default foreground color.

        Returns
        -------
        QPen
            The pen used to draw text, axes, ticks, and grid lines.
        """
        if self._pen is None:
            return fn.mkPen(getConfigOption('foreground'))
        return fn.mkPen(self._pen)

    def setPen(self, *args, **kwargs):
        """
        Set the pen used for drawing text, axes, ticks, and grid lines.
        
        If no arguments given, the default foreground color will be used.

        Parameters
        ----------
        *args : tuple
            Arguments relayed to :func:`~pyqtgraph.mkPen`.
        **kwargs : dict
            Arguments relayed to `:func:`~pyqtgraph.mkPen`.

        See Also
        --------
        :func:`setConfigOption <pyqtgraph.setConfigOption>`
            Option to change the default foreground color.
        """        
        self.picture = None
        if args or kwargs:
            self._pen = fn.mkPen(*args, **kwargs)
        else:
            self._pen = fn.mkPen(getConfigOption('foreground'))
        self.labelStyle['color'] = self._pen.color().name() #   #RRGGBB
        self._updateLabel()

    def textPen(self) -> QtGui.QPen:
        """
        Get the pen used for drawing text.

        If no custom text pen has been set, this method will return a pen with the
        default foreground color.

        Returns
        -------
        QPen
            The pen used to draw text.
        """
        if self._textPen is None:
            return fn.mkPen(getConfigOption('foreground'))
        return fn.mkPen(self._textPen)

    def setTextPen(self, *args, **kwargs):
        """
        Set the pen used for drawing text.

        If no arguments given, the default foreground color will be used.
        
        Parameters
        ----------
        *args : tuple
            Arguments relayed to :func:`~pyqtgraph.mkPen`.
        **kwargs : dict
            Arguments relayed to `:func:`~pyqtgraph.mkPen`.

        See Also
        --------
        :func:`setConfigOption <pyqtgraph.setConfigOption>`
            Option to change the default foreground color.
        """     
        self.picture = None
        if args or kwargs:
            self._textPen = fn.mkPen(*args, **kwargs)
        else:
            self._textPen = fn.mkPen(getConfigOption('foreground'))
        self.labelStyle['color'] = self._textPen.color().name() #   #RRGGBB
        self._updateLabel()

    def tickPen(self) -> QtGui.QPen:
        """
        Get the pen used for drawing ticks.

        If no custom tick pen has been set, this method will return the axis's
        main pen.

        Returns
        -------
        QPen
            The pen used to draw tick marks.
        """
        return self.pen() if self._tickPen is None else fn.mkPen(self._tickPen)

    def setTickPen(self, *args, **kwargs):
        """
        Set the pen used for drawing ticks.

        If no arguments given, the default foreground color will be used.
        
        Parameters
        ----------
        *args : tuple
            Arguments relayed to :func:`~pyqtgraph.mkPen`.
        **kwargs : dict
            Arguments relayed to `:func:`~pyqtgraph.mkPen`.

        See Also
        --------
        :func:`setConfigOption <pyqtgraph.setConfigOption>`
            Option to change the default foreground color.
        """   
        self.picture = None
        self._tickPen = fn.mkPen(*args, **kwargs) if args or kwargs else None
        self._updateLabel()

    def setScale(self, scale=1.0):
        """
        Set the value scaling for this axis.

        Setting this value causes the axis to draw ticks and tick labels as if the view
        coordinate system were scaled.

        Parameters
        ----------
        scale : float, optional
            Value to scale the drawing of ticks and tick labels as if the view
            coordinate system was scaled, by default 1.0.
        """
        if scale != self.scale:
            self.scale = scale
            self._updateLabel()

    def enableAutoSIPrefix(self, enable=True):
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

        Parameters
        ----------
        enable : bool, optional
            Enable Auto SI prefix, by default True.
        """

        self.autoSIPrefix = enable
        self.updateAutoSIPrefix()

    def updateAutoSIPrefix(self):
        scale = 1.0
        prefix = ''
        if self.label.isVisible():
            _range = 10**np.array(self.range) if self.logMode else self.range
            scaling_value = max(abs(_range[0]), abs(_range[1])) * self.scale
            if any(low <= scaling_value <= high for low, high in self.getSIPrefixEnableRanges()):
                (scale, prefix) = fn.siScale(scaling_value, power=self.unitPower)

        self.autoSIPrefixScale = scale
        self.labelUnitPrefix = prefix
        self._updateLabel()

    def setRange(self, mn: float, mx: float):
        """
        Set the range of values displayed by the axis.

        Usually this is handled automatically by linking the axis to a ViewBox with
        :func:`linkToView <pyqtgraph.AxisItem.linkToView>`.

        Parameters
        ----------
        mn : float
            Bottom value to set the range to.
        mx : float
            Top value to set the range to.

        Raises
        ------
        ValueError
            When non-finite values are passed.
        """

        if not isfinite(mn) or not isfinite(mx):
            raise ValueError(f"Not setting range to [{mn}, {mx}]")
        self.range = [mn, mx]
        if self.autoSIPrefix:
            # XXX: Will already update once!
            self.updateAutoSIPrefix()
        else:
            self.picture = None
            self.update()

    def linkedView(self):
        """
        Return the ViewBox linked to this axis.

        Returns
        -------
        ViewBox
            The linked ViewBox, or ``None`` if there is no ViewBox linked.
        """
        return None if self._linkedView is None else self._linkedView()

    def _linkToView_internal(self, view):
        # We need this code to be available without override,
        # even though DateAxisItem overrides the user-side linkToView method
        self.unlinkFromView()

        self._linkedView = weakref.ref(view)
        if self.orientation in ['right', 'left']:
            view.sigYRangeChanged.connect(self.linkedViewChanged)
        else:
            view.sigXRangeChanged.connect(self.linkedViewChanged)
        view.sigResized.connect(self.linkedViewChanged)

    def linkToView(self, view):
        """
        Link to a ViewBox, causing its displayed range to match the view range.

        This is usually called automatically by the ViewBox.

        Parameters
        ----------
        view : ViewBox
            The view to link to.
        """
        self._linkToView_internal(view)

    def unlinkFromView(self):
        """
        Unlink this axis from its linked ViewBox.
        """
        oldView = self.linkedView()
        self._linkedView = None
        if oldView is not None:
            oldView.sigResized.disconnect(self.linkedViewChanged)
            if self.orientation in ['right', 'left']:
                oldView.sigYRangeChanged.disconnect(self.linkedViewChanged)
            else:
                oldView.sigXRangeChanged.disconnect(self.linkedViewChanged)

    @QtCore.Slot(object)
    @QtCore.Slot(object, object)
    def linkedViewChanged(self, view, newRange=None):
        """
        Call when the linked view range has changed.

        Parameters
        ----------
        view : ViewBox
            The view whose range has changed.
        newRange : tuple of float, float, optional
            The new range of the view, by default None.
        """
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

    def boundingRect(self):
        m = 0
        hide_overlapping_labels = self.style['hideOverlappingLabels']
        if hide_overlapping_labels is True:
            pass # skip further checks
        elif hide_overlapping_labels is False:
            m = 15
        else:
            try:
                m = int( self.style['hideOverlappingLabels'] )
            except ValueError: pass # ignore any non-numeric value

        linkedView = self.linkedView()
        if linkedView is not None and self.grid is not False:
            return (
                self.mapRectFromParent(self.geometry()) | 
                linkedView.mapRectToItem(self, linkedView.boundingRect())
            )
        rect = self.mapRectFromParent(self.geometry())
        ## extend rect if ticks go in negative direction
        ## also extend to account for text that flows past the edges
        tl = self.style['tickLength']
        if self.orientation == 'left':
            rect = rect.adjusted(0, -m, -min(0,tl), m)
        elif self.orientation == 'right':
            rect = rect.adjusted(min(0,tl), -m, 0, m)
        elif self.orientation == 'top':
            rect = rect.adjusted(-m, 0, m, -min(0,tl))
        elif self.orientation == 'bottom':
            rect = rect.adjusted(-m, min(0,tl), m, 0)
        return rect

    def shape(self):
        # override shape() to exclude grid lines from getting mouse events
        rect = self.mapRectFromParent(self.geometry())
        path = QtGui.QPainterPath()
        path.addRect(rect)
        return path

    def paint(self, p, opt, widget):
        profiler = debug.Profiler()
        if self.picture is None:
            try:
                picture = QtGui.QPicture()
                painter = QtGui.QPainter(picture)
                if self.style["tickFont"]:
                    painter.setFont(self.style["tickFont"])
                specs = self.generateDrawSpecs(painter)
                profiler('generate specs')
                if specs is not None:
                    self.drawPicture(painter, *specs)
                    profiler('draw picture')
            finally:
                painter.end()
            self.picture = picture
        self.picture.play(p)


    def setTickDensity(self, density=1.0):
        """
        Set the density of ticks displayed on the axis.

        A higher density value means that more ticks will be displayed. The density
        value is used in conjunction with the tickSpacing method to determine the
        actual tick locations.

        Parameters
        ----------
        density : float, optional
            Density of ticks to display, by default 1.0.
        """
        self._tickDensity = density
        self.picture = None
        self.update()


    def setTicks(
        self,
        ticks: list[list[tuple[float, str]]] | None
    ):
        """
        Explicitly determine which ticks to display.

        This overrides the behavior specified by
        :meth:`~pyqtgraph.AxisItem.tickSpacing`, :meth:`~pyqtgraph.AxisItem.tickValues`,
        and :meth:`~pyqtgraph.AxisItem.tickStrings`.
        
        The format for *ticks* looks like::

            [
                [
                    (majorTickValue1, majorTickString1),
                    (majorTickValue2, majorTickString2),
                    ...
                ],
                [
                    (minorTickValue1, minorTickString1),
                    (minorTickValue2, minorTickString2),
                    ...
                ],
                ...
            ]
        
        The two levels of major and minor ticks are expected. A third tier of additional
        ticks is optional. If *ticks* is ``None``, then the default tick system will be
        used.

        Parameters
        ----------
        ticks : list of list of float, str or None
            Explicitly set tick display information.
        
        See Also
        --------
        :meth:`~pyqtgraph.AxisItem.tickSpacing`
            How tick spacing is configured.
        :meth:`~pyqtgraph.AxisItem.tickValues`
            How tick values are set.
        :meth:`~pyqtgraph.AxisItem.tickStrings`
            How tick strings are specified.
        """        

        self._tickLevels = ticks
        self.picture = None
        self.update()

    def setTickSpacing(
        self,
        major: float | None=None,
        minor: float | None=None,
        levels: list[tuple[float, float]] | None=None
    ):
        """
        Explicitly determine the spacing of major and minor ticks.

        This overrides the default behavior of the tickSpacing method, and disables
        the effect of setTicks(). Arguments may be either *major* and *minor*,
        or *levels* which is a list of ``(spacing, offset)`` tuples for each
        tick level desired. If no arguments are given, then the default
        behavior of tickSpacing is enabled.

        Parameters
        ----------
        major : float, optional
            Spacing for major ticks, by default None.
        minor : float, optional
            Spacing for minor ticks, by default None.
        levels : list of tuple of float, float, optional
            A list of (spacing, offset) tuples for each tick level, by default None.

        Examples
        --------
        .. code-block:: python

            # two levels, all offsets = 0
            axis.setTickSpacing(5., 1.)
            # three levels, all offsets = 0
            axis.setTickSpacing(levels=[(3., 0.), (1., 0.), (0.25, 0.)])
            # reset to default
            axis.setTickSpacing()
        """

        if levels is None:
            levels = None if major is None else [(major, 0.), (minor, 0.)]
        self._tickSpacing = levels
        self.picture = None
        self.update()

    def tickSpacing(self, minVal: float, maxVal: float, size: float):
        """
        Determine the spacing of ticks on the axis.

        This method is called whenever the axis needs to be redrawn and is a
        good method to override in subclasses that require control over tick locations.

        Parameters
        ----------
        minVal : float
            Minimum value being displayed on the axis.
        maxVal : float
            Maximum value being displayed on the axis.
        size : float
            Length of the axis in pixels.

        Returns
        -------
        list of tuple of float, float
            A list of tuples, one for each tick level.
            Each tuple contains two values: ``(spacing, offset)``.  The spacing value
            is the distance between ticks, and the offset is the first tick relative to
            *minVal*. For example, if ``result[0]`` is ``(10, 0)``, then major ticks 
            will be displayed every 10 units and the first major tick will correspond to
            ``minVal``. If instead ``result[0]`` is ``(10, 5)``, then major ticks will
            be displayed every 10 units, but the first major tick will correspond to
            ``minVal + 5``.

            .. code-block:: python

                [
                    (major_tick_spacing, offset),
                    (minor_tick_spacing, offset),
                    (sub_minor_tick_spacing, offset),
                    ...
                ]
        """
        # First check for explicit tick spacing
        if self._tickSpacing is not None:
            return self._tickSpacing

        dif = abs(maxVal - minVal)
        if dif == 0:
            return []

        ref_size = 300. # axes longer than this display more than the minimum number of major ticks
        minNumberOfIntervals = max(
            2.25,       # 2.0 ensures two tick marks. Fudged increase to 2.25 allows room for tick labels.
            2.25 * self._tickDensity * sqrt(size/ref_size) # sub-linear growth of tick spacing with size
        )

        majorMaxSpacing = dif / minNumberOfIntervals

        # We want to calculate the power of 10 just below the maximum spacing.
        # Then divide by ten so that the scale factors for subdivision all become intergers.
        # p10unit = 10**( floor( log10(majorMaxSpacing) ) ) / 10

        # And we want to do it without a log operation:
        mantissa, exp2 = frexp(majorMaxSpacing) # IEEE 754 float already knows its exponent, no need to calculate
        p10unit = 10. ** ( # approximate a power of ten base factor just smaller than the given number
            floor(            # int would truncate towards zero to give wrong results for negative exponents
                (exp2-1)      # IEEE 754 exponent is ceiling of true exponent --> estimate floor by subtracting 1
                / 3.32192809488736 # division by log2(10)=3.32 converts base 2 exponent to base 10 exponent
            ) - 1             # subtract one extra power of ten so that we can work with integer scale factors >= 5
        )
        # neglecting the mantissa can underestimate by one power of 10 when the true value is JUST above the threshold.
        if 100. * p10unit <= majorMaxSpacing: # Cheaper to check this than to use a more complicated approximation.
            majorScaleFactor = 10
            p10unit *= 10.
        else:
            for majorScaleFactor in (50, 20, 10):
                if majorScaleFactor * p10unit <= majorMaxSpacing:
                    break # find the first value that is smaller or equal
        majorInterval = majorScaleFactor * p10unit
        # manual sanity check: print(f"{majorMaxSpacing:.2e} > {majorInterval:.2e} = {majorScaleFactor:.2e} x {p10unit:.2e}")
        levels = [
            (majorInterval, 0),
        ]

        if self.style['maxTickLevel'] >= 1:
            minorMinSpacing = 2 * dif/size   # no more than one minor tick per two pixels
            trials = (5, 10) if majorScaleFactor == 10 else (10, 20, 50)
            for minorScaleFactor in trials:
                minorInterval = minorScaleFactor * p10unit
                if minorInterval >= minorMinSpacing:
                    break # find the first value that is larger or equal to allowed minimum of 1 per 2px
            levels.append((minorInterval, 0))

        # extra ticks at 10% of major interval are pretty, but eat up CPU
        if self.style['maxTickLevel'] >= 2: # consider only when enabled
            if majorScaleFactor == 10:
                trials = (1, 2, 5, 10) # start at 10% of major interval, increase if needed
            elif majorScaleFactor == 20:
                trials = (2, 5, 10, 20) # start at 10% of major interval, increase if needed
            elif majorScaleFactor == 50:
                trials = (5, 10, 50) # start at 10% of major interval, increase if needed
            else: # invalid value
                trials = () # skip extra interval
                extraInterval = minorInterval
            for extraScaleFactor in trials:
                extraInterval = extraScaleFactor * p10unit
                if extraInterval >= minorMinSpacing or extraInterval == minorInterval:
                    break # find the first value that is larger or equal to allowed minimum of 1 per 2px
            if extraInterval < minorInterval: # add extra interval only if it is visible
                levels.append((extraInterval, 0))
        return levels


    def tickValues(self, minVal:float, maxVal:float, size: float):
        """
        Return the values and spacing of ticks to draw.

        The values returned are essentially the same as those returned by
        :meth:`~pyqtgraph.AxisItem.tickSpacing`, but with the addition of
        explicit tick values for each tick level. This method is a good
        method to override in subclasses.

        Parameters
        ----------
        minVal : float
            Minimum value to generate tick values for.
        maxVal : float
            Maximum value to generate tick values for.
        size : float
            The length of the axis in pixels.

        Returns
        -------
        list of tuple of float, list of float
            A list of tuples, one for each tick level. Each tuple contains two
            values: ``(spacing, values)``, where *spacing* is the distance between
            ticks and *values* is a list of tick values.
        """
        minVal, maxVal = sorted((minVal, maxVal))

        minVal *= self.scale
        maxVal *= self.scale

        ticks = []
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
                np.isclose(
                    allValues,
                    values[:, np.newaxis],
                    rtol=0,
                    atol=spacing/self.scale*0.01
                ),
                axis=-1
            )
            values = values[~close]
            allValues = np.concatenate([allValues, values])
            ticks.append((spacing/self.scale, values.tolist()))
        if self.logMode:
            return self.logTickValues(minVal, maxVal, size, ticks)
        return ticks

    def logTickValues(self, minVal, maxVal, size, stdTicks):
        """
        Return tick values for log-scale axes.

        This method is called by :meth:`~pyqtgraph.AxisItem.tickValues` when the axis
        is in logarithmic mode. It is a good method to override in subclasses.

        Parameters
        ----------
        minVal : float
            Minimum value to generate tick values for.
        maxVal : float
            Maximum value to generate tick values for.
        size : float
            The length of the axis in pixels.
        stdTicks : list of tuple of float, float
            The tick values generated by the standard
            :meth:`~pyqtgraph.AxisItem.tickValues` method.

        Returns
        -------
        list of tuple of float, float or list of tuple of None, float
            A list of tuples, one for each tick level. Each tuple contains two
            values: ``(spacing, values)``, where *spacing* is the distance between
            ticks and *values* is a list of tick values.
        """

        ## start with the tick spacing given by tickValues().
        ## Any level whose spacing is < 1 needs to be converted to log scale
        ticks = [(spacing, t) for spacing, t in stdTicks if spacing >= 1.0]
        if len(ticks) < 3:
            v1 = int(floor(minVal))
            v2 = int(ceil(maxVal))

            # minor = [v + np.log10(np.arange(1, 10)) for v in range(v1, v2)]
            minor = []
            for v in range(v1, v2):
                minor.extend(v + np.log10(np.arange(1, 10)))
            minor = [x for x in minor if x > minVal and x < maxVal]
            ticks.append((None, minor))
        return ticks

    def tickStrings(self, values: list[float], scale: float, spacing: float):
        """
        Return the strings that should be displayed at each tick value.

        This method is used to generate tick strings, and is called automatically.

        Parameters
        ----------
        values : list of float
            List of tick values.
        scale : float
            The scaling factor for tick values.
        spacing : float
            The spacing between ticks.

        Returns
        -------
        list of str
            List of strings to display at each tick value.
        """

        if self.logMode:
            return self.logTickStrings(values, scale, spacing)

        places = max(0, ceil(-log10(spacing * scale)))
        strings = []
        for v in values:
            vs = v * scale
            if abs(vs) < .001 or abs(vs) >= 10000:
                vstr = "%g" % vs
            else:
                vstr = ("%%0.%df" % places) % vs
            strings.append(vstr)
        return strings

    def logTickStrings(self, values: list[float], scale: float, spacing: float):
        """
        Return the strings that should be displayed at each tick value in log mode.

        This method is called by :meth:`~pyqtgraph.AxisItem.tickStrings` when the axis
        is in logarithmic mode. It is a good method to override in subclasses.

        Parameters
        ----------
        values : list of float
            List of tick values.
        scale : float
            The scaling factor for tick values.
        spacing : float
            The spacing between ticks.

        Returns
        -------
        list of str
            List of strings to display at each tick value.
        """
        estrings = [
            "%0.1g"%x
            for x in 10 ** np.array(values).astype(float) * np.array(scale)
        ]
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
                v = "" if v == "1" else f"{v}·"
                dstrings.append(f"{v}10{sign}{pot}")
            else:
                dstrings.append(e)
        return dstrings

    def generateDrawSpecs(self, p):
        """
        Generate the drawing specifications for the axis, ticks, and labels.

        This method determines all the coordinates and other information needed to draw
        the axis, including tick positions, tick labels, and axis label. It returns a
        tuple of values that are used to draw the axis. This is a good method to
        override in subclasses that need more control over the appearance of the axis.

        Parameters
        ----------
        p : QPainter
            The painter used to draw the axis.

        Returns
        -------
        tuple
            A tuple containing the drawing specifications for the axis, ticks, and
            labels. The tuple contains the following values:

            - ``axisSpec``: A tuple containing the pen, start point, and end point of
              the axis line.
            - ``tickSpecs``: A list of tuples, one for each tick. Each tuple contains
              the pen, start point, and end point of the tick line.
            - ``textSpecs``: A list of tuples, one for each tick label. Each tuple
              contains the bounding rectangle, alignment flags, and text of the label.
        
        :meta private:
        """
        profiler = debug.Profiler()
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        bounds = self.mapRectFromParent(self.geometry())

        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            tickBounds = bounds
        else:
            tickBounds = linkedView.mapRectToItem(self, linkedView.boundingRect())

        left_offset = -1.0
        right_offset = 1.0
        top_offset = -1.0
        bottom_offset = 1.0
        if self.orientation == 'left':
            span = (bounds.topRight() + Point(left_offset, top_offset),
                    bounds.bottomRight() + Point(left_offset, bottom_offset))
            tickStart = tickBounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == 'right':
            span = (bounds.topLeft() + Point(right_offset, top_offset),
                    bounds.bottomLeft() + Point(right_offset, bottom_offset))
            tickStart = tickBounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == 'top':
            span = (bounds.bottomLeft() + Point(left_offset, top_offset),
                    bounds.bottomRight() + Point(right_offset, top_offset))
            tickStart = tickBounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == 'bottom':
            span = (bounds.topLeft() + Point(left_offset, bottom_offset),
                    bounds.topRight() + Point(right_offset, bottom_offset))
            tickStart = tickBounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1
        else:
            raise ValueError(
                "self.orientation must be in {'left', 'right', 'top', 'bottom'}"
            )
        ## determine size of this item in pixels
        points = list(map(self.mapToDevice, span))
        if None in points:
            return
        lengthInPixels = Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return

        # Determine major / minor / subminor axis ticks
        if self._tickLevels is None:
            tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
            tickStrings = None
        else:
            ## parse self.tickLevels into the formats returned by tickLevels() and tickStrings()
            tickLevels = []
            tickStrings = []
            for level in self._tickLevels:
                values = []
                strings = []
                tickLevels.append((None, values))
                tickStrings.append(strings)
                for val, strn in level:
                    values.append(val)
                    strings.append(strn)

        ## determine mapping between tick values and local coordinates
        dif = self.range[1] - self.range[0]
        if dif == 0:
            xScale = 1
            offset = 0
        elif axis == 0:
            xScale = -bounds.height() / dif
            offset = self.range[0] * xScale - bounds.height()
        else:
            xScale = bounds.width() / dif
            offset = self.range[0] * xScale

        xRange = [x * xScale - offset for x in self.range]
        xMin = min(xRange)
        xMax = max(xRange)

        profiler('init')

        tickPositions = [] # remembers positions of previously drawn ticks

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
            tickPen = self.tickPen()
            if tickPen.brush().style() == QtCore.Qt.BrushStyle.SolidPattern: # only adjust simple color pens
                tickPen = QtGui.QPen(tickPen) # copy to a new QPen
                color = QtGui.QColor(tickPen.color()) # copy to a new QColor
                color.setAlpha(int(lineAlpha)) # adjust opacity
                tickPen.setColor(color)

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
        textSize2 = 0
        lastTextSize2 = 0
        textRects = []
        textSpecs = []  ## list of draw

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
            rects = []
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

            if textRects:
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

            # Determine exactly where tick text should be drawn
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None: ## this tick was ignored because it is out of bounds
                    continue
                x = tickPositions[i][j]
                textRect = rects[j]
                height = textRect.height()
                width = textRect.width()
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
                br = self.boundingRect()

                # br.contains(rect) suffers from floating point rounding errors
                if br & rect != rect:
                    continue

                textSpecs.append((rect, textFlags, vstr))
        profiler('compute text')

        ## update max text size if needed.
        self._updateMaxTextSize(lastTextSize2)

        return axisSpec, tickSpecs, textSpecs

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
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
            p.setFont(self.style['tickFont'])
        p.setPen(self.textPen())
        bounding = self.boundingRect().toAlignedRect()
        p.setClipRect(bounding)
        for rect, flags, text in textSpecs:
            p.drawText(rect, int(flags), text)

        profiler('draw text')

    def show(self):
        super().show()
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()

    def hide(self):
        super().hide()
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()

    def wheelEvent(self, event):
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

    def mouseDragEvent(self, event):
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

    def mouseClickEvent(self, event):
        lv = self.linkedView()
        if lv is None:
            return
        return lv.mouseClickEvent(event)
