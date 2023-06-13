import math
import numpy as np
from typing import Optional, Tuple, TypedDict, Union
import weakref

from .. import colormap
from .. import functions as fn
from .. import configStyle
from ..style.core import (
    ConfigColorHint,
    initItemStyle)
from ..Qt import QtCore, QtGui
from .ImageItem import ImageItem
from .LinearRegionItem import LinearRegionItem
from .PColorMeshItem import PColorMeshItem
from .PlotItem import PlotItem

__all__ = ['ColorBarItem']

Number = Union[float, int]

optsHint = TypedDict('optsHint',
                     {'width' : Number,
                      'handleColor' : ConfigColorHint,
                      'handleHoverColor' : ConfigColorHint,
                      'regionHoverBrush' : ConfigColorHint,
                      'orientation' : str},
                     total=False)
# kwargs are not typed because mypy has not yet included Unpack[Typeddict]

class ColorBarItem(PlotItem):
    """
    **Bases:** :class:`PlotItem <pyqtgraph.PlotItem>`

    :class:`ColorBarItem` controls the application of a
    :ref:`color map <apiref_colormap>` to one (or more)
    :class:`~pyqtgraph.ImageItem`. It is a simpler, compact alternative to
    :class:`~pyqtgraph.HistogramLUTItem`, without histogram or the
    option to adjust the colors of the look-up table.

    A labeled axis is displayed directly next to the gradient to help identify values.
    Handles included in the color bar allow for interactive adjustment.

    A ColorBarItem can be assigned one or more :class:`~pyqtgraph.ImageItem` s
    that will be displayed according to the selected color map and levels. The
    ColorBarItem can be used as a separate element in a
    :class:`~pyqtgraph.GraphicsLayout` or added to the layout of a
    :class:`~pyqtgraph.PlotItem` used to display image data with coordinate axes.

    =============================  =============================================
    **Signals:**
    sigLevelsChanged(self)         Emitted when the range sliders are moved
    sigLevelsChangeFinished(self)  Emitted when the range sliders are released
    =============================  =============================================
    """
    sigLevelsChanged = QtCore.Signal(object)
    sigLevelsChangeFinished = QtCore.Signal(object)

    def __init__(self, values=None, colorMap=None, label=None,
                 interactive=True, limits=None, rounding=1,
                 **kwargs):
        """
        Creates a new ColorBarItem.

        Parameters
        ----------
        colorMap: `str` or :class:`~pyqtgraph.ColorMap`
            Determines the color map displayed and applied to assigned ImageItem(s).
        values: tuple of float, optional
            The range of values that will be represented by the color bar, as ``(min, max)``.
            If no values are supplied, the default is to use user-specified values from
            an assigned image. If that does not exist, values will default to (0,1).
        label: str, optional
            Label applied to the color bar axis.
        interactive: bool, default=True
            If `True`, handles are displayed to interactively adjust the level range.
        limits: `tuple of float`, optional
            Limits the adjustment range to `(low, high)`, `None` disables the limit.
        rounding: float, default=1
            Adjusted range values are rounded to multiples of this value.
        **kwargs: optional
            style options , see setStyle() for accepted style parameters.
        """
        super().__init__()

        self.opts: optsHint = {}
        # Get default stylesheet
        initItemStyle(self, 'ColorBarItem', configStyle)
        # Update style if needed
        if len(kwargs)>0:
            self.setStyle(**kwargs)

        self.img_list  = [] # list of controlled ImageItems
        self._actively_adjusted_values = False
        if values is None:
            # Use default values
            # NOTE: User-specified values from the assigned item will be preferred over the default values of ColorBarItem
            values = (0,1)
        else:
            # The user explicitly entered values, prefer these over values from assigned image.
            self._actively_adjusted_values = True
        self.values    = values
        self._colorMap = None
        self.rounding  = rounding
        self.horizontal = bool( self.getOrientation() in ('h', 'horizontal') )

        self.lo_prv, self.hi_prv = self.values # remember previous values while adjusting range
        self.lo_lim = None
        self.hi_lim = None
        if limits is not None:
            self.lo_lim, self.hi_lim = limits
            # slightly expand the limits to match the rounding steps:
            if self.lo_lim is not None:
                self.lo_lim = self.rounding * math.floor( self.lo_lim/self.rounding )
            if self.hi_lim is not None:
                self.hi_lim = self.rounding * math.ceil( self.hi_lim/self.rounding )

        self.disableAutoRange()
        self.hideButtons()
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled( False)

        if self.horizontal:
            self.setRange( xRange=(0,256), yRange=(0,1), padding=0 )
            self.layout.setRowFixedHeight(2, self.getWidth())
        else:
            self.setRange( xRange=(0,1), yRange=(0,256), padding=0 )
            self.layout.setColumnFixedWidth(1, self.getWidth()) # width of color bar

        for key in ['left','right','top','bottom']:
            self.showAxis(key)
            axis = self.getAxis(key)
            axis.setZValue(0.5)
            # select main axis:
            if self.horizontal and key == 'bottom':
                self.axis = axis
            elif not self.horizontal and key == 'right':
                self.axis = axis
                self.axis.setWidth(45)
            else: # show other axes to create frame
                axis.setTicks( [] )
                axis.setStyle(showValues=False)
        self.axis.setStyle(showValues=True)
        self.axis.unlinkFromView()
        self.axis.setRange( self.values[0], self.values[1] )

        self.bar = ImageItem(axisOrder='col-major')
        if self.horizontal:
            self.bar.setImage( np.linspace(0, 1, 256).reshape( (-1,1) ) )
            if label is not None: self.getAxis('bottom').setLabel(label)
        else:
            self.bar.setImage( np.linspace(0, 1, 256).reshape( (1,-1) ) )
            if label is not None: self.getAxis('left').setLabel(label)
        self.addItem(self.bar)
        if colorMap is not None: self.setColorMap(colorMap)

        self.interactive = interactive
        if interactive:
            if self.horizontal:
                align = 'vertical'
            else:
                align = 'horizontal'
            self.region = LinearRegionItem(
                [63, 191], align, swapMode='block',
                # span=(0.15, 0.85),  # limited span looks better, but disables grabbing the region
                pen=self._getHandlePen(),
                brush=fn.mkBrush(None),
                hoverPen=self._getHandleHoverColor(),
                hoverBrush=self._getRegionHoverBrush() )
            self.region.setZValue(1000)
            self.region.lines[0].addMarker('<|>', size=6)
            self.region.lines[1].addMarker('<|>', size=6)
            self.region.sigRegionChanged.connect(self._regionChanging)
            self.region.sigRegionChangeFinished.connect(self._regionChanged)
            self.addItem(self.region)
            self.region_changed_enable = True
            self.region.setRegion( (63, 191) ) # place handles at 25% and 75% locations
        else:
            self.region = None
            self.region_changed_enable = False

    ##############################################################
    #
    #                   Style methods
    #
    ##############################################################

    def setWidth(self, width: Number) -> None:
        """
        Set the font size.
        """
        if not isinstance(width, float) and not isinstance(width, int):
            raise ValueError('width argument:{} is not a float or a int'.format(width))
        self.opts['width'] = width

    def getWidth(self) -> Number:
        """
        Get the current font size.
        """
        return self.opts['width']

    def setOrientation(self, orientation: str) -> None:
        """
        Set the orientation.
        """
        if not isinstance(orientation, str):
            raise ValueError('orientation argument:{} is not a string'.format(orientation))
        self.opts['orientation'] = orientation

    def getOrientation(self) -> str:
        """
        Get the current orientation.
        """
        return self.opts['orientation']

    def setHandleColor(self, handleColor: ConfigColorHint) -> None:
        """
        Set the handleColor.
        """
        self.opts['handleColor'] = handleColor


    def getHandleColor(self) -> ConfigColorHint:
        """
        Get the current handleColor.
        """
        return self.opts['handleColor']

    def setHandleHoverColor(self, handleHoverColor: ConfigColorHint) -> None:
        """
        Set the handleHoverColor.
        """
        self.opts['handleHoverColor'] = handleHoverColor


    def getHandleHoverColor(self) -> ConfigColorHint:
        """
        Get the current handleHoverColor.
        """
        return self.opts['handleHoverColor']

    def setRegionHoverBrush(self, regionHoverBrush: ConfigColorHint) -> None:
        """
        Set the regionHoverBrush.
        """
        self.opts['regionHoverBrush'] = regionHoverBrush

    def getRegionHoverBrush(self) -> ConfigColorHint:
        """
        Get the current regionHoverBrush.
        """
        return self.opts['regionHoverBrush']

    def _getHandlePen(self) -> QtGui.QPen:
        """
        Return the HandlePen following:
            1. the pen given by the user
            2. the default color define in the stylesheet
        """
        if hasattr(self, '_handlePen'):
            return self._handlePen
        else:
            return fn.mkPen(self.getHandleColor())

    def _getHandleHoverColor(self) -> QtGui.QPen:
        """
        Return the HandleHoverColor following:
            1. the pen given by the user
            2. the default color define in the stylesheet
        """
        if hasattr(self, '_handleHoverColor'):
            return self._handleHoverColor
        else:
            return fn.mkPen(self.getHandleHoverColor())

    def _getRegionHoverBrush(self) -> QtGui.QBrush:
        """
        Return the RegionHoverBrush following:
            1. the pen given by the user
            2. the default color define in the stylesheet
        """
        if hasattr(self, '_regionHoverBrush'):
            return self._regionHoverBrush
        else:
            return fn.mkBrush(self.getRegionHoverBrush())

    def setStyle(self, **kwargs) -> None:
        """
        Set the style of the ColorBarItem.

        Parameters
        ----------
        width: float
            The width of the displayed color bar.
        orientation: str
            'horizontal' or 'h' gives a horizontal color bar instead of the
            default vertical bar
        pen: :class:`QPen` or color_like
            Sets the color of adjustment handles in interactive mode.
        handleColor :
            Sets the color of adjustment handles in interactive mode.
        hoverPen: :class:`QPen` or color_like
            Sets the color of adjustment handles when hovered over.
        handleHoverColor :
            Sets the color of adjustment handles when hovered over.
        hoverBrush: :class:`QBrush` or color_like
            Sets the color of movable center region when hovered over.
        regionHoverBrush :
            Sets the color of movable center region when hovered over.
        """
        for k, v in kwargs.items():
            # If the key is a valid entry of the stylesheet
            if k in configStyle['ColorBarItem'].keys():
                fun = getattr(self, 'set{}{}'.format(k[:1].upper(), k[1:]))
                fun(v)
            # We save the different pen and brush to merge it later with the color
            elif 'pen' in kwargs.keys():
                self._pen = kwargs['pen']
            elif 'hoverPen' in kwargs.keys():
                self._hoverPen = kwargs['hoverPen']
            elif 'hoverBrush' in kwargs.keys():
                self._hoverBrush = kwargs['hoverBrush']
            else:
                raise ValueError('Your argument: "{}" is not a valid style argument.'.format(k))

    ##############################################################
    #
    #                   Item methods
    #
    ##############################################################

    def setImageItem(self, img: ImageItem,
                           insert_in: Optional[PlotItem]=None) -> None:
        """
        Assigns an item or list of items to be represented and controlled.
        Supported "image items": class:`~pyqtgraph.ImageItem`, class:`~pyqtgraph.PColorMeshItem`

        Parameters
        ----------
        image: :class:`~pyqtgraph.ImageItem` or list of :class:`~pyqtgraph.ImageItem`
            Assigns one or more image items to this ColorBarItem.
            If a :class:`~pyqtgraph.ColorMap` is defined for ColorBarItem, this will be assigned to the
            ImageItems. Otherwise, the ColorBarItem will attempt to retrieve a color map from the image items.
            In interactive mode, ColorBarItem will control the levels of the assigned image items,
            simultaneously if there is more than one.
            If the ColorBarItem was initialized without a specified ``values`` parameter, it will attempt
            to retrieve a set of user-defined ``levels`` from one of the image items. If this fails,
            the default values of ColorBarItem will be used as the (min, max) levels of the colorbar.
            Note that, for non-interactive ColorBarItems, levels may be overridden by image items with
            auto-scaling colors (defined by ``enableAutoLevels``). When using an interactive ColorBarItem
            in an animated plot, auto-scaling for its assigned image items should be *manually* disabled.
        insert_in: :class:`~pyqtgraph.PlotItem`, optional
            If a PlotItem is given, the color bar is inserted on the right
            or bottom of the plot, depending on the specified orientation.
        """
        try:
            self.img_list = [ weakref.ref(item) for item in img ]
        except TypeError: # failed to iterate, make a single-item list
            self.img_list = [ weakref.ref( img ) ]
        colormap_is_undefined = self._colorMap is None
        for img_weakref in self.img_list:
            img = img_weakref()
            if img is not None:
                if hasattr(img, "sigLevelsChanged"):
                    img.sigLevelsChanged.connect(self._levelsChangedHandler)

                if colormap_is_undefined and hasattr(img, 'getColorMap'): # check if one of the assigned images has a defined color map
                    img_cm = img.getColorMap()
                    if img_cm is not None:
                        self._colorMap = img_cm
                        colormap_is_undefined = False

                if not self._actively_adjusted_values:
                    # check if one of the assigned images has a non-default set of levels
                    if hasattr(img, 'getLevels'):
                        img_levels = img.getLevels()

                        if img_levels is not None:
                            self.setLevels(img_levels, update_items=False)


        if insert_in is not None:
            if self.horizontal:
                insert_in.layout.addItem( self, 5, 1 ) # insert in layout below bottom axis
                insert_in.layout.setRowFixedHeight(4, 10) # enforce some space to axis above
            else:
                insert_in.layout.addItem( self, 2, 5 ) # insert in layout after right-hand axis
                insert_in.layout.setColumnFixedWidth(4, 5) # enforce some space to axis on the left
        self._update_items( update_cmap = True )

    def setColorMap(self, colorMap: Union[str, colormap.ColorMap]) -> None:
        """
        Sets a color map to determine the ColorBarItem's look-up table. The same
        look-up table is applied to any assigned ImageItem.

        `colorMap` can be a :class:`~pyqtgraph.ColorMap` or a string argument that is passed to
        :func:`colormap.get() <pyqtgraph.colormap.get>`.
        """
        if isinstance(colorMap, str):
            colorMap = colormap.get(colorMap)
        self._colorMap = colorMap
        self._update_items( update_cmap = True )

    def colorMap(self) -> colormap.ColorMap:
        """
        Returns the assigned ColorMap object.
        """
        return self._colorMap

    def setLevels(self, values: Optional[Tuple[float, float]]=None,
                  low: Optional[float]=None,
                  high: Optional[float]=None,
                  update_items: Optional[bool]=True) -> None:
        """
        Sets the displayed range of image levels.

        Parameters
        ----------
        values: tuple of float
            Specifies levels as tuple ``(low, high)``. Either value can be `None` to leave
            the previous value unchanged. Takes precedence over `low` and `high` parameters.
        low: float
            Applies a new low level to color bar and assigned images
        high: float
            Applies a new high level to color bar and assigned images
        update_items: bool
            If true, update the iem
        """
        if values is not None: # values setting takes precendence
            low, high = values
        lo_new, hi_new = low, high
        lo_cur, hi_cur = self.values
        # allow None values to preserve original values:
        if lo_new is None: lo_new = lo_cur
        if hi_new is None: hi_new = hi_cur
        if lo_new > hi_new: # prevent reversal
            lo_new = hi_new = (lo_new + hi_new) / 2
        # clip to limits if set:
        if self.lo_lim is not None and lo_new < self.lo_lim: lo_new = self.lo_lim
        if self.hi_lim is not None and hi_new > self.hi_lim: hi_new = self.hi_lim
        self.values = self.lo_prv, self.hi_prv = (lo_new, hi_new)
        if update_items:
            self._update_items()
        else:
            # update color bar only:
            self.axis.setRange( self.values[0], self.values[1] )

    def levels(self) -> Tuple[float, float]:
        """ Returns the currently set levels as the tuple ``(low, high)``. """
        return self.values

    def _update_items(self, update_cmap: bool=False) -> None:
        """ internal: update color maps for bar and assigned ImageItems """
        # update color bar:
        self.axis.setRange( self.values[0], self.values[1] )
        if update_cmap and self._colorMap is not None:
            self.bar.setLookupTable( self._colorMap.getLookupTable(nPts=256) )
        # update assigned ImageItems, too:
        for img_weakref in self.img_list:
            img = img_weakref()
            if img is None: continue # dereference weakref
            img.setLevels( self.values ) # (min,max) tuple
            if update_cmap and self._colorMap is not None:
                if isinstance(img, PColorMeshItem):
                    img.setLookupTable( self._colorMap.getLookupTable(nPts=256, mode=self._colorMap.QCOLOR) )
                else:
                    img.setLookupTable( self._colorMap.getLookupTable(nPts=256) )

    def _levelsChangedHandler(self, levels: Optional[Tuple[float, float]]) -> None:
        """ internal: called when child item for some reason decides to update its levels without using ColorBarItem.
                      Will update colormap for the bar based on child items new levels """
        if levels != self.values:
            self.setLevels(levels, update_items=False)

    def _regionChanged(self) -> None:
        """ internal: snap adjusters back to default positions on release """
        self.lo_prv, self.hi_prv = self.values
        self.region_changed_enable = False # otherwise this affects the region again
        self.region.setRegion( (63, 191) )
        self.region_changed_enable = True
        self.sigLevelsChangeFinished.emit(self)

    def _regionChanging(self) -> None:
        """ internal: recalculate levels based on new position of adjusters """
        if not self.region_changed_enable: return
        bot, top = self.region.getRegion()
        bot = ( (bot -  63) / 64 ) # -1 to +1 over half-bar range
        top = ( (top - 191) / 64 ) # -1 to +1 over half-bar range
        bot = math.copysign( bot**2, bot ) # quadratic behaviour for sensitivity to small changes
        top = math.copysign( top**2, top )
        # These are the new values if adjuster is released now, rate of change depends on original separation
        span_prv = self.hi_prv - self.lo_prv # previous span of values
        hi_new = self.hi_prv + (span_prv + 2*self.rounding) * top # make sure that we can always
        lo_new = self.lo_prv + (span_prv + 2*self.rounding) * bot # reach 2x the minimal step

        # Alternative model with speed depending on level magnitude:
        # mean_val = abs(self.lo_prv) + abs(self.hi_prv) / 2
        # hi_new = self.hi_prv + (mean_val + 2*self.rounding) * top # make sure that we can always
        # lo_new = self.lo_prv + (mean_val + 2*self.rounding) * bot #    reach 2x the minimal step

        if self.hi_lim is not None:
            if hi_new > self.hi_lim: # limit maximum value
                hi_new = self.hi_lim
                if top!=0 and bot!=0:          # moving entire region?
                    lo_new = hi_new - span_prv # avoid collapsing the span against top limit
        if self.lo_lim is not None:
            if lo_new < self.lo_lim: # limit minimum value
                lo_new = self.lo_lim
                if top!=0 and bot!=0:          # moving entire region?
                    hi_new = lo_new + span_prv # avoid collapsing the span against bottom limit
        if hi_new-lo_new < self.rounding: # do not allow less than one "rounding" unit of span
            if   bot == 0: hi_new = lo_new + self.rounding
            elif top == 0: lo_new = hi_new - self.rounding
            else: # this should never happen, but let's try to recover if it does:
                mid = (hi_new + lo_new) / 2
                hi_new = mid + self.rounding / 2
                lo_new = mid - self.rounding / 2

        lo_new = self.rounding * round( lo_new/self.rounding )
        hi_new = self.rounding * round( hi_new/self.rounding )
        self.values = (lo_new, hi_new)
        self._update_items()
        self.sigLevelsChanged.emit(self)
