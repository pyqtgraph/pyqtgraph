import math
import weakref

import numpy as np

from .. import colormap
from .. import functions as fn
from ..Qt import QtCore
from .ImageItem import ImageItem
from .LinearRegionItem import LinearRegionItem
from .PlotItem import PlotItem

__all__ = ['ColorBarItem']

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

    def __init__(self, values=(0,1), width=25, colorMap=None, label=None,
                 interactive=True, limits=None, rounding=1,
                 orientation='vertical', pen='w', hoverPen='r', hoverBrush='#FF000080', cmap=None ):
        """
        Creates a new ColorBarItem.

        Parameters
        ----------
        colorMap: `str` or :class:`~pyqtgraph.ColorMap`
            Determines the color map displayed and applied to assigned ImageItem(s).
        values: tuple of float
            The range of image levels covered by the color bar, as ``(min, max)``.
        width: float, default=25.0
            The width of the displayed color bar.
        label: str, optional
            Label applied to the color bar axis.
        interactive: bool, default=True
            If `True`, handles are displayed to interactively adjust the level range.
        limits: `tuple of float`, optional
            Limits the adjustment range to `(low, high)`, `None` disables the limit.
        rounding: float, default=1
            Adjusted range values are rounded to multiples of this value.
        orientation: str, default 'vertical'
            'horizontal' or 'h' gives a horizontal color bar instead of the default vertical bar
        pen: :class:`QPen` or color_like
            Sets the color of adjustment handles in interactive mode.
        hoverPen: :class:`QPen` or color_like
            Sets the color of adjustment handles when hovered over.
        hoverBrush: :class:`QBrush` or color_like
            Sets the color of movable center region when hovered over.
        """
        super().__init__()
        self.img_list  = [] # list of controlled ImageItems
        self.values    = values
        self._colorMap = None
        self.rounding  = rounding
        self.horizontal = bool( orientation in ('h', 'horizontal') )

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
            self.layout.setRowFixedHeight(2, width)
        else:
            self.setRange( xRange=(0,1), yRange=(0,256), padding=0 )
            self.layout.setColumnFixedWidth(1, width) # width of color bar

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
                axis.setStyle( showValues=False )
        self.axis.setStyle( showValues=True )
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

        if interactive:
            if self.horizontal:
                align = 'vertical'
            else:
                align = 'horizontal'
            self.region = LinearRegionItem(
                [63, 191], align, swapMode='block',
                # span=(0.15, 0.85),  # limited span looks better, but disables grabbing the region
                pen=pen, brush=fn.mkBrush(None), hoverPen=hoverPen, hoverBrush=hoverBrush )
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

    def setImageItem(self, img, insert_in=None):
        """
        Assigns an ImageItem or list of ImageItems to be represented and controlled

        Parameters
        ----------
        image: :class:`~pyqtgraph.ImageItem` or list of `[ImageItem, ImageItem, ...]`
            Assigns one or more ImageItems to this ColorBarItem.
            If a :class:`~pyqtgraph.ColorMap` is defined for ColorBarItem, this will be assigned to the 
            ImageItems. Otherwise, the ColorBarItem will attempt to retrieve a color map from the ImageItems.
            In interactive mode, ColorBarItem will control the levels of the assigned ImageItems, 
            simultaneously if there is more than one.
        insert_in: :class:`~pyqtgraph.PlotItem`, optional
            If a PlotItem is given, the color bar is inserted on the right
            or bottom of the plot, depending on the specified orientation.
        """
        try:
            self.img_list = [ weakref.ref(item) for item in img ]
        except TypeError: # failed to iterate, make a single-item list
            self.img_list = [ weakref.ref( img ) ]
        if self._colorMap is None: # check if one of the assigned images has a defined color map
            for img_weakref in self.img_list:
                img = img_weakref()
                if img is not None:
                    img_cm = img.getColorMap()
                    if img_cm is not None:
                        self._colorMap = img_cm
                        break
        if insert_in is not None:
            if self.horizontal:
                insert_in.layout.addItem( self, 5, 1 ) # insert in layout below bottom axis
                insert_in.layout.setRowFixedHeight(4, 10) # enforce some space to axis above
            else:
                insert_in.layout.addItem( self, 2, 5 ) # insert in layout after right-hand axis
                insert_in.layout.setColumnFixedWidth(4, 5) # enforce some space to axis on the left
        self._update_items( update_cmap = True )

    def setColorMap(self, colorMap):
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
        
    def colorMap(self):
        """
        Returns the assigned ColorMap object.
        """
        return self._colorMap

    def setLevels(self, values=None, low=None, high=None ):
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
        self._update_items()

    def levels(self):
        """ Returns the currently set levels as the tuple ``(low, high)``. """
        return self.values

    def _update_items(self, update_cmap=False):
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
                img.setLookupTable( self._colorMap.getLookupTable(nPts=256) )

    def _regionChanged(self):
        """ internal: snap adjusters back to default positions on release """
        self.lo_prv, self.hi_prv = self.values
        self.region_changed_enable = False # otherwise this affects the region again
        self.region.setRegion( (63, 191) )
        self.region_changed_enable = True
        self.sigLevelsChangeFinished.emit(self)

    def _regionChanging(self):
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
