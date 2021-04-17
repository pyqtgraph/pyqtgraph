# -*- coding: utf-8 -*-
from ..Qt import QtCore
from .. import functions as fn
from .PlotItem import PlotItem
from .ImageItem import ImageItem
from .LinearRegionItem import LinearRegionItem

import weakref
import math
import numpy as np

__all__ = ['ColorBarItem']

class ColorBarItem(PlotItem):
    """
    **Bases:** :class:`PlotItem <pyqtgraph.PlotItem>`

    ColorBarItem is a simpler, compact alternative to HistogramLUTItem, without histogram
    or the option to adjust the look-up table.

    A labeled axis is displayed directly next to the gradient to help identify values.
    Handles included in the color bar allow for interactive adjustment.

    A ColorBarItem can be assigned one or more ImageItems that will be displayed according
    to the selected color map and levels. The ColorBarItem can be used as a separate
    element in a GraphicsLayout or added to the layout of a PlotItem used to display image
    data with coordinate axes.

    =============================  =============================================
    **Signals:**
    sigLevelsChanged(self)         Emitted when the range sliders are moved
    sigLevelsChangeFinished(self)  Emitted when the range sliders are released
    =============================  =============================================
    """
    sigLevelsChanged = QtCore.Signal(object)
    sigLevelsChangeFinished = QtCore.Signal(object)

    def __init__(self, values=(0,1), width=25, cmap=None, label=None,
                 interactive=True, limits=None, rounding=1,
                 orientation='vertical', pen='w', hoverPen='r', hoverBrush='#FF000080' ):
        """
        Create a new ColorBarItem.

        ==============  ===========================================================================
        **Arguments:**
        values          The range of values as tuple (min, max)
        width           (default=25) The width of the displayed color bar
        cmap            ColorMap object, look-up table is also applied to assigned ImageItem(s)
        label           (default=None) Label applied to color bar axis
        interactive     (default=True) Handles are displayed to interactively adjust level range
        limits          Limits to adjustment range as (low, high) tuple, None disables limit
        rounding        (default=1) Adjusted range values are rounded to multiples of this values
        orientation     (default='vertical') 'horizontal' gives a horizontal color bar
        pen             color of adjustement handles in interactive mode
        hoverPen        color of adjustement handles when hovered over
        hoverBrush      color of movable center region when hovered over
        ==============  ===========================================================================
        """
        super().__init__()
        self.img_list  = [] # list of controlled ImageItems
        self.values    = values
        self.cmap      = cmap
        self.rounding  = rounding
        self.horizontal = bool(orientation == 'horizontal')

        self.lo_prv, self.hi_prv = self.values # remember previous values while adjusting range
        if limits is None:
            self.lo_lim = None
            self.hi_lim = None
        else:
            self.lo_lim, self.hi_lim = limits

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
            axis.setZValue(1)
            # select main axis:
            if self.horizontal and key == 'bottom':
                self.axis = axis
            elif not self.horizontal and key == 'right':
                self.axis = axis
                self.axis.setWidth(45)
            else: # show other axes to create frame
                axis.setStyle( showValues=False, tickLength=0 )
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
        if cmap is not None: self.setCmap(cmap)

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
        assign ImageItem or list of ImageItems to be controlled

        ==============  ==========================================================================
        **Arguments:**
        image           ImageItem or list of [ImageItem, ImageItem, ...] that will be set to the
                        color map of the ColorBarItem. In interactive mode, the levels of all
                        assigned ImageItems will be controlled simultaneously.
        insert_in       If a PlotItem is given, the color bar is inserted on the right or bottom
                        of the plot
        ==============  ==========================================================================
        """
        try:
            self.img_list = [ weakref.ref(item) for item in img ]
        except TypeError: # failed to iterate, make a single-item list
            self.img_list = [ weakref.ref( img ) ]
        if insert_in is not None:
            if self.horizontal:
                insert_in.layout.addItem( self, 5, 1 ) # insert in layout below bottom axis
                insert_in.layout.setRowFixedHeight(4, 10) # enforce some space to axis above
            else:
                insert_in.layout.addItem( self, 2, 5 ) # insert in layout after right-hand axis
                insert_in.layout.setColumnFixedWidth(4, 5) # enforce some space to axis on the left
        self._update_items( update_cmap = True )

    def setCmap(self, cmap):
        """
        sets a ColorMap object to determine the ColorBarItem's look-up table. The same
        look-up table is applied to any assigned ImageItem.
        """
        self.cmap = cmap
        self._update_items( update_cmap = True )

    def setLevels(self, values=None, low=None, high=None ):
        """
        Sets the displayed range of levels as specified.

        ==============  ===========================================================================
        **Arguments:**
        values          Specify levels by tuple (low, high). Either value can be None to leave
                        to previous value unchanged. Takes precedence over low and high parameters.
        low             new low level to be applied to color bar and assigned images
        high            new high level to be applied to color bar and assigned images
        ==============  ===========================================================================
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
        """ returns the currently set levels as the tuple (low, high). """
        return self.values

    def _update_items(self, update_cmap=False):
        """ internal: update color maps for bar and assigned ImageItems """
        # update color bar:
        self.axis.setRange( self.values[0], self.values[1] )
        if update_cmap and self.cmap is not None:
            self.bar.setLookupTable( self.cmap.getLookupTable(nPts=256) )
        # update assigned ImageItems, too:
        for img_weakref in self.img_list:
            img = img_weakref()
            if img is None: continue # dereference weakref
            img.setLevels( self.values ) # (min,max) tuple
            if update_cmap and self.cmap is not None:
                img.setLookupTable( self.cmap.getLookupTable(nPts=256) )

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
        lo_new = self.lo_prv + (span_prv + 2*self.rounding) * bot #   reach 2x the minimal step
        # Alternative model with speed depending on level magnitude:
        # mean_val = abs(self.lo_prv) + abs(self.hi_prv) / 2
        # hi_new = self.hi_prv + (mean_val + 2*self.rounding) * top # make sure that we can always
        # lo_new = self.lo_prv + (mean_val + 2*self.rounding) * bot #    reach 2x the minimal step

        if self.hi_lim is not None and hi_new > self.hi_lim: # limit maximum value
            # print('lim +')
            hi_new = self.hi_lim
            if lo_new > hi_new - span_prv: # avoid collapsing the span against top or bottom limits
                lo_new = hi_new - span_prv
        if self.lo_lim is not None and lo_new < self.lo_lim: # limit minimum value
            # print('lim -')
            lo_new = self.lo_lim
            if hi_new < lo_new + span_prv: # avoid collapsing the span against top or bottom limits
                hi_new = lo_new + span_prv
        if lo_new + self.rounding > hi_new: # do not allow less than one "rounding" unit of span
            # print('lim X')
            if   bot == 0: hi_new = lo_new + self.rounding
            elif top == 0: lo_new = hi_new - self.rounding
            else:
                lo_new = (lo_new + hi_new - self.rounding) / 2
                hi_new = lo_new + self.rounding
        lo_new = self.rounding * round( lo_new/self.rounding )
        hi_new = self.rounding * round( hi_new/self.rounding )
        # if hi_new == lo_new: hi_new = lo_new + self.rounding # hack solution if axis span still collapses
        self.values = (lo_new, hi_new)
        self._update_items()
        self.sigLevelsChanged.emit(self)
