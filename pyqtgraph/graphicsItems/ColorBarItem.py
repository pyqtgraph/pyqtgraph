# -*- coding: utf-8 -*-
from ..Qt import QtCore, QtWidgets
from .. import functions as fn
from .PlotItem import PlotItem
from .ImageItem import ImageItem
from .LinearRegionItem import LinearRegionItem

import weakref, math
import numpy as np

__all__ = ['ColorBarItem']

class ColorBarItem(PlotItem):
    """
    **Bases:** :class:`PlotItem <pyqtgraph.PlotItem>`
    
    ColorBarItem is a simpler, compact alternative to HistogramLUTItem, without histogram
    or the option to adjust the look-up table. 
    
    A labeled axis is displayed directly next to the gradient to help identifying values.
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
    
    def __init__(self, values=(0,1), width=25, cmap=None, 
                 adjustable=True, limits=None, rounding=1, pen='w'):
        """
        Create a new ColorBarItem.
        
        ==============  =============================================================================
        **Arguments:**
        values          The range of values as tuple (min, max)
        width           (default=25) The width of the displayed color bar
        cmap            ColorMap object, look-up table will also be applied to assigned ImageItem(s)
        adjustable      (default=True) Display handles to interactively adjust image data range
        limits          Limits to adjustment range as (low, high) tuple
        rounding        (default=1) Range limits are rounded to multiples of this values
        pen             color of adjustement handles
        ==============  =============================================================================

        """
        super().__init__()
        self.img_list = [] # list of controlled ImageItems
        self.cmap       = cmap
        self.rounding   = rounding
        self.values = values
        self.lo_prv, self.hi_prv = self.values # remember previous values while adusting range
        if limits is None:
            self.lo_lim = None
            self.hi_lim = None
        else:
            self.lo_lim, self.hi_lim = limits
                
        self.setRange( xRange=(0,1), yRange=(0,256), padding=0 )
        self.disableAutoRange()
        self.hideButtons()
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled( False)

        self.layout.setColumnFixedWidth(1, width) # width of color bar
        
        for key in ['left','top','bottom']:
            self.showAxis(key)
            self.getAxis(key).setStyle( showValues=False )
        self.getAxis('left').setWidth(4)

        self.showAxis('right')
        self.axis = self.getAxis('right')
        self.axis.setStyle( showValues=True )
        self.axis.setWidth(44)
        self.axis.unlinkFromView()
        self.axis.setRange( self.values[0], self.values[1] )

        self.bar = ImageItem()
        self.bar.setZValue(-100)
        self.bar.setImage( np.linspace(0, 1, 256).reshape( (1,-1) ) )
        self.addItem(self.bar)
        
        if cmap is not None: self.setcmap( cmap )
            
        if adjustable:
            self.region = LinearRegionItem(
                [63, 191], 'horizontal', swapMode='block', 
                # span=(0.15, 0.85),  # limited span looks better, but disables grabbing the region
                pen=pen, brush=fn.mkBrush(None) )
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
        insert_in       If a PlotItem is given, the color bar is inserted to show to the right of 
                        the plot
        ==============  ==========================================================================
        """
        try:
            self.img_list = [ weakref.ref(item) for item in img ]
        except TypeError: # failed to iterate, make a single-item list
            self.img_list = [ weakref.ref( img ) ]
        if insert_in is not None:
            insert_in.layout.addItem( self, 2, 4 ) # insert in layout after right-hand axis
        self._update_items( update_cmap = True )

    def setcmap(self, cmap):
        """ 
        sets a ColorMap object to determine the ColorBarItem's look-up table. The same
        look-up table is applied to any assigned ImageItem. 
        """
        self.cmap = cmap
        self._update_items( update_cmap = True )
        
    def setLevels(self, values=None, low=None, high=None ):
        """
        Sets the displayed range of levels as specified.

        ==============  ==========================================================================
        **Arguments:**
        values          Specify levels by tuple (low, high). Either value can be None to leave
                        to previous value unchanged.
        low             new low level to be applied to color bar and assigned images
        high            new high level to be applied to color bar and assignes images
        ==============  ==========================================================================
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
        """ update color maps for bar and assigned ImageItems """
        # update color bar:
        self.axis.setRange( self.values[0], self.values[1] )
        if update_cmap and self.cmap is not None:
            self.bar.setLookupTable( self.cmap.getLookupTable() ) ## send function pointer, not the result
        # update assigned ImageItems, too:
        for img_weakref in self.img_list:
            img = img_weakref()
            if img is None: continue # dereference weakref
            img.setLevels( self.values ) # (min,max) tuple
            if update_cmap and self.cmap is not None:
                img.setLookupTable( self.cmap.getLookupTable() ) ## send function pointer, not the result

    def _regionChanged(self):
        """ snap adjusters back to default positions on release """
        self.lo_prv, self.hi_prv = self.values
        self.region_changed_enable = False # otherwise this affects the region again
        self.region.setRegion( (63, 191) )
        self.region_changed_enable = True
        self.sigLevelsChangeFinished.emit(self)

    def _regionChanging(self):
        """ recalculate levels based on new position of adjusters """
        if not self.region_changed_enable: return
        bot, top = self.region.getRegion()
        bot = ( (bot -  63) / 64 ) # -1 to +1 over half-bar range
        top = ( (top - 191) / 64 ) # -1 to +1 over half-bar range
        bot = math.copysign( bot**2, bot ) # quadratic behaviour for better sensitivity to small changes
        top = math.copysign( top**2, top )
        # These are the new values if adjuster is released now, rate of change depends on original separation
        span_prv = self.hi_prv - self.lo_prv # previous span of values
        hi_new = self.hi_prv + (span_prv + 2*self.rounding) * top # make sure that we can always reach 2x the minimal step
        lo_new = self.lo_prv + (span_prv + 2*self.rounding) * bot
        # Alternative model with speed depending on level magnitude:
        # mean_val = abs(self.lo_prv) + abs(self.hi_prv) / 2
        # hi_new = self.hi_prv + (mean_val + 2*self.rounding) * top # make sure that we can always reach 2x the minimal step
        # lo_new = self.lo_prv + (mean_val + 2*self.rounding) * bot
        
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
