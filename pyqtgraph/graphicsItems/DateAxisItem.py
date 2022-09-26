import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np

from .AxisItem import AxisItem

__all__ = ['DateAxisItem']

MS_SPACING = 1/1000.0
SECOND_SPACING = 1
MINUTE_SPACING = 60
HOUR_SPACING = 3600
DAY_SPACING = 24 * HOUR_SPACING
WEEK_SPACING = 7 * DAY_SPACING
MONTH_SPACING = 30 * DAY_SPACING
YEAR_SPACING = 365 * DAY_SPACING

if sys.platform == 'win32':
    _epoch = datetime.utcfromtimestamp(0)
    def utcfromtimestamp(timestamp):
        return _epoch + timedelta(seconds=timestamp)
else:
    utcfromtimestamp = datetime.utcfromtimestamp

MIN_REGULAR_TIMESTAMP = (datetime(1, 1, 1) - datetime(1970,1,1)).total_seconds()
MAX_REGULAR_TIMESTAMP = (datetime(9999, 1, 1) - datetime(1970,1,1)).total_seconds()
SEC_PER_YEAR = 365.25*24*3600


# The stepper functions provide
#   'first' == True: The first tick value for 'val' being the minimum of the current view.
#   'first' == False: The next tick value for 'val' being the previous tick value.


def makeMSStepper(stepSize):
    def stepper(val, n, first: bool):
        if val < MIN_REGULAR_TIMESTAMP or val > MAX_REGULAR_TIMESTAMP:
            return np.inf

        if first:
            val *= 1000
            f = stepSize * 1000
            return (val // (n * f) + 1) * (n * f) / 1000.0
        else:
            return val + n * stepSize

    return stepper


def makeSStepper(stepSize):
    def stepper(val, n, first: bool):
        if val < MIN_REGULAR_TIMESTAMP or val > MAX_REGULAR_TIMESTAMP:
            return np.inf

        if first:
            return (val // (n * stepSize) + 1) * (n * stepSize)
        else:
            return val + n * stepSize

    return stepper


def makeMStepper(stepSize):
    def stepper(val, n, first: bool):
        if val < MIN_REGULAR_TIMESTAMP or val > MAX_REGULAR_TIMESTAMP:
            return np.inf

        d = utcfromtimestamp(val)
        base0m = d.month + n * stepSize - 1
        d = datetime(d.year + base0m // 12, base0m % 12 + 1, 1)
        return (d - datetime(1970, 1, 1)).total_seconds()

    return stepper


def makeYStepper(stepSize):
    def stepper(val, n, first: bool):
        if val < MIN_REGULAR_TIMESTAMP or val > MAX_REGULAR_TIMESTAMP:
            return np.inf

        d = utcfromtimestamp(val)
        next_year = (d.year // (n * stepSize) + 1) * (n * stepSize)
        if next_year > 9999:
            return np.inf
        next_date = datetime(next_year, 1, 1)
        return (next_date - datetime(1970, 1, 1)).total_seconds()

    return stepper

class TickSpec:
    """ Specifies the properties for a set of date ticks and computes ticks
    within a given utc timestamp range """
    def __init__(self, spacing, stepper, format, autoSkip=None):
        """
        ============= ==========================================================
        Arguments
        spacing       approximate (average) tick spacing
        stepper       a stepper function that takes a utc time stamp and a step
                      steps number n to compute the start of the next unit. You
                      can use the make_X_stepper functions to create common
                      steppers.
        format        a strftime compatible format string which will be used to
                      convert tick locations to date/time strings
        autoSkip      list of step size multipliers to be applied when the tick
                      density becomes too high. The tick spec automatically
                      applies additional powers of 10 (10, 100, ...) to the list
                      if necessary. Set to None to switch autoSkip off
        ============= ==========================================================

        """
        self.spacing = spacing
        self.step = stepper
        self.format = format
        self.autoSkip = autoSkip

    def makeTicks(self, minVal, maxVal, minSpc):
        ticks = []
        n = self.skipFactor(minSpc)
        x = self.step(minVal, n, first=True)
        while x <= maxVal:
            ticks.append(x)
            x = self.step(x, n, first=False)
        return (np.array(ticks), n)

    def skipFactor(self, minSpc):
        if self.autoSkip is None or minSpc < self.spacing:
            return 1
        factors = np.array(self.autoSkip, dtype=np.float64)
        while True:
            for f in factors:
                spc = self.spacing * f
                if spc > minSpc:
                    return int(f)
            factors *= 10
        

class ZoomLevel:
    """ Generates the ticks which appear in a specific zoom level """
    def __init__(self, tickSpecs, exampleText):
        """
        ============= ==========================================================
        tickSpecs     a list of one or more TickSpec objects with decreasing
                      coarseness
        ============= ==========================================================

        """
        self.tickSpecs = tickSpecs
        self.utcOffset = 0
        self.exampleText = exampleText

    def tickValues(self, minVal, maxVal, minSpc):
        # return tick values for this format in the range minVal, maxVal
        # the return value is a list of tuples (<avg spacing>, [tick positions])
        # minSpc indicates the minimum spacing (in seconds) between two ticks
        # to fullfill the maxTicksPerPt constraint of the DateAxisItem at the
        # current zoom level. This is used for auto skipping ticks.
        allTicks = np.array([])
        valueSpecs = []
        # back-project (minVal maxVal) to UTC, compute ticks then offset to
        # back to local time again
        utcMin = minVal - self.utcOffset
        utcMax = maxVal - self.utcOffset
        for spec in self.tickSpecs:
            ticks, skipFactor = spec.makeTicks(utcMin, utcMax, minSpc)
            # reposition tick labels to local time coordinates
            ticks += self.utcOffset
            # remove any ticks that were present in higher levels
            # we assume here that if the difference between a tick value and a previously seen tick value
            # is less than min-spacing/100, then they are 'equal' and we can ignore the new tick.
            close = np.any(
                np.isclose(allTicks, ticks[:, np.newaxis], rtol=0, atol=minSpc * 0.01),
                axis=-1,
            )
            ticks = ticks[~close]
            allTicks = np.concatenate([allTicks, ticks])
            valueSpecs.append((spec.spacing, ticks.tolist()))
            # if we're skipping ticks on the current level there's no point in
            # producing lower level ticks
            if skipFactor > 1:
                break
        return valueSpecs


YEAR_MONTH_ZOOM_LEVEL = ZoomLevel([
    TickSpec(YEAR_SPACING, makeYStepper(1), '%Y', autoSkip=[1, 5, 10, 25]),
    TickSpec(MONTH_SPACING, makeMStepper(1), '%b')
], "YYYY")
MONTH_DAY_ZOOM_LEVEL = ZoomLevel([
    TickSpec(MONTH_SPACING, makeMStepper(1), '%b'),
    TickSpec(DAY_SPACING, makeSStepper(DAY_SPACING), '%d', autoSkip=[1, 5])
], "MMM")
DAY_HOUR_ZOOM_LEVEL = ZoomLevel([
    TickSpec(DAY_SPACING, makeSStepper(DAY_SPACING), '%a %d'),
    TickSpec(HOUR_SPACING, makeSStepper(HOUR_SPACING), '%H:%M', autoSkip=[1, 6])
], "MMM 00")
HOUR_MINUTE_ZOOM_LEVEL = ZoomLevel([
    TickSpec(DAY_SPACING, makeSStepper(DAY_SPACING), '%a %d'),
    TickSpec(MINUTE_SPACING, makeSStepper(MINUTE_SPACING), '%H:%M',
             autoSkip=[1, 5, 15])
], "MMM 00")
HMS_ZOOM_LEVEL = ZoomLevel([
    TickSpec(SECOND_SPACING, makeSStepper(SECOND_SPACING), '%H:%M:%S',
             autoSkip=[1, 5, 15, 30])
], "99:99:99")
MS_ZOOM_LEVEL = ZoomLevel([
    TickSpec(MINUTE_SPACING, makeSStepper(MINUTE_SPACING), '%H:%M:%S'),
    TickSpec(MS_SPACING, makeMSStepper(MS_SPACING), '%S.%f',
             autoSkip=[1, 5, 10, 25])
], "99:99:99")


def getOffsetFromUtc():
    """Retrieve the utc offset respecting the daylight saving time"""
    ts = time.localtime()
    if ts.tm_isdst:
        utc_offset = time.altzone
    else:
        utc_offset = time.timezone
    return utc_offset


class DateAxisItem(AxisItem):
    """
    **Bases:** :class:`AxisItem <pyqtgraph.AxisItem>`
    
    An AxisItem that displays dates from unix timestamps.

    The display format is adjusted automatically depending on the current time
    density (seconds/point) on the axis. For more details on changing this
    behaviour, see :func:`setZoomLevelForDensity() <pyqtgraph.DateAxisItem.setZoomLevelForDensity>`.
    
    Can be added to an existing plot e.g. via 
    :func:`setAxisItems({'bottom':axis}) <pyqtgraph.PlotItem.setAxisItems>`.

    """

    def __init__(self, orientation='bottom', utcOffset=None, **kwargs):
        """
        Create a new DateAxisItem.
        
        For `orientation` and `**kwargs`, see
        :func:`AxisItem.__init__ <pyqtgraph.AxisItem.__init__>`.
        
        """

        super(DateAxisItem, self).__init__(orientation, **kwargs)
        # Set the zoom level to use depending on the time density on the axis
        if utcOffset is None:
            utcOffset = getOffsetFromUtc()
        self.utcOffset = utcOffset
        
        self.zoomLevels = OrderedDict([
            (np.inf,      YEAR_MONTH_ZOOM_LEVEL),
            (5 * 3600*24, MONTH_DAY_ZOOM_LEVEL),
            (6 * 3600,    DAY_HOUR_ZOOM_LEVEL),
            (15 * 60,     HOUR_MINUTE_ZOOM_LEVEL),
            (30,          HMS_ZOOM_LEVEL),
            (1,           MS_ZOOM_LEVEL),
            ])
        self.autoSIPrefix = False
    
    def tickStrings(self, values, scale, spacing):
        tickSpecs = self.zoomLevel.tickSpecs
        tickSpec = next((s for s in tickSpecs if s.spacing == spacing), None)
        try:
            dates = [utcfromtimestamp(v - self.utcOffset) for v in values]
        except (OverflowError, ValueError, OSError):
            # should not normally happen
            return ['%g' % ((v-self.utcOffset)//SEC_PER_YEAR + 1970) for v in values]
            
        formatStrings = []
        for x in dates:
            try:
                s = x.strftime(tickSpec.format)
                if '%f' in tickSpec.format:
                    # we only support ms precision
                    s = s[:-3]
                elif '%Y' in tickSpec.format:
                    s = s.lstrip('0')
                formatStrings.append(s)
            except ValueError:  # Windows can't handle dates before 1970
                formatStrings.append('')
        return formatStrings

    def tickValues(self, minVal, maxVal, size):
        density = (maxVal - minVal) / size
        self.setZoomLevelForDensity(density)
        values = self.zoomLevel.tickValues(minVal, maxVal, minSpc=self.minSpacing)
        return values

    def setZoomLevelForDensity(self, density):
        """
        Setting `zoomLevel` and `minSpacing` based on given density of seconds per pixel
        
        The display format is adjusted automatically depending on the current time
        density (seconds/point) on the axis. You can customize the behaviour by 
        overriding this function or setting a different set of zoom levels
        than the default one. The `zoomLevels` variable is a dictionary with the
        maximal distance of ticks in seconds which are allowed for each zoom level
        before the axis switches to the next coarser level. To customize the zoom level
        selection, override this function.
        """
        padding = 10
        
        # Size in pixels a specific tick label will take
        if self.orientation in ['bottom', 'top']:
            def sizeOf(text):
                return self.fontMetrics.boundingRect(text).width() + padding
        else:
            def sizeOf(text):
                return self.fontMetrics.boundingRect(text).height() + padding
        
        # Fallback zoom level: Years/Months
        self.zoomLevel = YEAR_MONTH_ZOOM_LEVEL
        for maximalSpacing, zoomLevel in self.zoomLevels.items():
            size = sizeOf(zoomLevel.exampleText)

            # Test if zoom level is too fine grained
            if maximalSpacing/size < density:
                break
            
            self.zoomLevel = zoomLevel
        
        # Set up zoomLevel
        self.zoomLevel.utcOffset = self.utcOffset
        
        # Calculate minimal spacing of items on the axis
        size = sizeOf(self.zoomLevel.exampleText)
        self.minSpacing = density*size
        
    def linkToView(self, view):
        """Link this axis to a ViewBox, causing its displayed range to match the visible range of the view."""
        self._linkToView_internal(view) # calls original linkToView code
        
        # Set default limits
        _min = MIN_REGULAR_TIMESTAMP
        _max = MAX_REGULAR_TIMESTAMP
        
        if self.orientation in ['right', 'left']:
            view.setLimits(yMin=_min, yMax=_max)
        else:
            view.setLimits(xMin=_min, xMax=_max)
        
    def generateDrawSpecs(self, p):
        # Get font metrics from QPainter
        # Not happening in "paint", as the QPainter p there is a different one from the one here,
        # so changing that font could cause unwanted side effects
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        
        self.fontMetrics = p.fontMetrics()
        
        # Get font scale factor by current window resolution
        
        return super(DateAxisItem, self).generateDrawSpecs(p)
