import numpy as np
import time
from datetime import datetime
from .AxisItem import AxisItem

__all__ = ['DateAxisItem', 'ZoomLevel']

SECOND_SPACING = 1
MINUTE_SPACING = 60
HOUR_SPACING = 3600
DAY_SPACING = 24 * HOUR_SPACING
WEEK_SPACING = 7 * DAY_SPACING
MONTH_SPACING = 30 * DAY_SPACING
YEAR_SPACING = 365 * DAY_SPACING

def makeSStepper(n):
    def stepper(val):
        return int(val / n + 1) * n
    return stepper

def makeMStepper(n):
    def stepper(val):
        d = datetime.utcfromtimestamp(val)
        base0m = (d.month + n - 1)
        d = datetime(d.year + base0m / 12, base0m % 12 + 1, 1)
        return (d - datetime(1970, 1, 1)).total_seconds()
    return stepper

def makeYStepper(n):
    def stepper(val):
        d = datetime.utcfromtimestamp(val)
        new_date = datetime(d.year + n, 1, 1)
        return (new_date - datetime(1970, 1, 1)).total_seconds()
    return stepper


class TickSpec:
    """ Specifies the properties for a set of date ticks and provides and
    computes ticks within a given utc timestamp range """
    def __init__(self, spacing, stepper, format):
        """
        ============= ==========================================================
        Arguments
        spacing       approximate (average) tick spacing
        stepper       a stepper function that takes a utc time stamp and returns
                      the start of the next unit, as created by the
                      make_X_stepper functions
        format        a strftime compatible format string which will be used to
                      convert tick locations to date/time strings
        ============= ==========================================================

        """
        self.spacing = spacing
        self.step = stepper
        self.format = format

    def makeTicks(self, minVal, maxVal):
        ticks = []
        x = self.step(minVal)
        while x <= maxVal:
            ticks.append(x)
            x = self.step(x)
        return np.array(ticks)


class ZoomLevel:
    """ Generates the ticks which appear in a specific zoom level """
    def __init__(self, tickSpecs):
        """
        ============= ==========================================================
        tickSpecs     a list of one or more TickSpec objects to with decreasing
                      coarseness
        ============= ==========================================================

        """
        self.tickSpecs = tickSpecs
        self.utcOffset = 0

    def tickValues(self, minVal, maxVal):
        # return tick values for this format in the range minVal, maxVal
        # the return value is a list of tuples (<avg spacing>, [tick positions])
        allTicks = []
        valueSpecs = []
        # back-project (minVal maxVal) to UTC, compute ticks then offset to
        # back to local time again
        utcMin = minVal - self.utcOffset
        utcMax = maxVal - self.utcOffset
        for spec in self.tickSpecs:
            ticks = spec.makeTicks(utcMin, utcMax)
            # reposition tick labels to local time coordinates
            ticks += self.utcOffset
            # remove any ticks that were present in higher levels
            tick_list = [x for x in ticks.tolist() if x not in allTicks]
            allTicks.extend(tick_list)
            valueSpecs.append((spec.spacing, tick_list))
        return valueSpecs


YEAR_MONTH_ZOOM_LEVEL = ZoomLevel([
    TickSpec(YEAR_SPACING, makeYStepper(1), '%Y'),
    TickSpec(MONTH_SPACING, makeMStepper(1), '%b')
])
MONTH_DAY_ZOOM_LEVEL = ZoomLevel([
    TickSpec(MONTH_SPACING, makeMStepper(1), '%b %d'),
    TickSpec(DAY_SPACING, makeSStepper(DAY_SPACING), '%d')
])
DAY_6HOUR_ZOOM_LEVEL = ZoomLevel([
    TickSpec(DAY_SPACING, makeSStepper(DAY_SPACING), '%a %d'),
    TickSpec(6*HOUR_SPACING, makeSStepper(6*HOUR_SPACING), '%H:%M')
])
DAY_HOUR_ZOOM_LEVEL = ZoomLevel([
    TickSpec(DAY_SPACING, makeSStepper(DAY_SPACING), '%a %d'),
    TickSpec(HOUR_SPACING, makeSStepper(HOUR_SPACING), '%H:%M')
])
HOUR_15MIN_ZOOM_LEVEL = ZoomLevel([
    TickSpec(HOUR_SPACING, makeSStepper(HOUR_SPACING), '%H:%M'),
    TickSpec(15*MINUTE_SPACING, makeSStepper(15*MINUTE_SPACING), '%H:%M')
])
HOUR_MINUTE_ZOOM_LEVEL = ZoomLevel([
    TickSpec(DAY_SPACING, makeSStepper(DAY_SPACING), '%a %d'),
    TickSpec(MINUTE_SPACING, makeSStepper(MINUTE_SPACING), '%H:%M')
])


class DateAxisItem(AxisItem):
    """ An AxisItem that displays dates from unix timestamps

    The display format is adjusted automatically depending on the current time
    density (seconds/point) on the axis.
    You can customize the behaviour by specifying a different set of zoom levels
    than the default one. The zoomLevels variable is a dictionary with the
    maximum number of seconds/point which are allowed for each ZoomLevel
    before the axis switches to the next coarser level.

    """

    def __init__(self, orientation, **kvargs):
        super(DateAxisItem, self).__init__(orientation, **kvargs)
         # Set the zoom level to use depending on the time density on the axis
        self.utcOffset = time.timezone
        self.zoomLevel = YEAR_MONTH_ZOOM_LEVEL
        # 50 pt is a reasonable spacing for short labels
        pt = 50.0
        self.zoomLevels = {
            60/pt:          HOUR_MINUTE_ZOOM_LEVEL,
            15*60/pt:       HOUR_15MIN_ZOOM_LEVEL,
            3600/pt:        DAY_HOUR_ZOOM_LEVEL,
            6*3600/pt:      DAY_6HOUR_ZOOM_LEVEL,
            3600*24/pt:     MONTH_DAY_ZOOM_LEVEL,
            3600*24*30/pt:  YEAR_MONTH_ZOOM_LEVEL
        }

    def tickStrings(self, values, scale, spacing):
        tickSpecs = self.zoomLevel.tickSpecs
        tickSpec = next((s for s in tickSpecs if s.spacing == spacing), None)
        dates = [datetime.utcfromtimestamp(v - self.utcOffset) for v in values]
        formatStrings = []
        for x in dates:
            try:
                formatStrings.append(x.strftime(tickSpec.format))
            except ValueError:  # Windows can't handle dates before 1970
                formatStrings.append('')
        return formatStrings

    def tickValues(self, minVal, maxVal, size):
        density = (maxVal - minVal) / size
        self.setZoomLevelForDensity(density)
        values = self.zoomLevel.tickValues(minVal, maxVal)
        return values

    def setZoomLevelForDensity(self, density):
        keys = sorted(self.zoomLevels.iterkeys())
        key = next((k for k in keys if density < k), keys[-1])
        self.zoomLevel = self.zoomLevels[key]
        self.zoomLevel.utcOffset = self.utcOffset
