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

def make_s_stepper(n):
    def stepper(val):
        return int(val / n + 1) * n
    return stepper

def make_m_stepper(n):
    def stepper(val):
        d = datetime.utcfromtimestamp(val)
        base0m = (d.month + n - 1)
        d = datetime(d.year + base0m / 12, base0m % 12 + 1, 1)
        return (d - datetime(1970, 1, 1)).total_seconds()
    return stepper

def make_y_stepper(n):
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

    def make_ticks(self, min_val, max_val):
        ticks = []
        x = self.step(min_val)
        while x <= max_val:
            ticks.append(x)
            x = self.step(x)
        return np.array(ticks)


class ZoomLevel:
    """ Generates the ticks which appear in a specific zoom level """
    def __init__(self, tick_specs):
        """
        ============= ==========================================================
        tick_specs    a list of one or more TickSpec objects to with decreasing
                      coarseness
        ============= ==========================================================

        """
        self.tick_specs = tick_specs
        self.utc_offset = 0

    def tick_values(self, min_val, max_val):
        # return tick values for this format in the range minVal, maxVal
        # the return value is a list of tuples (<avg spacing>, [tick positions])
        all_ticks = []
        value_specs = []
        # back-project (minVal maxVal) to UTC, compute ticks then offset to
        # back to local time again
        utc_min = min_val - self.utc_offset
        utc_max = max_val - self.utc_offset
        for spec in self.tick_specs:
            ticks = spec.make_ticks(utc_min, utc_max)
            # reposition tick labels to local time coordinates
            ticks += self.utc_offset
            # remove any ticks that were present in higher levels
            tick_list = [x for x in ticks.tolist() if x not in all_ticks]
            all_ticks.extend(tick_list)
            value_specs.append((spec.spacing, tick_list))
        return value_specs


YEAR_MONTH_ZOOM_LEVEL = ZoomLevel([
    TickSpec(YEAR_SPACING, make_y_stepper(1), '%Y'),
    TickSpec(MONTH_SPACING, make_m_stepper(1), '%b')
])
MONTH_DAY_ZOOM_LEVEL = ZoomLevel([
    TickSpec(MONTH_SPACING, make_m_stepper(1), '%b %d'),
    TickSpec(DAY_SPACING, make_s_stepper(DAY_SPACING), '%d')
])
DAY_6HOUR_ZOOM_LEVEL = ZoomLevel([
    TickSpec(DAY_SPACING, make_s_stepper(DAY_SPACING), '%a %d'),
    TickSpec(6*HOUR_SPACING, make_s_stepper(6*HOUR_SPACING), '%H:%M')
])
DAY_HOUR_ZOOM_LEVEL = ZoomLevel([
    TickSpec(DAY_SPACING, make_s_stepper(DAY_SPACING), '%a %d'),
    TickSpec(HOUR_SPACING, make_s_stepper(HOUR_SPACING), '%H:%M')
])
HOUR_15MIN_ZOOM_LEVEL = ZoomLevel([
    TickSpec(HOUR_SPACING, make_s_stepper(HOUR_SPACING), '%H:%M'),
    TickSpec(15*MINUTE_SPACING, make_s_stepper(15*MINUTE_SPACING), '%H:%M')
])
HOUR_MINUTE_ZOOM_LEVEL = ZoomLevel([
    TickSpec(DAY_SPACING, make_s_stepper(DAY_SPACING), '%a %d'),
    TickSpec(MINUTE_SPACING, make_s_stepper(MINUTE_SPACING), '%H:%M')
])


class DateAxisItem(AxisItem):
    """ An AxisItem that displays dates from unix timestamps

    The display format is adjusted automatically depending on the current time
    density (seconds/point) on the axis.
    You can customize the behaviour by specifying a different set of zoom levels
    than the default one. The zoom_levels variable is a dictionary with the
    maximum number of seconds/point which are allowed for each ZoomLevel
    before the axis switches to the next coarser level.

    """

    def __init__(self, orientation, **kvargs):
        super(DateAxisItem, self).__init__(orientation, **kvargs)
         # Set the zoom level to use depending on the time density on the axis
        self.utc_offset = time.timezone
        self.zoom_level = YEAR_MONTH_ZOOM_LEVEL
        # 50 pt is a reasonable spacing for short labels
        pt = 50.0
        self.zoom_levels = {
            60/pt:          HOUR_MINUTE_ZOOM_LEVEL,
            15*60/pt:       HOUR_15MIN_ZOOM_LEVEL,
            3600/pt:        DAY_HOUR_ZOOM_LEVEL,
            6*3600/pt:      DAY_6HOUR_ZOOM_LEVEL,
            3600*24/pt:     MONTH_DAY_ZOOM_LEVEL,
            3600*24*30/pt:  YEAR_MONTH_ZOOM_LEVEL
        }

    def tickStrings(self, values, scale, spacing):
        tick_specs = self.zoom_level.tick_specs
        tick_spec = next((s for s in tick_specs if s.spacing == spacing), None)
        dates = [datetime.utcfromtimestamp(v - self.utc_offset) for v in values]
        format_strings = []
        for x in dates:
            try:
                format_strings.append(x.strftime(tick_spec.format))
            except ValueError:  # Windows can't handle dates before 1970
                format_strings.append('')
        return format_strings

    def tickValues(self, minVal, maxVal, size):
        density = (maxVal - minVal) / size
        self._set_zoom_level_for_density(density)
        values = self.zoom_level.tick_values(minVal, maxVal)
        return values

    def _set_zoom_level_for_density(self, density):
        keys = sorted(self.zoom_levels.iterkeys())
        key = next((k for k in keys if density < k), keys[-1])
        self.zoom_level = self.zoom_levels[key]
        self.zoom_level.utc_offset = self.utc_offset
