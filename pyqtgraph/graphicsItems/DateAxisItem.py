import sys
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import warnings

import numpy as np

from ..Qt.QtCore import QDateTime
from .AxisItem import AxisItem

__all__ = ['DateAxisItem', 'TimeDeltaAxisItem']

MS_SPACING = 1/1000.0
SECOND_SPACING = 1
MINUTE_SPACING = 60
HOUR_SPACING = 3600
DAY_SPACING = 24 * HOUR_SPACING
WEEK_SPACING = 7 * DAY_SPACING
MONTH_SPACING = 30 * DAY_SPACING
YEAR_SPACING = 365 * DAY_SPACING

if sys.platform == 'win32':
    _epoch = datetime.fromtimestamp(0, timezone.utc)
    def utcfromtimestamp(timestamp):
        return _epoch + timedelta(seconds=timestamp)
else:
    def utcfromtimestamp(timestamp):
        return datetime.fromtimestamp(timestamp, timezone.utc)

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
                      can use the makeXStepper functions to create common
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
        self.utcOffset = None
        self.exampleText = exampleText

    def extendTimeRangeForSpacing(
            self, spacing: int, minVal: int | float, maxVal: int | float,
    ) -> tuple[int | float, int | float]:
        if spacing < HOUR_SPACING:
            return minVal, maxVal

        extendedMax = maxVal + abs(getPreferredOffsetFromUtc(maxVal, self.utcOffset))
        extendedMin = minVal - abs(getPreferredOffsetFromUtc(minVal, self.utcOffset))
        return extendedMin, extendedMax

    def moveTicksToLocalTimeCoords(
            self, ticks: np.ndarray, spacing: int, skipFactor: int,
    ) -> np.ndarray:
        if len(ticks) == 0:
            return ticks

        if (spacing == HOUR_SPACING and skipFactor > 1) or spacing > HOUR_SPACING:
            ticks += [applyOffsetToUtc(tick, self.utcOffset) for tick in ticks]
        elif spacing == HOUR_SPACING:
            ticks += [offsetToLocalHour(tick) for tick in ticks]
            ticks = np.array([tick for tick in ticks if offsetToLocalHour(tick) == 0])
        return ticks

    def tickValues(self, minVal, maxVal, minSpc):
        # return tick values for this format in the range minVal, maxVal
        # the return value is a list of tuples (<avg spacing>, [tick positions])
        # minSpc indicates the minimum spacing (in seconds) between two ticks
        # to fullfill the maxTicksPerPt constraint of the DateAxisItem at the
        # current zoom level. This is used for auto skipping ticks.
        allTicks = np.array([])
        valueSpecs = []

        for spec in self.tickSpecs:
            # extend time range, so that if distance to certain local hour
            # stretches due to DST change, this hour is still included in ticks
            extendedRange = self.extendTimeRangeForSpacing(spec.spacing, minVal, maxVal)
            ticks, skipFactor = spec.makeTicks(*extendedRange, minSpc)
            ticks = self.moveTicksToLocalTimeCoords(ticks, spec.spacing, skipFactor)
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


def fromSecsSinceEpoch(timestamp: float | int) -> QDateTime:
    try:
        return QDateTime.fromSecsSinceEpoch(round(timestamp))
    except OverflowError:
        return QDateTime()


def calculateUtcOffset(timestamp: float | int) -> int:
    return -fromSecsSinceEpoch(timestamp).offsetFromUtc()


def getPreferredOffsetFromUtc(
        timestamp: float | int,
        preferred_offset: int | None = None,
) -> int:
    """Retrieve the utc offset respecting the daylight saving time"""
    if preferred_offset is not None:
        return preferred_offset
    return calculateUtcOffset(timestamp)


def adjustTimestampToPreferredUtcOffset(
        timestamp: float | int,
        offest: int | None = None,
) -> int | float:
    return timestamp - getPreferredOffsetFromUtc(timestamp, offest)


def offsetToLocalHour(timestamp: float | int) -> int:
    local = fromSecsSinceEpoch(timestamp)
    roundedToHour = local.time()
    roundedToHour.setHMS(roundedToHour.hour(), 0, 0)
    return -roundedToHour.secsTo(local.time())


def applyOffsetFromUtc(timestamp: float | int) -> int:
    """
    UTC+4
    1970-01-02 02:00 (local) == 1970-01-01 22:00 (UTC) -> 1970-01-01 22:00 (local)

    NB: it won't work correctly for timestamps that represent same local time
    (when time goes backwards)
    """
    local = fromSecsSinceEpoch(timestamp)
    utcDate = local.toUTC().date()
    utcTime = local.toUTC().time()
    repositioned = QDateTime(utcDate, utcTime, local.timeZone())
    return repositioned.toSecsSinceEpoch()


def applyOffsetToUtc(
        timestamp: float | int,
        preferred_offset: int | None = None,
) -> int:
    delocalized = applyOffsetFromUtc(timestamp)
    return getPreferredOffsetFromUtc(delocalized, preferred_offset)


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
        self.utcOffset = utcOffset
        # Set the zoom level to use depending on the time density on the axis
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
            dates = [
                utcfromtimestamp(adjustTimestampToPreferredUtcOffset(v, self.utcOffset))
                for v in values
            ]
        except (OverflowError, ValueError, OSError):
            # should not normally happen
            offset = self.utcOffset or 0
            return ['%g' % ((v-offset)//SEC_PER_YEAR + 1970) for v in values]

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


def _format_hms_timedelta(seconds, show_seconds=False):
    """
    Format time with hours that can exceed 24

    Formats timestamps into HH:MM or HH:MM:SS format while allowing hours to
    exceed 24 (unlike standard time formatting functions),
    e.g. 81:16 for 81 hours and 16 minutes.

    Args:
        timestamp: Time in seconds
        show_seconds: Whether to include seconds in the output

    Returns:
        str: Formatted time string (either HH:MM or HH:MM:SS)
    """
    # Extract whole seconds and milliseconds
    seconds_whole = int(seconds)
    milliseconds = int((seconds - seconds_whole) * 1000)

    # Calculate hours, minutes, seconds
    hours = seconds_whole // 3600
    minutes = (seconds_whole % 3600) // 60
    secs = seconds_whole % 60

    if not milliseconds == 0:
        # For HH:MM:SS format, ensure we don't have partial seconds
        warnings.warn(
            f"Truncating milliseconds ({milliseconds} ms), "
            "this may lead to an incorrect label for the tick."
        )

    if show_seconds:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    if not np.isclose(secs, 0, atol=1e-10):
        # For HH:MM format, ensure we're on a minute boundary
        warnings.warn(
            f"Truncating seconds ({secs} s), "
            "this may lead to an incorrect label for the tick."
        )

    return f"{hours:02d}:{minutes:02d}"


def _format_day_timedelta(seconds):
    """
    Format time to hour string.

    Args:
        timestamp: Time in seconds

    Returns:
        str: Formatted time string (e.g. '21 d')
    """
    hours = int(seconds // (3600 * 24))
    missing_seconds = hours * 3600 * 24 - seconds
    if not missing_seconds == 0:
        warnings.warn(
            f"Truncating seconds ({missing_seconds} s), "
            "this may lead to an incorrect label for the tick."
        )

    return f"{hours:d} d"


DAY_DT_ZOOM_LEVEL = ZoomLevel(
    [TickSpec(DAY_SPACING, makeSStepper(DAY_SPACING), None, autoSkip=[2, 5, 10, 20, 30])],
    "123 d",
)

H_DT_ZOOM_LEVEL = ZoomLevel(
    [TickSpec(HOUR_SPACING, makeSStepper(HOUR_SPACING), None, autoSkip=[1, 5, 15, 30])],
    "99:99",
)
HM_DT_ZOOM_LEVEL = ZoomLevel(
    [TickSpec(MINUTE_SPACING, makeSStepper(MINUTE_SPACING), None, autoSkip=[1, 5, 15, 30])],
    "99:99",
)


class TimeDeltaAxisItem(DateAxisItem):
    """
    **Bases:** :class:`DateAxisItem <pyqtgraph.AxisItem>`

    An AxisItem that displays time-deltas provided in seconds.

    The display format is adjusted automatically depending on the current time
    density (seconds/point) on the axis. For more details on changing this
    behaviour, see :func:`setZoomLevelForDensity() <pyqtgraph.DateAxisItem.setZoomLevelForDensity>`.

    Can be added to an existing plot e.g. via
    :func:`setAxisItems({'bottom':axis}) <pyqtgraph.PlotItem.setAxisItems>`.

    """

    def __init__(self, orientation="bottom", utcOffset=None, **kwargs):
        """
        Create a new TimeDeltaAxisItem.

        For `orientation` and `**kwargs`, see
        :func:`AxisItem.__init__ <pyqtgraph.AxisItem.__init__>`.

        """
        super().__init__(orientation, utcOffset, **kwargs)

        # Set the zoom level to use depending on the time density on the axis
        self.zoomLevels = OrderedDict(
            [
                (np.inf, DAY_DT_ZOOM_LEVEL),  # days
                (24 * 3600, H_DT_ZOOM_LEVEL),  # HH:00 with hour-spacing
                (1800, HM_DT_ZOOM_LEVEL),  # HH:MM
                (100, HMS_ZOOM_LEVEL),  # HH:MM:SS
                (10, MS_ZOOM_LEVEL),  # SS.ms
            ]
        )
        self.autoSIPrefix = False

    def tickStrings(self, values, scale, spacing):
        tickSpecs = self.zoomLevel.tickSpecs
        tickSpec = next((s for s in tickSpecs if s.spacing == spacing), None)
        if tickSpec is None:
            return super(TimeDeltaAxisItem, self).tickStrings(values, scale, spacing)

        if tickSpec.spacing == DAY_SPACING:
            self.labelUnits = "day"
            self._updateLabel()
            return [_format_day_timedelta(value) for value in values]
        elif tickSpec.spacing == HOUR_SPACING or tickSpec.spacing == MINUTE_SPACING:
            self.labelUnits = "hour:minute"
            self._updateLabel()
            return [_format_hms_timedelta(value, show_seconds=False) for value in values]
        elif tickSpec.spacing == 1:
            self.labelUnits = "hour:minute:sec"
            self._updateLabel()
            return [_format_hms_timedelta(value, show_seconds=True) for value in values]

        # For sub-second precision, use the parent implementation
        else:
            self.labelUnits = "seconds"
            self._updateLabel()
            return super(TimeDeltaAxisItem, self).tickStrings(values, scale, spacing)
