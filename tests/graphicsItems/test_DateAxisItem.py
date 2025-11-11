import locale
from contextlib import contextmanager
from functools import lru_cache
from itertools import product
from unittest import mock
import numpy as np

import pytest

import pyqtgraph as pg
from pyqtgraph.graphicsItems.DateAxisItem import (
    DAY_HOUR_ZOOM_LEVEL,
    DAY_SPACING,
    HMS_ZOOM_LEVEL,
    HOUR_MINUTE_ZOOM_LEVEL,
    HOUR_SPACING,
    MINUTE_SPACING,
    MONTH_SPACING,
    MS_SPACING,
    MS_ZOOM_LEVEL,
    SEC_PER_YEAR,
    SECOND_SPACING,
    WEEK_SPACING,
    YEAR_MONTH_ZOOM_LEVEL,
    YEAR_SPACING,
    ZoomLevel,
    TickSpec,
    applyOffsetFromUtc,
    calculateUtcOffset,
    getPreferredOffsetFromUtc,
)
from pyqtgraph.Qt.QtCore import QDate, QDateTime, QTime, QTimeZone
from pyqtgraph.Qt.QtGui import QFont, QFontMetrics

app = pg.mkQApp()


def makeDateAxis():
    axis = pg.DateAxisItem()
    axis.fontMetrics = QFontMetrics(QFont())
    axis.zoomLevel = YEAR_MONTH_ZOOM_LEVEL
    return axis


@lru_cache
def densityForZoomLevel(level):
    axis = makeDateAxis()
    density = 3600
    while axis.zoomLevel != level and density > 1:
        axis.setZoomLevelForDensity(density)
        density -= 1
    return density


def getViewLengthInPxForZoomLevel(level, valuesRange):
    return valuesRange / densityForZoomLevel(level)


def assert_subarray(subarray, array):
    start = array.index(subarray[0])
    assert array[start : start + len(subarray)] == subarray


@contextmanager
def inTimezone(timezone):

    def fromSecsSinceEpochLocal(timestamp):
        return QDateTime.fromMSecsSinceEpoch(timestamp * 1000).toTimeZone(timezone)

    with mock.patch.object(QDateTime, "fromSecsSinceEpoch", fromSecsSinceEpochLocal):
        yield


@pytest.fixture(autouse=True)
def reset_zoom_levels_utc_offsets():
    for level in (
        DAY_HOUR_ZOOM_LEVEL,
        HOUR_MINUTE_ZOOM_LEVEL,
        HMS_ZOOM_LEVEL,
        MS_ZOOM_LEVEL,
    ):
        level.utcOffset = None


@pytest.fixture
def dateAxis():
    return makeDateAxis()


@pytest.fixture(autouse=True)
def use_c_locale():
    locale.setlocale(locale.LC_TIME, "C")


def test_preferred_utc_offset_respects_chosen_offset():
    assert getPreferredOffsetFromUtc(0, 7200) == 7200
    assert getPreferredOffsetFromUtc(0, -7200) == -7200


@pytest.mark.qt_no_exception_capture
def test_preferred_utc_offset_doesnt_break_with_big_timestamps():
    timestamp = SEC_PER_YEAR**13

    assert -16 * 3600 <= getPreferredOffsetFromUtc(timestamp) <= 16 * 3600
    assert getPreferredOffsetFromUtc(timestamp, 3600) == 3600

    assert -16 * 3600 <= getPreferredOffsetFromUtc(-timestamp) <= 16 * 3600
    assert getPreferredOffsetFromUtc(-timestamp, -1800) == -1800


def test_utc_offset_works_with_float_timestamp():
    assert -16 * 3600 <= calculateUtcOffset(123456.0734) <= 16 * 3600


def test_applyOffsetFromUtc_does_what_it_promises_to_do():
    timeZone = QTimeZone(b"UTC+4")

    startDate = QDateTime(QDate(1970, 1, 2), QTime(2, 0), timeZone)
    goalDate = QDateTime(QDate(1970, 1, 1), QTime(22, 0), timeZone)
    assert (
        startDate.toUTC().time() == goalDate.time()
        and startDate.toUTC().date() == goalDate.date()
    )

    with inTimezone(timeZone):
        shifted = applyOffsetFromUtc(startDate.toSecsSinceEpoch())

    assert shifted == goalDate.toSecsSinceEpoch()


@pytest.mark.parametrize(
    ("timeZone", "transitionDate", "expectedDayTickStrings", "expectedHourTickStrings"),
    (
        (
            QTimeZone(b"Europe/Berlin"),
            QDate(2022, 10, 30),
            ["Sun 30"],
            ["01:00", "02:00", "02:00", "03:00", "04:00"],
        ),
        (
            QTimeZone(b"Europe/Berlin"),
            QDate(2023, 3, 26),
            ["Sun 26"],
            ["01:00", "03:00", "04:00", "05:00", "06:00"],
        ),
        (
            QTimeZone(b"Pacific/Chatham"),
            QDate(2024, 4, 7),
            ["Sun 07"],
            ["01:00", "02:00", "03:00", "03:00", "04:00"],
        ),
        (
            QTimeZone(b"Pacific/Chatham"),
            QDate(2022, 9, 25),
            ["Sun 25"],
            ["01:00", "02:00", "04:00", "05:00", "06:00"],
        ),
        (
            QTimeZone(b"America/St_Johns"),
            QDate(2012, 11, 4),
            ["Sun 04"],
            ["01:00", "01:00", "02:00", "03:00", "04:00"],
        ),
        (
            QTimeZone(b"America/St_Johns"),
            QDate(1995, 4, 2),
            ["Sun 02"],
            ["02:00", "03:00", "04:00", "05:00", "06:00"],
        ),
        (
            QTimeZone(b"Australia/Lord_Howe"),
            QDate(2007, 3, 25),
            ["Sun 25"],
            ["01:00", "02:00", "03:00", "04:00", "05:00"],
        ),
        (
            QTimeZone(b"Australia/Lord_Howe"),
            QDate(2010, 10, 3),
            ["Sun 03"],
            ["01:00", "03:00", "04:00", "05:00", "06:00"],
        ),
    ),
    ids=(
        f"{zone}-{direction}"
        for zone, direction in product(
            ("Berlin", "Chatham", "St_Johns", "Lord_Howe"),
            ("backward", "forward"),
        )
    ),
)
def test_maps_tick_values_to_local_times(
    timeZone,
    transitionDate,
    expectedDayTickStrings,
    expectedHourTickStrings,
    dateAxis,
):
    minTime = QDateTime(transitionDate, QTime(0, 0, 0, 0), timeZone).toSecsSinceEpoch()
    maxTime = QDateTime(transitionDate, QTime(4, 0, 0, 0), timeZone).toSecsSinceEpoch()

    xvals = list(range(minTime, maxTime + 3600, 3600))
    timeRange = maxTime - minTime
    lengthInPixels = getViewLengthInPxForZoomLevel(HOUR_MINUTE_ZOOM_LEVEL, timeRange)

    with inTimezone(timeZone):
        tickValues = dateAxis.tickValues(xvals[0] - 1, xvals[-1] + 1, lengthInPixels)
        for spacing, ticks in tickValues:
            if spacing == DAY_SPACING:
                tickStrings = dateAxis.tickStrings(ticks, 1, DAY_SPACING)
                assert_subarray(expectedDayTickStrings, tickStrings)
            elif spacing == HOUR_SPACING:
                tickStrings = dateAxis.tickStrings(ticks, 1, spacing)
                assert_subarray(expectedHourTickStrings, tickStrings)


@pytest.mark.parametrize(
    ("timeZone"),
    (
        QTimeZone(b"Europe/Berlin"),
        QTimeZone(b"Pacific/Chatham"),
        QTimeZone(b"America/St_Johns"),
        QTimeZone(b"Australia/Lord_Howe"),
    ),
    ids=("Berlin", "Chatham", "St_Johns", "Lord_Howe"),
)
def test_maps_hour_ticks_to_local_times_when_skip_greater_than_one(timeZone, dateAxis):
    date = QDate(2023, 5, 10)
    minTime = QDateTime(date, QTime(0, 0, 0, 0), timeZone).toSecsSinceEpoch()
    maxTime = QDateTime(date, QTime(18, 0, 0, 0), timeZone).toSecsSinceEpoch()

    xvals = list(range(minTime, maxTime + 3600, 3600))
    timeRange = maxTime - minTime
    lengthInPixels = getViewLengthInPxForZoomLevel(DAY_HOUR_ZOOM_LEVEL, timeRange)

    with inTimezone(timeZone):
        tickValues = dateAxis.tickValues(xvals[0] - 1, xvals[-1] + 1, lengthInPixels)
        for spacing, ticks in tickValues:
            if spacing == HOUR_SPACING:
                tickStrings = dateAxis.tickStrings(ticks, 1, spacing)
                assert_subarray(["06:00", "12:00", "18:00"], tickStrings)


@pytest.mark.parametrize(
    ("zoomLevel", "expectedHourTickStrings"),
    (
        (HOUR_MINUTE_ZOOM_LEVEL, ["01:00", "02:00", "03:00", "04:00", "05:00"]),
        (DAY_HOUR_ZOOM_LEVEL, ["06:00", "12:00", "18:00"]),
    ),
)
def test_custom_utc_offset_works(zoomLevel, expectedHourTickStrings, dateAxis):
    maxHour = 4 if zoomLevel == HOUR_MINUTE_ZOOM_LEVEL else 18

    utcZone = QTimeZone(b"UTC")
    date = QDate(2001, 1, 1)
    minTime = QDateTime(date, QTime(0, 0, 0, 0), utcZone).toSecsSinceEpoch()
    maxTime = QDateTime(date, QTime(maxHour, 0, 0, 0), utcZone).toSecsSinceEpoch()

    size_px = getViewLengthInPxForZoomLevel(zoomLevel, maxTime - minTime)
    xvals = list(range(minTime, maxTime + 3600, 3600))
    dateAxis.utcOffset = -3600

    for spacing, ticks in dateAxis.tickValues(xvals[0] - 1, xvals[-1] + 1, size_px):
        if spacing == DAY_SPACING:
            tickStrings = dateAxis.tickStrings(ticks, 1, DAY_SPACING)
            assert_subarray(["Mon 01"], tickStrings)
        elif spacing == HOUR_SPACING:
            tickStrings = dateAxis.tickStrings(ticks, 1, spacing)
            assert_subarray(expectedHourTickStrings, tickStrings)


@pytest.mark.parametrize(
    ("localZone", "spacing", "expectedExtentionInHours"),
    (
        (QTimeZone(b"UTC+7"), MS_SPACING, 0),
        (QTimeZone(b"UTC+5"), SECOND_SPACING, 0),
        (QTimeZone(b"UTC+3"), MINUTE_SPACING, 0),
        (QTimeZone(b"UTC+7"), HOUR_SPACING, 7),
        (QTimeZone(b"UTC+6"), DAY_SPACING, 6),
        (QTimeZone(b"UTC+5"), WEEK_SPACING, 5),
        (QTimeZone(b"UTC+4"), MONTH_SPACING, 4),
        (QTimeZone(b"UTC+3"), YEAR_SPACING, 3),
        (QTimeZone(b"UTC"), MS_SPACING, 0),
        (QTimeZone(b"UTC"), SECOND_SPACING, 0),
        (QTimeZone(b"UTC"), MINUTE_SPACING, 0),
        (QTimeZone(b"UTC"), HOUR_SPACING, 0),
        (QTimeZone(b"UTC"), DAY_SPACING, 0),
        (QTimeZone(b"UTC"), WEEK_SPACING, 0),
        (QTimeZone(b"UTC"), MONTH_SPACING, 0),
        (QTimeZone(b"UTC"), YEAR_SPACING, 0),
    ),
)
def test_extendTimeRangeForSpacing_repsects_utc_offset(
    localZone,
    spacing,
    expectedExtentionInHours,
):
    utcZone = QTimeZone(b"UTC")
    date = QDate(2001, 1, 1)
    minTime = QDateTime(date, QTime(0, 0, 0, 0), utcZone).toSecsSinceEpoch()
    maxTime = QDateTime(date, QTime(18, 0, 0, 0), utcZone).toSecsSinceEpoch()

    zoom = ZoomLevel([], "")

    with inTimezone(localZone):
        extMin, extMax = zoom.extendTimeRangeForSpacing(spacing, minTime, maxTime)
    assert extMax - maxTime == expectedExtentionInHours * 3600
    assert minTime - extMin == expectedExtentionInHours * 3600


@pytest.mark.parametrize(
    ("autoskip", "minspc", "expectedSkipFactor"),
    (
        (np.array([1.0, 4.0, 8.0]), 3, 4),
        (np.array([1, 4, 8]), 16, 8),
        (np.array([1, 4, 8]), 17, 10),
        (np.array([1, 4, 8]), 501, 400),
        (np.array([1, 4, 8]), 0.01, 1),
    ),
)
def test_skipFactor(autoskip, minspc, expectedSkipFactor):
    tickSpec = TickSpec(2.0, None, None, autoskip)
    result = tickSpec.skipFactor(minspc)

    assert result == expectedSkipFactor
