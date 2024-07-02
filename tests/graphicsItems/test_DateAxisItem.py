from itertools import product
from unittest import mock

import pytest

import pyqtgraph as pg
from pyqtgraph.graphicsItems.DateAxisItem import (
    DAY_SPACING,
    HMS_ZOOM_LEVEL,
    HOUR_MINUTE_ZOOM_LEVEL,
    HOUR_SPACING,
    MS_ZOOM_LEVEL,
    SEC_PER_YEAR,
    calculateUtcOffset,
    getPreferredOffsetFromUtc,
    shiftLocalTimeToUtcTime,
)
from pyqtgraph.Qt.QtCore import QDate, QDateTime, QTime, QTimeZone
from pyqtgraph.Qt.QtGui import QFont, QFontMetrics

app = pg.mkQApp()


@pytest.fixture
def date_axis():
    axis = pg.DateAxisItem()
    axis.fontMetrics = QFontMetrics(QFont())
    yield axis


def assert_subarray(subarray, array):
    start = array.index(subarray[0])
    assert array[start:start+len(subarray)] == subarray


def test_preferred_utc_offset_respects_chosen_offset():
    assert getPreferredOffsetFromUtc(0, 7200) == 7200
    assert getPreferredOffsetFromUtc(0, -7200) == -7200


def test_preferred_utc_offset_doesnt_break_with_big_timestamps():
    timestamp = SEC_PER_YEAR ** 13

    assert -16 * 3600 <= getPreferredOffsetFromUtc(timestamp) <= 16 * 3600
    assert getPreferredOffsetFromUtc(timestamp, 3600) == 3600

    assert -16 * 3600 <= getPreferredOffsetFromUtc(-timestamp) <= 16 * 3600
    assert getPreferredOffsetFromUtc(-timestamp, -1800) == -1800


def test_utc_offset_works_with_float_timestamp():
    assert -16 * 3600 <= calculateUtcOffset(123456.0734) <= 16 * 3600


def test_shift_local_time_to_utc_time_does_what_it_promises_to_do():
    timeZone = QTimeZone(b"UTC+4")

    def fromSecsSinceEpochLocal(timestamp):
        return QDateTime.fromMSecsSinceEpoch(timestamp * 1000).toTimeZone(timeZone)

    startDate = QDateTime(QDate(1970, 1, 2), QTime(2, 0), timeZone)
    goalDate = QDateTime(QDate(1970, 1, 1), QTime(22, 0), timeZone)
    assert (
        startDate.toUTC().time() == goalDate.time()
        and startDate.toUTC().date() == goalDate.date()
    )

    fromEpochSecs = "pyqtgraph.graphicsItems.DateAxisItem.QDateTime.fromSecsSinceEpoch"
    with mock.patch(fromEpochSecs, fromSecsSinceEpochLocal):
        shifted = shiftLocalTimeToUtcTime(startDate.toSecsSinceEpoch())
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
        # Qt wants to go backwards at 00:01 to 23:01,
        # but according to
        # https://www.timeanddate.com/time/change/canada/st-johns?year=1986
        # it should be from 02:00 to 01:00. Maybe Qt bug, maybe website is wrong
        # maybe those are different time zones and everything is fine
        (
            QTimeZone(b"America/St_Johns"),
            QDate(1986, 10, 26),
            ["Sun 26"],
            ["00:00", "01:00", "02:00", "03:00", "04:00"],
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
        for zone, direction
        in product(
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
    date_axis,
):

    def fromSecsSinceEpochLocal(timestamp):
        return QDateTime.fromMSecsSinceEpoch(timestamp * 1000).toTimeZone(timeZone)

    minTime = QDateTime(transitionDate, QTime(0, 0, 0, 0), timeZone).toSecsSinceEpoch()
    maxTime = QDateTime(transitionDate, QTime(4, 0, 0, 0), timeZone).toSecsSinceEpoch()

    fromEpochSecs = "pyqtgraph.graphicsItems.DateAxisItem.QDateTime.fromSecsSinceEpoch"
    with mock.patch(fromEpochSecs, fromSecsSinceEpochLocal):
        xvals = [x for x in range(minTime, maxTime + 3600, 3600)]
        lengthInPixels = 600
        tickValues = date_axis.tickValues(xvals[0] - 1, xvals[-1] + 1, lengthInPixels)
        for spacing, ticks in tickValues:
            if spacing == DAY_SPACING:
                tickStrings = date_axis.tickStrings(ticks, 1, DAY_SPACING)
                assert_subarray(expectedDayTickStrings, tickStrings)
            elif spacing == HOUR_SPACING:
                tickStrings = date_axis.tickStrings(ticks, 1, spacing)
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
def test_maps_hour_ticks_to_local_times_when_skip_greater_than_one(timeZone, date_axis):

    def fromSecsSinceEpochLocal(timestamp):
        return QDateTime.fromMSecsSinceEpoch(timestamp * 1000).toTimeZone(timeZone)

    date = QDate(2023, 5, 10)
    minTime = QDateTime(date, QTime(0, 0, 0, 0), timeZone).toSecsSinceEpoch()
    maxTime = QDateTime(date, QTime(18, 0, 0, 0), timeZone).toSecsSinceEpoch()

    fromEpochSecs = "pyqtgraph.graphicsItems.DateAxisItem.QDateTime.fromSecsSinceEpoch"
    with mock.patch(fromEpochSecs, fromSecsSinceEpochLocal):
        xvals = [x for x in range(minTime, maxTime + 3600, 3600)]

        lengthInPixels = 200
        tickValues = date_axis.tickValues(xvals[0] - 1, xvals[-1] + 1, lengthInPixels)
        for spacing, ticks in tickValues:
            if spacing == HOUR_SPACING:
                tickStrings = date_axis.tickStrings(ticks, 1, spacing)
                assert_subarray(["06:00", "12:00", "18:00"], tickStrings)


@pytest.mark.parametrize(
    ("autoSkip", "expectedHourTickStrings"),
    (
        (1, ["01:00", "02:00", "03:00", "04:00", "05:00"]),
        (6, ["06:00", "12:00", "18:00"]),
    ),
)
def test_custom_utc_offset_works(autoSkip, expectedHourTickStrings, date_axis):
    size_px = 600 if autoSkip == 1 else 200
    maxHour = 4 if autoSkip == 1 else 18

    utcZone = QTimeZone(b"UTC")
    date = QDate(2001, 1, 1)
    minTime = QDateTime(date, QTime(0, 0, 0, 0), utcZone).toSecsSinceEpoch()
    maxTime = QDateTime(date, QTime(maxHour, 0, 0, 0), utcZone).toSecsSinceEpoch()

    xvals = [x for x in range(minTime, maxTime + 3600, 3600)]

    date_axis.utcOffset = -3600
    for spacing, ticks in date_axis.tickValues(xvals[0] - 1, xvals[-1] + 1, size_px):
        if spacing == DAY_SPACING:
            tickStrings = date_axis.tickStrings(ticks, 1, DAY_SPACING)
            assert_subarray(["Mon 01"], tickStrings)
        elif spacing == HOUR_SPACING:
            tickStrings = date_axis.tickStrings(ticks, 1, spacing)
            assert_subarray(expectedHourTickStrings, tickStrings)


def test_time_range_is_not_extended_for_minutes_and_ms():
    utcZone = QTimeZone(b"UTC")
    date = QDate(2001, 1, 1)
    minTime = QDateTime(date, QTime(0, 0, 0, 0), utcZone).toSecsSinceEpoch()
    maxTime = QDateTime(date, QTime(18, 0, 0, 0), utcZone).toSecsSinceEpoch()

    for zoom in (MS_ZOOM_LEVEL, HOUR_MINUTE_ZOOM_LEVEL, HMS_ZOOM_LEVEL):
        for spec in zoom.tickSpecs:
            if spec.spacing >= HOUR_SPACING:
                continue
            newRange = zoom.extendTimeRangeForSpec(spec, minTime, maxTime)
            assert newRange == (minTime, maxTime)


def test_time_range_is_extended_for_hour():
    utcZone = QTimeZone(b"UTC")
    date = QDate(2001, 1, 1)
    minTime = QDateTime(date, QTime(0, 0, 0, 0), utcZone).toSecsSinceEpoch()
    maxTime = QDateTime(date, QTime(18, 0, 0, 0), utcZone).toSecsSinceEpoch()

    for spec in HOUR_MINUTE_ZOOM_LEVEL.tickSpecs:
        if spec.spacing < HOUR_SPACING:
            continue
        newRange = HOUR_MINUTE_ZOOM_LEVEL.extendTimeRangeForSpec(spec, minTime, maxTime)
        extMin, extMax = newRange
        assert extMin < minTime and extMax > maxTime
