import numpy as np
import pytest
from pytestqt.qtbot import QtBot

import pyqtgraph as pg
from pyqtgraph.graphicsItems.DateAxisItem import (
    DAY_SPACING,
    HOUR_SPACING,
    MINUTE_SPACING,
    TimeDeltaAxisItem,
    _format_day_timedelta,
    _format_hms_timedelta,
)

app = pg.mkQApp()


def test_initialization():
    """Test that TimeDeltaAxisItem initializes correctly."""
    axis = TimeDeltaAxisItem()
    assert axis is not None
    assert axis.orientation == "bottom"
    assert len(axis.zoomLevels) == 5  # Verify all zoom levels are present


def test_format_hms_timedelta_hours_minutes():
    """Test the _format_hms_timedelta function with hours:minutes format."""
    assert _format_hms_timedelta(3600) == "01:00"
    assert _format_hms_timedelta(3660) == "01:01"
    assert _format_hms_timedelta(86400) == "24:00"
    assert _format_hms_timedelta(90000) == "25:00"


def test_format_hms_timedelta_with_seconds():
    """Test the _format_hms_timedelta function with hours:minutes:seconds format."""
    assert _format_hms_timedelta(3600, show_seconds=True) == "01:00:00"
    assert _format_hms_timedelta(3661, show_seconds=True) == "01:01:01"
    assert _format_hms_timedelta(90061, show_seconds=True) == "25:01:01"


def test_format_hms_timedelta_fractional_seconds():
    """Test handling of fractional seconds (should trigger warning)."""
    with pytest.warns(UserWarning, match="Truncating milliseconds"):
        result = _format_hms_timedelta(3600.5)
        assert result == "01:00"


def test_format_hms_timedelta_with_seconds_warning():
    """Test warning when seconds are present in hours:minutes format."""
    with pytest.warns(UserWarning, match="Truncating seconds"):
        result = _format_hms_timedelta(3661, show_seconds=False)
        assert result == "01:01"


def test_format_day_timedelta_exact():
    """Test _format_day_timedelta with exact day values."""
    assert _format_day_timedelta(86400) == "1 d"
    assert _format_day_timedelta(172800) == "2 d"
    assert _format_day_timedelta(864000) == "10 d"


def test_format_day_timedelta_fractional():
    """Test _format_day_timedelta with fractional days (should trigger warning)."""
    with pytest.warns(UserWarning, match="Truncating seconds"):
        result = _format_day_timedelta(86400 + 3600)  # 1 day and 1 hour
        assert result == "1 d"


def test_tick_strings_days():
    """Test tickStrings method with day-level spacing."""
    axis = TimeDeltaAxisItem()
    # Mock the zoom level to day level
    axis.zoomLevel = axis.zoomLevels[np.inf]

    values = [86400, 172800, 259200]  # 1, 2, 3 days
    result = axis.tickStrings(values, 1, DAY_SPACING)

    assert result == ["1 d", "2 d", "3 d"]
    assert axis.labelUnits == "day"


def test_tick_strings_hours():
    """Test tickStrings method with hour-level spacing."""
    axis = TimeDeltaAxisItem()
    # Mock the zoom level to hour level
    axis.zoomLevel = axis.zoomLevels[24 * 3600]

    values = [3600, 7200, 10800]  # 1, 2, 3 hours
    result = axis.tickStrings(values, 1, HOUR_SPACING)

    assert result == ["01:00", "02:00", "03:00"]
    assert axis.labelUnits == "hour:minute"


def test_tick_strings_minutes():
    """Test tickStrings method with minute-level spacing."""
    axis = TimeDeltaAxisItem()
    # Mock the zoom level to minute level
    axis.zoomLevel = axis.zoomLevels[1800]

    values = [60, 120, 180]  # 1, 2, 3 minutes
    result = axis.tickStrings(values, 1, MINUTE_SPACING)

    assert result == ["00:01", "00:02", "00:03"]
    assert axis.labelUnits == "hour:minute"


def test_tick_strings_seconds():
    """Test tickStrings method with second-level spacing."""
    axis = TimeDeltaAxisItem()
    # Mock the zoom level to second level
    axis.zoomLevel = axis.zoomLevels[100]

    values = [1, 2, 3]  # 1, 2, 3 seconds
    result = axis.tickStrings(values, 1, 1)  # Spacing of 1 second

    assert result == ["00:00:01", "00:00:02", "00:00:03"]
    assert axis.labelUnits == "hour:minute:sec"


def test_large_time_values():
    """Test handling of very large time values."""
    # Test with large hour values (>99 hours)
    large_time = 3600 * 100  # 100 hours
    assert _format_hms_timedelta(large_time) == "100:00"

    # Test with extremely large values
    very_large_time = 3600 * 1000  # 1000 hours
    assert _format_hms_timedelta(very_large_time) == "1000:00"


@pytest.mark.parametrize(
    "timestamp,expected",
    [
        (0, "00:00"),
        (60, "00:01"),
        (3600, "01:00"),
        (3660, "01:01"),
        (86400, "24:00"),
        (172800, "48:00"),
    ],
)
def test_format_hms_timedelta_parametrized(timestamp, expected):
    """Parametrized test for _format_hms_timedelta."""
    assert _format_hms_timedelta(timestamp) == expected


def test_smoke_integration_with_plot(qtbot: QtBot):
    """Smoke test that the axis can be added to a plot."""
    # Create a plot with our custom axis
    plt = pg.PlotWidget()
    time_axis = TimeDeltaAxisItem(orientation="bottom")
    plt.setAxisItems({"bottom": time_axis})

    # Add some data
    x = np.arange(0, 24 * 3600, 3600)  # 24 hours in seconds
    y = np.sin(x * 2 * np.pi / (24 * 3600))  # One period over 24 hours
    assert not time_axis.labelUnits
    plt.plot(x, y)

    # Make sure the plot auto-scales to show all data
    plt.autoRange()

    # Register with qtbot and show the plot
    qtbot.addWidget(plt)
    plt.show()

    # Force the plot to do a full layout update
    plt.resize(800, 600)

    # wait for PlotItem & AxisItem to be drawn (maybe there is a better way?)
    qtbot.waitForWindowShown(plt)

    # Ensure that AxisItem has been updated, units are correct,
    # and everything runs without errors.
    assert time_axis.labelUnits == "hour:minute"
    assert plt.plotItem.getAxis("bottom") is time_axis

    plt.close()
