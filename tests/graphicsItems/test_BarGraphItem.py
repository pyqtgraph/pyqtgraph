import numpy as np

import pyqtgraph as pg

app = pg.mkQApp()


def test_logY():
    plot = pg.PlotWidget()
    y1 = np.arange(10, 101, 10, dtype=np.float64)
    y0 = np.ones(10)
    y0[0] = 0
    x = np.arange(1, 11)
    width = 0.5
    bar = pg.BarGraphItem(x=x, width=width, y0=y0, y1=y1)
    plot.addItem(bar)
    normalized_x0, normalized_y0, normalized_x1, normalized_y1 = (
        bar._getNormalizedCoords())

    expected_x0 = x - (width/2)
    expected_x1 = x + (width/2)
    assert np.array_equal(expected_x0, normalized_x0)
    assert np.array_equal(expected_x1, normalized_x1)

    assert np.array_equal(y0, normalized_y0)
    assert np.array_equal(y1, normalized_y1)

    # With LogMode on Y.
    bar.setLogMode(x=False, y=True)

    normalized_x0, normalized_y0, normalized_x1, normalized_y1 = (
        bar._getNormalizedCoords())

    assert np.array_equal(expected_x0, normalized_x0)
    assert np.array_equal(expected_x1, normalized_x1)

    expected_y0 = np.zeros(10)
    expected_y0[0] = np.nan
    expected_y1 = np.log10(y1)
    assert np.array_equal(expected_y0, normalized_y0, equal_nan=True)
    assert np.array_equal(expected_y1, normalized_y1)
