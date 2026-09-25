import numpy as np
import pytest

import pyqtgraph as pg

pg.mkQApp()


def test_log_mode_maps_bar_edges():
    bar = pg.BarGraphItem(
        x=[1, 2, 3],
        y0=[0.1, 0.1, 0.1],
        y1=[1, 10, 100],
        width=1,
    )

    assert bar.dataBounds(1) == (0.1, 100)

    bar.setLogMode(False, True)

    np.testing.assert_allclose(bar.dataBounds(1), (-1, 2))
    rects = bar._rectarray.ndarray()
    np.testing.assert_allclose(rects[:, 1], [-1, -1, -1])
    np.testing.assert_allclose(rects[:, 3], [1, 2, 3])


@pytest.mark.qt_log_ignore("Populating font family aliases took .*")
def test_plotitem_applies_log_mode_to_bar_graph_item():
    plot = pg.PlotItem()
    plot.setLogMode(y=True)
    bar = pg.BarGraphItem(x=[1], y0=[1], y1=[10], width=1)

    plot.addItem(bar)

    assert bar.dataBounds(1) == (0, 1)


def test_log_mode_skips_bars_with_non_positive_edges():
    bar = pg.BarGraphItem(
        x=[1, 2],
        y0=[0, 1],
        y1=[10, 100],
        width=1,
        brushes=["r", "g"],
    )

    bar.setLogMode(False, True)

    assert bar._rectIndices.tolist() == [1]
    np.testing.assert_allclose(bar.dataBounds(1), (0, 2))
