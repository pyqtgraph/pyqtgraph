import numpy as np

import pyqtgraph as pg

app = pg.mkQApp()


def test_ErrorBarItem_defer_data():
    plot = pg.PlotWidget()
    plot.show()

    # plot some data away from the origin to set the view rect
    x = np.arange(5) + 10
    curve = pg.PlotCurveItem(x=x, y=x)
    plot.addItem(curve)
    app.processEvents()
    app.processEvents()
    r_no_ebi = plot.viewRect()

    # ErrorBarItem with no data shouldn't affect the view rect
    err = pg.ErrorBarItem()
    plot.addItem(err)
    app.processEvents()
    app.processEvents()
    r_empty_ebi = plot.viewRect()

    assert r_no_ebi.height() == r_empty_ebi.height()

    err.setData(x=x, y=x, bottom=x, top=x)
    app.processEvents()
    app.processEvents()
    r_ebi = plot.viewRect()

    assert r_ebi.height() > r_empty_ebi.height()

    # unset data, ErrorBarItem disappears and view rect goes back to original
    err.setData(x=None, y=None)
    app.processEvents()
    app.processEvents()
    r_clear_ebi = plot.viewRect()

    assert r_clear_ebi.height() == r_empty_ebi.height()

    plot.close()


def test_ErrorBarItem_limit_arrows_render():
    """Limits on each side should render without errors and not affect viewRect height."""
    plot = pg.PlotWidget()
    plot.show()

    x = np.arange(5, dtype=float)
    y = np.zeros(5)
    top = np.ones(5)
    bottom = np.ones(5)
    left = np.ones(5)
    right = np.ones(5)

    topLimit = np.array([True, False, False, False, False])
    bottomLimit = np.array([False, True, False, False, False])
    leftLimit = np.array([False, False, True, False, False])
    rightLimit = np.array([False, False, False, True, False])

    err = pg.ErrorBarItem(
        x=x, y=y, top=top, bottom=bottom, left=left, right=right, beam=0.2,
        topLimit=topLimit, bottomLimit=bottomLimit,
        leftLimit=leftLimit, rightLimit=rightLimit,
    )
    plot.addItem(err)
    app.processEvents()
    app.processEvents()

    # The error bars themselves still contribute to the view rect.
    assert plot.viewRect().height() > 0
    plot.close()


def test_ErrorBarItem_limit_mask_scalar_and_array():
    """Scalar True/False and per-point arrays both work for limit kwargs."""
    err = pg.ErrorBarItem()
    x = np.arange(3, dtype=float)
    y = np.zeros(3)

    # Scalar True -> all points are limits
    err.setData(x=x, y=y, top=np.ones(3), beam=0.1, topLimit=True)
    mask = err._limitMask('topLimit', 3)
    assert mask is not None and mask.all()

    # Scalar False -> no limits (treated as None)
    err.setData(topLimit=False)
    assert err._limitMask('topLimit', 3) is None

    # All-False array -> no limits
    err.setData(topLimit=np.zeros(3, dtype=bool))
    assert err._limitMask('topLimit', 3) is None

    # Mixed array -> mask returned as-is
    err.setData(topLimit=np.array([True, False, True]))
    mask = err._limitMask('topLimit', 3)
    assert mask is not None
    assert mask.tolist() == [True, False, True]


def test_ErrorBarItem_no_limits_backwards_compat():
    """Pre-existing usage (no limit kwargs) should be unchanged."""
    plot = pg.PlotWidget()
    plot.show()
    x = np.arange(5, dtype=float)
    err = pg.ErrorBarItem(x=x, y=x, top=np.ones(5), bottom=np.ones(5), beam=0.5)
    plot.addItem(err)
    app.processEvents()
    app.processEvents()
    # drawPath populates self.path with no exception
    assert err.path is not None
    plot.close()
