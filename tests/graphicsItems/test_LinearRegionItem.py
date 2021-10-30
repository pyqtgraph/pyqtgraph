import math

import numpy as np
import pytest

import pyqtgraph as pg

app = pg.mkQApp()


def check_region(lr, bounds, exact=False, rtol=0.005):
    """Optionally tolerant LinearRegionItem region check"""
    reg = lr.getRegion()
    if exact:
        assert reg[0] == bounds[0]
        assert reg[1] == bounds[1]
    else:
        assert math.isclose(reg[0], bounds[0], rel_tol=rtol)
        assert math.isclose(reg[1], bounds[1], rel_tol=rtol)


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_clip_to_plot_data_item(orientation):
    """Vertical and horizontal LRIs clipping both bounds to a PlotDataItem"""
    # initial bounds for the LRI
    init_vals = (-1.5, 1.5)

    # data for a PlotDataItem to clip to, both inside the inial bounds
    x = np.linspace(-1, 1, 10)
    y = np.linspace(1, 1.2, 10)

    p = pg.PlotWidget()
    pdi = p.plot(x=x, y=y)

    lr = pg.LinearRegionItem(init_vals, clipItem=pdi, orientation=orientation)
    p.addItem(lr)

    app.processEvents()

    if orientation == "vertical":
        check_region(lr, x[[0, -1]])
    else:
        check_region(lr, y[[0, -1]])


def test_disable_clip_item():
    """LRI clipItem (ImageItem) disabled by explicit call to setBounds"""
    init_vals = (5, 40)

    p = pg.PlotWidget()
    img = pg.ImageItem(image=np.eye(20, 20))
    p.addItem(img)

    lr = pg.LinearRegionItem(init_vals, clipItem=img)
    p.addItem(lr)

    app.processEvents()

    # initial bound that was out of range snaps to the imageitem bound
    check_region(lr, (init_vals[0], img.height()), exact=True)

    # disable clipItem, move the right InfiniteLine beyond the new bound
    lr.setBounds(init_vals)
    lr.lines[1].setPos(init_vals[1] + 10)

    app.processEvents()

    check_region(lr, init_vals, exact=True)


def test_clip_to_item_in_other_vb():
    """LRI clip to item in a different ViewBox"""
    init_vals = (10, 50)
    img_shape = (20, 20)

    win = pg.GraphicsLayoutWidget()

    # "remote" PlotDataItem to bound to
    p1 = win.addPlot()
    img = pg.ImageItem(image=np.eye(*img_shape))
    p1.addItem(img)

    # plot the LRI lives in
    p2 = win.addPlot()
    x2 = np.linspace(-200, 200, 100)
    p2.plot(x=x2, y=x2)

    lr = pg.LinearRegionItem(init_vals)
    p2.addItem(lr)
    app.processEvents()

    # no clip item set yet, should be initial bounds
    check_region(lr, init_vals, exact=True)

    lr.setClipItem(img)
    app.processEvents()

    # snap to image width
    check_region(lr, (init_vals[0], img_shape[1]), exact=True)


def test_clip_item_override_init_bounds():
    """clipItem overrides bounds provided in the constructor"""
    init_vals = (-10, 10)
    init_bounds = (-5, 5)
    img_shape = (5, 5)

    p = pg.PlotWidget()
    img = pg.ImageItem(image=np.eye(*img_shape))
    p.addItem(img)

    lr = pg.LinearRegionItem(init_vals, clipItem=img, bounds=init_bounds)
    p.addItem(lr)

    app.processEvents()
    check_region(lr, (0, img_shape[1]), exact=True)
