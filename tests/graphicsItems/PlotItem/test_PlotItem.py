import warnings

import numpy as np
import pytest

import pyqtgraph as pg

app = pg.mkQApp()


@pytest.mark.parametrize('orientation', ['left', 'right', 'top', 'bottom'])
def test_PlotItem_shared_axis_items(orientation):
    """Adding an AxisItem to multiple plots raises RuntimeError"""
    ax1 = pg.AxisItem(orientation)
    ax2 = pg.AxisItem(orientation)

    layout = pg.GraphicsLayoutWidget()

    _ = layout.addPlot(axisItems={orientation: ax1})

    pi2 = layout.addPlot()
    # left or bottom replaces, right or top adds new
    pi2.setAxisItems({orientation: ax2})

    with pytest.raises(RuntimeError):
        pi2.setAxisItems({orientation: ax1})


def test_PlotItem_maxTraces():
    item = pg.PlotItem()

    curve1 = pg.PlotDataItem(np.random.normal(size=10))
    item.addItem(curve1)
    assert curve1.isVisible(), "curve1 should be visible"

    item.ctrl.maxTracesCheck.setChecked(True)
    item.ctrl.maxTracesSpin.setValue(0)
    assert not curve1.isVisible(), "curve1 should not be visible"

    item.ctrl.maxTracesCheck.setChecked(False)
    assert curve1.isVisible(), "curve1 should be visible"

    curve2 = pg.PlotDataItem(np.random.normal(size=10))
    item.addItem(curve2)
    assert curve2.isVisible(), "curve2 should be visible"

    item.ctrl.maxTracesCheck.setChecked(True)
    item.ctrl.maxTracesSpin.setValue(1)
    assert curve2.isVisible(), "curve2 should be visible"
    assert not curve1.isVisible(), "curve1 should not be visible"
    assert curve1 in item.curves, "curve1 should be in the item's curves"

    item.ctrl.forgetTracesCheck.setChecked(True)
    assert curve2 in item.curves, "curve2 should be in the item's curves"
    assert curve1 not in item.curves, "curve1 should not be in the item's curves"


def test_PlotItem_preserve_external_visibility_control():
    item = pg.PlotItem()
    curve1 = pg.PlotDataItem(np.random.normal(size=10))
    curve2 = pg.PlotDataItem(np.random.normal(size=10))
    item.addItem(curve1)
    curve1.hide()
    item.addItem(curve2)
    assert not curve1.isVisible()
    item.removeItem(curve2)
    assert not curve1.isVisible()


def test_plotitem_menu_initialize():
    """Test the menu initialization of the plotitem"""
    item = pg.PlotItem()
    assert item.menuEnabled() is True
    viewbox = item.vb
    assert viewbox is not None
    assert viewbox.menu is not None
    assert viewbox.menuEnabled() is True

    item = pg.PlotItem(enableMenu=False)
    assert item.menuEnabled() is False
    viewbox = item.vb
    assert viewbox is not None
    assert viewbox.menu is None
    assert viewbox.menuEnabled() is False

    viewbox = pg.ViewBox()
    item = pg.PlotItem(viewBox=viewbox, enableMenu=False)
    assert item.menuEnabled() is False
    viewbox = item.vb
    assert viewbox is not None
    assert viewbox.menu is not None
    assert viewbox.menuEnabled() is True

    viewbox = pg.ViewBox(enableMenu=False)
    item = pg.PlotItem(viewBox=viewbox)
    assert item.menuEnabled() is True
    viewbox = item.vb
    assert viewbox is not None
    assert viewbox.menu is None
    assert viewbox.menuEnabled() is False


def test_fft():
    f = 20.
    x = np.linspace(0, 1, 1000)
    y = np.sin(2 * np.pi * f * x)
    pi = pg.PlotItem()
    pd = pi.plot(x, y)
    # TODO remove all these deprecated calls (move this test up to PlotItem?)
    pi.setDataTransformState("Power Spectrum (FFT)", True)
    x, y = pd.getData()
    assert abs(x[np.argmax(y)] - f) < 0.03

    x = np.linspace(0, 1, 1001)
    y = np.sin(2 * np.pi * f * x)
    pd.setData(x, y)
    x, y = pd.getData()
    assert abs(x[np.argmax(y)] - f) < 0.03

    pi.setDataTransformState("Log X", True)
    x, y = pd.getData()
    assert abs(x[np.argmax(y)] - np.log10(f)) < 0.01


def test_nonfinite():
    def _assert_equal_arrays(a1, a2):
        assert a1.shape == a2.shape
        for (xtest, xgood) in zip(a1, a2):
            assert ((xtest == xgood) or (np.isnan(xtest) and np.isnan(xgood)))

    x = np.array([-np.inf, 0.0, 1.0, 2.0, np.nan, 4.0, np.inf])
    y = np.array([1.0, 0.0, -1.0, np.inf, 2.0, np.nan, 0.0])
    pi = pg.PlotItem()
    pdi = pi.plot(x, y)
    dataset = pdi.getDisplayDataset()
    _assert_equal_arrays(dataset.x, x)
    _assert_equal_arrays(dataset.y, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_log = np.log10(x)
        y_log = np.log10(y)
    x_log[~np.isfinite(x_log)] = np.nan
    y_log[~np.isfinite(y_log)] = np.nan

    pi.setDataTransformState("Log X", True)
    pi.setDataTransformState("Log Y", True)
    dataset = pdi.getDisplayDataset()
    _assert_equal_arrays(dataset.x, x_log)
    _assert_equal_arrays(dataset.y, y_log)


def test_data_transforms_restore():
    item = pg.PlotItem()

    def transform(x, y, foo):
        return x + foo, y

    item.addDataTransformOption("test", transform, params=[{"name": "foo", "type": "float"}])
    item.setDataTransformState("test", True)
    item.setDataTransformParams("test", foo=1.3)
    state = item.saveState()
    item.setDataTransformState("test", False)
    item.setDataTransformParams("test", foo=301)
    assert not item._transforms["test"]["checkbox"].isChecked()
    assert item._paramsForDataTransform("test")["foo"] == 301
    item.restoreState(state)
    assert item._transforms["test"]["checkbox"].isChecked()
    assert item._paramsForDataTransform("test")["foo"] == 1.3
