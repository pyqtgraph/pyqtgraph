# -*- coding: utf-8 -*-
import pytest
import pyqtgraph as pg

app = pg.mkQApp()


@pytest.mark.parametrize('orientation', ['left', 'right', 'top', 'bottom'])
def test_PlotItem_shared_axis_items(orientation):
    """Adding an AxisItem to multiple plots raises RuntimeError"""
    ax1 = pg.AxisItem(orientation)
    ax2 = pg.AxisItem(orientation)

    layout = pg.GraphicsLayoutWidget()

    pi1 = layout.addPlot(axisItems={orientation: ax1})

    pi2 = layout.addPlot()
    # left or bottom replaces, right or top adds new
    pi2.setAxisItems({orientation: ax2})

    with pytest.raises(RuntimeError):
        pi2.setAxisItems({orientation: ax1})


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
