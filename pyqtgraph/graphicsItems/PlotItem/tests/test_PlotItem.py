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
