import pyqtgraph as pg

from pyqtgraph.graphicsItems.LegendItem import get_toggle_pen_brush


def test_legend_item_basics():
    pg.mkQApp()

    legend = pg.LegendItem()

    assert legend.opts['pen'] == pg.mkPen(None)
    assert legend.opts['brush'] == pg.mkBrush(None)
    assert legend.opts['labelTextColor'] is None
    assert legend.opts['labelTextSize'] == '9pt'
    assert legend.opts['offset'] is None

    assert legend.columnCount == 1
    assert legend.rowCount == 1
    assert legend.curRow == 0

    assert legend.labelTextColor() is None
    assert legend.labelTextSize() == '9pt'
    assert legend.brush() == pg.mkBrush(None)
    assert legend.pen() == pg.mkPen(None)

    # Set brush
    # ----------------------------------------------------

    brush = pg.mkBrush('b')
    legend.setBrush(brush)
    assert legend.brush() == brush
    assert legend.opts['brush'] == brush

    # Set pen
    # ----------------------------------------------------

    pen = pg.mkPen('b')
    legend.setPen(pen)
    assert legend.pen() == pen
    assert legend.opts['pen'] == pen

    # Set labelTextColor
    # ----------------------------------------------------

    text_color = pg.mkColor('b')
    legend.setLabelTextColor(text_color)
    assert legend.labelTextColor() == text_color
    assert legend.opts['labelTextColor'] == text_color

    # Set labelTextSize
    # ----------------------------------------------------

    text_size = '12pt'
    legend.setLabelTextSize(text_size)
    assert legend.labelTextSize() == text_size
    assert legend.opts['labelTextSize'] == text_size

    # Add items
    # ----------------------------------------------------

    assert len(legend.items) == 0
    plot = pg.PlotDataItem(name="Plot")
    legend.addItem(plot, name="Plot")
    assert len(legend.items) == 1

    scatter = pg.PlotDataItem(name="Scatter")
    legend.addItem(scatter, name="Scatter")
    assert len(legend.items) == 2
    assert legend.columnCount == 1
    assert legend.rowCount == 1

    curve = pg.PlotDataItem(name="Curve")
    legend.addItem(curve, name="Curve")
    assert len(legend.items) == 3

    scrabble = pg.PlotDataItem(name="Scrabble")
    legend.addItem(scrabble, name="Scrabble")
    assert len(legend.items) == 4

    assert legend.layout.rowCount() == 4
    legend.setColumnCount(2)
    assert legend.rowCount == 2

    assert legend.layout.rowCount() == 3

    # Remove items
    # ----------------------------------------------------

    legend.removeItem(scrabble)
    assert legend.rowCount == 2
    assert legend.layout.rowCount() == 2
    assert scrabble not in legend.items
    assert len(legend.items) == 3

    legend.removeItem(curve)
    assert legend.rowCount == 2
    assert legend.layout.rowCount() == 1
    assert curve not in legend.items
    assert len(legend.items) == 2

    # LegendItem style
    # ----------------------------------------------------

    assert legend.itemStyle() == 0
    legend.setItemStyle(pg.LegendStyle.Toggle)
    assert legend.itemStyle() == 1
    assert len(legend.items) == 2

    # LegendItem clear
    # ----------------------------------------------------
    legend.clear()
    assert legend.items == []


def test_get_toggle_pen_brush():
    pg.mkQApp()

    item = pg.PlotDataItem(name="Plot", pen="r")
    pen, brush = get_toggle_pen_brush(item)
    assert pen == pg.mkPen("r")
    assert brush == pg.mkBrush("r")

    item = pg.BarGraphItem(brush='b', pen='w', name='bar')
    pen, brush = get_toggle_pen_brush(item)
    assert pen == pg.mkPen("b")
    assert brush == pg.mkBrush("b")

    item = pg.PlotDataItem(pen='r', symbol='t', symbolPen='r', symbolBrush='g',
                           name='Symbol')
    pen, brush = get_toggle_pen_brush(item)
    assert pen == pg.mkPen('r')
    assert brush == pg.mkBrush('g')

    item = pg.PlotDataItem(pen='g', fillLevel=0, fillBrush=(70, 255, 255, 30),
                           name='Fill')
    pen, brush = get_toggle_pen_brush(item)
    assert pen == pg.mkPen('g')
    assert brush == pg.mkBrush('g')

    item = pg.ScatterPlotItem(size=10, pen='c', brush=(255, 255, 255, 120),
                              name='Scatter')
    pen, brush = get_toggle_pen_brush(item)
    assert pen == pg.mkPen('c')
    assert brush == pg.mkBrush(color=(255, 255, 255, 120))
