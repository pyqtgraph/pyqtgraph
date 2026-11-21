from unittest import mock

import pyqtgraph as pg

pg.mkQApp()

def test_legend_item_basics():

    legend = pg.LegendItem()

    assert legend.opts['pen'] == pg.mkPen(None)
    assert legend.opts['brush'] == pg.mkBrush(None)
    assert legend.opts['labelTextColor'] is None
    assert legend.opts['labelTextSize'] == '9pt'
    assert legend.opts['offset'] is None

    assert legend.columnCount == 1
    assert legend.rowCount == 1

    assert legend.labelTextColor() is None
    assert legend.labelTextSize() == '9pt'
    assert legend.brush() == pg.mkBrush(None)
    assert legend.pen() == pg.mkPen(None)
    assert legend.sampleType is pg.ItemSample

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
    assert legend.rowCount == 2

    curve = pg.PlotDataItem(name="Curve")
    legend.addItem(curve, name="Curve")
    assert len(legend.items) == 3
    assert legend.rowCount == 3

    scrabble = pg.PlotDataItem(name="Scrabble")
    legend.addItem(scrabble, name="Scrabble")
    assert len(legend.items) == 4

    assert legend.layout().rowCount() == 4
    assert legend.rowCount == 4
    legend.setColumnCount(2)
    assert legend.columnCount == 2
    assert legend.rowCount == 2

    assert legend.layout().rowCount() == 2

    # Remove items
    # ----------------------------------------------------

    legend.removeItem(scrabble)
    assert legend.rowCount == 2
    assert legend.layout().rowCount() == 2
    assert scrabble not in legend.items
    assert len(legend.items) == 3

    legend.removeItem(curve)
    assert legend.rowCount == 2 # rowCount will never decrease when removing
    assert legend.layout().rowCount() == 1
    assert curve not in legend.items
    assert len(legend.items) == 2

    legend.clear()
    assert legend.items == []


def test_legend_label_updates_on_setdata():
    # Regression test for https://github.com/pyqtgraph/pyqtgraph/issues/1000
    # Calling PlotDataItem.setData() with a new name (or style) must refresh
    # the corresponding legend entry instead of leaving a stale label/sample.
    win = pg.GraphicsLayoutWidget()
    plot = win.addPlot()
    legend = plot.addLegend()

    item = plot.plot([1, 2, 3], [1, 2, 3], name="Old Name")
    assert [label.text for _, label in legend.items] == ["Old Name"]

    item.setData([1, 2, 3], [3, 2, 1], name="New Name")
    assert [label.text for _, label in legend.items] == ["New Name"]
    assert legend.getLabel(item).text == "New Name"
    # setData must not add or remove legend entries
    assert len(legend.items) == 1

    # a style-only update keeps the entry and refreshes the sample
    sample = legend.items[0][0]
    with mock.patch.object(sample, "update") as update_mock:
        item.setData([4, 5, 6], pen=pg.mkPen("g"), symbol="s")
        update_mock.assert_called()
    assert len(legend.items) == 1
    assert legend.getLabel(item).text == "New Name"
    assert legend.items[0][0] is sample

    # removing the item from the plot also removes the legend entry
    plot.removeItem(item)
    assert legend.items == []

    win.close()


def test_legend_addItem_existing_item_updates_label():
    # Adding the same item again (with a new name) must update the existing
    # entry instead of creating a duplicate.
    legend = pg.LegendItem()
    item = pg.PlotDataItem([1, 2, 3], name="First")
    legend.addItem(item, name="First")
    assert len(legend.items) == 1

    legend.addItem(item, name="Second")
    assert len(legend.items) == 1
    assert legend.items[0][1].text == "Second"

    # setData() with a new name refreshes the label of the existing entry
    item.setData([4, 5, 6], name="Renamed")
    assert len(legend.items) == 1
    assert legend.items[0][1].text == "Renamed"
    assert legend.getLabel(item).text == "Renamed"

    # after removeItem(), the legend must no longer react to the item
    label = legend.items[0][1]
    legend.removeItem(item)
    assert legend.items == []
    item.setData([7, 8, 9], name="Removed")
    assert label.text == "Renamed"


def test_legend_clear_disconnects_items():
    legend = pg.LegendItem()
    item = pg.PlotDataItem([1, 2, 3], name="First")
    other = pg.PlotDataItem([3, 2, 1], name="Other")
    legend.addItem(item, name="First")
    legend.addItem(other, name="Other")
    label = legend.getLabel(item)

    legend.clear()
    assert legend.items == []

    # clearing the legend disconnects the update slot: no stale updates
    item.setData([4, 5, 6], name="After Clear")
    assert label.text == "First"

    # re-adding the item after clear() works and reconnects updates
    legend.addItem(item, name="First")
    item.setData([7, 8, 9], name="Re-added")
    assert legend.getLabel(item).text == "Re-added"
