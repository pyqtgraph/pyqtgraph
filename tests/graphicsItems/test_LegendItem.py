import numpy as np
import pyqtgraph as pg

app = pg.mkQApp()


def test_LegendItem_dynamic_update():
    plot = pg.PlotItem()
    legend = plot.addLegend()

    # Create curve with initial name
    curve = pg.PlotDataItem(np.array([1, 2, 3]), name="Old Name")
    plot.addItem(curve)

    # Check that it is added to the legend with the correct label
    label = legend.getLabel(curve)
    assert label is not None
    assert label.text == "Old Name"

    # Call setData with a new name
    curve.setData(np.array([4, 5, 6]), name="New Name")

    # Check that the legend updated dynamically!
    assert label.text == "New Name"
