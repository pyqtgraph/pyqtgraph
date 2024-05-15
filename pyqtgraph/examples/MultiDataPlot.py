import traceback
import numpy as np

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ScatterPlotItem import name_list
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import interact, ParameterTree, Parameter
import random

pg.mkQApp()

rng = np.random.default_rng(10)
random.seed(10)


def sortedRandint(low, high, size):
    return np.sort(rng.integers(low, high, size))


def isNoneOrScalar(value):
    return value is None or np.isscalar(value[0])


values = {
    # Convention 1
    "None (replaced by integer indices)": None,
    # Convention 2
    "Single curve values": sortedRandint(0, 20, 15),
    # Convention 3 list form
    "container of (optionally) mixed-size curve values": [
        sortedRandint(0, 20, 15),
        *[sortedRandint(0, 20, 15) for _ in range(4)],
    ],
    # Convention 3 array form
    "2D matrix": np.row_stack([sortedRandint(20, 40, 15) for _ in range(6)]),
}


def next_plot(xtype="random", ytype="random", symbol="o", symbolBrush="#f00"):
    constKwargs = locals()
    x = y = None
    if xtype == "random":
        xtype = random.choice(list(values))
    if ytype == "random":
        ytype = random.choice(list(values))
    x = values[xtype]
    y = values[ytype]
    textbox.setValue(f"x={xtype}\ny={ytype}")
    pltItem.clear()
    try:
        pltItem.multiDataPlot(
            x=x, y=y, pen=cmap.getLookupTable(nPts=6), constKwargs=constKwargs
        )
    except Exception as e:
        QtWidgets.QMessageBox.critical(widget, "Error", traceback.format_exc())


cmap = pg.colormap.get("viridis")
widget = pg.PlotWidget()
pltItem: pg.PlotItem = widget.plotItem

xytype = dict(type="list", values=list(values))
topParam = interact(
    next_plot,
    symbolBrush=dict(type="color"),
    symbol=dict(type="list", values=name_list),
    xtype=xytype,
    ytype=xytype,
)
tree = ParameterTree()
tree.setMinimumWidth(150)
tree.addParameters(topParam, showTop=True)

textbox = Parameter.create(name="text", type="text", readonly=True)
tree.addParameters(textbox)

win = QtWidgets.QWidget()
win.setLayout(lay := QtWidgets.QHBoxLayout())
lay.addWidget(widget)
lay.addWidget(tree)
if __name__ == "__main__":
    win.show()
    pg.exec()
