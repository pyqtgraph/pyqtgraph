import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

app = pg.mkQApp()
mw = QtWidgets.QMainWindow()
mw.resize(800, 800)
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

mpw = pg.MultiAxisPlotWidget()

mw.setCentralWidget(mpw)
mw.show()

# LEGEND
mpw.addLegend(offset=(0, 0))
# TITLE
mpw.setTitle("MultiAxisPlotWidget Example")
# AXYS
ax1 = mpw.addAxis("sx1", "bottom", text="Samples1", units="sx1")
ax2 = mpw.addAxis("sx2", "bottom", text="Samples2", units="sx2")
ax3 = mpw.addAxis("sx3", "bottom", text="Samples3", units="sx3")
ay1 = mpw.addAxis("sy1", "left", text="Data1", units="sy1")
ay2 = mpw.addAxis("sy2", "left", text="Data2", units="sy2")
ay3 = mpw.addAxis("sy3", "left", text="Data3", units="sy3")
# CHARTS
c0, _ = mpw.addChart("Dataset 0")
c1, pi1 = mpw.addChart("Dataset 1", xAxisName="sx1", yAxisName="sy1")
c2, pi2 = mpw.addChart("Dataset 2", xAxisName="sx2", yAxisName="sy1")
c3, pi3 = mpw.addChart("Dataset 3", xAxisName="sx2", yAxisName="sy2")
c4, pi4 = mpw.addChart("Dataset 4", xAxisName="sx3", yAxisName="sy3")
c5, pi5 = mpw.addChart("Dataset 5")  # will create it's axes automatically
c6, pi6 = mpw.addChart("Dataset 6")  # will not be displayed (not in the makeLayout's chart list)
# make and display chart
mpw.makeLayout(
    axes=["sx2", "sx3", "sx1", "sy2", "sy3", "sy1", ],  # optional, selects and orders axes displayed
    charts=["Dataset 0", "Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4", "Dataset 5"]  # optional, selects charts displayed
)

for i, c in enumerate([c0, c1, c2, c3, c4, c5, c6, ]):
    c.setData(np.array(np.sin(np.linspace(0, i * np.pi, num=1000))))

mpw.update()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    pg.exec()
