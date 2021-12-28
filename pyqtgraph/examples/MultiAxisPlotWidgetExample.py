import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import Qt
from pyqtgraph.Qt.QtGui import QBrush, QColor, QGradient, QLinearGradient, QPen
from pyqtgraph.Qt.QtWidgets import QMainWindow


def mkStripedPen(colors, segment_lenght=20):
    gradient = QLinearGradient(0, 0, segment_lenght, 0)
    gradient.setCoordinateMode(QGradient.LogicalMode)
    gradient.setSpread(QGradient.RepeatSpread)
    stops = []
    previous = None
    for i, color in enumerate(colors + [None]):
        if previous is not None:
            stops.append([i / len(colors) - 0.001, previous])
        if color is not None:
            stops.append([i / len(colors), color])
        previous = color
    gradient.setStops(stops)
    brush = QBrush(gradient)
    pen = QPen(brush, 2, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
    pen.setCosmetic(True)
    return pen


app = pg.mkQApp()
mw = QMainWindow()
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
ax1c = QColor("red")
ax1.setPen(ax1c)
ax2 = mpw.addAxis("sx2", "bottom", text="Samples2", units="sx2")
ax2c = QColor("green")
ax2.setPen(ax2c)
ay1 = mpw.addAxis("sy1", "left", text="Data1", units="sy1")
ay1c = QColor("cyan")
ay1.setPen(ay1c)
ay2 = mpw.addAxis("sy2", "left", text="Data2", units="sy2")
ay2c = QColor("magenta")
ay2.setPen(ay2c)
# CHARTS
c0, pi0 = mpw.addChart("Dataset 0")
c0.setPen("black")
c1, pi1 = mpw.addChart("Dataset 1", xAxisName="sx1", yAxisName="sy1")
c1.setPen(mkStripedPen([ax1c, ay1c]))
c2, pi2 = mpw.addChart("Dataset 2", xAxisName="sx2", yAxisName="sy1")
c2.setPen(mkStripedPen([ax2c, ay1c]))
c3, pi3 = mpw.addChart("Dataset 3", xAxisName="sx2", yAxisName="sy2")
c3.setPen(mkStripedPen([ax2c, ay2c]))
# make and display chart
mpw.makeLayout(
    # optional, selects and orders axes displayed.
    # remember to include the default axes if used.
    axes=["bottom", "sx1", "sx2", "sy2", "sy1", "left"],
    # optional, selects charts displayed
    charts=["Dataset 0", "Dataset 1", "Dataset 2", "Dataset 3"]
)

for i, c in enumerate([c0, c1, c2, c3, ], start=1):
    c.setData(np.array(np.sin(np.linspace(0, i * np.pi, num=1000))))

mpw.update()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    pg.exec()
