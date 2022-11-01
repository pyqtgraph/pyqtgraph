import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt.QtWidgets import QMainWindow


def mkStripedPen(colors, blending=0.0001, span=[0, 15], orientation="horizontal", width=2):
    stops = []
    stops_colors = []
    previous = None
    for i, color in enumerate(colors + [None]):
        pos = i / len(colors)
        if previous is not None:
            stops.append(pos - blending)
            stops_colors.append(previous)
        if color is not None:
            stops.append(pos)
            stops_colors.append(color)
        previous = color
    return pg.ColorMap(stops, stops_colors, mapping=pg.ColorMap.REPEAT).getPen(span=span, orientation=orientation, width=width)


app = pg.mkQApp()
mw = QMainWindow()
mw.resize(800, 400)
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
ax1 = mpw.addAxis("bottom", text="Samples1", units="sx1")
ax1c = "red"
ax1.setPen(ax1c)
ax2 = mpw.addAxis("bottom", text="Samples2", units="sx2")
ax2c = "green"
ax2.setPen(ax2c)
# ax3 = mpw.addAxis("bottom", text="Samples3", units="sx3")
# ax3c = "grey"
# ax3.setPen(ax3c)
ay1 = mpw.addAxis("left", text="Data1", units="sy1")
ay1c = "cyan"
ay1.setPen(ay1c)
# ay2 = mpw.addAxis("left", text="Data2", units="sy2")
# ay2c = "magenta"
# ay2.setPen(ay2c)
ay3 = mpw.addAxis("left", text="Data3", units="sy3")
ay3c = "yellow"
ay3.setPen(ay3c)
# CHARTS
c0, pi0 = mpw.addChart()
c0.setPen("black")
c1, pi1 = mpw.addChart(xAxis=ax1, yAxis=ay1)
c1.setPen(mkStripedPen([ax1c, ay1c]))
c2, pi2 = mpw.addChart(xAxis=ax2, yAxis=ay1)
c2.setPen(mkStripedPen([ax2c, ay1c]))
c3, pi3 = mpw.addChart(xAxis=ax2, yAxis=ay3)
c3.setPen(mkStripedPen([ax2c, ay3c]))
mpw.enableAxisAutoRange()

for i, c in enumerate([c0, c1, c2, c3, ], start=1):
# for i, c in enumerate([c0,  ], start=1):
    c.setData(np.array(np.sin(np.linspace(0, i * 2 * np.pi, num=1000))))

mpw.update()

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    pg.exec()
