"""
Simple example using BarGraphItem
"""

import numpy as np

import pyqtgraph as pg

win = pg.plot()
win.setWindowTitle('pyqtgraph example: BarGraphItem')

x = np.arange(10)
y1 = np.sin(x)
y2 = 1.1 * np.sin(x+1)
y3 = 1.2 * np.sin(x+2)

bg1 = pg.BarGraphItem(x=x, height=y1, width=0.3, brush='r')
bg2 = pg.BarGraphItem(x=x+0.33, height=y2, width=0.3, brush='g')
bg3 = pg.BarGraphItem(x=x+0.66, height=y3, width=0.3, brush='b')

win.addItem(bg1)
win.addItem(bg2)
win.addItem(bg3)


# Final example shows how to handle mouse clicks:
class BarGraph(pg.BarGraphItem):
    def mouseClickEvent(self, event):
        print("clicked")


bg = BarGraph(x=x, y=y1*0.3+2, height=0.4+y1*0.2, width=0.8)
win.addItem(bg)

if __name__ == '__main__':
    pg.exec()
