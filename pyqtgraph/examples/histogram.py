"""
In this example we draw two different kinds of histogram.
"""

import numpy as np

import pyqtgraph as pg

win = pg.GraphicsLayoutWidget(show=True)
win.resize(800,480)
win.setWindowTitle('pyqtgraph example: Histogram')
plt1 = win.addPlot()
plt2 = win.addPlot()

## make interesting distribution of values
vals = np.hstack([np.random.normal(size=500), np.random.normal(size=260, loc=4)])

## compute standard histogram
y,x = np.histogram(vals, bins=np.linspace(-3, 8, 40))

## Using stepMode="center" causes the plot to draw two lines for each sample.
## notice that len(x) == len(y)+1
plt1.plot(x, y, stepMode="center", fillLevel=0, fillOutline=True, brush=(0,0,255,150))

## Now draw all points as a nicely-spaced scatter plot
psy = pg.pseudoScatter(vals, spacing=0.15)
plt2.plot(vals, psy, pen=None, symbol='o', symbolSize=5, symbolPen=(255,255,255,200), symbolBrush=(0,0,255,150))

# draw histogram using BarGraphItem
win.nextRow()
plt3 = win.addPlot()
bgi = pg.BarGraphItem(x0=x[:-1], x1=x[1:], height=y, pen='w', brush=(0,0,255,150))
plt3.addItem(bgi)

if __name__ == '__main__':
    pg.exec()
