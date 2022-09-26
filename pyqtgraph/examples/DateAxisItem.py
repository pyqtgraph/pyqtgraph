"""
Demonstrates the usage of DateAxisItem to display properly-formatted 
timestamps on x-axis which automatically adapt to current zoom level.

"""

import time

import numpy as np

import pyqtgraph as pg

app = pg.mkQApp("DateAxisItem Example")

# Create a plot with a date-time axis
w = pg.PlotWidget(axisItems = {'bottom': pg.DateAxisItem()})
w.showGrid(x=True, y=True)

# Plot sin(1/x^2) with timestamps in the last 100 years
now = time.time()
x = np.linspace(2*np.pi, 1000*2*np.pi, 8301)
w.plot(now-(2*np.pi/x)**2*100*np.pi*1e7, np.sin(x), symbol='o')

w.setWindowTitle('pyqtgraph example: DateAxisItem')
w.show()

if __name__ == '__main__':
    pg.exec()
