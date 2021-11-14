"""
Demonstrates basic use of ErrorBarItem
"""

import numpy as np

import pyqtgraph as pg

pg.setConfigOptions(antialias=True)

x = np.arange(10)
y = np.arange(10) %3
top = np.linspace(1.0, 3.0, 10)
bottom = np.linspace(2, 0.5, 10)

plt = pg.plot()
plt.setWindowTitle('pyqtgraph example: ErrorBarItem')
err = pg.ErrorBarItem(x=x, y=y, top=top, bottom=bottom, beam=0.5)
plt.addItem(err)
plt.plot(x, y, symbol='o', pen={'color': 0.8, 'width': 2})

if __name__ == '__main__':
    pg.exec()
