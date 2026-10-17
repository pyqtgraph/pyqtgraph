"""
Demonstrates basic use of ErrorBarItem, including upper/lower limit arrows.
"""

import numpy as np

import pyqtgraph as pg

pg.setConfigOptions(antialias=True)

x = np.arange(10)
y = np.arange(10) % 3
top = np.linspace(1.0, 3.0, 10)
bottom = np.linspace(2, 0.5, 10)

# mark a couple of points as upper / lower limits (arrows instead of beams)
topLimit = np.zeros(10, dtype=bool)
topLimit[2] = True
bottomLimit = np.zeros(10, dtype=bool)
bottomLimit[7] = True

plt = pg.plot()
plt.setWindowTitle('pyqtgraph example: ErrorBarItem')
err = pg.ErrorBarItem(
    x=x, y=y, top=top, bottom=bottom, beam=0.5,
    topLimit=topLimit, bottomLimit=bottomLimit,
)
plt.addItem(err)
plt.plot(x, y, symbol='o', pen={'color': 0.8, 'width': 2})

if __name__ == '__main__':
    pg.exec()
