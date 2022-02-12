"""
CLI Example
===========

Display a plot and an image with minimal setup.

``pg.plot()`` and ``pg.image()`` are indended to be used from an interactive prompt to
allow easy data inspection.
"""

import numpy as np

import pyqtgraph as pg

# %%
# Make a line plot
data = np.random.normal(size=1000)
pg.plot(data, title="Simplest possible plotting example")

# %%
# Plot image data
data = np.random.normal(size=(500, 500))
pg.image(data, title="Simplest possible image example")

if __name__ == '__main__':
    pg.exec()
