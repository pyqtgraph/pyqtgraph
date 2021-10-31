"""
Demonstrates ScaleBar
"""

import numpy as np

import pyqtgraph as pg

pg.mkQApp()
win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle('pyqtgraph example: ScaleBar')

vb = win.addViewBox()
vb.setAspectLocked()

img = pg.ImageItem()
img.setImage(np.random.normal(size=(100,100)))
img.setScale(0.01)
vb.addItem(img)

scale = pg.ScaleBar(size=0.1)
scale.setParentItem(vb)
scale.anchor((1, 1), (1, 1), offset=(-20, -20))

if __name__ == '__main__':
    pg.exec()
