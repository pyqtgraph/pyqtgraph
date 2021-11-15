"""
Use a HistogramLUTWidget to control the contrast / coloration of an image.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

app = pg.mkQApp("Histogram Lookup Table Example")
win = QtWidgets.QMainWindow()
win.resize(880, 600)
win.show()
win.setWindowTitle('pyqtgraph example: Histogram LUT')

cw = QtWidgets.QWidget()
win.setCentralWidget(cw)

layout = QtWidgets.QGridLayout()
cw.setLayout(layout)
layout.setSpacing(0)

view = pg.GraphicsView()
vb = pg.ViewBox()
vb.setAspectLocked()
view.setCentralItem(vb)
layout.addWidget(view, 0, 1, 3, 1)

hist = pg.HistogramLUTWidget(gradientPosition="left")
layout.addWidget(hist, 0, 2)


monoRadio = QtWidgets.QRadioButton('mono')
rgbaRadio = QtWidgets.QRadioButton('rgba')
layout.addWidget(monoRadio, 1, 2)
layout.addWidget(rgbaRadio, 2, 2)
monoRadio.setChecked(True)


def setLevelMode():
    mode = 'mono' if monoRadio.isChecked() else 'rgba'
    hist.setLevelMode(mode)


monoRadio.toggled.connect(setLevelMode)

data = pg.gaussianFilter(np.random.normal(size=(256, 256, 3)), (20, 20, 0))
for i in range(32):
    for j in range(32):
        data[i*8, j*8] += .1
img = pg.ImageItem(data)
vb.addItem(img)
vb.autoRange()

hist.setImageItem(img)

if __name__ == '__main__':
    pg.exec()
