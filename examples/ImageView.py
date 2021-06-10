# -*- coding: utf-8 -*-
"""
This example demonstrates the use of ImageView with 3-color image stacks.
ImageView is a high-level widget for displaying and analyzing 2D and 3D data.
ImageView provides:

  1. A zoomable region (ViewBox) for displaying the image
  2. A combination histogram and gradient editor (HistogramLUTItem) for
     controlling the visual appearance of the image
  3. A timeline for selecting the currently displayed frame (for 3D data only).
  4. Tools for very basic analysis of image data (see ROI and Norm buttons)

"""
## Add path to library (just for examples; you do not need this)
import initExample

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

app = pg.mkQApp("ImageView Example")

## Create window with ImageView widget
win = QtGui.QMainWindow()
win.resize(800,800)
imv = pg.ImageView(discreteTimeLine=True)
win.setCentralWidget(imv)
win.show()
win.setWindowTitle('pyqtgraph example: ImageView')
imv.setHistogramLabel("Histogram label goes here")
imv.setHistogramPrintView(printView=False)

## Create random 3D data set with time varying signals
dataRed = np.ones((100, 200, 200)) * np.linspace(90, 150, 100)[:, np.newaxis, np.newaxis]
dataRed += pg.gaussianFilter(np.random.normal(size=(200, 200)), (5, 5)) * 100
dataGrn = np.ones((100, 200, 200)) * np.linspace(90, 180, 100)[:, np.newaxis, np.newaxis]
dataGrn += pg.gaussianFilter(np.random.normal(size=(200, 200)), (5, 5)) * 100
dataBlu = np.ones((100, 200, 200)) * np.linspace(180, 90, 100)[:, np.newaxis, np.newaxis]
dataBlu += pg.gaussianFilter(np.random.normal(size=(200, 200)), (5, 5)) * 100

data = np.concatenate(
    (dataRed[:, :, :, np.newaxis], dataGrn[:, :, :, np.newaxis], dataBlu[:, :, :, np.newaxis]), axis=3
)


# Display the data and assign each frame a time value from 1.0 to 3.0
imv.setImage(data, xvals=np.linspace(1., 3., data.shape[0]))

## Set a custom color map
colors = [
    (0, 0, 0),
    (45, 5, 61),
    (84, 42, 55),
    (150, 87, 60),
    (208, 171, 141),
    (255, 255, 255)
]
cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
imv.setColorMap(cmap)

# Start up with an ROI
imv.ui.roiBtn.setChecked(True)
imv.roiClicked()

if __name__ == '__main__':
    pg.exec()
