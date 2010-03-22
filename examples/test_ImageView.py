# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pyqtgraph.ImageView import *
from numpy import random
from PyQt4 import QtCore, QtGui
from scipy.ndimage import *

app = QtGui.QApplication([])

## Create window with ImageView widget
win = QtGui.QMainWindow()
imv = ImageView()
win.setCentralWidget(imv)
win.show()

## Create random 3D data set
img = gaussian_filter(random.random((200, 200)), (5, 5)) * 5
data = random.random((100, 200, 200))
data += img
for i in range(data.shape[0]):
    data[i] += exp(-(2.*i)/data.shape[0])
data += 10    

## Display the data
imv.setImage(data)

app.exec_()
