# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pyqtgraph.ImageView import *
from numpy import random, linspace
from PyQt4 import QtCore, QtGui
from scipy.ndimage import *

app = QtGui.QApplication([])

## Create window with ImageView widget
win = QtGui.QMainWindow()
imv = ImageView()
win.setCentralWidget(imv)
win.show()

## Create random 3D data set
img = gaussian_filter(random.normal(size=(200, 200)), (5, 5)) * 20 + 100
img = img[newaxis,:,:]
decay = exp(-linspace(0,0.3,100))[:,newaxis,newaxis]
data = random.normal(size=(100, 200, 200))
data += img * decay

#for i in range(data.shape[0]):
    #data[i] += 10*exp(-(2.*i)/data.shape[0])
data += 2

## Add time-varying signal
sig = zeros(data.shape[0])
sig[30:] += exp(-linspace(1,10, 70))
sig[40:] += exp(-linspace(1,10, 60))
sig[70:] += exp(-linspace(1,10, 30))

sig = sig[:,newaxis,newaxis] * 3
data[:,50:60,50:60] += sig


## Display the data
imv.setImage(data, xvals=linspace(1., 3., data.shape[0]))

app.exec_()
