# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)
import initExample

import numpy as np
import scipy
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

app = QtGui.QApplication([])

## Create window with ImageView widget
win = QtGui.QMainWindow()
win.resize(800,800)
imv = pg.ImageView()
win.setCentralWidget(imv)
win.show()

## Create random 3D data set with noisy signals
img = scipy.ndimage.gaussian_filter(np.random.normal(size=(200, 200)), (5, 5)) * 20 + 100
img = img[np.newaxis,:,:]
decay = np.exp(-np.linspace(0,0.3,100))[:,np.newaxis,np.newaxis]
data = np.random.normal(size=(100, 200, 200))
data += img * decay

#for i in range(data.shape[0]):
    #data[i] += 10*exp(-(2.*i)/data.shape[0])
data += 2

## Add time-varying signal
sig = np.zeros(data.shape[0])
sig[30:] += np.exp(-np.linspace(1,10, 70))
sig[40:] += np.exp(-np.linspace(1,10, 60))
sig[70:] += np.exp(-np.linspace(1,10, 30))

sig = sig[:,np.newaxis,np.newaxis] * 3
data[:,50:60,50:60] += sig


## Display the data
imv.setImage(data, xvals=np.linspace(1., 3., data.shape[0]))

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
