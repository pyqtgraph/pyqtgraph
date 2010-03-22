# -*- coding: utf-8 -*-

from ImageView import *
from numpy import random
from PyQt4 import QtCore, QtGui
from scipy.ndimage import *

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
imv = ImageView()
win.setCentralWidget(imv)
win.show()

img = gaussian_filter(random.random((200, 200)), (5, 5)) * 5
data = random.random((100, 200, 200))
data += img

for i in range(data.shape[0]):
    data[i] += exp(-(2.*i)/data.shape[0])
data += 10    
imv.setImage(data)