# -*- coding: utf-8 -*-

from GradientWidget import *
from PyQt4 import QtGui

app = QtGui.QApplication([])
w = QtGui.QMainWindow()
w.show()
w.resize(400,400)
cw = QtGui.QWidget()
w.setCentralWidget(cw)

l = QtGui.QGridLayout()
l.setSpacing(0)
cw.setLayout(l)

w1 = GradientWidget(orientation='top')
w2 = GradientWidget(orientation='right', allowAdd=False)
w2.setTickColor(1, QtGui.QColor(255,255,255))
w3 = GradientWidget(orientation='bottom')
w4 = TickSlider(orientation='left')

l.addWidget(w1, 0, 1)
l.addWidget(w2, 1, 2)
l.addWidget(w3, 2, 1)
l.addWidget(w4, 1, 0)
