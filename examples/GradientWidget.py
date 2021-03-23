# -*- coding: utf-8 -*-
"""
Demonstrates the appearance / interactivity of GradientWidget
(without actually doing anything useful with it)

"""
import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np



app = pg.mkQApp("Gradiant Widget Example")
w = QtGui.QMainWindow()
w.show()
w.setWindowTitle('pyqtgraph example: GradientWidget')
w.setGeometry(10, 50, 400, 400)
cw = QtGui.QWidget()
w.setCentralWidget(cw)

l = QtGui.QGridLayout()
l.setSpacing(0)
cw.setLayout(l)

w1 = pg.GradientWidget(orientation='top')
w2 = pg.GradientWidget(orientation='right', allowAdd=False)
#w2.setTickColor(1, QtGui.QColor(255,255,255))
w3 = pg.GradientWidget(orientation='bottom', allowAdd=False, allowRemove=False)
w4 = pg.GradientWidget(orientation='left')
w4.loadPreset('spectrum')
label = QtGui.QLabel("""
- Click a triangle to change its color
- Drag triangles to move
- Right-click a gradient to load triangle presets
- Click in an empty area to add a new color
    (adding is disabled for the bottom-side and right-side widgets)
- Right click a triangle to remove
    (only possible if more than two triangles are visible)
    (removing is disabled for the bottom-side widget)
""")

l.addWidget(w1, 0, 1)
l.addWidget(w2, 1, 2)
l.addWidget(w3, 2, 1)
l.addWidget(w4, 1, 0)
l.addWidget(label, 1, 1)

if __name__ == '__main__':
    pg.mkQApp().exec_()



