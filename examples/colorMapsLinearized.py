# -*- coding: utf-8 -*-
"""
This example demonstrates linearized ColorMap objects using colormap.makeMonochrome()
or using the `ColorMap`'s `linearize()` method.
"""
# Add path to library (just for examples; you do not need this)
import initExample

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

name_list = (
    'warm','neutral','cool',
    'green','amber','blue','red','pink','lavender',
    (0.5, 0.2, 0.1, 0.8)
)
ramp_list = [
    pg.colormap.makeMonochrome(name)
    for name in name_list
]

cm_list = []
# Create a gray ramp for demonstrating the idea:
cm = pg.ColorMap( None, [
    QtGui.QColor(  0,   0,   0),
    QtGui.QColor( 10,  10,  10),
    QtGui.QColor(127, 127, 127),
    QtGui.QColor(240, 240, 240),
    QtGui.QColor(255, 255, 255)
])
cm_list.append(('Distorted gray ramp',cm))

# Create a rainbow scale in HSL color space:
length = 41
col_list = []
for idx in range(length):
    frac = idx/(length-1)
    qcol = QtGui.QColor()
    qcol.setHslF( (2*frac-0.15)%1.0, 0.8, 0.5-0.5*np.cos(np.pi*frac) )
    col_list.append(qcol)
cm = pg.ColorMap( None, col_list )
cm_list.append( ('Distorted HSL spiral', cm) )

# Create some random examples:
for example_idx in range(3):
    previous = None
    col_list = []
    for idx in range(8):
        values = np.random.random((3))
        if previous is not None:
            intermediate = (values + previous) / 2
            qcol = QtGui.QColor()
            qcol.setRgbF( *intermediate )
            col_list.append( qcol)
        qcol1 = QtGui.QColor()
        qcol1.setRgbF( *values )
        col_list.append( qcol1)
        previous = values
    cm = pg.ColorMap( None, col_list )
    cm_list.append( (f'random {example_idx+1}', cm) )

app = pg.mkQApp()
win = QtGui.QMainWindow()
win.resize(1000,800)

lw = pg.GraphicsLayoutWidget()
lw.setFixedWidth(1000)
lw.setSizePolicy(QtGui.QSizePolicy.Policy.Expanding, QtGui.QSizePolicy.Policy.Expanding)

scr = QtGui.QScrollArea()
scr.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
scr.setWidget(lw)
win.setCentralWidget(scr)
win.setWindowTitle('pyqtgraph example: Linearized color maps')
win.show()

bar_width = 32
bar_data = pg.colormap.modulatedBarData(width=bar_width)

num_bars = 0

def add_heading(lw, name):
    global num_bars
    lw.addLabel('=== '+name+' ===')
    num_bars += 1
    lw.nextRow()

def add_bar(lw, name, cm):
    global num_bars
    lw.addLabel(name)
    imi = pg.ImageItem( bar_data )
    imi.setLookupTable( cm.getLookupTable(alpha=True) )
    vb = lw.addViewBox(lockAspect=True, enableMouse=False)
    vb.addItem( imi )
    num_bars += 1
    lw.nextRow()

add_heading(lw, 'ramp generator')
for cm in ramp_list:
    add_bar(lw, cm.name, cm)

add_heading(lw, 'linearizer demonstration')
for (name, cm) in cm_list:
    add_bar(lw, name, cm)
    cm.linearize()
    add_bar(lw, '> linearized', cm)

add_heading(lw, 'consistency with included maps')
for name in ('CET-C3', 'CET-L17', 'CET-L2'):
    # lw.addLabel(str(name))
    cm = pg.colormap.get(name)
    add_bar(lw, name, cm)
    cm.linearize()
    add_bar(lw, '> linearized', cm)

lw.setFixedHeight(num_bars * (bar_width+5) )

if __name__ == '__main__':
    pg.exec()
