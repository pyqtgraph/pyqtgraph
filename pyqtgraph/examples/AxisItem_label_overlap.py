"""
This example demonstrates many of the 2D plotting capabilities
in pyqtgraph. All of the plots may be panned/scaled by dragging with 
the left/right mouse buttons. Right click on any plot to show a context menu.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

app = pg.mkQApp("AxisItem - label overlap settings")

win = pg.GraphicsLayoutWidget(show=True, title="AxisItem - label overlap settings")
win.resize(800,600)
win.setWindowTitle("AxisItem - label overlap settings")

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

x_data = np.arange(101) + 1_000.
y_data = np.random.normal(scale=40., size=101)

font = QtGui.QFont()
font.setPointSize(14) # A larger font makes the effects more visible

p1 = win.addPlot(title="Default settings: Overlap allowed for y axis", x=x_data, y=y_data)
for axis_key in ('top', 'bottom', 'left', 'right'):
    ax = p1.getAxis(axis_key)
    ax.setTickFont( font )

p2 = win.addPlot(title="Overlap allowed for X axis", x=x_data, y=y_data)
for axis_key, hide_overlap in (
    ('top'   , False), 
    ('bottom', False),
    ('left'  , True ),
    ('right' , True )
):
    ax = p2.getAxis(axis_key)
    ax.setStyle( hideOverlappingLabels = hide_overlap )
    ax.setTickFont( font )

win.nextRow()

p3 = win.addPlot(title="All overlap disabled", x=x_data, y=y_data)
for axis_key in ('top', 'bottom', 'left', 'right'):
    ax = p3.getAxis(axis_key)
    ax.setStyle( hideOverlappingLabels = True )
    ax.setTickFont( font )

p4 = win.addPlot(title="All overlap enabled, custom tolerances", x=x_data, y=y_data)
for axis_key, tolerance in (
    ('top'   ,  15 ),
    ('bottom', 200 ),
    ('left'  , 100 ),
    ('right' ,  15 )
):
    ax = p4.getAxis(axis_key)
    ax.setStyle( hideOverlappingLabels = tolerance )
    ax.setTickFont( font )

# Link all axes and set viewing range with no padding:
for p in (p1, p2, p3, p4):
    p.showAxes(True, showValues=(True, True, True, True))
    if p != p1:
        p.setXLink(p1)
        p.setYLink(p1)
    ax.setTickFont( font )
p1.setXRange( 1_000., 1_100., padding=0.0)
p1.setYRange(-60.,  60., padding=0.0)


if __name__ == '__main__':
    pg.exec()
