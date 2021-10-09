# -*- coding: utf-8 -*-
"""
Demonstrates a way to put multiple axes around a single plot.
"""
import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg

pg.mkQApp()

# Create a new plot as normal
pw = pg.PlotWidget()
pw.show()
pw.setWindowTitle('pyqtgraph example: MultiplePlotAxes')
p1 = pw.plotItem
p1.setLabels(left='axis 1')
p1.getViewBox().setMouseMode(p1.vb.RectMode)
p1.plot([1, 2, 4, 8, 16, 32])

# Now create a couple of additional axes
ax2 = pg.AxisItem('right')
ax2.setLabel('axis2', color='#0000ff')
ax3 = pg.AxisItem('right')
ax3.setLabel('axis 3', color='#ff0000')

# Add axis 2 to the plot and associated a data curve with it at the same time
p1.addAxis(ax2, 'right1', pg.PlotCurveItem([10, 20, 40, 80, 40, 20], pen='b'))

# An example of linking a data curve to an axis which already exists on the plot
p1.addAxis(ax3, 'right2')
p1.linkDataToAxis(pg.PlotCurveItem([3200, 1600, 800, 400, 200, 100], pen='r'), 'right2')

if __name__ == '__main__':
    pg.exec()
