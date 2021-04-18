# -*- coding: utf-8 -*-
"""
This example demonstrates ViewBox and AxisItem configuration to plot a correlation matrix.
"""
## Add path to library (just for examples; you do not need this)
import initExample

import numpy as np
import pyqtgraph as pg
import numpy as np

corrMatrix = np.array([
    [ 1.        ,  0.5184571 , -0.70188642],
    [ 0.5184571 ,  1.        , -0.86094096],
    [-0.70188642, -0.86094096,  1.        ]
])
columns = ["A", "B", "C"]

app = pg.mkQApp()
win = pg.PlotWidget()

# create correlation matrix image with correct orientation:
correlogram = pg.ImageItem(image=corrMatrix, axisOrder='row-major')

plotItem = win.getPlotItem()        # get PlotItem of the main PlotWidget
plotItem.getViewBox().invertY(True) # orient y axis to run top-to-bottom
plotItem.setDefaultPadding(0.0)     # plot without padding data range
plotItem.addItem(correlogram)       # display correlogram
# show full frame, label tick marks at top and left sides, with some extra space for labels
plotItem.showAxes( True, showValues=(True, True, False, False), size=20 )

# define major tick marks and labels:
ticks = [ (idx+0.5, label) for idx, label in enumerate( columns ) ]
for side in ('left','top','right','bottom'):
    plotItem.getAxis(side).setTicks( (ticks, []) ) # add list of major ticks and no minor ticks
plotItem.getAxis('bottom').setHeight(10) # include some additional space at bottom of figure

colorMap = pg.colormap.get("CET-D1")     # choose a perceptually uniform, diverging color map
bar = pg.ColorBarItem( values=(-1,1), cmap=colorMap) # generate an adjustabled color bar, initially spanning -1 to 1
bar.setImageItem(correlogram, insert_in=plotItem)    # link color bar and color map to correlogram, and show it in plotItem

win.show()
app.exec_()
