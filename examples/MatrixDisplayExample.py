# -*- coding: utf-8 -*-
"""
This example demonstrates ViewBox and AxisItem configuration to plot a correlation matrix.
"""
## Add path to library (just for examples; you do not need this)
import initExample

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, mkQApp, QtGui

class MainWindow(QtWidgets.QMainWindow):
    """ example application main window """
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        gr_wid = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(gr_wid)
        self.setWindowTitle('pyqtgraph example: Correlation matrix display')
        self.resize(600,500)
        self.show()

        corrMatrix = np.array([
            [ 1.        ,  0.5184571 , -0.70188642],
            [ 0.5184571 ,  1.        , -0.86094096],
            [-0.70188642, -0.86094096,  1.        ]
        ])
        columns = ["A", "B", "C"]

        pg.setConfigOption('imageAxisOrder', 'row-major') # Switch default order to Row-major
        
        correlogram = pg.ImageItem()
        # create transform to center the corner element on the origin, for any assigned image:
        tr = QtGui.QTransform().translate(-0.5, -0.5) 
        correlogram.setTransform(tr)
        correlogram.setImage(corrMatrix)

        plotItem = gr_wid.addPlot()      # add PlotItem to the main GraphicsLayoutWidget
        plotItem.invertY(True)           # orient y axis to run top-to-bottom
        plotItem.setDefaultPadding(0.0)  # plot without padding data range
        plotItem.addItem(correlogram)    # display correlogram
        
        # show full frame, label tick marks at top and left sides, with some extra space for labels:
        plotItem.showAxes( True, showValues=(True, True, False, False), size=20 )

        # define major tick marks and labels:
        ticks = [ (idx, label) for idx, label in enumerate( columns ) ]
        for side in ('left','top','right','bottom'):
            plotItem.getAxis(side).setTicks( (ticks, []) ) # add list of major ticks; no minor ticks
        plotItem.getAxis('bottom').setHeight(10) # include some additional space at bottom of figure

        colorMap = pg.colormap.get("CET-D1")     # choose perceptually uniform, diverging color map
        # generate an adjustabled color bar, initially spanning -1 to 1:
        bar = pg.ColorBarItem( values=(-1,1), cmap=colorMap) 
        # link color bar and color map to correlogram, and show it in plotItem:
        bar.setImageItem(correlogram, insert_in=plotItem)    

mkQApp("Correlation matrix display")
main_window = MainWindow()

## Start Qt event loop
if __name__ == '__main__':
    mkQApp().exec_()
