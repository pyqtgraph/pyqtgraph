# -*- coding: utf-8 -*-
"""
Example for using ColorBarItem.
"""

# local testing environment
# import os, sys
# LIBRARY_PATH = os.path.dirname( os.path.abspath(__file__) )
# LIBRARY_PATH = os.path.normpath( os.path.join(LIBRARY_PATH, '..','..') )
# sys.path.insert(0, LIBRARY_PATH)

# import dev_pyqtgraph.pyqtgraph as pg
# from dev_pyqtgraph.pyqtgraph.Qt import QtCore, QtWidgets

## Add path to library (just for examples; you do not need this)
import initExample

from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

import sys
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.resize( QtCore.QSize( 600, 600 ) )
        gr_wid = pg.GraphicsLayoutWidget(self)
        self.setCentralWidget( gr_wid)

        ## Create image items
        data = np.fromfunction(lambda i, j: (1+0.3*np.sin(i)) * (i)**2 + (j)**2, (100, 100))
        noisy_data = data * (1 + 0.2 * np.random.random(data.shape) )
        noisy_transposed = noisy_data.transpose()
        
        #--- add non-interactive image with integrated color -----------------
        i1 = pg.ImageItem(image=data)
        p1 = gr_wid.addPlot(title="non-interactive")
        p1.addItem( i1 )
        p1.setMouseEnabled( x=False, y=False)
        p1.disableAutoRange()
        p1.hideButtons()
        p1.setRange(xRange=(0,100), yRange=(0,100), padding=0.01)
        for key in ['right','top']:
            p1.showAxis(key)
            p1.getAxis(key).setStyle( showValues=False )

        cmap = pg.colormap.get( 'bgyw', source='colorcet' )
        bar = pg.ColorBarItem( interactive=False, values= (0, 30_000), cmap=cmap )
        bar.setImageItem( i1, insert_in=p1 ) # 

        #--- add interactive image with integrated color ---------------------
        i2 = pg.ImageItem(image=noisy_data)
        p2 = gr_wid.addPlot(1,0, 1,1, title="interactive")
        p2.addItem( i2, title='' )
        # inserted color bar also works with labels on the right.
        p2.showAxis('right')
        p2.getAxis('left').setStyle( showValues=False )

        cmap = pg.colormap.get( 'fire', source='colorcet' )
        bar = pg.ColorBarItem( 
            values = (0, 30_000),
            limits = (0, None), 
            rounding=1000, 
            pen = '#8888FF',
            cmap=cmap )
        bar.setImageItem( i2, insert_in=p2 )

        #--- multiple images adjusted by a separate color bar ----------------
        i3 = pg.ImageItem(image=noisy_data)
        p3 = gr_wid.addPlot(0,1, 1,1, title="shared 1")
        p3.addItem( i3 )

        i4 = pg.ImageItem(image=noisy_transposed)
        p4 = gr_wid.addPlot(1,1, 1,1, title="shared 2")
        p4.addItem( i4 )
        
        cmap = pg.colormap.get( 'bmy', source='colorcet' )
        bar = pg.ColorBarItem( 
            # values = (-15_000, 15_000),
            limits = (-30_000, 30_000), 
            rounding=1000, 
            width = 10,
            cmap=cmap )
        bar.setImageItem( [i3, i4] )
        bar.setLevels( low=-15_000, high=15_000)

        # manually adjust reserved space at top and bottom to align with plot
        bar.getAxis('bottom').setHeight(21)
        bar.getAxis('top').setHeight(31)
        gr_wid.addItem(bar, 0,2, 2,1) # large bar spanning both rows
        self.show()

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
