# -*- coding: utf-8 -*-
"""
This example demonstrates the use of ColorBarItem, which displays a simple interactive color bar.
"""
## Add path to library (just for examples; you do not need this)
import initExample

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

app = pg.mkQApp()

## Create window with ImageView widget
win = QtGui.QMainWindow()
win.resize(600,600)

gr_wid = pg.GraphicsLayoutWidget()
win.setCentralWidget( gr_wid)
win.show()
win.setWindowTitle('pyqtgraph example: Interactive color bar')

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

cmap = pg.colormap.get('CET-L9')
bar = pg.ColorBarItem( interactive=False, values= (0, 30_000), cmap=cmap )
bar.setImageItem( i1, insert_in=p1 ) # 

#--- add interactive image with integrated color ---------------------
i2 = pg.ImageItem(image=noisy_data)
p2 = gr_wid.addPlot(1,0, 1,1, title="interactive")
p2.addItem( i2, title='' )
# inserted color bar also works with labels on the right.
p2.showAxis('right')
p2.getAxis('left').setStyle( showValues=False )

cmap = pg.colormap.get('CET-L4')
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

cmap = pg.colormap.get('CET-L8')
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

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
