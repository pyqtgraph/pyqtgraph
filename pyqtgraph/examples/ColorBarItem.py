"""
This example demonstrates the use of ColorBarItem, which displays a simple interactive color bar.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, mkQApp


class MainWindow(QtWidgets.QMainWindow):
    """ example application main window """
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        gr_wid = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(gr_wid)
        self.setWindowTitle('pyqtgraph example: Interactive color bar')
        self.resize(800,700)
        self.show()

        ## Create image items
        data = np.fromfunction(lambda i, j: (1+0.3*np.sin(i)) * (i)**2 + (j)**2, (100, 100))
        noisy_data = data * (1 + 0.2 * np.random.random(data.shape) )
        noisy_transposed = noisy_data.transpose()

        #--- add non-interactive image with integrated color ------------------
        p1 = gr_wid.addPlot(title="non-interactive")
        # Basic steps to create a false color image with color bar:
        i1 = pg.ImageItem(image=data)
        p1.addItem( i1 )
        p1.addColorBar( i1, colorMap='CET-L9', values=(0, 30_000)) # , interactive=False)
        #
        p1.setMouseEnabled( x=False, y=False)
        p1.disableAutoRange()
        p1.hideButtons()
        p1.setRange(xRange=(0,100), yRange=(0,100), padding=0)
        p1.showAxes(True, showValues=(True,False,False,True) )

        #--- add interactive image with integrated horizontal color bar -------
        i2 = pg.ImageItem(image=noisy_data)
        p2 = gr_wid.addPlot(1,0, 1,1, title="interactive")
        p2.addItem( i2, title='' )
        # inserted color bar also works with labels on the right.
        p2.showAxis('right')
        p2.getAxis('left').setStyle( showValues=False )
        p2.getAxis('bottom').setLabel('bottom axis label')
        p2.getAxis('right').setLabel('right axis label')

        bar = pg.ColorBarItem(
            values = (0, 30_000),
            colorMap='CET-L4',
            label='horizontal color bar',
            limits = (0, None),
            rounding=1000,
            orientation = 'h',
            pen='#8888FF', hoverPen='#EEEEFF', hoverBrush='#EEEEFF80'
        )
        bar.setImageItem( i2, insert_in=p2 )

        #--- multiple images adjusted by a separate color bar ------------------------
        i3 = pg.ImageItem(image=noisy_data)
        p3 = gr_wid.addPlot(0,1, 1,1, title="shared 1")
        p3.addItem( i3 )

        i4 = pg.ImageItem(image=noisy_transposed)
        p4 = gr_wid.addPlot(1,1, 1,1, title="shared 2")
        p4.addItem( i4 )

        cmap = pg.colormap.get('CET-L8')
        bar = pg.ColorBarItem(
            # values = (-15_000, 15_000),
            limits = (-30_000, 30_000), # start with full range...
            rounding=1000,
            width = 10,
            colorMap=cmap )
        bar.setImageItem( [i3, i4] )
        bar.setLevels( low=-5_000, high=15_000) # ... then adjust to retro sunset.

        # manually adjust reserved space at top and bottom to align with plot
        bar.getAxis('bottom').setHeight(21)
        bar.getAxis('top').setHeight(31)
        gr_wid.addItem(bar, 0,2, 2,1) # large bar spanning both rows

mkQApp("ColorBarItem Example")
main_window = MainWindow()

## Start Qt event loop
if __name__ == '__main__':
    pg.exec()
