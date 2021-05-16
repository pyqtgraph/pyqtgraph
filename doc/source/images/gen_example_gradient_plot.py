"""
generates 'example_gradient_plot.png'
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters as exp
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

class MainWindow(pg.GraphicsLayoutWidget):
    """ example application main window """
    def __init__(self):
        super().__init__()
        self.resize(420,400)
        self.show()

        # Prepare demonstration data
        raw = np.linspace(0.0, 2.0, 400)
        y_data1 = ( (raw+0.1)%1 ) ** 4
        y_data2 = ( (raw+0.1)%1 ) ** 4 - ( (raw+0.6)%1 ) ** 4

        # Example 1: Gradient pen
        cm = pg.colormap.get('CET-L17') # prepare a linear color map
        cm.reverse()                    # reverse it to put light colors at the top 
        pen = cm.getPen( span=(0.0,1.0), width=5 ) # gradient from blue (y=0) to white (y=1)
        # plot a curve drawn with a pen colored according to y value:
        curve1 = pg.PlotDataItem( y=y_data1, pen=pen )

        # Example 2: Gradient brush
        cm = pg.colormap.get('CET-D1') # prepare a diverging color map
        cm.setMappingMode('diverging') # set mapping mode
        brush = cm.getBrush( span=(-1., 1.) ) # gradient from blue at -1 to red at +1
        # plot a curve that is filled to zero with the gradient brush:
        curve2 = pg.PlotDataItem( y=y_data2, pen='w', brush=brush, fillLevel=0.0 )

        for idx, curve in enumerate( (curve1, curve2) ):
            plot = self.addPlot(row=idx, col=0)
            plot.getAxis('left').setWidth(25)
            plot.addItem( curve )
            
        self.timer = pg.QtCore.QTimer( singleShot=True )
        self.timer.timeout.connect(self.export)
        self.timer.start(100)
    
    def export(self):
        print('exporting')
        exporter = exp.ImageExporter(self.scene())
        exporter.parameters()['width'] = 420
        exporter.export('example_gradient_plot.png')

mkQApp("Gradient plotting example")
main_window = MainWindow()

## Start Qt event loop
if __name__ == '__main__':
    mkQApp().exec_()
