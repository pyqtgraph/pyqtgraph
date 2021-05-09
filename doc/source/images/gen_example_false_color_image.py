"""
generates 'example_false_color_image.png'
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

        plot = self.addPlot() # title="non-interactive")

        # prepare demonstration data:
        data = np.fromfunction(lambda i, j: (1+0.3*np.sin(i)) * (i)**2 + (j)**2, (100, 100))
        noisy_data = data * (1 + 0.2 * np.random.random(data.shape) )

        # Example: False color image with interactive level adjustment
        img = pg.ImageItem(image=noisy_data) # create monochrome image from demonstration data
        plot.addItem( img )            # add to PlotItem 'plot'
        cm = pg.colormap.get('CET-L9') # prepare a linear color map
        bar = pg.ColorBarItem( values= (0, 20_000), cmap=cm ) # prepare interactive color bar
        # Have ColorBarItem control colors of img and appear in 'plot':
        bar.setImageItem( img, insert_in=plot ) 

        self.timer = pg.QtCore.QTimer( singleShot=True )
        self.timer.timeout.connect(self.export)
        self.timer.start(100)
    
    def export(self):
        print('exporting')
        exporter = exp.ImageExporter(self.scene())
        exporter.parameters()['width'] = 420
        exporter.export('example_false_color_image.png')

mkQApp("False color image example")
main_window = MainWindow()

## Start Qt event loop
if __name__ == '__main__':
    mkQApp().exec_()
