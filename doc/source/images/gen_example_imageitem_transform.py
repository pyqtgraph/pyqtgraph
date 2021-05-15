"""
generates 'example_false_color_image.png'
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters as exp
from pyqtgraph.Qt import QtGui, mkQApp

class MainWindow(pg.GraphicsLayoutWidget):
    """ example application main window """
    def __init__(self):
        super().__init__()
        self.resize(420,400)
        self.show()

        plot = self.addPlot()
        # Example: Transformed display of ImageItem

        tr = QtGui.QTransform()  # prepare ImageItem transformation:
        tr.scale(6.0, 3.0)       # scale horizontal and vertical axes
        tr.translate(-1.5, -1.5) # move 3x3 image to locate center at axis origin

        img = pg.ImageItem( image=np.eye(3), levels=(0,1) ) # create example image
        img.setTransform(tr) # assign transform

        plot.addItem( img )  # add ImageItem to PlotItem
        plot.showAxes(True)  # frame it with a full set of axes
        plot.invertY(True)   # vertical axis counts top to bottom

        self.timer = pg.QtCore.QTimer( singleShot=True )
        self.timer.timeout.connect(self.export)
        self.timer.start(100)
    
    def export(self):
        print('exporting')
        exporter = exp.ImageExporter(self.scene())
        exporter.parameters()['width'] = 420
        exporter.export('example_imageitem_transform.png')

mkQApp("ImageItem transform example")
main_window = MainWindow()

## Start Qt event loop
if __name__ == '__main__':
    mkQApp().exec_()
