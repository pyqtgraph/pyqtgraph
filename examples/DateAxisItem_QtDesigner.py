"""
Demonstrates the usage of DateAxisItem in a layout created with Qt Designer.

The spotlight here is on the 'setAxisItems' method, without which
one would have to subclass plotWidget in order to attach a dateaxis to it.

"""
import initExample ## Add path to library (just for examples; you do not need this)

import sys
import time

import numpy as np
from PyQt5 import QtWidgets, QtCore, uic
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

BLUE = pg.mkPen('#1f77b4')

Design, _ = uic.loadUiType('DateAxisItem_QtDesigner.ui')

class ExampleApp(QtWidgets.QMainWindow, Design):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        now = time.time()
        # Plot random values with timestamps in the last 6 months
        timestamps = np.linspace(now - 6*30*24*3600, now, 100)
        self.curve = self.plotWidget.plot(x=timestamps, y=np.random.rand(100), 
                                          symbol='o', symbolSize=5, pen=BLUE)
        # 'o' circle  't' triangle  'd' diamond  '+' plus  's' square
        self.plotWidget.setAxisItems({'bottom': pg.DateAxisItem()})
        self.plotWidget.showGrid(x=True, y=True)

app = QtWidgets.QApplication(sys.argv)
app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
app.setPalette(QtWidgets.QApplication.style().standardPalette())
window = ExampleApp()
window.show()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app.exec_()
