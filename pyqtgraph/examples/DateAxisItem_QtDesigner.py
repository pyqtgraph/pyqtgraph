"""
Demonstrates the usage of DateAxisItem in a layout created with Qt Designer.

The spotlight here is on the 'setAxisItems' method, without which
one would have to subclass plotWidget in order to attach a dateaxis to it.
"""

import os
import time

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, loadUiType

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

BLUE = pg.mkPen('#1f77b4')

path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'DateAxisItem_QtDesigner.ui')
Design, _ = loadUiType(uiFile)

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

app = pg.mkQApp("DateAxisItem_QtDesigner Example")
window = ExampleApp()
window.setWindowTitle('pyqtgraph example: DateAxisItem_QtDesigner')
window.show()

if __name__ == '__main__':
    pg.exec()
