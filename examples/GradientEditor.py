# -*- coding: utf-8 -*-
## Add path to library (just for examples; you do not need this)                                                                           
import initExample

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


app = pg.mkQApp("Gradiant Editor Example")
mw = pg.GraphicsView()
mw.resize(800,800)
mw.show()

#ts = pg.TickSliderItem()
#mw.setCentralItem(ts)
#ts.addTick(0.5, 'r')
#ts.addTick(0.9, 'b')

ge = pg.GradientEditorItem()
mw.setCentralItem(ge)

if __name__ == '__main__':
    pg.Qt.QtWidgets.QApplication.exec_()
