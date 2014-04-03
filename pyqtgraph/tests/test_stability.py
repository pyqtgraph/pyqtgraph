"""
PyQt/PySide stress test:

Create lots of random widgets and graphics items, connect them together randomly,
the tear them down repeatedly. 

The purpose of this is to attempt to generate segmentation faults.
"""
import pyqtgraph as pg
import random

random.seed(12345)

widgetTypes = [pg.PlotWidget, pg.ImageView, pg.GraphicsView, pg.QtGui.QWidget,
               pg.QtGui.QTreeWidget, pg.QtGui.QPushButton]

itemTypes = [pg.PlotCurveItem, pg.ImageItem, pg.PlotDataItem, pg.ViewBox,
             pg.QtGui.QGraphicsRectItem]

while True:
    action = random.randint(0,5)
    if action == 0:
        # create a widget
        pass
    elif action == 1:
        # set parent (widget or None), possibly create a reference in either direction
        pass
    elif action == 2:
        # 
        pass
    elif action == 3:
        pass
        






