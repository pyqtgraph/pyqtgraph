# -*- coding: utf-8 -*-
"""
This example demonstrates the creation of a plot with 
DateAxisItem and a customized ViewBox. 
"""


import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import time

class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        kwds['enableMenu'] = False
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)
        
    ## reimplement right-click to zoom out
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            self.autoRange()
    
    ## reimplement mouseDragEvent to disable continuous axis zoom
    def mouseDragEvent(self, ev, axis=None):
        if axis is not None and ev.button() == QtCore.Qt.RightButton:
            ev.ignore()
        else:
            pg.ViewBox.mouseDragEvent(self, ev, axis=axis)

class CustomTickSliderItem(pg.TickSliderItem):
    def __init__(self, *args, **kwds):
        pg.TickSliderItem.__init__(self, *args, **kwds)
        
        self.all_ticks = []
        self.visible_ticks = {}
        self._range = [0,1]
    
    def setTicks(self, ticks):
        for tick, pos in self.listTicks():
            self.removeTick(tick)
        self.visible_ticks = {}
        
        self.all_ticks = ticks
        
        self.updateRange(None, self._range)
    
    def updateRange(self, vb, viewRange):
        origin = self.tickSize/2.
        length = self.length

        lengthIncludingPadding = length + self.tickSize + 2
        
        self._range = viewRange
        
        for pos in self.all_ticks:
        
            if pos not in self.visible_ticks:
                self.visible_ticks[pos] = self.addTick(pos, movable=False, color="333333")
            
            tick = self.visible_ticks[pos]
            
            tickValueIncludingPadding = (pos - viewRange[0]) / (viewRange[1] - viewRange[0])
            tickValue = (tickValueIncludingPadding*lengthIncludingPadding - origin) / length
            
            visible = tickValue >= 0 and tickValue <= 1
            
            if visible:
                self.setTickValue(tick, tickValue)
            elif pos in self.visible_ticks:
                self.removeTick(self.visible_ticks[pos])
                del self.visible_ticks[pos]


app = pg.mkQApp()

axis = pg.DateAxisItem(orientation='bottom')
vb = CustomViewBox()

pw = pg.PlotWidget(viewBox=vb, axisItems={'bottom': axis}, enableMenu=False, title="PlotItem with DateAxisItem, custom ViewBox and markers on x axis<br>Menu disabled, mouse behavior changed: left-drag to zoom, right-click to reset zoom")

dates = np.arange(8) * (3600*24*356)
pw.plot(x=dates, y=[1,6,2,4,3,5,6,8], symbol='o')

# Using allowAdd and allowRemove to limit user interaction
tickViewer = CustomTickSliderItem(allowAdd=False, allowRemove=False)
vb.sigXRangeChanged.connect(tickViewer.updateRange)
pw.plotItem.layout.addItem(tickViewer, 4, 1)

tickViewer.setTicks( [dates[0], dates[2], dates[-1]] )

pw.show()
pw.setWindowTitle('pyqtgraph example: customPlot')

r = pg.PolyLineROI([(0,0), (10, 10)])
pw.addItem(r)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
