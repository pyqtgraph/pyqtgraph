# -*- coding: utf-8 -*-
"""
PlotWidget.py -  Convenience class--GraphicsView widget displaying a single PlotItem
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""

from GraphicsView import *
from PlotItem import *
import exceptions

class PlotWidget(GraphicsView):
    """Widget implementing a graphicsView with a single PlotItem inside."""
    def __init__(self, parent=None):
        GraphicsView.__init__(self, parent)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.enableMouse(False)
        self.plotItem = PlotItem()
        self.setCentralItem(self.plotItem)
        ## Explicitly wrap methods from plotItem
        for m in ['addItem', 'autoRange', 'clear']:
            setattr(self, m, getattr(self.plotItem, m))
        QtCore.QObject.connect(self.plotItem, QtCore.SIGNAL('viewChanged'), self.viewChanged)
                
    def __getattr__(self, attr):  ## implicitly wrap methods from plotItem
        if hasattr(self.plotItem, attr):
            m = getattr(self.plotItem, attr)
            if hasattr(m, '__call__'):
                return m
        raise exceptions.NameError(attr)
            
    def viewChanged(self, *args):
        self.emit(QtCore.SIGNAL('viewChanged'), *args)

    def widgetGroupInterface(self):
        return (None, PlotWidget.saveState, PlotWidget.restoreState)

    def saveState(self):
        return self.plotItem.saveState()
        
    def restoreState(self, state):
        return self.plotItem.restoreState(state)
        
