# -*- coding: utf-8 -*-
"""
MultiPlotWidget.py -  Convenience class--GraphicsView widget displaying a MultiPlotItem
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""
from ..Qt import QtCore
from .GraphicsView import GraphicsView
from ..graphicsItems import MultiPlotItem as MultiPlotItem

__all__ = ['MultiPlotWidget']
class MultiPlotWidget(GraphicsView):
    """Widget implementing a graphicsView with a single MultiPlotItem inside."""
    def __init__(self, parent=None):
        self.minPlotHeight = 150
        self.mPlotItem = MultiPlotItem.MultiPlotItem()
        GraphicsView.__init__(self, parent)
        self.enableMouse(False)
        self.setCentralItem(self.mPlotItem)
        ## Explicitly wrap methods from mPlotItem
        #for m in ['setData']:
            #setattr(self, m, getattr(self.mPlotItem, m))
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                
    def __getattr__(self, attr):  ## implicitly wrap methods from plotItem
        if hasattr(self.mPlotItem, attr):
            m = getattr(self.mPlotItem, attr)
            if hasattr(m, '__call__'):
                return m
        raise AttributeError(attr)
    

    def widgetGroupInterface(self):
        return (None, MultiPlotWidget.saveState, MultiPlotWidget.restoreState)

    def saveState(self):
        return {}
        #return self.plotItem.saveState()
        
    def restoreState(self, state):
        pass
        #return self.plotItem.restoreState(state)

    def close(self):
        self.mPlotItem.close()
        self.mPlotItem = None
        self.setParent(None)
        GraphicsView.close(self)

    def setRange(self, *args, **kwds):
        GraphicsView.setRange(self, *args, **kwds)
        if self.centralWidget is not None:
            r = self.range
            minHeight = len(self.mPlotItem.plots) * self.minPlotHeight
            if r.height() < minHeight:
                r.setHeight(minHeight)
                r.setWidth(r.width() - 25)
            self.centralWidget.setGeometry(r)

    def resizeEvent(self, ev):
        if self.closed:
            return
        if self.autoPixelRange:
            self.range = QtCore.QRectF(0, 0, self.size().width(), self.size().height())
        MultiPlotWidget.setRange(self, self.range, padding=0, disableAutoPixel=False)  ## we do this because some subclasses like to redefine setRange in an incompatible way.
        self.updateMatrix()
