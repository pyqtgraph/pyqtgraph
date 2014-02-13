# -*- coding: utf-8 -*-
"""
BarWidget.py -  Convenience class--GraphicsView widget displaying a single BarGraphItem
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""
from ..Qt import QtCore, QtGui
from .GraphicsView import *
from ..graphicsItems.BarGraphItem import BarGraphItem

__all__ = ['BarWidget']


class BarWidget(GraphicsView):

    # signals wrapped from BarGraphItem
    sigRangeChanged = QtCore.Signal(object)
    sigTransformChanged = QtCore.Signal(object)

    def __init__(self, parent=None, background='default', *args, **kargs):
        """When initializing BarWidget, *parent* and *background* are passed to
        :func:`GraphicsWidget.__init__() <pyqtgraph.GraphicsWidget.__init__>`
        and all others are passed
        to :func:`BarGraphItem.__init__() <pyqtgraph.BarGraphItem.__init__>`."""
        GraphicsView.__init__(self, parent, background=background)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.enableMouse(False)
        self.item = BarGraphItem(**kargs)
        self.setCacheMode(self.CacheNone)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setFrameStyle(QtGui.QFrame.NoFrame | QtGui.QFrame.Plain)

    def __getattr__(self, attr):
        ### wrap methods from BarGraphItem
        return getattr(self.item, attr)


