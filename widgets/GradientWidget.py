# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtGui, QtCore
from .GraphicsView import GraphicsView
from pyqtgraph.graphicsItems.GradientEditorItem import GradientEditorItem
import weakref
import numpy as np

__all__ = ['TickSlider', 'GradientWidget', 'BlackWhiteSlider']


class GradientWidget(GraphicsView):
    
    sigGradientChanged = QtCore.Signal(object)
    
    def __init__(self, parent=None, orientation='bottom',  *args, **kargs):
        GraphicsView.__init__(self, parent, useOpenGL=False, background=None)
        self.maxDim = 31
        kargs['tickPen'] = 'k'
        self.item = GradientEditorItem(*args, **kargs)
        self.item.sigGradientChanged.connect(self.sigGradientChanged)
        self.setCentralItem(self.item)
        self.setOrientation(orientation)
        self.setCacheMode(self.CacheNone)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setFrameStyle(QtGui.QFrame.NoFrame | QtGui.QFrame.Plain)
        #self.setBackgroundRole(QtGui.QPalette.NoRole)
        #self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        #self.setAutoFillBackground(False)
        #self.setAttribute(QtCore.Qt.WA_PaintOnScreen, False)
        #self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def setOrientation(self, ort):
        self.item.setOrientation(ort)
        self.orientation = ort
        self.setMaxDim()
        
    def setMaxDim(self, mx=None):
        if mx is None:
            mx = self.maxDim
        else:
            self.maxDim = mx
            
        if self.orientation in ['bottom', 'top']:
            self.setFixedHeight(mx)
            self.setMaximumWidth(16777215)
        else:
            self.setFixedWidth(mx)
            self.setMaximumHeight(16777215)
        
    def __getattr__(self, attr):
        ### wrap methods from GradientEditorItem
        return getattr(self.item, attr)


