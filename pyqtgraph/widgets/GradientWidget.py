# -*- coding: utf-8 -*-
from ..Qt import QtGui, QtCore, QtWidgets, QT_LIB
from .GraphicsView import GraphicsView
from ..graphicsItems.GradientEditorItem import GradientEditorItem
import weakref
import numpy as np

__all__ = ['GradientWidget']


class GradientWidget(GraphicsView):
    """
    Widget displaying an editable color gradient. The user may add, move, recolor,
    or remove colors from the gradient. Additionally, a context menu allows the 
    user to select from pre-defined gradients.
    """
    sigGradientChanged = QtCore.Signal(object)
    sigGradientChangeFinished = QtCore.Signal(object)
    
    def __init__(self, parent=None, orientation='bottom',  *args, **kargs):
        """
        The *orientation* argument may be 'bottom', 'top', 'left', or 'right' 
        indicating whether the gradient is displayed horizontally (top, bottom)
        or vertically (left, right) and on what side of the gradient the editable 
        ticks will appear.
        
        All other arguments are passed to 
        :func:`GradientEditorItem.__init__ <pyqtgraph.GradientEditorItem.__init__>`.
        
        Note: For convenience, this class wraps methods from 
        :class:`GradientEditorItem <pyqtgraph.GradientEditorItem>`.
        """
        GraphicsView.__init__(self, parent, useOpenGL=False, background=None)
        self.maxDim = 31
        kargs['tickPen'] = 'k'
        self.item = GradientEditorItem(*args, **kargs)
        self.item.sigGradientChanged.connect(self.sigGradientChanged)
        self.item.sigGradientChangeFinished.connect(self.sigGradientChangeFinished)
        self.setCentralItem(self.item)
        self.setOrientation(orientation)
        self.setCacheMode(self.CacheNone)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)

        if QT_LIB == 'PyQt6':
            # PyQt6 doesn't allow or-ing of different enum types
            # so we need to take its value property
            NoFrame = QtWidgets.QFrame.Shape.NoFrame.value
            Plain = QtWidgets.QFrame.Shadow.Plain.value
        else:
            NoFrame = QtWidgets.QFrame.NoFrame
            Plain = QtWidgets.QFrame.Plain
        frame_style = NoFrame | Plain

        self.setFrameStyle(frame_style)
        #self.setBackgroundRole(QtGui.QPalette.NoRole)
        #self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        #self.setAutoFillBackground(False)
        #self.setAttribute(QtCore.Qt.WA_PaintOnScreen, False)
        #self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def setOrientation(self, ort):
        """Set the orientation of the widget. May be one of 'bottom', 'top', 
        'left', or 'right'."""
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

    def widgetGroupInterface(self):
        return (self.sigGradientChanged, self.saveState, self.restoreState)


