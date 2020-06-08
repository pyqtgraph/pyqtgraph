# -*- coding: utf-8 -*-
"""
DEPRECATED:  The classes below are convenience classes that create a new window
containting a single, specific widget. These classes are now unnecessary because
it is possible to place any widget into its own window by simply calling its
show() method.
"""

from .Qt import QtCore, QtGui, mkQApp
from .widgets.PlotWidget import *
from .imageview import *
from .widgets.GraphicsLayoutWidget import GraphicsLayoutWidget
from .widgets.GraphicsView import GraphicsView


class GraphicsWindow(GraphicsLayoutWidget):
    """
    (deprecated; use :class:`~pyqtgraph.GraphicsLayoutWidget` instead)
    
    Convenience subclass of :class:`~pyqtgraph.GraphicsLayoutWidget`. This class
    is intended for use from the interactive python prompt.
    """
    def __init__(self, title=None, size=(800,600), **kargs):
        mkQApp()
        GraphicsLayoutWidget.__init__(self, **kargs)
        self.resize(*size)
        if title is not None:
            self.setWindowTitle(title)
        self.show()
        

class TabWindow(QtGui.QMainWindow):
    """
    (deprecated)
    """
    def __init__(self, title=None, size=(800,600)):
        mkQApp()
        QtGui.QMainWindow.__init__(self)
        self.resize(*size)
        self.cw = QtGui.QTabWidget()
        self.setCentralWidget(self.cw)
        if title is not None:
            self.setWindowTitle(title)
        self.show()
        
    def __getattr__(self, attr):
        return getattr(self.cw, attr)
    

class PlotWindow(PlotWidget):
    sigClosed = QtCore.Signal(object)

    """
    (deprecated; use :class:`~pyqtgraph.PlotWidget` instead)
    """
    def __init__(self, title=None, **kargs):
        mkQApp()
        PlotWidget.__init__(self, **kargs)
        if title is not None:
            self.setWindowTitle(title)
        self.show()

    def closeEvent(self, event):
        PlotWidget.closeEvent(self, event)
        self.sigClosed.emit(self)


class ImageWindow(ImageView):
    sigClosed = QtCore.Signal(object)

    """
    (deprecated; use :class:`~pyqtgraph.ImageView` instead)
    """
    def __init__(self, *args, **kargs):
        mkQApp()
        ImageView.__init__(self)
        if 'title' in kargs:
            self.setWindowTitle(kargs['title'])
            del kargs['title']
        if len(args) > 0 or len(kargs) > 0:
            self.setImage(*args, **kargs)
        self.show()

    def closeEvent(self, event):
        ImageView.closeEvent(self, event)
        self.sigClosed.emit(self)
