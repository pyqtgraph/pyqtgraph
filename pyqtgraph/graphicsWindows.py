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
    (deprecated; use GraphicsLayoutWidget instead)
    
    Convenience subclass of :class:`GraphicsLayoutWidget 
    <pyqtgraph.GraphicsLayoutWidget>`. This class is intended for use from 
    the interactive python prompt.
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
    """
    (deprecated; use PlotWidget instead)
    """
    def __init__(self, title=None, **kargs):
        mkQApp()
        self.win = QtGui.QMainWindow()
        PlotWidget.__init__(self, **kargs)
        self.win.setCentralWidget(self)
        for m in ['resize']:
            setattr(self, m, getattr(self.win, m))
        if title is not None:
            self.win.setWindowTitle(title)
        self.win.show()


class ImageWindow(ImageView):
    """
    (deprecated; use ImageView instead)
    """
    def __init__(self, *args, **kargs):
        mkQApp()
        self.win = QtGui.QMainWindow()
        self.win.resize(800,600)
        if 'title' in kargs:
            self.win.setWindowTitle(kargs['title'])
            del kargs['title']
        ImageView.__init__(self, self.win)
        if len(args) > 0 or len(kargs) > 0:
            self.setImage(*args, **kargs)
        self.win.setCentralWidget(self)
        for m in ['resize']:
            setattr(self, m, getattr(self.win, m))
        #for m in ['setImage', 'autoRange', 'addItem', 'removeItem', 'blackLevel', 'whiteLevel', 'imageItem']:
            #setattr(self, m, getattr(self.cw, m))
        self.win.show()
