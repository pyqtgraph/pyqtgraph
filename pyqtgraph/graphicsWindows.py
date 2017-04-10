# -*- coding: utf-8 -*-
"""
graphicsWindows.py -  Convenience classes which create a new window with PlotWidget or ImageView.
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""

from .Qt import QtCore, QtGui
from .widgets.PlotWidget import *
from .imageview import *
from .widgets.GraphicsLayoutWidget import GraphicsLayoutWidget
from .widgets.GraphicsView import GraphicsView
QAPP = None

def mkQApp():
    if QtGui.QApplication.instance() is None:
        global QAPP
        QAPP = QtGui.QApplication([])


class GraphicsWindow(GraphicsLayoutWidget):
    """
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
        if hasattr(self.cw, attr):
            return getattr(self.cw, attr)
        else:
            raise NameError(attr)
    

class PlotWindow(PlotWidget):
    def __init__(self, title=None, **kargs):
        mkQApp()
        PlotWidget.__init__(self, **kargs)
        if title is not None:
            self.setWindowTitle(title)
        self.show()


class ImageWindow(ImageView):
    def __init__(self, *args, **kargs):
        mkQApp()
        ImageView.__init__(self)
        if 'title' in kargs:
            self.setWindowTitle(kargs['title'])
            del kargs['title']
        if len(args) > 0 or len(kargs) > 0:
            self.setImage(*args, **kargs)
        self.show()
