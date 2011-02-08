# -*- coding: utf-8 -*-
"""
graphicsWindows.py -  Convenience classes which create a new window with PlotWidget or ImageView.
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""

from PyQt4 import QtCore, QtGui
from PlotWidget import *
from ImageView import *
QAPP = None

def mkQApp():
    if QtGui.QApplication.instance() is None:
        global QAPP
        QAPP = QtGui.QApplication([])

class GraphicsLayoutWidget(GraphicsView):
    def __init__(self):
        GraphicsView.__init__(self)
        self.items = {}
        self.currentRow = 0
        self.currentCol = 0
    
    def nextRow(self):
        """Advance to next row for automatic item placement"""
        self.currentRow += 1
        self.currentCol = 0
        
    def nextCol(self, colspan=1):
        """Advance to next column, while returning the current column number 
        (generally only for internal use)"""
        self.currentCol += colspan
        return self.currentCol-colspan
        
    def addPlot(self, row=None, col=None, rowspan=1, colspan=1, **kargs):
        plot = PlotItem(**kargs)
        self.addItem(plot, row, col, rowspan, colspan)
        return plot

    def addItem(self, item, row=None, col=None, rowspan=1, colspan=1):
        if row not in self.items:
            self.items[row] = {}
        self.items[row][col] = item
        
        if row is None:
            row = self.currentRow
        if col is None:
            col = self.nextCol(colspan)
        self.centralLayout.addItem(item, row, col, rowspan, colspan)

    def getItem(self, row, col):
        return self.items[row][col]


class GraphicsWindow(GraphicsLayoutWidget):
    def __init__(self, title=None, size=(800,600)):
        mkQApp()
        self.win = QtGui.QMainWindow()
        GraphicsLayoutWidget.__init__(self)
        self.win.setCentralWidget(self)
        self.win.resize(*size)
        if title is not None:
            self.win.setWindowTitle(title)
        self.win.show()
        

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
    

#class PlotWindow(QtGui.QMainWindow):
    #def __init__(self, title=None, **kargs):
        #mkQApp()
        #QtGui.QMainWindow.__init__(self)
        #self.cw = PlotWidget(**kargs)
        #self.setCentralWidget(self.cw)
        #for m in ['plot', 'autoRange', 'addItem', 'removeItem', 'setLabel', 'clear', 'viewRect']:
            #setattr(self, m, getattr(self.cw, m))
        #if title is not None:
            #self.setWindowTitle(title)
        #self.show()


class PlotWindow(PlotWidget):
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
    def __init__(self, *args, **kargs):
        mkQApp()
        self.win = QtGui.QMainWindow()
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
