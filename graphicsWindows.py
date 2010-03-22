# -*- coding: utf-8 -*-
"""
graphicsWindows.py -  Convenience classes which create a new window with PlotWidget or ImageView.
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""

from PyQt4 import QtCore, QtGui
from PlotWidget import *
from ImageView import *

class PlotWindow(QtGui.QMainWindow):
    def __init__(self, title=None):
        QtGui.QMainWindow.__init__(self)
        self.cw = PlotWidget()
        self.setCentralWidget(self.cw)
        for m in ['plot', 'autoRange', 'addItem', 'setLabel', 'clear']:
            setattr(self, m, getattr(self.cw, m))
        if title is not None:
            self.setWindowTitle(title)
        self.show()

class ImageWindow(QtGui.QMainWindow):
    def __init__(self, title=None):
        QtGui.QMainWindow.__init__(self)
        self.cw = ImageView()
        self.setCentralWidget(self.cw)
        for m in ['setImage', 'autoRange', 'addItem']:
            setattr(self, m, getattr(self.cw, m))
        if title is not None:
            self.setWindowTitle(title)
        self.show()
