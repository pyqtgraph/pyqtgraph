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
class PlotWindow(QtGui.QMainWindow):
    def __init__(self, title=None):
        if QtGui.QApplication.instance() is None:
            global QAPP
            QAPP = QtGui.QApplication([])
        QtGui.QMainWindow.__init__(self)
        self.cw = PlotWidget()
        self.setCentralWidget(self.cw)
        for m in ['plot', 'autoRange', 'addItem', 'removeItem', 'setLabel', 'clear']:
            setattr(self, m, getattr(self.cw, m))
        if title is not None:
            self.setWindowTitle(title)
        self.show()

class ImageWindow(QtGui.QMainWindow):
    def __init__(self, title=None):
        if QtGui.QApplication.instance() is None:
            global QAPP
            QAPP = QtGui.QApplication([])
        QtGui.QMainWindow.__init__(self)
        self.cw = ImageView()
        self.setCentralWidget(self.cw)
        for m in ['setImage', 'autoRange', 'addItem', 'removeItem', 'blackLevel', 'whiteLevel', 'imageItem']:
            setattr(self, m, getattr(self.cw, m))
        if title is not None:
            self.setWindowTitle(title)
        self.show()
