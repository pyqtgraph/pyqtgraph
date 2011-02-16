# -*- coding: utf-8 -*-
### import all the goodies and add some helper functions for easy CLI use

from functions import *
from graphicsItems import *
from graphicsWindows import *
#import PlotWidget
#import ImageView
from PyQt4 import QtGui

plots = []
images = []
QAPP = None

def plot(*args, **kargs):
    mkQApp()
    if 'title' in kargs:
        w = PlotWindow(title=kargs['title'])
        del kargs['title']
    else:
        w = PlotWindow()
    w.plot(*args, **kargs)
    plots.append(w)
    w.show()
    return w
    
def show(*args, **kargs):
    mkQApp()
    w = ImageWindow(*args, **kargs)
    images.append(w)
    w.show()
    return w
    
def mkQApp():
    if QtGui.QApplication.instance() is None:
        global QAPP
        QAPP = QtGui.QApplication([])