# -*- coding: utf-8 -*-
### import all the goodies and add some helper functions for easy CLI use

## 'Qt' is a local module; it is intended mainly to cover up the differences
## between PyQt4 and PySide.
from Qt import QtGui 

## not really safe.
#if QtGui.QApplication.instance() is None:
    #app = QtGui.QApplication([])

CONFIG_OPTIONS = {
    'leftButtonPan': True  ## if false, left button drags a rubber band for zooming in viewbox
}

def setConfigOption(opt, value):
    CONFIG_OPTIONS[opt] = value

def getConfigOption(opt):
    return CONFIG_OPTIONS[opt]

## Import almost everything to make it available from a single namespace
## don't import the more complex systems--canvas, parametertree, flowchart, dockarea
## these must be imported separately.

import os
def importAll(path):
    d = os.path.join(os.path.split(__file__)[0], path)
    files = []
    for f in os.listdir(d):
        if os.path.isdir(os.path.join(d, f)):
            files.append(f)
        elif f[-3:] == '.py' and f != '__init__.py':
            files.append(f[:-3])
        
    for modName in files:
        mod = __import__(path+"."+modName, globals(), locals(), fromlist=['*'])
        if hasattr(mod, '__all__'):
            names = mod.__all__
        else:
            names = [n for n in dir(mod) if n[0] != '_']
        for k in names:
            if hasattr(mod, k):
                globals()[k] = getattr(mod, k)

importAll('graphicsItems')
importAll('widgets')

from imageview import *
from WidgetGroup import *
from Point import Point
from Transform import Transform
from functions import *
from graphicsWindows import *
from SignalProxy import *




## Convenience functions for command-line use



plots = []
images = []
QAPP = None

def plot(*args, **kargs):
    """
    | Create and return a PlotWindow (this is just a window with PlotWidget inside), plot data in it.
    | Accepts a *title* argument to set the title of the window.
    | All other arguments are used to plot data. (see :func:`PlotItem.plot() <pyqtgraph.PlotItem.plot>`)
    """
    mkQApp()
    if 'title' in kargs:
        w = PlotWindow(title=kargs['title'])
        del kargs['title']
    else:
        w = PlotWindow()
    if len(args)+len(kargs) > 0:
        w.plot(*args, **kargs)
    plots.append(w)
    w.show()
    return w
    
def image(*args, **kargs):
    """
    | Create and return an ImageWindow (this is just a window with ImageView widget inside), show image data inside.
    | Will show 2D or 3D image data.
    | Accepts a *title* argument to set the title of the window.
    | All other arguments are used to show data. (see :func:`ImageView.setImage() <pyqtgraph.ImageView.setImage>`)
    """
    mkQApp()
    w = ImageWindow(*args, **kargs)
    images.append(w)
    w.show()
    return w
show = image  ## for backward compatibility
    
    
def mkQApp():
    if QtGui.QApplication.instance() is None:
        global QAPP
        QAPP = QtGui.QApplication([])