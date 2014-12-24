# -*- coding: utf-8 -*-
"""
Demonstrates adding a custom context menu to a GraphicsItem
and extending the context menu of a ViewBox.

PyQtGraph implements a system that allows each item in a scene to implement its 
own context menu, and for the menus of its parent items to be automatically 
displayed as well. 

"""
import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

win = pg.GraphicsWindow()
win.setWindowTitle('pyqtgraph example: context menu')


view = win.addViewBox()

# add two new actions to the ViewBox context menu:
zoom1 = view.menu.addAction('Zoom to box 1')
zoom2 = view.menu.addAction('Zoom to box 2')

# define callbacks for these actions
def zoomTo1():
    # note that box1 is defined below
    view.autoRange(items=[box1])
zoom1.triggered.connect(zoomTo1)

def zoomTo2():
    # note that box1 is defined below
    view.autoRange(items=[box2])
zoom2.triggered.connect(zoomTo2)



class MenuBox(pg.GraphicsObject):
    """
    This class draws a rectangular area. Right-clicking inside the area will
    raise a custom context menu which also includes the context menus of
    its parents.    
    """
    def __init__(self, name):
        self.name = name
        self.pen = pg.mkPen('r')
        
        # menu creation is deferred because it is expensive and often
        # the user will never see the menu anyway.
        self.menu = None
        
        # note that the use of super() is often avoided because Qt does not 
        # allow to inherit from multiple QObject subclasses.
        pg.GraphicsObject.__init__(self) 

    
    # All graphics items must have paint() and boundingRect() defined.
    def boundingRect(self):
        return QtCore.QRectF(0, 0, 10, 10)
    
    def paint(self, p, *args):
        p.setPen(self.pen)
        p.drawRect(self.boundingRect())
    
    
    # On right-click, raise the context menu
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()

    def raiseContextMenu(self, ev):
        menu = self.getContextMenus()
        
        # Let the scene add on to the end of our context menu
        # (this is optional)
        menu = self.scene().addParentContextMenus(self, menu, ev)
        
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))
        return True

    # This method will be called when this item's _children_ want to raise
    # a context menu that includes their parents' menus.
    def getContextMenus(self, event=None):
        if self.menu is None:
            self.menu = QtGui.QMenu()
            self.menu.setTitle(self.name+ " options..")
            
            green = QtGui.QAction("Turn green", self.menu)
            green.triggered.connect(self.setGreen)
            self.menu.addAction(green)
            self.menu.green = green
            
            blue = QtGui.QAction("Turn blue", self.menu)
            blue.triggered.connect(self.setBlue)
            self.menu.addAction(blue)
            self.menu.green = blue
            
            alpha = QtGui.QWidgetAction(self.menu)
            alphaSlider = QtGui.QSlider()
            alphaSlider.setOrientation(QtCore.Qt.Horizontal)
            alphaSlider.setMaximum(255)
            alphaSlider.setValue(255)
            alphaSlider.valueChanged.connect(self.setAlpha)
            alpha.setDefaultWidget(alphaSlider)
            self.menu.addAction(alpha)
            self.menu.alpha = alpha
            self.menu.alphaSlider = alphaSlider
        return self.menu

    # Define context menu callbacks
    def setGreen(self):
        self.pen = pg.mkPen('g')
        # inform Qt that this item must be redrawn.
        self.update()

    def setBlue(self):
        self.pen = pg.mkPen('b')
        self.update()

    def setAlpha(self, a):
        self.setOpacity(a/255.)


# This box's context menu will include the ViewBox's menu
box1 = MenuBox("Menu Box #1")
view.addItem(box1)

# This box's context menu will include both the ViewBox's menu and box1's menu
box2 = MenuBox("Menu Box #2")
box2.setParentItem(box1)
box2.setPos(5, 5)
box2.scale(0.2, 0.2)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
