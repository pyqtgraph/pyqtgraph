# -*- coding: utf-8 -*-
from ..Qt import QtGui, QtCore
from ..widgets.GraphicsView import GraphicsView
from ..GraphicsScene import GraphicsScene
from ..graphicsItems.ViewBox import ViewBox

translate = QtCore.QCoreApplication.translate

class FlowchartGraphicsView(GraphicsView):
    
    sigHoverOver = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object)
    
    def __init__(self, widget, *args):
        GraphicsView.__init__(self, *args, useOpenGL=False)
        self._vb = FlowchartViewBox(widget, lockAspect=True, invertY=True)
        self.setCentralItem(self._vb)
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
    
    def viewBox(self):
        return self._vb
    
        
class FlowchartViewBox(ViewBox):
    
    def __init__(self, widget, *args, **kwargs):
        ViewBox.__init__(self, *args, **kwargs)
        self.widget = widget
        
    def getMenu(self, ev):
        ## called by ViewBox to create a new context menu
        self._fc_menu = QtGui.QMenu()
        self._subMenus = self.getContextMenus(ev)
        for menu in self._subMenus:
            self._fc_menu.addMenu(menu)
        return self._fc_menu
    
    def getContextMenus(self, ev):
        ## called by scene to add menus on to someone else's context menu
        menu = self.widget.buildMenu(ev.scenePos())
        menu.setTitle(translate("Context Menu", "Add node"))
        return [menu, ViewBox.getMenu(self, ev)]
