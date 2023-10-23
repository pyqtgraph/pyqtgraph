# type: ignore
from typing import Callable, Unpack

from .Node import Node
from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.GraphicsView import GraphicsView

translate: Callable = QtCore.QCoreApplication.translate


class FlowchartGraphicsView(GraphicsView):
    sigHoverOver = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object)

    def __init__(self, widget: Node, *args: Unpack) -> None:
        GraphicsView.__init__(self, *args, useOpenGL=False)
        self._vb = FlowchartViewBox(widget, lockAspect=True, invertY=True)
        self.setCentralItem(self._vb)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

    def viewBox(self) -> 'FlowchartViewBox':
        return self._vb


class FlowchartViewBox(ViewBox):

    def __init__(self, widget: Node, *args: Unpack, **kwargs: Unpack) -> None:
        ViewBox.__init__(self, *args, **kwargs)
        self.widget = widget

    def getMenu(self, ev) -> QtWidgets.QMenu:
        ## called by ViewBox to create a new context menu
        self._fc_menu = QtWidgets.QMenu()
        self._subMenus = self.getContextMenus(ev)
        for menu in self._subMenus:
            self._fc_menu.addMenu(menu)
        return self._fc_menu

    def getContextMenus(self, ev) -> list:
        ## called by scene to add menus on to someone else's context menu
        menu = self.widget.buildMenu(ev.scenePos())
        menu.setTitle(translate("Context Menu", "Add node"))
        return [menu, ViewBox.getMenu(self, ev)]
