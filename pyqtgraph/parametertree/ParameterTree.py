from .parameterTypes import GroupParameterItem
from ..Qt import QtCore, QtWidgets, mkQApp
from ..widgets.TreeWidget import TreeWidget
from .ParameterItem import ParameterItem


class ParameterTree(TreeWidget):
    """Widget used to display or control data from a hierarchy of Parameters"""
    
    def __init__(self, parent=None, showHeader=True):
        """
        ============== ========================================================
        **Arguments:**
        parent         (QWidget) An optional parent widget
        showHeader     (bool) If True, then the QTreeView header is displayed.
        ============== ========================================================
        """
        TreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.setAnimated(False)
        self.setColumnCount(2)
        self.setHeaderLabels(["Parameter", "Value"])
        self.setAlternatingRowColors(True)
        self.paramSet = None
        self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.setHeaderHidden(not showHeader)
        self.itemChanged.connect(self.itemChangedEvent)
        self.itemExpanded.connect(self.itemExpandedEvent)
        self.itemCollapsed.connect(self.itemCollapsedEvent)
        self.lastSel = None
        self.setRootIsDecorated(False)

        app = mkQApp()
        app.paletteChanged.connect(self.updatePalette)

    def setParameters(self, param, showTop=True):
        """
        Set the top-level :class:`Parameter <pyqtgraph.parametertree.Parameter>`
        to be displayed in this ParameterTree.

        If *showTop* is False, then the top-level parameter is hidden and only 
        its children will be visible. This is a convenience method equivalent 
        to::
        
            tree.clear()
            tree.addParameters(param, showTop)
        """
        self.clear()
        self.addParameters(param, showTop=showTop)
        
    def addParameters(self, param, root=None, depth=0, showTop=True):
        """
        Adds one top-level :class:`Parameter <pyqtgraph.parametertree.Parameter>`
        to the view. 
        
        ============== ==========================================================
        **Arguments:** 
        param          The :class:`Parameter <pyqtgraph.parametertree.Parameter>` 
                       to add.
        root           The item within the tree to which *param* should be added.
                       By default, *param* is added as a top-level item.
        showTop        If False, then *param* will be hidden, and only its 
                       children will be visible in the tree.
        ============== ==========================================================
        """
        item = param.makeTreeItem(depth=depth)
        if root is None:
            root = self.invisibleRootItem()
            ## Hide top-level item
            if not showTop:
                item.setText(0, '')
                item.setSizeHint(0, QtCore.QSize(1,1))
                item.setSizeHint(1, QtCore.QSize(1,1))
                depth -= 1
        root.addChild(item)
        item.treeWidgetChanged()
            
        for ch in param:
            self.addParameters(ch, root=item, depth=depth+1)

    def clear(self):
        """
        Remove all parameters from the tree.        
        """
        self.invisibleRootItem().takeChildren()        
            
    def focusNext(self, item, forward=True):
        """Give input focus to the next (or previous) item after *item*
        """
        while True:
            parent = item.parent()
            if parent is None:
                return
            nextItem = self.nextFocusableChild(parent, item, forward=forward)
            if nextItem is not None:
                nextItem.setFocus()
                self.setCurrentItem(nextItem)
                return
            item = parent

    def focusPrevious(self, item):
        self.focusNext(item, forward=False)

    def nextFocusableChild(self, root, startItem=None, forward=True):
        if startItem is None:
            if forward:
                index = 0
            else:
                index = root.childCount()-1
        else:
            if forward:
                index = root.indexOfChild(startItem) + 1
            else:
                index = root.indexOfChild(startItem) - 1
            
        if forward:
            inds = list(range(index, root.childCount()))
        else:
            inds = list(range(index, -1, -1))
            
        for i in inds:
            item = root.child(i)
            if hasattr(item, 'isFocusable') and item.isFocusable():
                return item
            else:
                item = self.nextFocusableChild(item, forward=forward)
                if item is not None:
                    return item
        return None

    def contextMenuEvent(self, ev):
        item = self.currentItem()
        if hasattr(item, 'contextMenuEvent'):
            item.contextMenuEvent(ev)
            
    def updatePalette(self):
        """
        called when application palette changes
        This should ensure that the color theme of the OS is applied to the GroupParameterItems
        should the theme chang while the application is running.
        """
        for item in self.listAllItems():
            if isinstance(item, GroupParameterItem):
                item.updateDepth(item.depth)

    def itemChangedEvent(self, item, col):
        if hasattr(item, 'columnChangedEvent'):
            item.columnChangedEvent(col)
    
    def itemExpandedEvent(self, item):
        if hasattr(item, 'expandedChangedEvent'):
            item.expandedChangedEvent(True)
    
    def itemCollapsedEvent(self, item):
        if hasattr(item, 'expandedChangedEvent'):
            item.expandedChangedEvent(False)
            
    def selectionChanged(self, *args):
        sel = self.selectedItems()
        if len(sel) != 1:
            sel = None
        if self.lastSel is not None and isinstance(self.lastSel, ParameterItem):
            self.lastSel.selected(False)
        if sel is None:
            self.lastSel = None
            return
        self.lastSel = sel[0]
        if hasattr(sel[0], 'selected'):
            sel[0].selected(True)
        return super().selectionChanged(*args)
        
    # commented out due to being unreliable
    # def wheelEvent(self, ev):
    #     self.clearSelection()
    #     return super().wheelEvent(ev)

    def sizeHint(self):
        w, h = 0, 0
        ind = self.indentation()
        for x in self.listAllItems():
            if x.isHidden():
                continue
            try:
                depth = x.depth
            except AttributeError:
                depth = 0

            s0 = x.sizeHint(0)
            s1 = x.sizeHint(1)
            w = max(w, depth * ind + max(0, s0.width()) + max(0, s1.width()))
            h += max(0, s0.height(), s1.height())
            # typ = x.param.opts['type'] if isinstance(x, ParameterItem) else x
            # print(typ, depth * ind, (s0.width(), s0.height()), (s1.width(), s1.height()), (w, h))

        # todo: find out if this alternative can be made to work (currently fails when color or colormap are present)
        # print('custom', (w, h))
        # w = self.sizeHintForColumn(0) + self.sizeHintForColumn(1)
        # h = self.viewportSizeHint().height()
        # print('alternative', (w, h))

        if not self.header().isHidden():
            h += self.header().height()

        return QtCore.QSize(w, h)
