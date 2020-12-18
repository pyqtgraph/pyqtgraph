# -*- coding: utf-8 -*-
from ..Qt import QtGui, QtCore
from weakref import *

__all__ = ['TreeWidget', 'TreeWidgetItem']


class TreeWidget(QtGui.QTreeWidget):
    """Extends QTreeWidget to allow internal drag/drop with widgets in the tree.
    Also maintains the expanded state of subtrees as they are moved.
    This class demonstrates the absurd lengths one must go to to make drag/drop work."""
    
    sigItemMoved = QtCore.Signal(object, object, object) # (item, parent, index)
    sigItemCheckStateChanged = QtCore.Signal(object, object)
    sigItemTextChanged = QtCore.Signal(object, object)
    sigColumnCountChanged = QtCore.Signal(object, object)  # self, count
    
    def __init__(self, parent=None):
        QtGui.QTreeWidget.__init__(self, parent)
        
        # wrap this item so that we can propagate tree change information
        # to children.
        self._invRootItem = InvisibleRootItem(QtGui.QTreeWidget.invisibleRootItem(self))
        
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setEditTriggers(QtGui.QAbstractItemView.EditKeyPressed|QtGui.QAbstractItemView.SelectedClicked)
        self.placeholders = []
        self.childNestingLimit = None
        self.itemClicked.connect(self._itemClicked)

    def setItemWidget(self, item, col, wid):
        """
        Overrides QTreeWidget.setItemWidget such that widgets are added inside an invisible wrapper widget.
        This makes it possible to move the item in and out of the tree without its widgets being automatically deleted.
        """
        w = QtGui.QWidget()  ## foster parent / surrogate child widget
        l = QtGui.QVBoxLayout()
        l.setContentsMargins(0,0,0,0)
        w.setLayout(l)
        w.setSizePolicy(wid.sizePolicy())
        w.setMinimumHeight(wid.minimumHeight())
        w.setMinimumWidth(wid.minimumWidth())
        l.addWidget(wid)
        w.realChild = wid
        self.placeholders.append(w)
        QtGui.QTreeWidget.setItemWidget(self, item, col, w)

    def itemWidget(self, item, col):
        w = QtGui.QTreeWidget.itemWidget(self, item, col)
        if w is not None and hasattr(w, 'realChild'):
            w = w.realChild
        return w

    def dropMimeData(self, parent, index, data, action):
        item = self.currentItem()
        p = parent
        #print "drop", item, "->", parent, index
        while True:
            if p is None:
                break
            if p is item:
                return False
                #raise Exception("Can not move item into itself.")
            p = p.parent()
        
        if not self.itemMoving(item, parent, index):
            return False
        
        currentParent = item.parent()
        if currentParent is None:
            currentParent = self.invisibleRootItem()
        if parent is None:
            parent = self.invisibleRootItem()
            
        if currentParent is parent and index > parent.indexOfChild(item):
            index -= 1
            
        self.prepareMove(item)
            
        currentParent.removeChild(item)
        #print "  insert child to index", index
        parent.insertChild(index, item)  ## index will not be correct
        self.setCurrentItem(item)
        
        self.recoverMove(item)
        #self.emit(QtCore.SIGNAL('itemMoved'), item, parent, index)
        self.sigItemMoved.emit(item, parent, index)
        return True

    def itemMoving(self, item, parent, index):
        """Called when item has been dropped elsewhere in the tree.
        Return True to accept the move, False to reject."""
        return True
        
    def prepareMove(self, item):
        item.__widgets = []
        item.__expanded = item.isExpanded()
        for i in range(self.columnCount()):
            w = self.itemWidget(item, i)
            item.__widgets.append(w)
            if w is None:
                continue
            w.setParent(None)
        for i in range(item.childCount()):
            self.prepareMove(item.child(i))
        
    def recoverMove(self, item):
        for i in range(self.columnCount()):
            w = item.__widgets[i]
            if w is None:
                continue
            self.setItemWidget(item, i, w)
        for i in range(item.childCount()):
            self.recoverMove(item.child(i))
        
        item.setExpanded(False)  ## Items do not re-expand correctly unless they are collapsed first.
        QtGui.QApplication.instance().processEvents()
        item.setExpanded(item.__expanded)
        
    def collapseTree(self, item):
        item.setExpanded(False)
        for i in range(item.childCount()):
            self.collapseTree(item.child(i))
            
    def removeTopLevelItem(self, item):
        for i in range(self.topLevelItemCount()):
            if self.topLevelItem(i) is item:
                self.takeTopLevelItem(i)
                return
        raise Exception("Item '%s' not in top-level items." % str(item))
    
    def listAllItems(self, item=None):
        items = []
        if item is not None:
            items.append(item)
        else:
            item = self.invisibleRootItem()
        
        for cindex in range(item.childCount()):
            foundItems = self.listAllItems(item=item.child(cindex))
            for f in foundItems:
                items.append(f)
        return items
            
    def dropEvent(self, ev):
        QtGui.QTreeWidget.dropEvent(self, ev)
        self.updateDropFlags()

    def updateDropFlags(self):
        ### intended to put a limit on how deep nests of children can go.
        ### self.childNestingLimit is upheld when moving items without children, but if the item being moved has children/grandchildren, the children/grandchildren
        ### can end up over the childNestingLimit. 
        if self.childNestingLimit == None:
            pass # enable drops in all items (but only if there are drops that aren't enabled? for performance...)
        else:
            items = self.listAllItems()
            for item in items:
                parentCount = 0
                p = item.parent()
                while p is not None:
                    parentCount += 1
                    p = p.parent()
                if parentCount >= self.childNestingLimit:
                    item.setFlags(item.flags() & (~QtCore.Qt.ItemIsDropEnabled))
                else:
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsDropEnabled)

    @staticmethod
    def informTreeWidgetChange(item):
        if hasattr(item, 'treeWidgetChanged'):
            item.treeWidgetChanged()
        for i in range(item.childCount()):
            TreeWidget.informTreeWidgetChange(item.child(i))
        
    def addTopLevelItem(self, item):
        QtGui.QTreeWidget.addTopLevelItem(self, item)
        self.informTreeWidgetChange(item)

    def addTopLevelItems(self, items):
        QtGui.QTreeWidget.addTopLevelItems(self, items)
        for item in items:
            self.informTreeWidgetChange(item)
            
    def insertTopLevelItem(self, index, item):
        QtGui.QTreeWidget.insertTopLevelItem(self, index, item)
        self.informTreeWidgetChange(item)

    def insertTopLevelItems(self, index, items):
        QtGui.QTreeWidget.insertTopLevelItems(self, index, items)
        for item in items:
            self.informTreeWidgetChange(item)
            
    def takeTopLevelItem(self, index):
        item = self.topLevelItem(index)
        if item is not None:
            self.prepareMove(item)
        item = QtGui.QTreeWidget.takeTopLevelItem(self, index)
        self.prepareMove(item)
        self.informTreeWidgetChange(item)
        return item

    def topLevelItems(self):
        return [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        
    def clear(self):
        items = self.topLevelItems()
        for item in items:
            self.prepareMove(item)
        QtGui.QTreeWidget.clear(self)
        
        ## Why do we want to do this? It causes RuntimeErrors. 
        #for item in items:
            #self.informTreeWidgetChange(item)

    def invisibleRootItem(self):
        return self._invRootItem
        
    def itemFromIndex(self, index):
        """Return the item and column corresponding to a QModelIndex.
        """
        col = index.column()
        rows = []
        while index.row() >= 0:
            rows.insert(0, index.row())
            index = index.parent()
        item = self.topLevelItem(rows[0])
        for row in rows[1:]:
            item = item.child(row)
        return item, col

    def setColumnCount(self, c):
        QtGui.QTreeWidget.setColumnCount(self, c)
        self.sigColumnCountChanged.emit(self, c)

    def _itemClicked(self, item, col):
        if hasattr(item, 'itemClicked'):
            item.itemClicked(col)


class TreeWidgetItem(QtGui.QTreeWidgetItem):
    """
    TreeWidgetItem that keeps track of its own widgets and expansion state.
    
    * Widgets may be added to columns before the item is added to a tree.
    * Expanded state may be set before item is added to a tree.
    * Adds setCheked and isChecked methods.
    * Adds addChildren, insertChildren, and takeChildren methods.
    """
    def __init__(self, *args):
        QtGui.QTreeWidgetItem.__init__(self, *args)
        self._widgets = {}  # col: widget
        self._tree = None
        self._expanded = False
        
    def setChecked(self, column, checked):
        self.setCheckState(column, QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)

    def isChecked(self, col):
        return self.checkState(col) == QtCore.Qt.Checked
        
    def setExpanded(self, exp):
        self._expanded = exp
        QtGui.QTreeWidgetItem.setExpanded(self, exp)
        
    def isExpanded(self):
        return self._expanded
        
    def setWidget(self, column, widget):
        if column in self._widgets:
            self.removeWidget(column)
        self._widgets[column] = widget
        tree = self.treeWidget()
        if tree is None:
            return
        else:
            tree.setItemWidget(self, column, widget)
            
    def removeWidget(self, column):
        del self._widgets[column]
        tree = self.treeWidget()
        if tree is None:
            return
        tree.removeItemWidget(self, column)
            
    def treeWidgetChanged(self):
        tree = self.treeWidget()
        if self._tree is tree:
            return
        self._tree = self.treeWidget()
        if tree is None:
            return
        for col, widget in self._widgets.items():
            tree.setItemWidget(self, col, widget)
        QtGui.QTreeWidgetItem.setExpanded(self, self._expanded)
    
    def childItems(self):
        return [self.child(i) for i in range(self.childCount())]
    
    def addChild(self, child):
        QtGui.QTreeWidgetItem.addChild(self, child)
        TreeWidget.informTreeWidgetChange(child)
            
    def addChildren(self, childs):
        QtGui.QTreeWidgetItem.addChildren(self, childs)
        for child in childs:
            TreeWidget.informTreeWidgetChange(child)

    def insertChild(self, index, child):
        QtGui.QTreeWidgetItem.insertChild(self, index, child)
        TreeWidget.informTreeWidgetChange(child)
    
    def insertChildren(self, index, childs):
        QtGui.QTreeWidgetItem.addChildren(self, index, childs)
        for child in childs:
            TreeWidget.informTreeWidgetChange(child)
    
    def removeChild(self, child):
        QtGui.QTreeWidgetItem.removeChild(self, child)
        TreeWidget.informTreeWidgetChange(child)
            
    def takeChild(self, index):
        child = QtGui.QTreeWidgetItem.takeChild(self, index)
        TreeWidget.informTreeWidgetChange(child)
        return child
    
    def takeChildren(self):
        childs = QtGui.QTreeWidgetItem.takeChildren(self)
        for child in childs:
            TreeWidget.informTreeWidgetChange(child)
        return childs
        
    def setData(self, column, role, value):
        # credit: ekhumoro
        #   http://stackoverflow.com/questions/13662020/how-to-implement-itemchecked-and-itemunchecked-signals-for-qtreewidget-in-pyqt4
        checkstate = self.checkState(column)
        text = self.text(column)
        QtGui.QTreeWidgetItem.setData(self, column, role, value)
        
        treewidget = self.treeWidget()
        if treewidget is None:
            return
        if (role == QtCore.Qt.CheckStateRole and checkstate != self.checkState(column)):
            treewidget.sigItemCheckStateChanged.emit(self, column)
        elif (role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole) and text != self.text(column)):
            treewidget.sigItemTextChanged.emit(self, column)

    def itemClicked(self, col):
        """Called when this item is clicked on.
        
        Override this method to react to user clicks.
        """

            
class InvisibleRootItem(object):
    """Wrapper around a TreeWidget's invisible root item that calls
    TreeWidget.informTreeWidgetChange when child items are added/removed.
    """
    def __init__(self, item):
        self._real_item = item
        
    def addChild(self, child):
        self._real_item.addChild(child)
        TreeWidget.informTreeWidgetChange(child)
            
    def addChildren(self, childs):
        self._real_item.addChildren(childs)
        for child in childs:
            TreeWidget.informTreeWidgetChange(child)

    def insertChild(self, index, child):
        self._real_item.insertChild(index, child)
        TreeWidget.informTreeWidgetChange(child)
    
    def insertChildren(self, index, childs):
        self._real_item.addChildren(index, childs)
        for child in childs:
            TreeWidget.informTreeWidgetChange(child)
    
    def removeChild(self, child):
        self._real_item.removeChild(child)
        TreeWidget.informTreeWidgetChange(child)
            
    def takeChild(self, index):
        child = self._real_item.takeChild(index)
        TreeWidget.informTreeWidgetChange(child)
        return child
    
    def takeChildren(self):
        childs = self._real_item.takeChildren()
        for child in childs:
            TreeWidget.informTreeWidgetChange(child)
        return childs

    def __getattr__(self, attr):
        return getattr(self._real_item, attr)
