# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtGui, QtCore
from weakref import *

__all__ = ['TreeWidget']
class TreeWidget(QtGui.QTreeWidget):
    """Extends QTreeWidget to allow internal drag/drop with widgets in the tree.
    Also maintains the expanded state of subtrees as they are moved.
    This class demonstrates the absurd lengths one must go to to make drag/drop work."""
    
    sigItemMoved = QtCore.Signal(object, object, object) # (item, parent, index)
    
    def __init__(self, parent=None):
        QtGui.QTreeWidget.__init__(self, parent)
        #self.itemWidgets = WeakKeyDictionary()
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setEditTriggers(QtGui.QAbstractItemView.EditKeyPressed|QtGui.QAbstractItemView.SelectedClicked)
        self.placeholders = []
        self.childNestingLimit = None

    def setItemWidget(self, item, col, wid):
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
        if w is not None:
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
        if item != None:
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
            
