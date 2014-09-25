# -*- coding: utf-8 -*-
from ..Qt import QtGui, QtCore
from ..pgcollections import OrderedDict
from .TableWidget import TableWidget
import types, traceback
import numpy as np

try:
    import metaarray
    HAVE_METAARRAY = True
except:
    HAVE_METAARRAY = False

__all__ = ['DataTreeWidget']

class DataTreeWidget(QtGui.QTreeWidget):
    """
    Widget for displaying hierarchical python data structures
    (eg, nested dicts, lists, and arrays)
    """
    
    
    def __init__(self, parent=None, data=None):
        QtGui.QTreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.setData(data)
        self.setColumnCount(3)
        self.setHeaderLabels(['key / index', 'type', 'value'])
        
    def setData(self, data, hideRoot=False):
        """data should be a dictionary."""
        self.clear()
        self.tables = []
        self.buildTree(data, self.invisibleRootItem(), hideRoot=hideRoot)
        self.expandToDepth(3)
        self.resizeColumnToContents(0)
        
    def buildTree(self, data, parent, name='', hideRoot=False):
        if hideRoot:
            node = parent
        else:
            typeStr = type(data).__name__
            if typeStr == 'instance':
                typeStr += ": " + data.__class__.__name__
            node = QtGui.QTreeWidgetItem([name, typeStr, ""])
            parent.addChild(node)
        
        if isinstance(data, types.TracebackType):  ## convert traceback to a list of strings
            data = list(map(str.strip, traceback.format_list(traceback.extract_tb(data))))
        elif HAVE_METAARRAY and (hasattr(data, 'implements') and data.implements('MetaArray')):
            data = {
                'data': data.view(np.ndarray),
                'meta': data.infoCopy()
            }
            
        if isinstance(data, dict):
            for k in data.keys():
                self.buildTree(data[k], node, str(k))
        elif isinstance(data, list) or isinstance(data, tuple):
            for i in range(len(data)):
                self.buildTree(data[i], node, str(i))
        elif isinstance(data, np.ndarray):
            desc = "<%s shape=%s dtype=%s>" % (data.__class__.__name__, data.shape, data.dtype)
            node.setText(2, desc)
            subnode = QtGui.QTreeWidgetItem(["", "", ""])
            node.addChild(subnode)
            table = TableWidget()
            table.setData(data)
            table.setMaximumHeight(200)
            self.setItemWidget(subnode, 2, table)
            self.tables.append(table)
        else:
            node.setText(2, str(data))
