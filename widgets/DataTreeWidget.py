# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtGui, QtCore
from collections import OrderedDict
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
        self.buildTree(data, self.invisibleRootItem(), hideRoot=hideRoot)
        #node = self.mkNode('', data)
        #while node.childCount() > 0:
            #c = node.child(0)
            #node.removeChild(c)
            #self.invisibleRootItem().addChild(c)
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
            data = map(str.strip, traceback.format_list(traceback.extract_tb(data)))
        elif HAVE_METAARRAY and isinstance(data, metaarray.MetaArray):
            data = {
                'data': data.view(np.ndarray),
                'meta': data.infoCopy()
            }
            
        if isinstance(data, dict):
            for k in data:
                self.buildTree(data[k], node, str(k))
        elif isinstance(data, list) or isinstance(data, tuple):
            for i in range(len(data)):
                self.buildTree(data[i], node, str(i))
        else:
            node.setText(2, str(data))
        
        
    #def mkNode(self, name, v):
        #if type(v) is list and len(v) > 0 and isinstance(v[0], dict):
            #inds = map(unicode, range(len(v)))
            #v = OrderedDict(zip(inds, v))
        #if isinstance(v, dict):
            ##print "\nadd tree", k, v
            #node = QtGui.QTreeWidgetItem([name])
            #for k in v:
                #newNode = self.mkNode(k, v[k])
                #node.addChild(newNode)
        #else:
            ##print "\nadd value", k, str(v)
            #node = QtGui.QTreeWidgetItem([unicode(name), unicode(v)])
        #return node
        
        
if __name__ == '__main__':
    app = QtGui.QApplication([])
    d = {
        'list1': [1,2,3,4,5,6, {'nested1': 'aaaaa', 'nested2': 'bbbbb'}, "seven"],
        'dict1': {
            'x': 1,
            'y': 2,
            'z': 'three'
        },
        'array1 (20x20)': np.ones((10,10))
    }
        
    tree = DataTreeWidget(data=d)
    tree.show()
    tree.resize(600,600)
    
    
    ## Start Qt event loop unless running in interactive mode.
    import sys
    if sys.flags.interactive != 1:
        app.exec_()
        