# -*- coding: utf-8 -*-
from ..Qt import QtGui, QtCore
from ..pgcollections import OrderedDict
from .TableWidget import TableWidget
from ..python2_3 import asUnicode
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
        self.setAlternatingRowColors(True)
        
    def setData(self, data, hideRoot=False):
        """data should be a dictionary."""
        self.clear()
        self.widgets = []
        self.nodes = {}
        self.buildTree(data, self.invisibleRootItem(), hideRoot=hideRoot)
        self.expandToDepth(3)
        self.resizeColumnToContents(0)
        
    def buildTree(self, data, parent, name='', hideRoot=False, path=()):
        if hideRoot:
            node = parent
        else:
            node = QtGui.QTreeWidgetItem([name, "", ""])
            parent.addChild(node)
        
        # record the path to the node so it can be retrieved later
        # (this is used by DiffTreeWidget)
        self.nodes[path] = node

        typeStr, desc, childs, widget = self.parse(data)
        node.setText(1, typeStr)
        node.setText(2, desc)
            
        # Truncate description and add text box if needed
        if len(desc) > 100:
            desc = desc[:97] + '...'
            if widget is None:
                widget = QtGui.QPlainTextEdit(asUnicode(data))
                widget.setMaximumHeight(200)
                widget.setReadOnly(True)
        
        # Add widget to new subnode
        if widget is not None:
            self.widgets.append(widget)
            subnode = QtGui.QTreeWidgetItem(["", "", ""])
            node.addChild(subnode)
            self.setItemWidget(subnode, 0, widget)
            self.setFirstItemColumnSpanned(subnode, True)
            
        # recurse to children
        for key, data in childs.items():
            self.buildTree(data, node, asUnicode(key), path=path+(key,))

    def parse(self, data):
        """
        Given any python object, return:
        * type
        * a short string representation
        * a dict of sub-objects to be parsed
        * optional widget to display as sub-node
        """
        # defaults for all objects
        typeStr = type(data).__name__
        if typeStr == 'instance':
            typeStr += ": " + data.__class__.__name__
        widget = None
        desc = ""
        childs = {}
        
        # type-specific changes
        if isinstance(data, dict):
            desc = "length=%d" % len(data)
            if isinstance(data, OrderedDict):
                childs = data
            else:
                childs = OrderedDict(sorted(data.items()))
        elif isinstance(data, (list, tuple)):
            desc = "length=%d" % len(data)
            childs = OrderedDict(enumerate(data))
        elif HAVE_METAARRAY and (hasattr(data, 'implements') and data.implements('MetaArray')):
            childs = OrderedDict([
                ('data', data.view(np.ndarray)),
                ('meta', data.infoCopy())
            ])
        elif isinstance(data, np.ndarray):
            desc = "shape=%s dtype=%s" % (data.shape, data.dtype)
            table = TableWidget()
            table.setData(data)
            table.setMaximumHeight(200)
            widget = table
        elif isinstance(data, types.TracebackType):  ## convert traceback to a list of strings
            frames = list(map(str.strip, traceback.format_list(traceback.extract_tb(data))))
            #childs = OrderedDict([
                #(i, {'file': child[0], 'line': child[1], 'function': child[2], 'code': child[3]})
                #for i, child in enumerate(frames)])
            #childs = OrderedDict([(i, ch) for i,ch in enumerate(frames)])
            widget = QtGui.QPlainTextEdit(asUnicode('\n'.join(frames)))
            widget.setMaximumHeight(200)
            widget.setReadOnly(True)
        else:
            desc = asUnicode(data)
        
        return typeStr, desc, childs, widget
        