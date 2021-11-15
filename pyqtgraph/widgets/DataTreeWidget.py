import traceback
import types
from collections import OrderedDict

import numpy as np

from ..Qt import QtWidgets
from .TableWidget import TableWidget

try:
    import metaarray  # noqa
    HAVE_METAARRAY = True
except:
    HAVE_METAARRAY = False

__all__ = ['DataTreeWidget']

class DataTreeWidget(QtWidgets.QTreeWidget):
    """
    Widget for displaying hierarchical python data structures
    (eg, nested dicts, lists, and arrays)
    """
    def __init__(self, parent=None, data=None):
        QtWidgets.QTreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
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
            node = QtWidgets.QTreeWidgetItem([name, "", ""])
            parent.addChild(node)
        
        # record the path to the node so it can be retrieved later
        # (this is used by DiffTreeWidget)
        self.nodes[path] = node

        typeStr, desc, childs, widget = self.parse(data)
            
        # Truncate description and add text box if needed
        if len(desc) > 100:
            desc = desc[:97] + '...'
            if widget is None:
                widget = QtWidgets.QPlainTextEdit(str(data))
                widget.setMaximumHeight(200)
                widget.setReadOnly(True)

        node.setText(1, typeStr)
        node.setText(2, desc)
        
        # Add widget to new subnode
        if widget is not None:
            self.widgets.append(widget)
            subnode = QtWidgets.QTreeWidgetItem(["", "", ""])
            node.addChild(subnode)
            self.setItemWidget(subnode, 0, widget)
            subnode.setFirstColumnSpanned(True)
            
        # recurse to children
        for key, data in childs.items():
            self.buildTree(data, node, str(key), path=path+(key,))

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
                try:
                    childs = OrderedDict(sorted(data.items()))
                except TypeError: # if sorting falls
                    childs = OrderedDict(data.items())
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
            widget = QtWidgets.QPlainTextEdit('\n'.join(frames))
            widget.setMaximumHeight(200)
            widget.setReadOnly(True)
        else:
            desc = str(data)
        
        return typeStr, desc, childs, widget
        
