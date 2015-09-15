# -*- coding: utf-8 -*-
from ...pgcollections import OrderedDict
import os, types
from ...debug import printExc
from ..NodeLibrary import NodeLibrary, isNodeClass
from ... import reload as reload


# Build default library
LIBRARY = NodeLibrary()

# For backward compatibility, expose the default library's properties here:
NODE_LIST = LIBRARY.nodeList
NODE_TREE = LIBRARY.nodeTree
registerNodeType = LIBRARY.addNodeType
getNodeTree = LIBRARY.getNodeTree
getNodeType = LIBRARY.getNodeType

# Add all nodes to the default library
from . import Data, Display, Filters, Operators
for mod in [Data, Display, Filters, Operators]:
    nodes = [getattr(mod, name) for name in dir(mod) if isNodeClass(getattr(mod, name))]
    for node in nodes:
        LIBRARY.addNodeType(node, [(mod.__name__.split('.')[-1],)])
