# -*- coding: utf-8 -*-
from pyqtgraph.pgcollections import OrderedDict
#from pyqtgraph import importModules
import os, types
from pyqtgraph.debug import printExc
#from ..Node import Node
from ..NodeLibrary import NodeLibrary, isNodeClass
import pyqtgraph.reload as reload


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
    #mod = getattr(__import__('', fromlist=[modName], level=1), modName)
    #mod = __import__(modName, level=1)
    nodes = [getattr(mod, name) for name in dir(mod) if isNodeClass(getattr(mod, name))]
    for node in nodes:
        LIBRARY.addNodeType(node, [(mod.__name__.split('.')[-1],)])
    
#NODE_LIST = OrderedDict()  ## maps name:class for all registered Node subclasses
#NODE_TREE = OrderedDict()  ## categorized tree of Node subclasses

#def getNodeType(name):
    #try:
        #return NODE_LIST[name]
    #except KeyError:
        #raise Exception("No node type called '%s'" % name)

#def getNodeTree():
    #return NODE_TREE

#def registerNodeType(cls, paths, override=False):
    #"""
    #Register a new node type. If the type's name is already in use,
    #an exception will be raised (unless override=True).
    
    #Arguments:
        #cls - a subclass of Node (must have typ.nodeName)
        #paths - list of tuples specifying the location(s) this 
                #type will appear in the library tree.
        #override - if True, overwrite any class having the same name
    #"""
    #if not isNodeClass(cls):
        #raise Exception("Object %s is not a Node subclass" % str(cls))
    
    #name = cls.nodeName
    #if not override and name in NODE_LIST:
        #raise Exception("Node type name '%s' is already registered." % name)
    
    #NODE_LIST[name] = cls
    #for path in paths:
        #root = NODE_TREE
        #for n in path:
            #if n not in root:
                #root[n] = OrderedDict()
            #root = root[n]
        #root[name] = cls



#def isNodeClass(cls):
    #try:
        #if not issubclass(cls, Node):
            #return False
    #except:
        #return False
    #return hasattr(cls, 'nodeName')

#def loadLibrary(reloadLibs=False, libPath=None):
    #"""Import all Node subclasses found within files in the library module."""

    #global NODE_LIST, NODE_TREE
    
    #if reloadLibs:
        #reload.reloadAll(libPath)
        
    #mods = importModules('', globals(), locals())
        
    #for name, mod in mods.items():
        #nodes = []
        #for n in dir(mod):
            #o = getattr(mod, n)
            #if isNodeClass(o):
                #registerNodeType(o, [(name,)], override=reloadLibs)
    
#def reloadLibrary():
    #loadLibrary(reloadLibs=True)
    
#loadLibrary()



