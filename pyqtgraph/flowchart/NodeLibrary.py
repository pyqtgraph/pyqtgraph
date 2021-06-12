from collections import OrderedDict
from .Node import Node

def isNodeClass(cls):
    try:
        if not issubclass(cls, Node):
            return False
    except:
        return False
    return hasattr(cls, 'nodeName')



class NodeLibrary:
    """
    A library of flowchart Node types. Custom libraries may be built to provide 
    each flowchart with a specific set of allowed Node types.
    """

    def __init__(self):
        self.nodeList = OrderedDict()
        self.nodeTree = OrderedDict()
        
    def addNodeType(self, nodeClass, paths, override=False):
        """
        Register a new node type. If the type's name is already in use,
        an exception will be raised (unless override=True).
        
        ============== =========================================================
        **Arguments:**
        
        nodeClass      a subclass of Node (must have typ.nodeName)
        paths          list of tuples specifying the location(s) this 
                       type will appear in the library tree.
        override       if True, overwrite any class having the same name
        ============== =========================================================
        """
        if not isNodeClass(nodeClass):
            raise Exception("Object %s is not a Node subclass" % str(nodeClass))
        
        name = nodeClass.nodeName
        if not override and name in self.nodeList:
            raise Exception("Node type name '%s' is already registered." % name)
        
        self.nodeList[name] = nodeClass
        for path in paths:
            root = self.nodeTree
            for n in path:
                if n not in root:
                    root[n] = OrderedDict()
                root = root[n]
            root[name] = nodeClass

    def getNodeType(self, name):
        try:
            return self.nodeList[name]
        except KeyError:
            raise Exception("No node type called '%s'" % name)

    def getNodeTree(self):
        return self.nodeTree

    def copy(self):
        """
        Return a copy of this library.
        """
        lib = NodeLibrary()
        lib.nodeList = self.nodeList.copy()
        lib.nodeTree = self.treeCopy(self.nodeTree)
        return lib

    @staticmethod
    def treeCopy(tree):
        copy = OrderedDict()
        for k,v in tree.items():
            if isNodeClass(v):
                copy[k] = v
            else:
                copy[k] = NodeLibrary.treeCopy(v)
        return copy

    def reload(self):
        """
        Reload Node classes in this library.
        """
        raise NotImplementedError()
