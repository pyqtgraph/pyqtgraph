from pyqtgraph.Qt import QtGui, QtCore

class GLGraphicsItem(QtCore.QObject):
    def __init__(self, parentItem=None):
        QtCore.QObject.__init__(self)
        self.__parent = None
        self.__view = None
        self.__children = set()
        self.setParentItem(parentItem)
        self.setDepthValue(0)
        
    def setParentItem(self, item):
        if self.__parent is not None:
            self.__parent.__children.remove(self)
        if item is not None:
            item.__children.add(self)
        self.__parent = item
        
    def parentItem(self):
        return self.__parent
        
    def childItems(self):
        return list(self.__children)
        
    def _setView(self, v):
        self.__view = v
        
    def view(self):
        return self.__view
        
    def setDepthValue(self, value):
        """
        Sets the depth value of this item. Default is 0.
        This controls the order in which items are drawn--those with a greater depth value will be drawn later.
        Items with negative depth values are drawn before their parent.
        (This is analogous to QGraphicsItem.zValue)
        The depthValue does NOT affect the position of the item or the values it imparts to the GL depth buffer.
        '"""
        self.__depthValue = value
        
    def depthValue(self):
        """Return the depth value of this item. See setDepthValue for mode information."""
        return self.__depthValue
        
    def initializeGL(self):
        """
        Called after an item is added to a GLViewWidget. 
        The widget's GL context is made current before this method is called.
        (So this would be an appropriate time to generate lists, upload textures, etc.)
        """
        pass
    
    def paint(self):
        """
        Called by the GLViewWidget to draw this item.
        It is the responsibility of the item to set up its own modelview matrix,
        but the caller will take care of pushing/popping.
        """
        pass
        