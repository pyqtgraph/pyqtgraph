from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph import Transform3D

class GLGraphicsItem(QtCore.QObject):
    def __init__(self, parentItem=None):
        QtCore.QObject.__init__(self)
        self.__parent = None
        self.__view = None
        self.__children = set()
        self.__transform = Transform3D()
        self.__visible = True
        self.setParentItem(parentItem)
        self.setDepthValue(0)
        
    def setParentItem(self, item):
        if self.__parent is not None:
            self.__parent.__children.remove(self)
        if item is not None:
            item.__children.add(self)
        self.__parent = item
        
        if self.__parent is not None and self.view() is not self.__parent.view():
            if self.view() is not None:
                self.view().removeItem(self)
            self.__parent.view().addItem(self)
        
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
        
    def setTransform(self, tr):
        self.__transform = Transform3D(tr)
        self.update()
        
    def resetTransform(self):
        self.__transform.setToIdentity()
        self.update()
        
    def applyTransform(self, tr, local):
        """
        Multiply this object's transform by *tr*. 
        If local is True, then *tr* is multiplied on the right of the current transform:
            newTransform = transform * tr
        If local is False, then *tr* is instead multiplied on the left:
            newTransform = tr * transform
        """
        if local:
            self.setTransform(self.transform() * tr)
        else:
            self.setTransform(tr * self.transform())
        
    def transform(self):
        return self.__transform
        
    def viewTransform(self):
        tr = self.__transform
        p = self
        while True:
            p = p.parentItem()
            if p is None:
                break
            tr = p.transform() * tr
        return Transform3D(tr)
        
    def translate(self, dx, dy, dz, local=False):
        """
        Translate the object by (*dx*, *dy*, *dz*) in its parent's coordinate system.
        If *local* is True, then translation takes place in local coordinates.
        """
        tr = Transform3D()
        tr.translate(dx, dy, dz)
        self.applyTransform(tr, local=local)
        
    def rotate(self, angle, x, y, z, local=False):
        """
        Rotate the object around the axis specified by (x,y,z).
        *angle* is in degrees.
        
        """
        tr = Transform3D()
        tr.rotate(angle, x, y, z)
        self.applyTransform(tr, local=local)
    
    def scale(self, x, y, z, local=True):
        """
        Scale the object by (*dx*, *dy*, *dz*) in its local coordinate system.
        If *local* is False, then scale takes place in the parent's coordinates.
        """
        tr = Transform3D()
        tr.scale(x, y, z)
        self.applyTransform(tr, local=local)
    
    
    def hide(self):
        self.setVisible(False)
        
    def show(self):
        self.setVisible(True)
    
    def setVisible(self, vis):
        self.__visible = vis
        self.update()
        
    def visible(self):
        return self.__visible
    
    
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
        
    def update(self):
        v = self.view()
        if v is None:
            return
        v.updateGL()
        
    def mapToParent(self, point):
        tr = self.transform()
        if tr is None:
            return point
        return tr.map(point)
        
    def mapFromParent(self, point):
        tr = self.transform()
        if tr is None:
            return point
        return tr.inverted()[0].map(point)
        
    def mapToView(self, point):
        tr = self.viewTransform()
        if tr is None:
            return point
        return tr.map(point)
        
    def mapFromView(self, point):
        tr = self.viewTransform()
        if tr is None:
            return point
        return tr.inverted()[0].map(point)
        
        
        