from OpenGL import GL

from .. import Transform3D
from ..Qt import QtCore, QtGui

GLOptions = {
    'opaque': {
        GL.GL_DEPTH_TEST: True,
        GL.GL_BLEND: False,
        GL.GL_CULL_FACE: False,
    },
    'translucent': {
        GL.GL_DEPTH_TEST: True,
        GL.GL_BLEND: True,
        GL.GL_CULL_FACE: False,
        'glBlendFuncSeparate': (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                                GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA),
    },
    'additive': {
        GL.GL_DEPTH_TEST: False,
        GL.GL_BLEND: True,
        GL.GL_CULL_FACE: False,
        'glBlendFunc': (GL.GL_SRC_ALPHA, GL.GL_ONE),
    },
}    


class GLGraphicsItem(QtCore.QObject):
    _nextId = 0
    
    def __init__(self, parentItem: 'GLGraphicsItem' = None):
        super().__init__()
        self._id = GLGraphicsItem._nextId
        GLGraphicsItem._nextId += 1
        
        self.__parent: GLGraphicsItem | None = None
        self.__view = None
        self.__children: list[GLGraphicsItem] = list()
        self.__transform = Transform3D()
        self.__visible = True
        self.__initialized = False
        self.setParentItem(parentItem)
        self.setDepthValue(0)
        self.__glOpts = {}
        
    def setParentItem(self, item):
        """Set this item's parent in the scenegraph hierarchy."""
        if self.__parent is not None:
            self.__parent.__children.remove(self)
        if item is not None:
            item.__children.append(self)

        # if we had a __view, we were a top level object
        if self.__view is not None:
            self.__view.removeItem(self)

        # we are now either a child or an orphan.
        # either way, we don't have our own __view
        self.__parent = item
        self.__view = None
    
    def setGLOptions(self, opts):
        """
        Set the OpenGL state options to use immediately before drawing this item.
        (Note that subclasses must call setupGLState before painting for this to work)
        
        The simplest way to invoke this method is to pass in the name of
        a predefined set of options (see the GLOptions variable):
        
        ============= ======================================================
        opaque        Enables depth testing and disables blending
        translucent   Enables depth testing and blending
                      Elements must be drawn sorted back-to-front for
                      translucency to work correctly.
        additive      Disables depth testing, enables blending.
                      Colors are added together, so sorting is not required.
        ============= ======================================================
        
        It is also possible to specify any arbitrary settings as a dictionary. 
        This may consist of {'functionName': (args...)} pairs where functionName must 
        be a callable attribute of OpenGL.GL, or {GL_STATE_VAR: bool} pairs 
        which will be interpreted as calls to glEnable or glDisable(GL_STATE_VAR).
        
        For example::
            
            {
                GL_ALPHA_TEST: True,
                GL_CULL_FACE: False,
                'glBlendFunc': (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
            }
            
        
        """
        if isinstance(opts, str):
            opts = GLOptions[opts]
        self.__glOpts = opts.copy()
        self.update()
        
    def updateGLOptions(self, opts):
        """
        Modify the OpenGL state options to use immediately before drawing this item.
        *opts* must be a dictionary as specified by setGLOptions.
        Values may also be None, in which case the key will be ignored.
        """
        self.__glOpts.update(opts)
        
    
    def parentItem(self):
        """Return a this item's parent in the scenegraph hierarchy."""
        return self.__parent
        
    def childItems(self):
        """Return a list of this item's children in the scenegraph hierarchy."""
        return list(self.__children)
        
    def _setView(self, v):
        self.__view = v
        
    def view(self):
        if self.__parent is None:
            # top level object
            return self.__view
        else:
            # recurse
            return self.__parent.view()
        
    def setDepthValue(self, value):
        """
        Sets the depth value of this item. Default is 0.
        This controls the order in which items are drawn--those with a greater depth value will be drawn later.
        Items with negative depth values are drawn before their parent.
        (This is analogous to QGraphicsItem.zValue)
        The depthValue does NOT affect the position of the item or the values it imparts to the GL depth buffer.
        """
        self.__depthValue = value
        
    def depthValue(self):
        """Return the depth value of this item. See setDepthValue for more information."""
        return self.__depthValue
        
    def setTransform(self, tr):
        """Set the local transform for this object.

        Parameters
        ----------
        tr : pyqtgraph.Transform3D
            Tranformation from the local coordinate system to the parent's.
        """
        self.__transform = Transform3D(tr)
        self.update()
        
    def resetTransform(self):
        """Reset this item's transform to an identity transformation."""
        self.__transform.setToIdentity()
        self.update()
        
    def applyTransform(self, tr, local):
        """
        Multiply this object's transform by *tr*. 
        If local is True, then *tr* is multiplied on the right of the current transform::
        
            newTransform = transform * tr
            
        If local is False, then *tr* is instead multiplied on the left::
        
            newTransform = tr * transform
        """
        if local:
            self.setTransform(self.transform() * tr)
        else:
            self.setTransform(tr * self.transform())
        
    def transform(self):
        """Return this item's transform object."""
        return self.__transform
        
    def viewTransform(self):
        """Return the transform mapping this item's local coordinate system to the 
        view coordinate system."""
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
        """Hide this item. 
        This is equivalent to setVisible(False)."""
        self.setVisible(False)
        
    def show(self):
        """Make this item visible if it was previously hidden.
        This is equivalent to setVisible(True)."""
        self.setVisible(True)
    
    def setVisible(self, vis):
        """Set the visibility of this item."""
        self.__visible = vis
        self.update()
        
    def visible(self):
        """Return True if the item is currently set to be visible.
        Note that this does not guarantee that the item actually appears in the
        view, as it may be obscured or outside of the current view area."""
        return self.__visible
    
    def initialize(self):
        self.initializeGL()
        self.__initialized = True

    def isInitialized(self):
        return self.__initialized
    
    def initializeGL(self):
        """
        Called after an item is added to a GLViewWidget. 
        The widget's GL context is made current before this method is called.
        (So this would be an appropriate time to generate lists, upload textures, etc.)
        """
        pass
    
    def setupGLState(self):
        """
        This method is responsible for preparing the GL state options needed to render 
        this item (blending, depth testing, etc). The method is called immediately before painting the item.
        """
        for k,v in self.__glOpts.items():
            if v is None:
                continue
            if isinstance(k, str):
                func = getattr(GL, k)
                func(*v)
            else:
                if v is True:
                    GL.glEnable(k)
                else:
                    GL.glDisable(k)
    
    def paint(self):
        """
        Called by the GLViewWidget to draw this item.
        It is the responsibility of the item to set up its own modelview matrix,
        but the caller will take care of pushing/popping.
        """
        self.setupGLState()
        
    def update(self):
        """
        Indicates that this item needs to be redrawn, and schedules an update 
        with the view it is displayed in.
        """
        v = self.view()
        if v is None:
            return
        v.update()
        
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

    def modelViewMatrix(self) -> QtGui.QMatrix4x4:
        if (view := self.view()) is None:
            return QtGui.QMatrix4x4()
        return view.currentModelView()

    def projectionMatrix(self) -> QtGui.QMatrix4x4:
        if (view := self.view()) is None:
            return QtGui.QMatrix4x4()
        return view.currentProjection()

    def mvpMatrix(self) -> QtGui.QMatrix4x4:
        if (view := self.view()) is None:
            return QtGui.QMatrix4x4()
        return view.currentProjection() * view.currentModelView()
