import numpy as np

from OpenGL.GL import *
from .. GLGraphicsItem import GLGraphicsItem
from ... import QtGui

__all__ = ['GLGridItem']

class GLGridItem(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem>`
    
    Displays a wire-frame grid. 
    """
    
    def __init__(self, size=None, color=(1, 1, 1, .3), antialias=True, glOptions='translucent'):
        GLGraphicsItem.__init__(self)
        self.setGLOptions(glOptions)
        self.antialias = antialias
        if size is None:
            size = QtGui.QVector3D(20,20,1)
        self.setSize(size=size)
        self.setSpacing(1, 1, 1)
        self.color = color
    
    def setSize(self, x=None, y=None, z=None, size=None):
        """
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x,y,z]
        self.update()
        
    def size(self):
        return self.__size[:]

    def setSpacing(self, x=None, y=None, z=None, spacing=None):
        """
        Set the spacing between grid lines.
        Arguments can be x,y,z or spacing=QVector3D().
        """
        if spacing is not None:
            x = spacing.x()
            y = spacing.y()
            z = spacing.z()
        self.__spacing = [x,y,z]
        self.update() 
        
    def spacing(self):
        return self.__spacing[:]
        
    def paint(self):
        self.setupGLState()
        
        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            
        glBegin( GL_LINES )
        
        x,y,z = self.size()
        xs,ys,zs = self.spacing()
        xvals = np.arange(-x/2., x/2. + xs*0.001, xs) 
        yvals = np.arange(-y/2., y/2. + ys*0.001, ys)
        glColor4f(*self.color)
        for x in xvals:
            glVertex3f(x, yvals[0], 0)
            glVertex3f(x,  yvals[-1], 0)
        for y in yvals:
            glVertex3f(xvals[0], y, 0)
            glVertex3f(xvals[-1], y, 0)
        
        glEnd()
