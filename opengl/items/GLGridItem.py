from OpenGL.GL import *
from .. GLGraphicsItem import GLGraphicsItem
from pyqtgraph import QtGui

__all__ = ['GLGridItem']

class GLGridItem(GLGraphicsItem):
    def __init__(self, size=None, color=None):
        GLGraphicsItem.__init__(self)
        if size is None:
            size = QtGui.QVector3D(1,1,1)
        self.setSize(size=size)
    
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
    
    
    def paint(self):

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable( GL_BLEND )
        glEnable( GL_ALPHA_TEST )
        glEnable( GL_POINT_SMOOTH )
        #glDisable( GL_DEPTH_TEST )
        glBegin( GL_LINES )
        
        x,y,z = self.size()
        glColor4f(1, 1, 1, .3)
        for x in range(-10, 11):
            glVertex3f(x, -10, 0)
            glVertex3f(x,  10, 0)
        for y in range(-10, 11):
            glVertex3f(-10, y, 0)
            glVertex3f( 10, y, 0)
        
        glEnd()
