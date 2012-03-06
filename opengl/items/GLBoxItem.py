from OpenGL.GL import *
from .. GLGraphicsItem import GLGraphicsItem

__all__ = ['GLBoxItem']

class GLBoxItem(GLGraphicsItem):
    def paint(self):
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable( GL_BLEND )
        glEnable( GL_ALPHA_TEST )
        #glAlphaFunc( GL_ALWAYS,0.5 )
        glEnable( GL_POINT_SMOOTH )
        glDisable( GL_DEPTH_TEST )
        glBegin( GL_LINES )
        
        glColor4f(1, 1, 1, .3)
        w = 10
        glVertex3f(-w, -w, -w)
        glVertex3f(-w, -w,  w)
        glVertex3f( w, -w, -w)
        glVertex3f( w, -w,  w)
        glVertex3f(-w,  w, -w)
        glVertex3f(-w,  w,  w)
        glVertex3f( w,  w, -w)
        glVertex3f( w,  w,  w)

        glVertex3f(-w, -w, -w)
        glVertex3f(-w,  w, -w)
        glVertex3f( w, -w, -w)
        glVertex3f( w,  w, -w)
        glVertex3f(-w, -w,  w)
        glVertex3f(-w,  w,  w)
        glVertex3f( w, -w,  w)
        glVertex3f( w,  w,  w)
        
        glVertex3f(-w, -w, -w)
        glVertex3f( w, -w, -w)
        glVertex3f(-w,  w, -w)
        glVertex3f( w,  w, -w)
        glVertex3f(-w, -w,  w)
        glVertex3f( w, -w,  w)
        glVertex3f(-w,  w,  w)
        glVertex3f( w,  w,  w)
        
        glEnd()
        
        