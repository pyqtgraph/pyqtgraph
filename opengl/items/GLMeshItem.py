from OpenGL.GL import *
from .. GLGraphicsItem import GLGraphicsItem
from .. MeshData import MeshData
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from .. import shaders
import numpy as np



__all__ = ['GLMeshItem']

class GLMeshItem(GLGraphicsItem):
    """
    Displays a 3D triangle mesh. 
    
    """
    def __init__(self, faces, vertexes=None):
        
        """
        See MeshData for initialization arguments.
        """
        self.data = MeshData()
        self.data.setFaces(faces, vertexes)
        GLGraphicsItem.__init__(self)
        
    def initializeGL(self):
        self.shader = shaders.getShader('balloon')
        
        l = glGenLists(1)
        self.triList = l
        glNewList(l, GL_COMPILE)
        
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable( GL_BLEND )
        glEnable( GL_ALPHA_TEST )
        #glAlphaFunc( GL_ALWAYS,0.5 )
        glEnable( GL_POINT_SMOOTH )
        glDisable( GL_DEPTH_TEST )
        glColor4f(1, 1, 1, .1)
        glBegin( GL_TRIANGLES )
        for face in self.data:
            for (pos, norm, color) in face:
                glColor4f(*color)
                glNormal3f(norm.x(), norm.y(), norm.z())
                glVertex3f(pos.x(), pos.y(), pos.z())
        glEnd()
        glEndList()
        
        
        #l = glGenLists(1)
        #self.meshList = l
        #glNewList(l, GL_COMPILE)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glEnable( GL_BLEND )
        #glEnable( GL_ALPHA_TEST )
        ##glAlphaFunc( GL_ALWAYS,0.5 )
        #glEnable( GL_POINT_SMOOTH )
        #glEnable( GL_DEPTH_TEST )
        #glColor4f(1, 1, 1, .3)
        #glBegin( GL_LINES )
        #for f in self.faces:
            #for i in [0,1,2]:
                #j = (i+1) % 3
                #glVertex3f(*f[i])
                #glVertex3f(*f[j])
        #glEnd()
        #glEndList()

                
    def paint(self):
        shaders.glUseProgram(self.shader)
        glCallList(self.triList)
        shaders.glUseProgram(0)
        #glCallList(self.meshList)
