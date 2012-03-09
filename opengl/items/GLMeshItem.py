from OpenGL.GL import *
from .. GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from .. import shaders
import numpy as np



__all__ = ['GLMeshItem']

class GLMeshItem(GLGraphicsItem):
    def __init__(self, faces):
        self.faces = faces
        self.normals, self.faceNormals = pg.meshNormals(faces)
        
        GLGraphicsItem.__init__(self)
        
    def initializeGL(self):
        
        #balloonVertexShader = shaders.compileShader("""
        #varying vec3 normal;
        #void main() {
            #normal = normalize(gl_NormalMatrix * gl_Normal);
            #//vec4 color = normal;
            #//normal.w = min(color.w + 2.0 * color.w * pow(normal.x*normal.x + normal.y*normal.y, 2.0), 1.0);
            #gl_FrontColor = gl_Color;
            #gl_BackColor = gl_Color;
            #gl_Position = ftransform();
        #}""", GL_VERTEX_SHADER)
        #balloonFragmentShader = shaders.compileShader("""
        #varying vec3 normal;
        #void main() {
            #vec4 color = gl_Color;
            #color.w = min(color.w + 2.0 * color.w * pow(normal.x*normal.x + normal.y*normal.y, 5.0), 1.0);
            #gl_FragColor = color;
        #}""", GL_FRAGMENT_SHADER)
        #self.shader = shaders.compileProgram(balloonVertexShader, balloonFragmentShader)
        
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
        for i, f in enumerate(self.faces):
            pts = [QtGui.QVector3D(*x) for x in f]
            if pts[0] is None:
                print f
                continue
            #norm = QtGui.QVector3D.crossProduct(pts[1]-pts[0], pts[2]-pts[0])
            for j in [0,1,2]:
                norm = self.normals[self.faceNormals[i][j]]
                glNormal3f(norm.x(), norm.y(), norm.z())
                #j = (i+1) % 3
                glVertex3f(*f[j])
        glEnd()
        glEndList()
        
        
        l = glGenLists(1)
        self.meshList = l
        glNewList(l, GL_COMPILE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable( GL_BLEND )
        glEnable( GL_ALPHA_TEST )
        #glAlphaFunc( GL_ALWAYS,0.5 )
        glEnable( GL_POINT_SMOOTH )
        glEnable( GL_DEPTH_TEST )
        glColor4f(1, 1, 1, .3)
        glBegin( GL_LINES )
        for f in self.faces:
            for i in [0,1,2]:
                j = (i+1) % 3
                glVertex3f(*f[i])
                glVertex3f(*f[j])
        glEnd()
        glEndList()

                
    def paint(self):
        shaders.glUseProgram(self.shader)
        glCallList(self.triList)
        shaders.glUseProgram(0)
        #glCallList(self.meshList)
