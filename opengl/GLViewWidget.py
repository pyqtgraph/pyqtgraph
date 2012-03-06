from pyqtgraph.Qt import QtCore, QtGui, QtOpenGL
from OpenGL.GL import *
import numpy as np

Vector = QtGui.QVector3D

class GLViewWidget(QtOpenGL.QGLWidget):
    """
    Basic widget for displaying 3D data
        - Rotation/scale controls
        - Axis/grid display
        - Export options

    """
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.opts = {
            'center': Vector(0,0,0),  ## will always appear at the center of the widget
            'distance': 10.0,         ## distance of camera from center
            'fov':  60,               ## horizontal field of view in degrees
            'elevation':  30,         ## camera's angle of elevation in degrees
            'azimuth': 45,            ## camera's azimuthal angle in degrees 
                                      ## (rotation around z-axis 0 points along x-axis)
        }
        self.items = []

    def addItem(self, item):
        self.items.append(item)
        if hasattr(item, 'initializeGL'):
            self.makeCurrent()
            item.initializeGL()
        item._setView(self)
        #print "set view", item, self, item.view()
        self.updateGL()
        
    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glEnable(GL_DEPTH_TEST)

        glEnable( GL_ALPHA_TEST )
        self.resizeGL(self.width(), self.height())
        self.generateAxes()
        #self.generatePoints()
        
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        #self.updateGL()

    def setProjection(self):
        ## Create the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        w = self.width()
        h = self.height()
        dist = self.opts['distance']
        fov = self.opts['fov']
        
        nearClip = dist * 0.001
        farClip = dist * 1000.
        
        r = nearClip * np.tan(fov)
        t = r * h / w
        glFrustum( -r, r, -t, t, nearClip, farClip)
        
    def setModelview(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef( 0.0, 0.0, -self.opts['distance'])
        glRotatef(self.opts['elevation']-90, 1, 0, 0)
        glRotatef(self.opts['azimuth']+90, 0, 0, -1)
        center = self.opts['center']
        glTranslatef(center.x(), center.y(), center.z())
        
        
    def paintGL(self):
        self.setProjection()
        self.setModelview()

        glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT )
        glDisable( GL_DEPTH_TEST )
        #print "draw list:", self.axisList
        glCallList(self.axisList)  ## draw axes
        #glCallList(self.pointList)
        #self.drawPoints()
        #self.drawAxes()
        
        self.drawItemTree()
        
    def drawItemTree(self, item=None):
        if item is None:
            items = [x for x in self.items if x.parentItem() is None]
        else:
            items = item.childItems()
            items.append(item)
        items.sort(lambda a,b: cmp(a.depthValue(), b.depthValue()))
        for i in items:
            if i is item:
                glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                i.paint()
                glMatrixMode(GL_MODELVIEW)
                glPopMatrix()
            else:
                self.drawItemTree(i)
            
        
    def cameraPosition(self):
        """Return current position of camera based on center, dist, elevation, and azimuth"""
        center = self.opts['center']
        dist = self.opts['distance']
        elev = self.opts['elevation'] * np.pi/180.
        azim = self.opts['azimuth'] * np.pi/180.
        
        pos = Vector(
            center.x() + dist * np.cos(elev) * np.cos(azim),
            center.y() + dist * np.cos(elev) * np.sin(azim),
            center.z() + dist * np.sin(elev)
        )
        
        return pos
        
        

    def generateAxes(self):
        self.axisList = glGenLists(1)
        glNewList(self.axisList, GL_COMPILE)

        #glShadeModel(GL_FLAT)
        #glFrontFace(GL_CCW)
        #glEnable( GL_LIGHT_MODEL_TWO_SIDE )
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable( GL_BLEND )
        glEnable( GL_ALPHA_TEST )
        #glAlphaFunc( GL_ALWAYS,0.5 )
        glEnable( GL_POINT_SMOOTH )
        glDisable( GL_DEPTH_TEST )
        glBegin( GL_LINES )
        
        glColor4f(1, 1, 1, .3)
        for x in range(-10, 11):
            glVertex3f(x, -10, 0)
            glVertex3f(x,  10, 0)
        for y in range(-10, 11):
            glVertex3f(-10, y, 0)
            glVertex3f( 10, y, 0)
        
        
        glColor4f(0, 1, 0, .6)  # z is green
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 5)

        glColor4f(1, 1, 0, .6)  # y is yellow
        glVertex3f(0, 0, 0)
        glVertex3f(0, 5, 0)

        glColor4f(0, 0, 1, .6)  # x is blue
        glVertex3f(0, 0, 0)
        glVertex3f(5, 0, 0)
        glEnd()
        glEndList()
        
    def generatePoints(self):
        self.pointList = glGenLists(1)
        glNewList(self.pointList, GL_COMPILE)
        width = 7
        alpha = 0.02
        n = 40
        glPointSize( width )
        glBegin(GL_POINTS)
        for x in range(-n, n+1):
            r = (n-x)/(2.*n)
            glColor4f(r, r, r, alpha)
            for y in range(-n, n+1):
                for z in range(-n, n+1):
                    glVertex3f(x, y, z)
        glEnd()
        glEndList()
        
        
    def mousePressEvent(self, ev):
        self.mousePos = ev.pos()
        
    def mouseMoveEvent(self, ev):
        diff = ev.pos() - self.mousePos
        self.mousePos = ev.pos()
        self.opts['azimuth'] -= diff.x()
        self.opts['elevation'] = np.clip(self.opts['elevation'] + diff.y(), -90, 90)
        #print self.opts['azimuth'], self.opts['elevation']
        self.updateGL()
        
    def mouseReleaseEvent(self, ev):
        pass
        
    def wheelEvent(self, ev):
        self.opts['distance'] *= 0.999**ev.delta()
        self.updateGL()



