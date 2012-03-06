from OpenGL.GL import *
from .. GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QtGui
import numpy as np

__all__ = ['GLVolumeItem']

class GLVolumeItem(GLGraphicsItem):
    def initializeGL(self):
        n = 128
        self.data = np.random.randint(0, 255, size=4*n**3).astype(np.uint8).reshape((n,n,n,4))
        self.data[...,3] *= 0.1
        for i in range(n):
            self.data[i,:,:,0] = i*256./n
        glEnable(GL_TEXTURE_3D)
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.texture)
        #glTexImage3D( GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, void *data );
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        #glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, )  ## black/transparent by default
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, n, n, n, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.data)
        glDisable(GL_TEXTURE_3D)
        
        self.lists = {}
        for ax in [0,1,2]:
            for d in [-1, 1]:
                l = glGenLists(1)
                self.lists[(ax,d)] = l
                glNewList(l, GL_COMPILE)
                self.drawVolume(ax, d)
                glEndList()

                
    def paint(self):
        
        glEnable(GL_TEXTURE_3D)
        glBindTexture(GL_TEXTURE_3D, self.texture)
        
        glDisable(GL_DEPTH_TEST)
        #glDisable(GL_CULL_FACE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable( GL_BLEND )
        glEnable( GL_ALPHA_TEST )

        view = self.view()
        cam = view.cameraPosition()
        cam = np.array([cam.x(), cam.y(), cam.z()])
        ax = np.argmax(abs(cam))
        d = 1 if cam[ax] > 0 else -1
        glCallList(self.lists[(ax,d)])  ## draw axes
        glDisable(GL_TEXTURE_3D)
                
    def drawVolume(self, ax, d):
        slices = 256
        N = 5
        
        imax = [0,1,2]
        imax.remove(ax)
        
        tp = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        vp = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        tp[0][imax[0]] = 0
        tp[0][imax[1]] = 0
        tp[1][imax[0]] = 1
        tp[1][imax[1]] = 0
        tp[2][imax[0]] = 1
        tp[2][imax[1]] = 1
        tp[3][imax[0]] = 0
        tp[3][imax[1]] = 1
        
        vp[0][imax[0]] = -N
        vp[0][imax[1]] = -N
        vp[1][imax[0]] = N
        vp[1][imax[1]] = -N
        vp[2][imax[0]] = N
        vp[2][imax[1]] = N
        vp[3][imax[0]] = -N
        vp[3][imax[1]] = N
        r = range(slices)
        if d == -1:
            r = r[::-1]
            
        glBegin(GL_QUADS)
        for i in r:
            z = float(i)/(slices-1.)
            w = float(i)*10./(slices-1.) - 5.
            
            tp[0][ax] = z
            tp[1][ax] = z
            tp[2][ax] = z
            tp[3][ax] = z
            
            vp[0][ax] = w
            vp[1][ax] = w
            vp[2][ax] = w
            vp[3][ax] = w
            
            
            glTexCoord3f(*tp[0])
            glVertex3f(*vp[0])
            glTexCoord3f(*tp[1])
            glVertex3f(*vp[1])
            glTexCoord3f(*tp[2])
            glVertex3f(*vp[2])
            glTexCoord3f(*tp[3])
            glVertex3f(*vp[3])
        glEnd()
        
        
        
        
        
        
        
        
        
        ## Interesting idea:
        ## remove projection/modelview matrixes, recreate in texture coords. 
        ## it _sorta_ works, but needs tweaking.
        #mvm = glGetDoublev(GL_MODELVIEW_MATRIX)
        #pm = glGetDoublev(GL_PROJECTION_MATRIX)
        #m = QtGui.QMatrix4x4(mvm.flatten()).inverted()[0]
        #p = QtGui.QMatrix4x4(pm.flatten()).inverted()[0]
        
        #glMatrixMode(GL_PROJECTION)
        #glPushMatrix()
        #glLoadIdentity()
        #N=1
        #glOrtho(-N,N,-N,N,-100,100)
        
        #glMatrixMode(GL_MODELVIEW)
        #glLoadIdentity()
        
        
        #glMatrixMode(GL_TEXTURE)
        #glLoadIdentity()
        #glMultMatrixf(m.copyDataTo())
        
        #view = self.view()
        #w = view.width()
        #h = view.height()
        #dist = view.opts['distance']
        #fov = view.opts['fov']
        #nearClip = dist * .1
        #farClip = dist * 5.
        #r = nearClip * np.tan(fov)
        #t = r * h / w
        
        #p = QtGui.QMatrix4x4()
        #p.frustum( -r, r, -t, t, nearClip, farClip)
        #glMultMatrixf(p.inverted()[0].copyDataTo())
        
        
        #glBegin(GL_QUADS)
        
        #M=1
        #for i in range(500):
            #z = i/500.
            #w = -i/500.
            #glTexCoord3f(-M, -M, z)
            #glVertex3f(-N, -N, w)
            #glTexCoord3f(M, -M, z)
            #glVertex3f(N, -N, w)
            #glTexCoord3f(M, M, z)
            #glVertex3f(N, N, w)
            #glTexCoord3f(-M, M, z)
            #glVertex3f(-N, N, w)
        #glEnd()
        #glDisable(GL_TEXTURE_3D)

        #glMatrixMode(GL_PROJECTION)
        #glPopMatrix()
        
        

