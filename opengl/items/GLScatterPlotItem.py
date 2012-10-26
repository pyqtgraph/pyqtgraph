from OpenGL.GL import *
from .. GLGraphicsItem import GLGraphicsItem
from pyqtgraph import QtGui
import numpy as np

__all__ = ['GLScatterPlotItem']

class GLScatterPlotItem(GLGraphicsItem):
    """Draws points at a list of 3D positions."""
    
    def __init__(self, **kwds):
        GLGraphicsItem.__init__(self)
        self.pos = []
        self.size = 10
        self.color = [1.0,1.0,1.0,0.5]
        self.pxMode = True
        self.setData(**kwds)
    
    def setData(self, **kwds):
        """
        Update the data displayed by this item. All arguments are optional; 
        for example it is allowed to update spot positions while leaving 
        colors unchanged, etc.
        
        ====================  ==================================================
        Arguments:
        ------------------------------------------------------------------------
        pos                   (N,3) array of floats specifying point locations.
        color                 (N,4) array of floats (0.0-1.0) specifying
                              spot colors OR a tuple of floats specifying
                              a single color for all spots.
        size                  (N,) array of floats specifying spot sizes or 
                              a single value to apply to all spots.
        pxMode                If True, spot sizes are expressed in pixels. 
                              Otherwise, they are expressed in item coordinates.
        ====================  ==================================================
        """
        args = ['pos', 'color', 'size', 'pxMode']
        for k in kwds.keys():
            if k not in args:
                raise Exception('Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))
        self.pos = kwds.get('pos', self.pos)
        self.color = kwds.get('color', self.color)
        self.size = kwds.get('size', self.size)
        self.pxMode = kwds.get('pxMode', self.pxMode)
        self.update()

        
    def initializeGL(self):
        
        ## Generate texture for rendering points
        w = 64
        def fn(x,y):
            r = ((x-w/2.)**2 + (y-w/2.)**2) ** 0.5
            return 200 * (w/2. - np.clip(r, w/2.-1.0, w/2.))
        pData = np.empty((w, w, 4))
        pData[:] = 255
        pData[:,:,3] = np.fromfunction(fn, pData.shape[:2])
        #print pData.shape, pData.min(), pData.max()
        pData = pData.astype(np.ubyte)
        
        self.pointTexture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.pointTexture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pData.shape[0], pData.shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, pData)
        
    def paint(self):
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable( GL_BLEND )
        glEnable( GL_ALPHA_TEST )
        glEnable( GL_POINT_SMOOTH )

        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        #glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, (0, 0, -1e-3))
        #glPointParameterfv(GL_POINT_SIZE_MAX, (65500,))
        #glPointParameterfv(GL_POINT_SIZE_MIN, (0,))
        
        glEnable(GL_POINT_SPRITE)
        glActiveTexture(GL_TEXTURE0)
        glEnable( GL_TEXTURE_2D )
        glBindTexture(GL_TEXTURE_2D, self.pointTexture)
    
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
        #glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)    ## use texture color exactly
        glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE )  ## texture modulates current color
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        if self.pxMode:     
            glVertexPointerf(self.pos)
            if isinstance(self.color, np.ndarray):
                glColorPointerf(self.color)
            else:
                if isinstance(self.color, QtGui.QColor):
                    glColor4f(*fn.glColor(self.color))
                else:
                    glColor4f(*self.color)
            
            if isinstance(self.size, np.ndarray):
                raise Exception('Array size not yet supported in pxMode (hopefully soon)')
            
            glPointSize(self.size)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glDrawArrays(GL_POINTS, 0, len(self.pos))
        else:
            
            
            for i in range(len(self.pos)):
                pos = self.pos[i]
                
                if isinstance(self.color, np.ndarray):
                    color = self.color[i]
                else:
                    color = self.color
                if isinstance(self.color, QtGui.QColor):
                    color = fn.glColor(self.color)
                    
                if isinstance(self.size, np.ndarray):
                    size = self.size[i]
                else:
                    size = self.size
                    
                pxSize = self.view().pixelSize(QtGui.QVector3D(*pos))
                
                glPointSize(size / pxSize)
                glBegin( GL_POINTS )
                glColor4f(*color)  # x is blue
                #glNormal3f(size, 0, 0)
                glVertex3f(*pos)
                glEnd()

        
        
        
        