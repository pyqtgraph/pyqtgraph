from OpenGL.GL import *  # noqa
import numpy as np

from ... import functions as fn
from ...Qt import QtGui
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem

__all__ = ['GLScatterPlotItem']

class GLScatterPlotItem(GLGraphicsItem):
    """Draws points at a list of 3D positions."""
    
    def __init__(self, parentItem=None, **kwds):
        super().__init__(parentItem=parentItem)
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        self.pos = None
        self.size = 10
        self.color = [1.0,1.0,1.0,0.5]
        self.pxMode = True
        self.setData(**kwds)
        self.shader = None
    
    def setData(self, **kwds):
        """
        Update the data displayed by this item. All arguments are optional; 
        for example it is allowed to update spot positions while leaving 
        colors unchanged, etc.
        
        ====================  ==================================================
        **Arguments:**
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
            
        if 'pos' in kwds:
            pos = kwds.pop('pos')
            self.pos = np.ascontiguousarray(pos, dtype=np.float32)
        if 'color' in kwds:
            color = kwds.pop('color')
            if isinstance(color, np.ndarray):
                color = np.ascontiguousarray(color, dtype=np.float32)
            self.color = color
        if 'size' in kwds:
            size = kwds.pop('size')
            if isinstance(size, np.ndarray):
                size = np.ascontiguousarray(size, dtype=np.float32)
            self.size = size
                
        self.pxMode = kwds.get('pxMode', self.pxMode)
        self.update()

    def initializeGL(self):
        if self.shader is not None:
            return
        
        ## Generate texture for rendering points
        w = 64
        def genTexture(x,y):
            r = np.hypot((x-(w-1)/2.), (y-(w-1)/2.))
            return 255 * (w / 2 - fn.clip_array(r, w / 2 - 1, w / 2))
        pData = np.empty((w, w, 4))
        pData[:] = 255
        pData[:,:,3] = np.fromfunction(genTexture, pData.shape[:2])
        pData = pData.astype(np.ubyte)
        
        if getattr(self, "pointTexture", None) is None:
            self.pointTexture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.pointTexture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pData.shape[0], pData.shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, pData)
        
        self.shader = shaders.getShaderProgram('pointSprite')
        
    #def setupGLState(self):
        #"""Prepare OpenGL state for drawing. This function is called immediately before painting."""
        ##glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  ## requires z-sorting to render properly.
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        #glEnable( GL_BLEND )
        #glEnable( GL_ALPHA_TEST )
        #glDisable( GL_DEPTH_TEST )
        
        ##glEnable( GL_POINT_SMOOTH )

        ##glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        ##glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, (0, 0, -1e-3))
        ##glPointParameterfv(GL_POINT_SIZE_MAX, (65500,))
        ##glPointParameterfv(GL_POINT_SIZE_MIN, (0,))
        
    def paint(self):
        if self.pos is None:
            return

        self.setupGLState()
        
        glEnable(GL_POINT_SPRITE)
        
        glActiveTexture(GL_TEXTURE0)
        glEnable( GL_TEXTURE_2D )
        glBindTexture(GL_TEXTURE_2D, self.pointTexture)
    
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
        #glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)    ## use texture color exactly
        #glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE )  ## texture modulates current color
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        
            
        with self.shader:
            #glUniform1i(self.shader.uniform('texture'), 0)  ## inform the shader which texture to use
            glEnableClientState(GL_VERTEX_ARRAY)
            try:
                pos = self.pos
                #if pos.ndim > 2:
                    #pos = pos.reshape((-1, pos.shape[-1]))
                glVertexPointerf(pos)
            
                if isinstance(self.color, np.ndarray):
                    glEnableClientState(GL_COLOR_ARRAY)
                    glColorPointerf(self.color)
                else:
                    color = self.color
                    if isinstance(color, QtGui.QColor):
                        color = color.getRgbF()
                    glColor4f(*color)
                
                if not self.pxMode or isinstance(self.size, np.ndarray):
                    glEnableClientState(GL_NORMAL_ARRAY)
                    norm = np.zeros(pos.shape, dtype=np.float32)
                    if self.pxMode:
                        norm[...,0] = self.size
                    else:
                        gpos = self.mapToView(pos.transpose()).transpose()
                        if self.view():
                            pxSize = self.view().pixelSize(gpos)
                        else:
                            pxSize = self.parentItem().view().pixelSize(gpos)
                        norm[...,0] = self.size / pxSize
        
                    glNormalPointerf(norm)
                else:
                    glNormal3f(self.size, 0, 0)  ## vertex shader uses norm.x to determine point size
                    #glPointSize(self.size)
                glDrawArrays(GL_POINTS, 0, pos.shape[0])
            finally:
                glDisableClientState(GL_NORMAL_ARRAY)
                glDisableClientState(GL_VERTEX_ARRAY)
                glDisableClientState(GL_COLOR_ARRAY)
                #posVBO.unbind()
                ##fixes #145
                glDisable( GL_TEXTURE_2D )
                                
        #for i in range(len(self.pos)):
            #pos = self.pos[i]
            
            #if isinstance(self.color, np.ndarray):
                #color = self.color[i]
            #else:
                #color = self.color
            #if isinstance(self.color, QtGui.QColor):
                #color = fn.glColor(self.color)
                
            #if isinstance(self.size, np.ndarray):
                #size = self.size[i]
            #else:
                #size = self.size
                
            #pxSize = self.view().pixelSize(QtGui.QVector3D(*pos))
            
            #glPointSize(size / pxSize)
            #glBegin( GL_POINTS )
            #glColor4f(*color)  # x is blue
            ##glNormal3f(size, 0, 0)
            #glVertex3f(*pos)
            #glEnd()

        
        
        
        
