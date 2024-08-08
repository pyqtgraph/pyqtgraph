import ctypes
import importlib

from OpenGL.GL import *  # noqa
import numpy as np

from ...Qt import QtGui, QT_LIB
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem

if QT_LIB in ["PyQt5", "PySide2"]:
    QtOpenGL = QtGui
else:
    QtOpenGL = importlib.import_module(f"{QT_LIB}.QtOpenGL")

__all__ = ['GLImageItem']

class GLImageItem(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem.GLGraphicsItem>`
    
    Displays image data as a textured quad.
    """
    
    
    def __init__(self, data, smooth=False, glOptions='translucent', parentItem=None):
        """
        
        ==============  =======================================================================================
        **Arguments:**
        data            Volume data to be rendered. *Must* be 3D numpy array (x, y, RGBA) with dtype=ubyte.
                        (See functions.makeRGBA)
        smooth          (bool) If True, the volume slices are rendered with linear interpolation 
        ==============  =======================================================================================
        """
        
        super().__init__()
        self.setGLOptions(glOptions)
        self.smooth = smooth
        self._needUpdate = False
        self.texture = None
        self.m_vbo_position = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.setParentItem(parentItem)
        self.setData(data)

    def setData(self, data):
        self.data = data
        self._needUpdate = True
        self.update()
        
    def _updateTexture(self):
        if self.texture is None:
            self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        if self.smooth:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        shape = self.data.shape
        
        context = QtGui.QOpenGLContext.currentContext()
        if not context.isOpenGLES():
            ## Test texture dimensions first
            glTexImage2D(GL_PROXY_TEXTURE_2D, 0, GL_RGBA, shape[0], shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 0:
                raise Exception("OpenGL failed to create 2D texture (%dx%d); too large for this hardware." % shape[:2])
        
        data = np.ascontiguousarray(self.data.transpose((1,0,2)))
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, shape[0], shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

        x, y = shape[:2]
        pos = np.array([
            [0, 0, 0, 0],
            [x, 0, 1, 0],
            [0, y, 0, 1],
            [x, y, 1, 1],
        ], dtype=np.float32)
        vbo = self.m_vbo_position
        if not vbo.isCreated():
            vbo.create()
        vbo.bind()
        vbo.allocate(pos, pos.nbytes)
        vbo.release()

    def paint(self):
        if self._needUpdate:
            self._updateTexture()
            self._needUpdate = False
        
        self.setupGLState()

        mat_mvp = self.mvpMatrix()
        mat_mvp = np.array(mat_mvp.data(), dtype=np.float32)

        shader = shaders.getShaderProgram('texture2d')
        loc_pos = glGetAttribLocation(shader.program(), "a_position")
        loc_tex = glGetAttribLocation(shader.program(), "a_texcoord")
        self.m_vbo_position.bind()
        glVertexAttribPointer(loc_pos, 2, GL_FLOAT, False, 4*4, None)
        glVertexAttribPointer(loc_tex, 2, GL_FLOAT, False, 4*4, ctypes.c_void_p(2*4))
        self.m_vbo_position.release()
        enabled_locs = [loc_pos, loc_tex]

        glBindTexture(GL_TEXTURE_2D, self.texture)

        for loc in enabled_locs:
            glEnableVertexAttribArray(loc)

        with shader:
            glUniformMatrix4fv(shader.uniform("u_mvp"), 1, False, mat_mvp)

            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        for loc in enabled_locs:
            glDisableVertexAttribArray(loc)

        glBindTexture(GL_TEXTURE_2D, 0)
