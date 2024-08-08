import importlib

from OpenGL.GL import *  # noqa
import numpy as np

from ... import functions as fn
from ...Qt import QtGui, QT_LIB
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem

if QT_LIB in ["PyQt5", "PySide2"]:
    QtOpenGL = QtGui
else:
    QtOpenGL = importlib.import_module(f"{QT_LIB}.QtOpenGL")

__all__ = ['GLScatterPlotItem']

class GLScatterPlotItem(GLGraphicsItem):
    """Draws points at a list of 3D positions."""
    
    def __init__(self, parentItem=None, **kwds):
        super().__init__()
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        self.pos = None
        self.size = 10
        self.color = [1.0,1.0,1.0,0.5]
        self.pxMode = True

        self.m_vbo_position = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.m_vbo_color = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.m_vbo_size = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.vbos_uploaded = False

        self.setParentItem(parentItem)
        self.setData(**kwds)

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
        self.vbos_uploaded = False
        self.update()

    def upload_vbo(self, vbo, arr):
        if arr is None:
            vbo.destroy()
            return
        if not vbo.isCreated():
            vbo.create()
        vbo.bind()
        vbo.allocate(arr, arr.nbytes)
        vbo.release()

    def paint(self):
        if self.pos is None:
            return

        self.setupGLState()

        mat_mvp = self.mvpMatrix()
        mat_mvp = np.array(mat_mvp.data(), dtype=np.float32)

        context = QtGui.QOpenGLContext.currentContext()
        sformat = context.format()

        if not self.vbos_uploaded:
            self.upload_vbo(self.m_vbo_position, self.pos)
            if isinstance(self.color, np.ndarray):
                self.upload_vbo(self.m_vbo_color, self.color)
            if self.pxMode:
                if isinstance(self.size, np.ndarray):
                    # for this case, size only needs to be uploaded once
                    self.upload_vbo(self.m_vbo_size, self.size)
            else:
                # for non-pxMode, upload a dummy array to allocate the buffer
                point_size = np.zeros(len(self.pos), dtype=np.float32)
                self.upload_vbo(self.m_vbo_size, point_size)
            self.vbos_uploaded = True

        if self.pxMode:
            # if size was an (static) array, it has already been uploaded
            point_size = self.size
        else:
            # whether size is a scalar or an array, non-pxMode ends up with
            # an array that needs to be re-uploaded on each paint.
            if not (view := self.view()):
                view = self.parentItem().view()
            gpos = self.mapToView(self.pos.T).T
            pxSize = view.pixelSize(gpos)
            point_size = np.array(self.size / pxSize, dtype=np.float32)

            self.m_vbo_size.bind()
            self.m_vbo_size.write(0, point_size, point_size.nbytes)
            self.m_vbo_size.release()

        if sformat.profile() == QtGui.QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile:
            # setting GL_POINT_SPRITE is only needed for OpenGL 2.1
            # In Core profiles, it is an error to set it even.
            glEnable(GL_POINT_SPRITE)

        if context.isOpenGLES():
            shader_name = 'pointSprite-es2'
        else:
            glEnable(GL_PROGRAM_POINT_SIZE)
            shader_name = 'pointSprite'
        shader = shaders.getShaderProgram(shader_name)

        enabled_locs = []

        if (loc := glGetAttribLocation(shader.program(), "a_position")) != -1:
            self.m_vbo_position.bind()
            glVertexAttribPointer(loc, 3, GL_FLOAT, False, 0, None)
            self.m_vbo_position.release()
            enabled_locs.append(loc)

        if (loc := glGetAttribLocation(shader.program(), "a_color")) != -1:
            if isinstance(self.color, np.ndarray):
                self.m_vbo_color.bind()
                glVertexAttribPointer(loc, 4, GL_FLOAT, False, 0, None)
                self.m_vbo_color.release()
                enabled_locs.append(loc)
            else:
                color = self.color
                if isinstance(color, QtGui.QColor):
                    color = color.getRgbF()
                glVertexAttrib4f(loc, *color)

        if (loc := glGetAttribLocation(shader.program(), "a_size")) != -1:
            if isinstance(point_size, np.ndarray):
                self.m_vbo_size.bind()
                glVertexAttribPointer(loc, 1, GL_FLOAT, False, 0, None)
                self.m_vbo_size.release()
                enabled_locs.append(loc)
            else:
                glVertexAttrib1f(loc, point_size)

        for loc in enabled_locs:
            glEnableVertexAttribArray(loc)

        with shader:
            glUniformMatrix4fv(shader.uniform("u_mvp"), 1, False, mat_mvp)

            glDrawArrays(GL_POINTS, 0, len(self.pos))

        for loc in enabled_locs:
            glDisableVertexAttribArray(loc)
