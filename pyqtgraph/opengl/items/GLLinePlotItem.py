import importlib

from OpenGL.GL import *  # noqa
import numpy as np

from ...Qt import QtGui, QT_LIB
from ... import functions as fn
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem

if QT_LIB in ["PyQt5", "PySide2"]:
    QtOpenGL = QtGui
else:
    QtOpenGL = importlib.import_module(f"{QT_LIB}.QtOpenGL")

__all__ = ['GLLinePlotItem']

class GLLinePlotItem(GLGraphicsItem):
    """Draws line plots in 3D."""
    
    def __init__(self, parentItem=None, **kwds):
        """All keyword arguments are passed to setData()"""
        super().__init__()
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        self.pos = None
        self.mode = 'line_strip'
        self.width = 1.
        self.color = (1.0,1.0,1.0,1.0)
        self.antialias = False

        self.m_vbo_position = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.m_vbo_color = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.vbos_uploaded = False

        self.setParentItem(parentItem)
        self.setData(**kwds)
    
    def setData(self, **kwds):
        """
        Update the data displayed by this item. All arguments are optional; 
        for example it is allowed to update vertex positions while leaving 
        colors unchanged, etc.
        
        ====================  ==================================================
        **Arguments:**
        ------------------------------------------------------------------------
        pos                   (N,3) array of floats specifying point locations.
        color                 (N,4) array of floats (0.0-1.0) or
                              tuple of floats specifying
                              a single color for the entire item.
        width                 float specifying line width
        antialias             enables smooth line drawing
        mode                  'lines': Each pair of vertexes draws a single line
                                       segment.
                              'line_strip': All vertexes are drawn as a
                                            continuous set of line segments.
        ====================  ==================================================
        """
        args = ['pos', 'color', 'width', 'mode', 'antialias']
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
        for k, v in kwds.items():
            setattr(self, k, v)

        if self.mode not in ['line_strip', 'lines']:
            raise ValueError("Unknown line mode '%s'. (must be 'lines' or 'line_strip')" % self.mode)

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

        if not self.vbos_uploaded:
            self.upload_vbo(self.m_vbo_position, self.pos)
            if isinstance(self.color, np.ndarray):
                self.upload_vbo(self.m_vbo_color, self.color)

        shader = shaders.getShaderProgram(None)

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
                if isinstance(color, str):
                    color = fn.mkColor(color)
                if isinstance(color, QtGui.QColor):
                    color = color.getRgbF()
                glVertexAttrib4f(loc, *color)

        glLineWidth(self.width)

        enable_aa = self.antialias and not context.isOpenGLES()

        if enable_aa:
            glEnable(GL_LINE_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        for loc in enabled_locs:
            glEnableVertexAttribArray(loc)

        with shader:
            glUniformMatrix4fv(shader.uniform("u_mvp"), 1, False, mat_mvp)

            if self.mode == 'line_strip':
                glDrawArrays(GL_LINE_STRIP, 0, len(self.pos))
            elif self.mode == 'lines':
                glDrawArrays(GL_LINES, 0, len(self.pos))

        for loc in enabled_locs:
            glDisableVertexAttribArray(loc)

        if enable_aa:
            glDisable(GL_LINE_SMOOTH)
            glDisable(GL_BLEND)
        
        glLineWidth(1.0)
