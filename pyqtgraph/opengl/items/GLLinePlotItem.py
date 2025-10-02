import enum
import importlib

from OpenGL import GL
from OpenGL.GL import shaders
import numpy as np

from ...Qt import QtGui, QT_LIB
from ... import functions as fn
from ..GLGraphicsItem import GLGraphicsItem

if QT_LIB in ["PyQt5", "PySide2"]:
    QtOpenGL = QtGui
else:
    QtOpenGL = importlib.import_module(f"{QT_LIB}.QtOpenGL")

__all__ = ['GLLinePlotItem']


class DirtyFlag(enum.Flag):
    POSITION = enum.auto()
    COLOR = enum.auto()


class GLLinePlotItem(GLGraphicsItem):
    """Draws line plots in 3D."""

    _shaderProgram = None

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
        self.dirty_bits = DirtyFlag(0)

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
            self.dirty_bits |= DirtyFlag.POSITION
        if 'color' in kwds:
            color = kwds.pop('color')
            if isinstance(color, np.ndarray):
                color = np.ascontiguousarray(color, dtype=np.float32)
                self.dirty_bits |= DirtyFlag.COLOR
            if isinstance(color, str):
                color = fn.mkColor(color)
            if isinstance(color, QtGui.QColor):
                color = color.getRgbF()
            self.color = color
        for k, v in kwds.items():
            setattr(self, k, v)

        if self.mode not in ['line_strip', 'lines']:
            raise ValueError("Unknown line mode '%s'. (must be 'lines' or 'line_strip')" % self.mode)

        self.update()

    def upload_vbo(self, vbo, arr):
        if arr is None:
            vbo.destroy()
            return
        if not vbo.isCreated():
            vbo.create()
        vbo.bind()
        if vbo.size() != arr.nbytes:
            vbo.allocate(arr, arr.nbytes)
        else:
            vbo.write(0, arr, arr.nbytes)
        vbo.release()

    @staticmethod
    def getShaderProgram():
        klass = GLLinePlotItem

        if klass._shaderProgram is not None:
            return klass._shaderProgram

        ctx = QtGui.QOpenGLContext.currentContext()
        fmt = ctx.format()

        if ctx.isOpenGLES():
            if fmt.version() >= (3, 0):
                glsl_version = "#version 300 es\n"
                sources = SHADER_CORE
            else:
                glsl_version = ""
                sources = SHADER_LEGACY
        else:
            if fmt.version() >= (3, 1):
                glsl_version = "#version 140\n"
                sources = SHADER_CORE
            else:
                glsl_version = ""
                sources = SHADER_LEGACY

        compiled = [shaders.compileShader([glsl_version, v], k) for k, v in sources.items()]
        program = shaders.compileProgram(*compiled)

        # bind generic vertex attrib 0 to "a_position" so that
        # vertex attrib 0 definitely gets enabled later.
        GL.glBindAttribLocation(program, 0, "a_position")
        GL.glBindAttribLocation(program, 1, "a_color")
        GL.glLinkProgram(program)

        klass._shaderProgram = program
        return program

    def paint(self):
        if self.pos is None:
            return
        self.setupGLState()

        mat_mvp = self.mvpMatrix()
        mat_mvp = np.array(mat_mvp.data(), dtype=np.float32)

        context = QtGui.QOpenGLContext.currentContext()

        if DirtyFlag.POSITION in self.dirty_bits:
            self.upload_vbo(self.m_vbo_position, self.pos)
        if DirtyFlag.COLOR in self.dirty_bits:
            self.upload_vbo(self.m_vbo_color, self.color)
        self.dirty_bits = DirtyFlag(0)

        program = self.getShaderProgram()

        enabled_locs = []

        loc = 0
        self.m_vbo_position.bind()
        GL.glVertexAttribPointer(loc, 3, GL.GL_FLOAT, False, 0, None)
        self.m_vbo_position.release()
        enabled_locs.append(loc)

        loc = 1
        if isinstance(self.color, np.ndarray):
            self.m_vbo_color.bind()
            GL.glVertexAttribPointer(loc, 4, GL.GL_FLOAT, False, 0, None)
            self.m_vbo_color.release()
            enabled_locs.append(loc)
        else:
            GL.glVertexAttrib4f(loc, *self.color)

        enable_aa = self.antialias and not context.isOpenGLES()

        if enable_aa:
            GL.glEnable(GL.GL_LINE_SMOOTH)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                                   GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)

        sfmt = context.format()
        core_forward_compatible = (
            sfmt.profile() == sfmt.OpenGLContextProfile.CoreProfile
            and not sfmt.testOption(sfmt.FormatOption.DeprecatedFunctions)
        )
        if not core_forward_compatible:
            # Core Forward Compatible profiles will return error for
            # any width that is not 1.0
            GL.glLineWidth(self.width)

        for loc in enabled_locs:
            GL.glEnableVertexAttribArray(loc)

        with program:
            loc = GL.glGetUniformLocation(program, "u_mvp")
            GL.glUniformMatrix4fv(loc, 1, False, mat_mvp)

            if self.mode == 'line_strip':
                GL.glDrawArrays(GL.GL_LINE_STRIP, 0, len(self.pos))
            elif self.mode == 'lines':
                GL.glDrawArrays(GL.GL_LINES, 0, len(self.pos))

        for loc in enabled_locs:
            GL.glDisableVertexAttribArray(loc)

        if enable_aa:
            GL.glDisable(GL.GL_LINE_SMOOTH)
            GL.glDisable(GL.GL_BLEND)
        
        GL.glLineWidth(1.0)


SHADER_LEGACY = {
    GL.GL_VERTEX_SHADER : """
        uniform mat4 u_mvp;
        attribute vec4 a_position;
        attribute vec4 a_color;
        varying vec4 v_color;
        void main() {
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """,
    GL.GL_FRAGMENT_SHADER : """
        #ifdef GL_ES
        precision mediump float;
        #endif
        varying vec4 v_color;
        void main() {
            gl_FragColor = v_color;
        }
    """,
}

SHADER_CORE = {
    GL.GL_VERTEX_SHADER : """
        uniform mat4 u_mvp;
        in vec4 a_position;
        in vec4 a_color;
        out vec4 v_color;
        void main() {
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """,
    GL.GL_FRAGMENT_SHADER : """
        #ifdef GL_ES
        precision mediump float;
        #endif
        in vec4 v_color;
        out vec4 fragColor;
        void main() {
            fragColor = v_color;
        }
    """,
}
