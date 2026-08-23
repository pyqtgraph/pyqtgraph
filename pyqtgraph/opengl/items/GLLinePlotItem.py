import enum

from OpenGL import GL
import numpy as np

from ...Qt import QtGui, QtOpenGL
from ... import functions as fn
from ..GLGraphicsItem import GLGraphicsItem

__all__ = ['GLLinePlotItem']


class DirtyFlag(enum.Flag):
    POSITION = enum.auto()
    COLOR = enum.auto()


class GLLinePlotItem(GLGraphicsItem):
    """Draws line plots in 3D."""

    _shaderProgram = None

    def __init__(self, parentItem=None, **kwargs):
        """All keyword arguments are passed to setData()"""
        super().__init__()
        glopts = kwargs.pop('glOptions', 'additive')
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
        self.setData(**kwargs)
    
    def setData(self, **kwargs):
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
        for k in kwargs.keys():
            if k not in args:
                raise Exception('Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))
        if 'pos' in kwargs:
            pos = kwargs.pop('pos')
            self.pos = np.ascontiguousarray(pos, dtype=np.float32)
            self.dirty_bits |= DirtyFlag.POSITION
        if 'color' in kwargs:
            color = kwargs.pop('color')
            if isinstance(color, np.ndarray):
                color = np.ascontiguousarray(color, dtype=np.float32)
                self.dirty_bits |= DirtyFlag.COLOR
            if isinstance(color, str):
                color = fn.mkColor(color)
            if isinstance(color, QtGui.QColor):
                color = color.getRgbF()
            self.color = color
        for k, v in kwargs.items():
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

        program = QtOpenGL.QOpenGLShaderProgram()
        for shader_type, src in sources.items():
            if not program.addShaderFromSourceCode(shader_type, glsl_version + src):
                raise RuntimeError(program.log())

        # bind generic vertex attribs 0 and 1 to "a_position" and "a_color"
        # so that they definitely get enabled later.
        program.bindAttributeLocation("a_position", 0)
        program.bindAttributeLocation("a_color", 1)
        if not program.link():
            raise RuntimeError(program.log())

        klass._shaderProgram = program
        return program

    def paint(self):
        if self.pos is None:
            return
        self.setupGLState()

        mat_mvp = self.mvpMatrix()

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
        program.setAttributeBuffer(loc, GL.GL_FLOAT, 0, 3)
        self.m_vbo_position.release()
        enabled_locs.append(loc)

        loc = 1
        if isinstance(self.color, np.ndarray):
            self.m_vbo_color.bind()
            program.setAttributeBuffer(loc, GL.GL_FLOAT, 0, 4)
            self.m_vbo_color.release()
            enabled_locs.append(loc)
        else:
            program.setAttributeValue(loc, QtGui.QColor.fromRgbF(*self.color))

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
            program.enableAttributeArray(loc)

        program.bind()
        program.setUniformValue("u_mvp", mat_mvp)

        if self.mode == 'line_strip':
            GL.glDrawArrays(GL.GL_LINE_STRIP, 0, len(self.pos))
        elif self.mode == 'lines':
            GL.glDrawArrays(GL.GL_LINES, 0, len(self.pos))

        program.release()

        for loc in enabled_locs:
            program.disableAttributeArray(loc)

        if enable_aa:
            GL.glDisable(GL.GL_LINE_SMOOTH)
            GL.glDisable(GL.GL_BLEND)
        
        GL.glLineWidth(1.0)


SHADER_LEGACY = {
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex : """
        uniform mat4 u_mvp;
        attribute vec4 a_position;
        attribute vec4 a_color;
        varying vec4 v_color;
        void main() {
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """,
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment : """
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
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex : """
        uniform mat4 u_mvp;
        in vec4 a_position;
        in vec4 a_color;
        out vec4 v_color;
        void main() {
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """,
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment : """
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
