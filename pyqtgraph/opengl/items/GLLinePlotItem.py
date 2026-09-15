import enum
import textwrap

import numpy as np

from ...Qt import QtGui, QtOpenGL
from ...Qt import OpenGLConstants as GLC
from ...Qt import OpenGLHelpers
from ...Qt.OpenGLHelpers import upload_vbo
from ... import functions as fn
from ..GLGraphicsItem import GLGraphicsItem

__all__ = ['GLLinePlotItem']


class DirtyFlag(enum.Flag):
    POSITION = enum.auto()
    COLOR = enum.auto()


class GLLinePlotItem(GLGraphicsItem):
    """Draws line plots in 3D."""

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

    def cleanupGL(self):
        self.m_vbo_position.destroy()
        self.m_vbo_color.destroy()
        self.dirty_bits = DirtyFlag.POSITION | DirtyFlag.COLOR
    
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

    def shaderProgram(self, view):
        klass = self.__class__
        cache_key = f'{klass.__module__}.{klass.__qualname__}'

        shaders_cache = self.shadersCache(view=view)

        if (program := shaders_cache.get(cache_key)) is None:
            program = OpenGLHelpers.compile_and_link(
                view.context(),
                sources_core=SHADER_CORE,
                sources_legacy=SHADER_LEGACY,
                attributes=dict(a_position=0, a_color=1),
            )
            shaders_cache[cache_key] = program

        return program

    def paint(self):
        if self.pos is None:
            return

        if (view := self.view()) is None:
            return
        context = view.context()
        glfn = self.glFunctions(context)

        self.setupGLState(context=context)

        mat_mvp = self.mvpMatrix(view=view)

        if DirtyFlag.POSITION in self.dirty_bits:
            upload_vbo(self.m_vbo_position, self.pos)
        if DirtyFlag.COLOR in self.dirty_bits:
            if isinstance(self.color, np.ndarray):
                upload_vbo(self.m_vbo_color, self.color)
        self.dirty_bits = DirtyFlag(0)

        program = self.shaderProgram(view)

        enabled_locs = []

        loc = 0
        self.m_vbo_position.bind()
        program.setAttributeBuffer(loc, GLC.GL_FLOAT, 0, 3)
        self.m_vbo_position.release()
        enabled_locs.append(loc)

        loc = 1
        if isinstance(self.color, np.ndarray):
            self.m_vbo_color.bind()
            program.setAttributeBuffer(loc, GLC.GL_FLOAT, 0, 4)
            self.m_vbo_color.release()
            enabled_locs.append(loc)
        else:
            program.setAttributeValue(loc, QtGui.QColor.fromRgbF(*self.color))

        enable_aa = self.antialias and not context.isOpenGLES()

        if enable_aa:
            glfn.glEnable(GLC.GL_LINE_SMOOTH)
            glfn.glEnable(GLC.GL_BLEND)
            glfn.glBlendFuncSeparate(GLC.GL_SRC_ALPHA, GLC.GL_ONE_MINUS_SRC_ALPHA,
                                   GLC.GL_ONE, GLC.GL_ONE_MINUS_SRC_ALPHA)
            glfn.glHint(GLC.GL_LINE_SMOOTH_HINT, GLC.GL_NICEST)

        sfmt = context.format()
        core_forward_compatible = (
            sfmt.profile() == sfmt.OpenGLContextProfile.CoreProfile
            and not sfmt.testOption(sfmt.FormatOption.DeprecatedFunctions)
        )
        if not core_forward_compatible:
            # Core Forward Compatible profiles will return error for
            # any width that is not 1.0
            glfn.glLineWidth(self.width)

        for loc in enabled_locs:
            program.enableAttributeArray(loc)

        program.bind()
        program.setUniformValue("u_mvp", mat_mvp)

        if self.mode == 'line_strip':
            glfn.glDrawArrays(GLC.GL_LINE_STRIP, 0, len(self.pos))
        elif self.mode == 'lines':
            glfn.glDrawArrays(GLC.GL_LINES, 0, len(self.pos))

        program.release()

        for loc in enabled_locs:
            program.disableAttributeArray(loc)

        if enable_aa:
            glfn.glDisable(GLC.GL_LINE_SMOOTH)
            glfn.glDisable(GLC.GL_BLEND)
        
        glfn.glLineWidth(1.0)


SHADER_LEGACY = {
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex : textwrap.dedent("""
        uniform mat4 u_mvp;
        attribute vec4 a_position;
        attribute vec4 a_color;
        varying vec4 v_color;
        void main() {
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """),
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment : textwrap.dedent("""
        #ifdef GL_ES
        precision mediump float;
        #endif
        varying vec4 v_color;
        void main() {
            gl_FragColor = v_color;
        }
    """),
}

SHADER_CORE = {
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex : textwrap.dedent("""
        uniform mat4 u_mvp;
        in vec4 a_position;
        in vec4 a_color;
        out vec4 v_color;
        void main() {
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """),
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment : textwrap.dedent("""
        #ifdef GL_ES
        precision mediump float;
        #endif
        in vec4 v_color;
        out vec4 fragColor;
        void main() {
            fragColor = v_color;
        }
    """),
}
