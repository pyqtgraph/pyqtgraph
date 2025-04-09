import enum
import math
import importlib

from OpenGL import GL
from OpenGL.GL import shaders
import numpy as np

from ...Qt import QtGui, QT_LIB
from ..GLGraphicsItem import GLGraphicsItem

if QT_LIB in ["PyQt5", "PySide2"]:
    QtOpenGL = QtGui
else:
    QtOpenGL = importlib.import_module(f"{QT_LIB}.QtOpenGL")

__all__ = ['GLScatterPlotItem']


class DirtyFlag(enum.Flag):
    POSITION = enum.auto()
    COLOR = enum.auto()
    SIZE = enum.auto()


class GLScatterPlotItem(GLGraphicsItem):
    """Draws points at a list of 3D positions."""
    
    _shaderProgram = None

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
        self.dirty_bits = DirtyFlag(0)

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
            self.dirty_bits |= DirtyFlag.POSITION
        if 'color' in kwds:
            color = kwds.pop('color')
            if isinstance(color, np.ndarray):
                color = np.ascontiguousarray(color, dtype=np.float32)
                self.dirty_bits |= DirtyFlag.COLOR
            if isinstance(color, QtGui.QColor):
                color = color.getRgbF()
            self.color = color
        if 'size' in kwds:
            size = kwds.pop('size')
            if isinstance(size, np.ndarray):
                size = np.ascontiguousarray(size, dtype=np.float32)
                self.dirty_bits |= DirtyFlag.SIZE
            self.size = size
                
        self.pxMode = kwds.get('pxMode', self.pxMode)
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
        klass = GLScatterPlotItem

        if klass._shaderProgram is not None:
            return klass._shaderProgram

        ctx = QtGui.QOpenGLContext.currentContext()
        fmt = ctx.format()

        if ctx.isOpenGLES():
            if fmt.version() >= (3, 0):
                glsl_version = "#version 300 es\n"
                sources = SHADER_CORE
            else:
                glsl_version = "#version 100\n"
                sources = SHADER_LEGACY
        else:
            if fmt.version() >= (3, 1):
                glsl_version = "#version 140\n"
                sources = SHADER_CORE
            else:
                glsl_version = "#version 120\n"
                sources = SHADER_LEGACY

        compiled = [shaders.compileShader([glsl_version, v], k) for k, v in sources.items()]
        program = shaders.compileProgram(*compiled)

        # bind generic vertex attrib 0 to "a_position" so that
        # vertex attrib 0 definitely gets enabled later.
        GL.glBindAttribLocation(program, 0, "a_position")
        GL.glBindAttribLocation(program, 1, "a_color")
        GL.glBindAttribLocation(program, 2, "a_size")
        GL.glLinkProgram(program)

        klass._shaderProgram = program
        return program

    def paint(self):
        if self.pos is None:
            return

        self.setupGLState()

        mat_mvp = self.mvpMatrix()
        mat_mvp = np.array(mat_mvp.data(), dtype=np.float32)

        mat_modelview = self.modelViewMatrix()
        mat_modelview = np.array(mat_modelview.data(), dtype=np.float32)

        view = self.view()
        if self.pxMode:
            scale = 0
        else:
            scale = 2.0 * math.tan(math.radians(0.5 * view.opts["fov"])) / view.width()

        context = QtGui.QOpenGLContext.currentContext()

        if DirtyFlag.POSITION in self.dirty_bits:
            self.upload_vbo(self.m_vbo_position, self.pos)
        if DirtyFlag.COLOR in self.dirty_bits:
            self.upload_vbo(self.m_vbo_color, self.color)
        if DirtyFlag.SIZE in self.dirty_bits:
            self.upload_vbo(self.m_vbo_size, self.size)
        self.dirty_bits = DirtyFlag(0)

        if not context.isOpenGLES():
            if _is_compatibility_profile(context):
                GL.glEnable(GL.GL_POINT_SPRITE)

            GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)

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

        loc = 2
        if isinstance(self.size, np.ndarray):
            self.m_vbo_size.bind()
            GL.glVertexAttribPointer(loc, 1, GL.GL_FLOAT, False, 0, None)
            self.m_vbo_size.release()
            enabled_locs.append(loc)
        else:
            GL.glVertexAttrib1f(loc, self.size)

        for loc in enabled_locs:
            GL.glEnableVertexAttribArray(loc)

        with program:
            loc = GL.glGetUniformLocation(program, "u_mvp")
            GL.glUniformMatrix4fv(loc, 1, False, mat_mvp)

            loc = GL.glGetUniformLocation(program, "u_modelview")
            GL.glUniformMatrix4fv(loc, 1, False, mat_modelview)
            loc = GL.glGetUniformLocation(program, "u_scale")
            GL.glUniform1f(loc, scale)

            GL.glDrawArrays(GL.GL_POINTS, 0, len(self.pos))

        for loc in enabled_locs:
            GL.glDisableVertexAttribArray(loc)


def _is_compatibility_profile(context):
    # https://stackoverflow.com/questions/73745603/detect-the-opengl-context-profile-before-version-3-2
    sformat = context.format()
    profile = sformat.profile()

    # >= 3.2 has {Compatibility,Core}Profile
    # <= 3.1 is NoProfile

    if profile == sformat.OpenGLContextProfile.CompatibilityProfile:
        compat = True
    elif profile == sformat.OpenGLContextProfile.CoreProfile:
        compat = False
    else:
        compat = False
        version = sformat.version()

        if version <= (2, 1):
            compat = True
        elif version == (3, 0):
            if sformat.testOption(sformat.FormatOption.DeprecatedFunctions):
                compat = True
        elif version == (3, 1):
            if context.hasExtension(b'GL_ARB_compatibility'):
                compat = True

    return compat


## See:
##
##  http://stackoverflow.com/questions/9609423/applying-part-of-a-texture-sprite-sheet-texture-map-to-a-point-sprite-in-ios
##  http://stackoverflow.com/questions/3497068/textured-points-in-opengl-es-2-0
##
##

SHADER_LEGACY = {
    GL.GL_VERTEX_SHADER : """
        uniform float u_scale;

        uniform mat4 u_modelview;
        uniform mat4 u_mvp;
        attribute vec4 a_position;
        attribute vec4 a_color;
        attribute float a_size;
        varying vec4 v_color;

        void main() {
            gl_Position = u_mvp * a_position;
            v_color = a_color;
            gl_PointSize = a_size;

            if (u_scale != 0.0) {
                // pxMode=False
                // the modelview matrix transforms the vertex to
                // camera space, where the camera is at (0, 0, 0).
                vec4 cpos = u_modelview * a_position;
                float dist = length(cpos.xyz);
                // equations:
                //   xDist = dist * 2.0 * tan(0.5 * fov)
                //   pxSize = xDist / view_width
                // let:
                //   u_scale = 2.0 * tan(0.5 * fov) / view_width
                // then:
                //   pxSize = dist * u_scale
                float pxSize = dist * u_scale;
                gl_PointSize /= pxSize;
            }
        }
    """,
    GL.GL_FRAGMENT_SHADER : """
        #ifdef GL_ES
        precision mediump float;
        #endif

        varying vec4 v_color;
        void main()
        {
            vec2 xy = (gl_PointCoord - 0.5) * 2.0;
            if (dot(xy, xy) <= 1.0) gl_FragColor = v_color;
            else discard;
        }
    """
}

SHADER_CORE = {
    GL.GL_VERTEX_SHADER : """
        uniform float u_scale;

        uniform mat4 u_modelview;
        uniform mat4 u_mvp;
        in vec4 a_position;
        in vec4 a_color;
        in float a_size;
        out vec4 v_color;

        void main() {
            gl_Position = u_mvp * a_position;
            v_color = a_color;
            gl_PointSize = a_size;

            if (u_scale != 0.0) {
                // pxMode=False
                // the modelview matrix transforms the vertex to
                // camera space, where the camera is at (0, 0, 0).
                vec4 cpos = u_modelview * a_position;
                float dist = length(cpos.xyz);
                // equations:
                //   xDist = dist * 2.0 * tan(0.5 * fov)
                //   pxSize = xDist / view_width
                // let:
                //   u_scale = 2.0 * tan(0.5 * fov) / view_width
                // then:
                //   pxSize = dist * u_scale
                float pxSize = dist * u_scale;
                gl_PointSize /= pxSize;
            }
        }
    """,
    GL.GL_FRAGMENT_SHADER : """
        #ifdef GL_ES
        precision mediump float;
        #endif

        in vec4 v_color;
        out vec4 fragColor;
        void main()
        {
            vec2 xy = (gl_PointCoord - 0.5) * 2.0;
            if (dot(xy, xy) <= 1.0) fragColor = v_color;
            else discard;
        }
    """
}
