import enum
import math

import numpy as np

from ...Qt import QtGui, QtOpenGL
from ...Qt import OpenGLConstants as GLC
from ..GLGraphicsItem import GLGraphicsItem

__all__ = ['GLScatterPlotItem']


class DirtyFlag(enum.Flag):
    POSITION = enum.auto()
    COLOR = enum.auto()
    SIZE = enum.auto()


class GLScatterPlotItem(GLGraphicsItem):
    """Draws points at a list of 3D positions."""
    
    _shaderProgram = None

    def __init__(self, parentItem=None, **kwargs):
        super().__init__()
        glopts = kwargs.pop('glOptions', 'additive')
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
        self.setData(**kwargs)

    def setData(self, **kwargs):
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
            if isinstance(color, QtGui.QColor):
                color = color.getRgbF()
            self.color = color
        if 'size' in kwargs:
            size = kwargs.pop('size')
            if isinstance(size, np.ndarray):
                size = np.ascontiguousarray(size, dtype=np.float32)
                self.dirty_bits |= DirtyFlag.SIZE
            self.size = size
                
        self.pxMode = kwargs.get('pxMode', self.pxMode)
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

        program = QtOpenGL.QOpenGLShaderProgram()
        for shader_type, src in sources.items():
            if not program.addShaderFromSourceCode(shader_type, glsl_version + src):
                raise RuntimeError(program.log())

        # bind generic vertex attribs 0, 1 and 2 to "a_position", "a_color"
        # and "a_size" so that they definitely get enabled later.
        program.bindAttributeLocation("a_position", 0)
        program.bindAttributeLocation("a_color", 1)
        program.bindAttributeLocation("a_size", 2)
        if not program.link():
            raise RuntimeError(program.log())

        klass._shaderProgram = program
        return program

    def paint(self):
        if self.pos is None:
            return

        self.setupGLState()

        mat_mvp = self.mvpMatrix()
        mat_modelview = self.modelViewMatrix()

        view = self.view()
        tan_half_fov = math.tan(math.radians(0.5 * view.opts["fov"]))

        context = QtGui.QOpenGLContext.currentContext()
        glfn = self.glFunctions()

        if DirtyFlag.POSITION in self.dirty_bits:
            self.upload_vbo(self.m_vbo_position, self.pos)
        if DirtyFlag.COLOR in self.dirty_bits:
            self.upload_vbo(self.m_vbo_color, self.color)
        if DirtyFlag.SIZE in self.dirty_bits:
            self.upload_vbo(self.m_vbo_size, self.size)
        self.dirty_bits = DirtyFlag(0)

        if not context.isOpenGLES():
            if _is_compatibility_profile(context):
                glfn.glEnable(GLC.GL_POINT_SPRITE)

            glfn.glEnable(GLC.GL_PROGRAM_POINT_SIZE)

        program = self.getShaderProgram()

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

        loc = 2
        if isinstance(self.size, np.ndarray):
            self.m_vbo_size.bind()
            program.setAttributeBuffer(loc, GLC.GL_FLOAT, 0, 1)
            self.m_vbo_size.release()
            enabled_locs.append(loc)
        else:
            # PySide6 errors on setAttributeValue() with a single float attribute
            program.setAttributeValue(loc, QtGui.QVector3D(self.size, 0.0, 0.0))

        for loc in enabled_locs:
            program.enableAttributeArray(loc)

        program.bind()
        program.setUniformValue("u_scale", 0.0 if self.pxMode else tan_half_fov, view.width())
        program.setUniformValue("u_mvp", mat_mvp)
        program.setUniformValue("u_modelview", mat_modelview)

        glfn.glDrawArrays(GLC.GL_POINTS, 0, len(self.pos))

        program.release()

        for loc in enabled_locs:
            program.disableAttributeArray(loc)


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
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex : """
        uniform vec2 u_scale;

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

            if (u_scale.x != 0.0) {
                // pxMode=False
                // the modelview matrix transforms the vertex to
                // camera space, where the camera is at (0, 0, 0).
                vec4 cpos = u_modelview * a_position;
                float dist = length(cpos.xyz);
                float tan_half_fov = u_scale.x;
                float view_width = u_scale.y;
                float pxSize = dist * 2.0 * tan_half_fov / view_width;
                gl_PointSize /= pxSize;
            }
        }
    """,
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment : """
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
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex : """
        uniform vec2 u_scale;

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

            if (u_scale.x != 0.0) {
                // pxMode=False
                // the modelview matrix transforms the vertex to
                // camera space, where the camera is at (0, 0, 0).
                vec4 cpos = u_modelview * a_position;
                float dist = length(cpos.xyz);
                float tan_half_fov = u_scale.x;
                float view_width = u_scale.y;
                float pxSize = dist * 2.0 * tan_half_fov / view_width;
                gl_PointSize /= pxSize;
            }
        }
    """,
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment : """
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
