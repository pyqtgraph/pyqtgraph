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

__all__ = ['GLImageItem']

class GLImageItem(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem.GLGraphicsItem>`
    
    Displays image data as a textured quad.
    """
    
    _shaderProgram = None
    
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
            self.texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        filt = GL.GL_LINEAR if self.smooth else GL.GL_NEAREST
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, filt)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, filt)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
        shape = self.data.shape
        
        context = QtGui.QOpenGLContext.currentContext()
        if not context.isOpenGLES():
            ## Test texture dimensions first
            GL.glTexImage2D(GL.GL_PROXY_TEXTURE_2D, 0, GL.GL_RGBA, shape[0], shape[1], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            if GL.glGetTexLevelParameteriv(GL.GL_PROXY_TEXTURE_2D, 0, GL.GL_TEXTURE_WIDTH) == 0:
                raise Exception("OpenGL failed to create 2D texture (%dx%d); too large for this hardware." % shape[:2])
        
        data = np.ascontiguousarray(self.data.transpose((1,0,2)))
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, shape[0], shape[1], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

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

    @staticmethod
    def getShaderProgram():
        klass = GLImageItem

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

        GL.glBindAttribLocation(program, 0, "a_position")
        GL.glBindAttribLocation(program, 1, "a_texcoord")
        GL.glLinkProgram(program)

        klass._shaderProgram = program
        return program

    def paint(self):
        if self._needUpdate:
            self._updateTexture()
            self._needUpdate = False
        
        self.setupGLState()

        mat_mvp = self.mvpMatrix()
        mat_mvp = np.array(mat_mvp.data(), dtype=np.float32)

        program = self.getShaderProgram()
        loc_pos, loc_tex = 0, 1
        self.m_vbo_position.bind()
        GL.glVertexAttribPointer(loc_pos, 2, GL.GL_FLOAT, False, 4*4, None)
        GL.glVertexAttribPointer(loc_tex, 2, GL.GL_FLOAT, False, 4*4, GL.GLvoidp(2*4))
        self.m_vbo_position.release()
        enabled_locs = [loc_pos, loc_tex]

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)

        for loc in enabled_locs:
            GL.glEnableVertexAttribArray(loc)

        with program:
            loc = GL.glGetUniformLocation(program, "u_mvp")
            GL.glUniformMatrix4fv(loc, 1, False, mat_mvp)

            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        for loc in enabled_locs:
            GL.glDisableVertexAttribArray(loc)

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)


SHADER_LEGACY = {
    GL.GL_VERTEX_SHADER : """
        uniform mat4 u_mvp;
        attribute vec4 a_position;
        attribute vec2 a_texcoord;
        varying vec2 v_texcoord;
        void main() {
            gl_Position = u_mvp * a_position;
            v_texcoord = a_texcoord;
        }
    """,
    GL.GL_FRAGMENT_SHADER : """
        #ifdef GL_ES
        precision mediump float;
        #endif
        uniform sampler2D u_texture;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(u_texture, v_texcoord);
        }
    """,
}

SHADER_CORE = {
    GL.GL_VERTEX_SHADER : """
        uniform mat4 u_mvp;
        in vec4 a_position;
        in vec2 a_texcoord;
        out vec2 v_texcoord;
        void main() {
            gl_Position = u_mvp * a_position;
            v_texcoord = a_texcoord;
        }
    """,
    GL.GL_FRAGMENT_SHADER : """
        #ifdef GL_ES
        precision mediump float;
        #endif
        uniform sampler2D u_texture;
        in vec2 v_texcoord;
        out vec4 fragColor;
        void main()
        {
            fragColor = texture(u_texture, v_texcoord);
        }
    """,
}
