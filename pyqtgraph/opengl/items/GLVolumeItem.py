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

__all__ = ['GLVolumeItem']

class GLVolumeItem(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem.GLGraphicsItem>`
    
    Displays volumetric data. 
    """
    
    _shaderProgram = None
    
    def __init__(self, data, sliceDensity=1, smooth=True, glOptions='translucent', parentItem=None):
        """
        ==============  =======================================================================================
        **Arguments:**
        data            Volume data to be rendered. *Must* be 4D numpy array (x, y, z, RGBA) with dtype=ubyte.
        sliceDensity    Density of slices to render through the volume. A value of 1 means one slice per voxel.
        smooth          (bool) If True, the volume slices are rendered with linear interpolation 
        ==============  =======================================================================================
        """
        
        super().__init__()
        self.setGLOptions(glOptions)
        self.sliceDensity = sliceDensity
        self.smooth = smooth
        self.data = None
        self._needUpload = False
        self.texture = None
        self.m_vbo_position = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.setParentItem(parentItem)
        self.setData(data)

    def setData(self, data):
        self.data = data
        self._needUpload = True
        self.update()
        
    def _uploadData(self):
        if self.texture is None:
            self.texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self.texture)
        filt = GL.GL_LINEAR if self.smooth else GL.GL_NEAREST
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, filt)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, filt)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_BORDER)
        shape = self.data.shape

        context = QtGui.QOpenGLContext.currentContext()
        if not context.isOpenGLES():
            ## Test texture dimensions first
            GL.glTexImage3D(GL.GL_PROXY_TEXTURE_3D, 0, GL.GL_RGBA, shape[0], shape[1], shape[2], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            if GL.glGetTexLevelParameteriv(GL.GL_PROXY_TEXTURE_3D, 0, GL.GL_TEXTURE_WIDTH) == 0:
                raise Exception("OpenGL failed to create 3D texture (%dx%dx%d); too large for this hardware." % shape[:3])
        
        data = np.ascontiguousarray(self.data.transpose((2,1,0,3)))
        GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGBA, shape[0], shape[1], shape[2], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
        
        all_vertices = []

        self.lists = {}
        for ax in [0,1,2]:
            for d in [-1, 1]:
                vertices = self.drawVolume(ax, d)
                self.lists[(ax,d)] = (len(all_vertices), len(vertices))
                all_vertices.extend(vertices)

        pos = np.array(all_vertices, dtype=np.float32)
        vbo = self.m_vbo_position
        if not vbo.isCreated():
            vbo.create()
        vbo.bind()
        vbo.allocate(pos, pos.nbytes)
        vbo.release()
        
        self._needUpload = False

    @staticmethod
    def getShaderProgram():
        klass = GLVolumeItem

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
        if self.data is None:
            return
        
        if self._needUpload:
            self._uploadData()
        
        self.setupGLState()

        mat_mvp = self.mvpMatrix()
        mat_mvp = np.array(mat_mvp.data(), dtype=np.float32)

        # calculate camera coordinates in this model's local space.
        # (in eye space, the camera is at the origin)
        modelview = self.modelViewMatrix()
        cam_local = modelview.inverted()[0].map(QtGui.QVector3D())

        # in local space, the model spans (0,0,0) to data.shape
        center = QtGui.QVector3D(*[x/2. for x in self.data.shape[:3]])
        cam = cam_local - center
        cam = np.array([cam.x(), cam.y(), cam.z()])
        ax = np.argmax(abs(cam))
        d = 1 if cam[ax] > 0 else -1
        offset, num_vertices = self.lists[(ax,d)]

        program = self.getShaderProgram()

        loc_pos, loc_tex = 0, 1
        self.m_vbo_position.bind()
        GL.glVertexAttribPointer(loc_pos, 3, GL.GL_FLOAT, False, 6*4, None)
        GL.glVertexAttribPointer(loc_tex, 3, GL.GL_FLOAT, False, 6*4, GL.GLvoidp(3*4))
        self.m_vbo_position.release()
        enabled_locs = [loc_pos, loc_tex]

        GL.glBindTexture(GL.GL_TEXTURE_3D, self.texture)

        for loc in enabled_locs:
            GL.glEnableVertexAttribArray(loc)

        with program:
            loc = GL.glGetUniformLocation(program, "u_mvp")
            GL.glUniformMatrix4fv(loc, 1, False, mat_mvp)

            GL.glDrawArrays(GL.GL_TRIANGLES, offset, num_vertices)

        for loc in enabled_locs:
            GL.glDisableVertexAttribArray(loc)

        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)

    def drawVolume(self, ax, d):
        imax = [0,1,2]
        imax.remove(ax)
        
        tp = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        vp = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        nudge = [0.5/x for x in self.data.shape]
        tp[0][imax[0]] = 0+nudge[imax[0]]
        tp[0][imax[1]] = 0+nudge[imax[1]]
        tp[1][imax[0]] = 1-nudge[imax[0]]
        tp[1][imax[1]] = 0+nudge[imax[1]]
        tp[2][imax[0]] = 1-nudge[imax[0]]
        tp[2][imax[1]] = 1-nudge[imax[1]]
        tp[3][imax[0]] = 0+nudge[imax[0]]
        tp[3][imax[1]] = 1-nudge[imax[1]]
        
        vp[0][imax[0]] = 0
        vp[0][imax[1]] = 0
        vp[1][imax[0]] = self.data.shape[imax[0]]
        vp[1][imax[1]] = 0
        vp[2][imax[0]] = self.data.shape[imax[0]]
        vp[2][imax[1]] = self.data.shape[imax[1]]
        vp[3][imax[0]] = 0
        vp[3][imax[1]] = self.data.shape[imax[1]]
        slices = self.data.shape[ax] * self.sliceDensity
        r = list(range(slices))
        if d == -1:
            r = r[::-1]

        vertices = []

        tzVals = np.linspace(nudge[ax], 1.0-nudge[ax], slices)
        vzVals = np.linspace(0, self.data.shape[ax], slices)
        for i in r:
            z = tzVals[i]
            w = vzVals[i]
            
            tp[0][ax] = z
            tp[1][ax] = z
            tp[2][ax] = z
            tp[3][ax] = z
            
            vp[0][ax] = w
            vp[1][ax] = w
            vp[2][ax] = w
            vp[3][ax] = w
            
            # assuming 0-1-2-3 are the BL, BR, TR, TL vertices of a quad
            for idx in [0, 1, 3, 1, 2, 3]:  # 2 triangles per quad
                vtx = tuple(vp[idx]) + tuple(tp[idx])
                vertices.append(vtx)

        return vertices


SHADER_LEGACY = {
    GL.GL_VERTEX_SHADER : """
        uniform mat4 u_mvp;
        attribute vec4 a_position;
        attribute vec3 a_texcoord;
        varying vec3 v_texcoord;
        void main() {
            gl_Position = u_mvp * a_position;
            v_texcoord = a_texcoord;
        }
    """,
    GL.GL_FRAGMENT_SHADER : """
        uniform sampler3D u_texture;
        varying vec3 v_texcoord;
        void main()
        {
            gl_FragColor = texture3D(u_texture, v_texcoord);
        }
    """,
}

SHADER_CORE = {
    GL.GL_VERTEX_SHADER : """
        uniform mat4 u_mvp;
        in vec4 a_position;
        in vec3 a_texcoord;
        out vec3 v_texcoord;
        void main() {
            gl_Position = u_mvp * a_position;
            v_texcoord = a_texcoord;
        }
    """,
    GL.GL_FRAGMENT_SHADER : """
        #ifdef GL_ES
        precision mediump float;
        precision lowp sampler3D;
        #endif
        uniform sampler3D u_texture;
        in vec3 v_texcoord;
        out vec4 fragColor;
        void main()
        {
            fragColor = texture(u_texture, v_texcoord);
        }
    """,
}
