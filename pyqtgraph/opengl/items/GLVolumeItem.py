import enum

import numpy as np

from ...Qt import QtGui, QtOpenGL
from ...Qt import OpenGLConstants as GLC
from ...Qt import OpenGLHelpers
from ..GLGraphicsItem import GLGraphicsItem

__all__ = ['GLVolumeItem']


class DirtyFlag(enum.Flag):
    POSITION = enum.auto()
    TEXTURE = enum.auto()


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
        OpenGLHelpers.suppress_texture_warning()
        self.setGLOptions(glOptions)
        self.sliceDensity = sliceDensity
        self.smooth = smooth
        self.data = None
        self.m_texture = QtOpenGL.QOpenGLTexture(QtOpenGL.QOpenGLTexture.Target.Target3D)
        self.m_vbo_position = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.dirty_bits = DirtyFlag(0)
        self.setParentItem(parentItem)
        self.setData(data)

    def setData(self, data):
        if self.data is None or data is None or self.data.shape != data.shape:
            # it becomes dirty when sliceDensity changes too,
            # but we will just treat sliceDensity as immutable once instantiated
            self.dirty_bits |= DirtyFlag.POSITION
        self.dirty_bits |= DirtyFlag.TEXTURE
        self.data = data
        self.update()

    def _uploadData(self):
        tex = self.m_texture

        data = np.ascontiguousarray(self.data.transpose((2,1,0,3)))
        d, h, w = data.shape[:3]

        if tex.isCreated() and (w != tex.width() or h != tex.height() or d != tex.depth()):
            tex.destroy()

        if not tex.isCreated():
            tex.setFormat(QtOpenGL.QOpenGLTexture.TextureFormat.RGBA8_UNorm)
            tex.setSize(w, h, d)
            tex.allocateStorage()
            if not tex.isStorageAllocated():
                raise RuntimeError("OpenGL failed to create 3D texture (%dx%dx%d); too large for this hardware." % (w, h, d))

        filt = QtOpenGL.QOpenGLTexture.Filter.Linear if self.smooth else QtOpenGL.QOpenGLTexture.Filter.Nearest
        tex.setMinMagFilters(filt, filt)
        tex.setWrapMode(QtOpenGL.QOpenGLTexture.WrapMode.ClampToBorder)

        tex.setData(
            QtOpenGL.QOpenGLTexture.PixelFormat.RGBA,
            QtOpenGL.QOpenGLTexture.PixelType.UInt8,
            data)

    def computeVertices(self):
        all_vertices = []

        offsets = {}
        for ax in [0,1,2]:
            for d in [-1, 1]:
                vertices = drawVolume(self.data.shape, ax, d, self.sliceDensity)
                offsets[(ax,d)] = (len(all_vertices), len(vertices))
                all_vertices.extend(vertices)

        return all_vertices, offsets

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

        program = QtOpenGL.QOpenGLShaderProgram()
        for shader_type, src in sources.items():
            if not program.addShaderFromSourceCode(shader_type, glsl_version + src):
                raise RuntimeError(program.log())

        program.bindAttributeLocation("a_position", 0)
        program.bindAttributeLocation("a_texcoord", 1)
        if not program.link():
            raise RuntimeError(program.log())

        klass._shaderProgram = program
        return program
        
    def paint(self):
        if self.data is None:
            return
        
        self.setupGLState()

        if DirtyFlag.POSITION in self.dirty_bits:
            vertices, self.lists = self.computeVertices()
            pos = np.array(vertices, dtype=np.float32)
            OpenGLHelpers.upload_vbo(self.m_vbo_position, pos)
        if DirtyFlag.TEXTURE in self.dirty_bits:
            self._uploadData()
        self.dirty_bits = DirtyFlag(0)

        mat_mvp = self.mvpMatrix()

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

        glfn = self.glFunctions()

        program = self.getShaderProgram()

        loc_pos, loc_tex = 0, 1
        self.m_vbo_position.bind()
        program.setAttributeBuffer(loc_pos, GLC.GL_FLOAT, 0*4, 3, 6*4)
        program.setAttributeBuffer(loc_tex, GLC.GL_FLOAT, 3*4, 3, 6*4)
        self.m_vbo_position.release()
        enabled_locs = [loc_pos, loc_tex]

        self.m_texture.bind()

        for loc in enabled_locs:
            program.enableAttributeArray(loc)

        program.bind()
        program.setUniformValue("u_mvp", mat_mvp)

        glfn.glDrawArrays(GLC.GL_TRIANGLES, offset, num_vertices)

        program.release()

        for loc in enabled_locs:
            program.disableAttributeArray(loc)

        self.m_texture.release()

def drawVolume(shape, ax, d, sliceDensity):
    imax = [0,1,2]
    imax.remove(ax)

    tp = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    vp = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    nudge = [0.5/x for x in shape]
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
    vp[1][imax[0]] = shape[imax[0]]
    vp[1][imax[1]] = 0
    vp[2][imax[0]] = shape[imax[0]]
    vp[2][imax[1]] = shape[imax[1]]
    vp[3][imax[0]] = 0
    vp[3][imax[1]] = shape[imax[1]]
    slices = shape[ax] * sliceDensity
    r = list(range(slices))
    if d == -1:
        r = r[::-1]

    vertices = []

    tzVals = np.linspace(nudge[ax], 1.0-nudge[ax], slices)
    vzVals = np.linspace(0, shape[ax], slices)
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
            vtx = vp[idx] + tp[idx]
            vertices.append(vtx)

    return vertices


SHADER_LEGACY = {
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex : """
        uniform mat4 u_mvp;
        attribute vec4 a_position;
        attribute vec3 a_texcoord;
        varying vec3 v_texcoord;
        void main() {
            gl_Position = u_mvp * a_position;
            v_texcoord = a_texcoord;
        }
    """,
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment : """
        uniform sampler3D u_texture;
        varying vec3 v_texcoord;
        void main()
        {
            gl_FragColor = texture3D(u_texture, v_texcoord);
        }
    """,
}

SHADER_CORE = {
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex : """
        uniform mat4 u_mvp;
        in vec4 a_position;
        in vec3 a_texcoord;
        out vec3 v_texcoord;
        void main() {
            gl_Position = u_mvp * a_position;
            v_texcoord = a_texcoord;
        }
    """,
    QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment : """
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
