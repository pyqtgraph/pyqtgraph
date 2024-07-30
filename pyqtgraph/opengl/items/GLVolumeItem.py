import ctypes
import importlib

from OpenGL.GL import *  # noqa
import numpy as np

from ...Qt import QtGui, QT_LIB
from .. import shaders
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
            self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.texture)
        if self.smooth:
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        else:
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        shape = self.data.shape

        context = QtGui.QOpenGLContext.currentContext()
        if not context.isOpenGLES():
            ## Test texture dimensions first
            glTexImage3D(GL_PROXY_TEXTURE_3D, 0, GL_RGBA, shape[0], shape[1], shape[2], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D, 0, GL_TEXTURE_WIDTH) == 0:
                raise Exception("OpenGL failed to create 3D texture (%dx%dx%d); too large for this hardware." % shape[:3])
        
        data = np.ascontiguousarray(self.data.transpose((2,1,0,3)))
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, shape[0], shape[1], shape[2], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        
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
        
    def paint(self):
        if self.data is None:
            return
        
        if self._needUpload:
            self._uploadData()
        
        self.setupGLState()

        mat_modelview = glGetFloatv(GL_MODELVIEW_MATRIX)
        mat_projection = glGetFloatv(GL_PROJECTION_MATRIX)
        mat_mvp = mat_modelview @ mat_projection

        view = self.view()
        center = QtGui.QVector3D(*[x/2. for x in self.data.shape[:3]])
        cam = self.mapFromParent(view.cameraPosition()) - center
        #print "center", center, "cam", view.cameraPosition(), self.mapFromParent(view.cameraPosition()), "diff", cam
        cam = np.array([cam.x(), cam.y(), cam.z()])
        ax = np.argmax(abs(cam))
        d = 1 if cam[ax] > 0 else -1
        offset, num_vertices = self.lists[(ax,d)]

        shader = shaders.getShaderProgram('texture3d')
        loc_pos = glGetAttribLocation(shader.program(), "a_position")
        loc_tex = glGetAttribLocation(shader.program(), "a_texcoord")
        self.m_vbo_position.bind()
        glVertexAttribPointer(loc_pos, 3, GL_FLOAT, False, 6*4, None)
        glVertexAttribPointer(loc_tex, 3, GL_FLOAT, False, 6*4, ctypes.c_void_p(3*4))
        self.m_vbo_position.release()
        enabled_locs = [loc_pos, loc_tex]

        glBindTexture(GL_TEXTURE_3D, self.texture)

        for loc in enabled_locs:
            glEnableVertexAttribArray(loc)

        with shader:
            glUniformMatrix4fv(shader.uniform("u_mvp"), 1, False, mat_mvp)

            glDrawArrays(GL_TRIANGLES, offset, num_vertices)

        for loc in enabled_locs:
            glDisableVertexAttribArray(loc)

        glBindTexture(GL_TEXTURE_3D, 0)

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
