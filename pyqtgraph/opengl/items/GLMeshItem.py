import importlib

from OpenGL.GL import *  # noqa
import numpy as np

from ...Qt import QtGui, QT_LIB
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem
from ..MeshData import MeshData

if QT_LIB in ["PyQt5", "PySide2"]:
    QtOpenGL = QtGui
else:
    QtOpenGL = importlib.import_module(f"{QT_LIB}.QtOpenGL")

__all__ = ['GLMeshItem']

class GLMeshItem(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem.GLGraphicsItem>`
    
    Displays a 3D triangle mesh. 
    """
    def __init__(self, parentItem=None, **kwds):
        """
        ============== =====================================================
        **Arguments:**
        meshdata       MeshData object from which to determine geometry for 
                       this item.
        color          Default face color used if no vertex or face colors 
                       are specified.
        edgeColor      Default edge color to use if no edge colors are
                       specified in the mesh data.
        drawEdges      If True, a wireframe mesh will be drawn. 
                       (default=False)
        drawFaces      If True, mesh faces are drawn. (default=True)
        shader         Name of shader program to use when drawing faces.
                       (None for no shader)
        smooth         If True, normal vectors are computed for each vertex
                       and interpolated within each face.
        computeNormals If False, then computation of normal vectors is 
                       disabled. This can provide a performance boost for 
                       meshes that do not make use of normals.
        ============== =====================================================
        """
        self.opts = {
            'meshdata': None,
            'color': (1., 1., 1., 1.),
            'drawEdges': False,
            'drawFaces': True,
            'edgeColor': (0.5, 0.5, 0.5, 1.0),
            'shader': None,
            'smooth': True,
            'computeNormals': True,
        }
        
        super().__init__(parentItem=parentItem)
        glopts = kwds.pop('glOptions', 'opaque')
        self.setGLOptions(glopts)
        shader = kwds.pop('shader', None)
        self.setShader(shader)
        
        self.setMeshData(**kwds)
        
        ## storage for data compiled from MeshData object
        self.vertexes = None
        self.normals = None
        self.colors = None
        self.faces = None

        self.m_vbo_position = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.m_vbo_normal = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.m_vbo_color = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.m_ibo_faces = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.IndexBuffer)
        self.m_vbo_edgeVerts = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.m_ibo_edges = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.IndexBuffer)

    def setShader(self, shader):
        """Set the shader used when rendering faces in the mesh. (see the GL shaders example)"""
        self.opts['shader'] = shader
        self.update()
        
    def shader(self):
        shader = self.opts['shader']
        if isinstance(shader, shaders.ShaderProgram):
            return shader
        else:
            return shaders.getShaderProgram(shader)
        
    def setColor(self, c):
        """Set the default color to use when no vertex or face colors are specified."""
        self.opts['color'] = c
        self.update()
        
    def setMeshData(self, **kwds):
        """
        Set mesh data for this item. This can be invoked two ways:
        
        1. Specify *meshdata* argument with a new MeshData object
        2. Specify keyword arguments to be passed to MeshData(..) to create a new instance.
        """
        md = kwds.get('meshdata', None)
        if md is None:
            opts = {}
            for k in ['vertexes', 'faces', 'edges', 'vertexColors', 'faceColors']:
                try:
                    opts[k] = kwds.pop(k)
                except KeyError:
                    pass
            md = MeshData(**opts)
        
        self.opts['meshdata'] = md
        self.opts.update(kwds)
        self.meshDataChanged()
        self.update()
        
    
    def meshDataChanged(self):
        """
        This method must be called to inform the item that the MeshData object
        has been altered.
        """
        
        self.vertexes = None
        self.faces = None
        self.normals = None
        self.colors = None
        self.edges = None
        self.edgeVerts = None
        self.edgeColors = None
        self.update()

    def upload_vertex_buffers(self):

        def upload_vbo(vbo, arr):
            if arr is None:
                vbo.destroy()
                return
            if not vbo.isCreated():
                vbo.create()
            vbo.bind()
            vbo.allocate(arr, arr.nbytes)
            vbo.release()

        upload_vbo(self.m_vbo_position, self.vertexes)
        upload_vbo(self.m_vbo_normal, self.normals)
        upload_vbo(self.m_vbo_color, self.colors)
        upload_vbo(self.m_ibo_faces, self.faces)

        upload_vbo(self.m_vbo_edgeVerts, self.edgeVerts)
        upload_vbo(self.m_ibo_edges, self.edges)

    def parseMeshData(self):
        ## interpret vertex / normal data before drawing
        ## This can:
        ##   - automatically generate normals if they were not specified
        ##   - pull vertexes/noormals/faces from MeshData if that was specified
        
        if self.vertexes is not None and self.normals is not None:
            return
        #if self.opts['normals'] is None:
            #if self.opts['meshdata'] is None:
                #self.opts['meshdata'] = MeshData(vertexes=self.opts['vertexes'], faces=self.opts['faces'])
        if self.opts['meshdata'] is not None:
            md = self.opts['meshdata']
            if self.opts['smooth'] and not md.hasFaceIndexedData():
                self.vertexes = md.vertexes()
                if self.opts['computeNormals']:
                    self.normals = md.vertexNormals()
                self.faces = md.faces().astype(np.uint32)
                if md.hasVertexColor():
                    self.colors = md.vertexColors()
                if md.hasFaceColor():
                    self.colors = md.faceColors()
            else:
                self.vertexes = md.vertexes(indexed='faces')
                if self.opts['computeNormals']:
                    if self.opts['smooth']:
                        self.normals = md.vertexNormals(indexed='faces')
                    else:
                        self.normals = md.faceNormals(indexed='faces')
                self.faces = None
                if md.hasVertexColor():
                    self.colors = md.vertexColors(indexed='faces')
                elif md.hasFaceColor():
                    self.colors = md.faceColors(indexed='faces')

            if self.opts['drawEdges']:
                if not md.hasFaceIndexedData():
                    self.edges = md.edges().astype(np.uint32)
                    self.edgeVerts = md.vertexes()
                else:
                    self.edges = md.edges().astype(np.uint32)
                    self.edgeVerts = md.vertexes(indexed='faces')

            # NOTE: it is possible for self.vertexes to be None at this point.
            #       this situation is encountered with the bundled animated
            #       GLSurfacePlot example. This occurs because it only sets the
            #       z component within update().
            self.upload_vertex_buffers()
    
    def paint(self):
        self.setupGLState()
        
        self.parseMeshData()        

        mat_mvp = self.mvpMatrix()
        mat_mvp = np.array(mat_mvp.data(), dtype=np.float32)
        mat_normal = self.modelViewMatrix().normalMatrix()
        mat_normal = np.array(mat_normal.data(), dtype=np.float32)

        if self.opts['drawFaces'] and self.vertexes is not None:
            shader = self.shader()

            enabled_locs = []

            if (loc := glGetAttribLocation(shader.program(), "a_position")) != -1:
                self.m_vbo_position.bind()
                glVertexAttribPointer(loc, 3, GL_FLOAT, False, 0, None)
                self.m_vbo_position.release()
                enabled_locs.append(loc)

            if (loc := glGetAttribLocation(shader.program(), "a_normal")) != -1:
                if self.normals is None:
                    # the shader needs a normal but the user set computeNormals=False...
                    glVertexAttrib3f(loc, 0, 0, 1)
                else:
                    self.m_vbo_normal.bind()
                    glVertexAttribPointer(loc, 3, GL_FLOAT, False, 0, None)
                    self.m_vbo_normal.release()
                    enabled_locs.append(loc)

            if (loc := glGetAttribLocation(shader.program(), "a_color")) != -1:
                if self.colors is None:
                    color = self.opts['color']
                    if isinstance(color, QtGui.QColor):
                        color = color.getRgbF()
                    glVertexAttrib4f(loc, *color)
                else:
                    self.m_vbo_color.bind()
                    glVertexAttribPointer(loc, 4, GL_FLOAT, False, 0, None)
                    self.m_vbo_color.release()
                    enabled_locs.append(loc)

            for loc in enabled_locs:
                glEnableVertexAttribArray(loc)

            with shader:
                glUniformMatrix4fv(shader.uniform("u_mvp"), 1, False, mat_mvp)
                if (uloc_normal := shader.uniform("u_normal")) != -1:
                    glUniformMatrix3fv(uloc_normal, 1, False, mat_normal)

                if (faces := self.faces) is None:
                    glDrawArrays(GL_TRIANGLES, 0, np.prod(self.vertexes.shape[:-1]))
                else:
                    self.m_ibo_faces.bind()
                    glDrawElements(GL_TRIANGLES, faces.size, GL_UNSIGNED_INT, None)
                    self.m_ibo_faces.release()

            for loc in enabled_locs:
                glDisableVertexAttribArray(loc)

        if self.opts['drawEdges']:
            shader = shaders.getShaderProgram(None)

            enabled_locs = []

            if (loc := glGetAttribLocation(shader.program(), "a_position")) != -1:
                self.m_vbo_edgeVerts.bind()
                glVertexAttribPointer(loc, 3, GL_FLOAT, False, 0, None)
                self.m_vbo_edgeVerts.release()
                enabled_locs.append(loc)

            # edge colors are always just one single color
            if (loc := glGetAttribLocation(shader.program(), "a_color")) != -1:
                color = self.opts['edgeColor']
                if isinstance(color, QtGui.QColor):
                    color = color.getRgbF()
                glVertexAttrib4f(loc, *color)

            for loc in enabled_locs:
                glEnableVertexAttribArray(loc)

            with shader:
                glUniformMatrix4fv(shader.uniform("u_mvp"), 1, False, mat_mvp)

                self.m_ibo_edges.bind()
                glDrawElements(GL_LINES, self.edges.size, GL_UNSIGNED_INT, None)
                self.m_ibo_edges.release()

            for loc in enabled_locs:
                glDisableVertexAttribArray(loc)


