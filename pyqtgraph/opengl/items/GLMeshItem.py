import enum
import importlib

from OpenGL import GL
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


class DirtyFlag(enum.Flag):
    POSITION = enum.auto()
    NORMAL = enum.auto()
    COLOR = enum.auto()
    FACES = enum.auto()
    EDGE_VERTS = enum.auto()
    EDGES = enum.auto()


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

    def upload_vertex_buffers(self, dirty_bits):

        def upload_vbo(vbo, arr):
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

        if DirtyFlag.POSITION in dirty_bits:
            upload_vbo(self.m_vbo_position, self.vertexes)
        if DirtyFlag.NORMAL in dirty_bits:
            upload_vbo(self.m_vbo_normal, self.normals)
        if DirtyFlag.COLOR in dirty_bits:
            upload_vbo(self.m_vbo_color, self.colors)
        if DirtyFlag.FACES in dirty_bits:
            upload_vbo(self.m_ibo_faces, self.faces)

        if DirtyFlag.EDGE_VERTS in dirty_bits:
            upload_vbo(self.m_vbo_edgeVerts, self.edgeVerts)
        if DirtyFlag.EDGES in dirty_bits:
            upload_vbo(self.m_ibo_edges, self.edges)

    def parseMeshData(self) -> DirtyFlag:
        ## interpret vertex / normal data before drawing
        
        dirty_bits = DirtyFlag(0)

        # self.vertexes acts as a flag to determine whether mesh data
        # has been parsed
        if self.vertexes is not None:
            return dirty_bits

        if self.opts['meshdata'] is not None:
            md = self.opts['meshdata']
            if self.opts['smooth'] and not md.hasFaceIndexedData():
                self.vertexes = md.vertexes()
                dirty_bits |= DirtyFlag.POSITION
                if self.opts['computeNormals']:
                    self.normals = md.vertexNormals()
                    dirty_bits |= DirtyFlag.NORMAL
                self.faces = md.faces().astype(np.uint32)
                dirty_bits |= DirtyFlag.FACES
                if md.hasVertexColor():
                    self.colors = md.vertexColors()
                    dirty_bits |= DirtyFlag.COLOR
                elif md.hasFaceColor():
                    self.colors = md.faceColors()
                    dirty_bits |= DirtyFlag.COLOR
            else:
                self.vertexes = md.vertexes(indexed='faces')
                dirty_bits |= DirtyFlag.POSITION
                if self.opts['computeNormals']:
                    if self.opts['smooth']:
                        self.normals = md.vertexNormals(indexed='faces')
                    else:
                        self.normals = md.faceNormals(indexed='faces')
                    dirty_bits |= DirtyFlag.NORMAL
                self.faces = None
                if md.hasVertexColor():
                    self.colors = md.vertexColors(indexed='faces')
                    dirty_bits |= DirtyFlag.COLOR
                elif md.hasFaceColor():
                    self.colors = md.faceColors(indexed='faces')
                    dirty_bits |= DirtyFlag.COLOR

            if self.opts['drawEdges']:
                if not md.hasFaceIndexedData():
                    self.edges = md.edges().astype(np.uint32)
                    self.edgeVerts = md.vertexes()
                else:
                    self.edges = md.edges().astype(np.uint32)
                    self.edgeVerts = md.vertexes(indexed='faces')
                dirty_bits |= DirtyFlag.EDGE_VERTS
                dirty_bits |= DirtyFlag.EDGES

            # NOTE: it is possible for self.vertexes to be None at this point.
            #       this situation is encountered with the bundled animated
            #       GLSurfacePlot example. This occurs because it only sets the
            #       z component within update().
    
        return dirty_bits

    def paint(self):
        self.setupGLState()
        
        if (dirty_bits := self.parseMeshData()):
            self.upload_vertex_buffers(dirty_bits)

        mat_mvp = self.mvpMatrix()
        mat_mvp = np.array(mat_mvp.data(), dtype=np.float32)
        mat_normal = self.modelViewMatrix().normalMatrix()
        mat_normal = np.array(mat_normal.data(), dtype=np.float32)

        context = QtGui.QOpenGLContext.currentContext()
        es2_compat = context.hasExtension(b'GL_ARB_ES2_compatibility')

        if self.opts['drawFaces'] and self.vertexes is not None:
            shader = self.shader()
            program = shader.program(es2_compat=es2_compat)

            enabled_locs = []

            if (loc := GL.glGetAttribLocation(program, "a_position")) != -1:
                self.m_vbo_position.bind()
                GL.glVertexAttribPointer(loc, 3, GL.GL_FLOAT, False, 0, None)
                self.m_vbo_position.release()
                enabled_locs.append(loc)

            if (loc := GL.glGetAttribLocation(program, "a_normal")) != -1:
                if self.normals is None:
                    # the shader needs a normal but the user set computeNormals=False...
                    GL.glVertexAttrib3f(loc, 0, 0, 1)
                else:
                    self.m_vbo_normal.bind()
                    GL.glVertexAttribPointer(loc, 3, GL.GL_FLOAT, False, 0, None)
                    self.m_vbo_normal.release()
                    enabled_locs.append(loc)

            if (loc := GL.glGetAttribLocation(program, "a_color")) != -1:
                if self.colors is None:
                    color = self.opts['color']
                    if isinstance(color, QtGui.QColor):
                        color = color.getRgbF()
                    GL.glVertexAttrib4f(loc, *color)
                else:
                    self.m_vbo_color.bind()
                    if self.colors.dtype == np.uint8:
                        GL.glVertexAttribPointer(loc, 4, GL.GL_UNSIGNED_BYTE, True, 0, None)
                    else:
                        GL.glVertexAttribPointer(loc, 4, GL.GL_FLOAT, False, 0, None)
                    self.m_vbo_color.release()
                    enabled_locs.append(loc)

            for loc in enabled_locs:
                GL.glEnableVertexAttribArray(loc)

            with shader:    # "with shader" will load extra uniforms
                loc = GL.glGetUniformLocation(program, "u_mvp")
                GL.glUniformMatrix4fv(loc, 1, False, mat_mvp)
                if (uloc_normal := GL.glGetUniformLocation(program, "u_normal")) != -1:
                    GL.glUniformMatrix3fv(uloc_normal, 1, False, mat_normal)

                if (faces := self.faces) is None:
                    GL.glDrawArrays(GL.GL_TRIANGLES, 0, np.prod(self.vertexes.shape[:-1]))
                else:
                    self.m_ibo_faces.bind()
                    GL.glDrawElements(GL.GL_TRIANGLES, faces.size, GL.GL_UNSIGNED_INT, None)
                    self.m_ibo_faces.release()

            for loc in enabled_locs:
                GL.glDisableVertexAttribArray(loc)

        if self.opts['drawEdges']:
            shader = shaders.getShaderProgram(None)
            program = shader.program(es2_compat=es2_compat)

            enabled_locs = []

            if (loc := GL.glGetAttribLocation(program, "a_position")) != -1:
                self.m_vbo_edgeVerts.bind()
                GL.glVertexAttribPointer(loc, 3, GL.GL_FLOAT, False, 0, None)
                self.m_vbo_edgeVerts.release()
                enabled_locs.append(loc)

            # edge colors are always just one single color
            if (loc := GL.glGetAttribLocation(program, "a_color")) != -1:
                color = self.opts['edgeColor']
                if isinstance(color, QtGui.QColor):
                    color = color.getRgbF()
                GL.glVertexAttrib4f(loc, *color)

            for loc in enabled_locs:
                GL.glEnableVertexAttribArray(loc)

            with program:
                loc = GL.glGetUniformLocation(program, "u_mvp")
                GL.glUniformMatrix4fv(loc, 1, False, mat_mvp)

                self.m_ibo_edges.bind()
                GL.glDrawElements(GL.GL_LINES, self.edges.size, GL.GL_UNSIGNED_INT, None)
                self.m_ibo_edges.release()

            for loc in enabled_locs:
                GL.glDisableVertexAttribArray(loc)


