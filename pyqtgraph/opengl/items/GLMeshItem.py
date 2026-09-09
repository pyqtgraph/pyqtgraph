import copy
import enum
import warnings

import numpy as np

from ...Qt import QtGui, QtOpenGL, QT_LIB, QtVersionInfo, compat
from ...Qt import OpenGLConstants as GLC
from ...Qt import OpenGLHelpers
from ...Qt.OpenGLHelpers import upload_vbo
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem
from ..MeshData import MeshData

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
    def __init__(self, parentItem=None, **kwargs):
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
        polygonOffset  If True, polygon offset is enabled, this is useful 
                       when drawing edges on top of faces.
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
            'polygonOffset': False,
        }
        
        super().__init__(parentItem=parentItem)
        glopts = kwargs.pop('glOptions', 'opaque')
        self.setGLOptions(glopts)
        self._shaderProgram : shaders.ShaderProgram | None = None
        shader = kwargs.pop('shader', None)
        self.setShader(shader)
        
        self.setMeshData(**kwargs)
        
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
        self.dirty_bits = DirtyFlag(0)

        self._glUniform1fv = None

    def cleanupGL(self):
        self.m_vbo_position.destroy()
        self.m_vbo_normal.destroy()
        self.m_vbo_color.destroy()
        self.m_ibo_faces.destroy()
        self.m_vbo_edgeVerts.destroy()
        self.m_ibo_edges.destroy()

        self.dirty_bits = (
            DirtyFlag.POSITION   |
            DirtyFlag.NORMAL     |
            DirtyFlag.COLOR      |
            DirtyFlag.FACES      |
            DirtyFlag.EDGE_VERTS |
            DirtyFlag.EDGES
        )

        self._glUniform1fv = None

    def setShader(self, shader):
        """Set the shader used when rendering faces in the mesh. (see the GL shaders example)"""
        self.opts['shader'] = shader

        if shader is None:
            shader = 'default'

        if isinstance(shader, str):
            # load from internal shaders
            shader = shaders.getShaderProgram(shader)

        self._shaderProgram = copy.deepcopy(shader)

        self.update()
        
    def shader(self):
        return self._shaderProgram

    def shaderProgram(self, view, shader_program):
        name = shader_program.name
        klass = self.__class__
        cache_key = f'{klass.__module__}.{klass.__qualname__}.{name}'

        sources = { sp.shaderType() : sp.sourceCode() for sp in shader_program.shaders }

        shaders_cache = self.shadersCache(view=view)

        if (program := shaders_cache.get(cache_key)) is None:
            program = OpenGLHelpers.compile_and_link(
                view.context(),
                sources_core=None,
                sources_legacy=sources,
                attributes=dict(a_position=0),
            )
            shaders_cache[cache_key] = program

        return program

    def setColor(self, c):
        """Set the default color to use when no vertex or face colors are specified."""
        self.opts['color'] = c
        self.update()
        
    def setPolygonOffset(self, enable):
        """Enable or disable polygon offset for this mesh item."""
        self.opts['polygonOffset'] = enable
        self.update()

    def setMeshData(self, **kwargs):
        """
        Set mesh data for this item. This can be invoked two ways:
        
        1. Specify *meshdata* argument with a new MeshData object
        2. Specify keyword arguments to be passed to MeshData(..) to create a new instance.
        """
        md = kwargs.get('meshdata', None)
        if md is None:
            opts = {}
            for k in ['vertexes', 'faces', 'edges', 'vertexColors', 'faceColors']:
                try:
                    opts[k] = kwargs.pop(k)
                except KeyError:
                    pass
            md = MeshData(**opts)
        
        self.opts['meshdata'] = md
        self.opts.update(kwargs)
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
        self.update()

    def upload_vertex_buffers(self, dirty_bits):
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
        if (view := self.view()) is None:
            return
        self.setupGLState()
        glfn = self.glFunctions(view=view)

        NULL = compat.voidptr(0) if QT_LIB.startswith('PySide') else None

        if self.opts['polygonOffset']:
            glfn.glEnable(GLC.GL_POLYGON_OFFSET_FILL)
            glfn.glPolygonOffset(1.0, 1.0)

        self.dirty_bits |= self.parseMeshData()
        self.upload_vertex_buffers(self.dirty_bits)
        self.dirty_bits = DirtyFlag(0)

        mat_mvp = self.mvpMatrix()
        mat_normal = self.modelViewMatrix().normalMatrix()

        if self.opts['drawFaces'] and self.vertexes is not None:
            shader_program = self.shader()
            program = self.shaderProgram(view, shader_program)

            enabled_locs = []

            if (loc := program.attributeLocation("a_position")) != -1:
                self.m_vbo_position.bind()
                program.setAttributeBuffer(loc, GLC.GL_FLOAT, 0, 3)
                self.m_vbo_position.release()
                enabled_locs.append(loc)

            if (loc := program.attributeLocation("a_normal")) != -1:
                if self.normals is None:
                    # the shader needs a normal but the user set computeNormals=False...
                    program.setAttributeValue(loc, QtGui.QVector3D(0, 0, 1))
                else:
                    self.m_vbo_normal.bind()
                    program.setAttributeBuffer(loc, GLC.GL_FLOAT, 0, 3)
                    self.m_vbo_normal.release()
                    enabled_locs.append(loc)

            if (loc := program.attributeLocation("a_color")) != -1:
                if self.colors is None:
                    color = self.opts['color']
                    if not isinstance(color, QtGui.QColor):
                        color = QtGui.QColor.fromRgbF(*color)
                    program.setAttributeValue(loc, color)
                else:
                    self.m_vbo_color.bind()
                    if self.colors.dtype == np.uint8:
                        program.setAttributeBuffer(loc, GLC.GL_UNSIGNED_BYTE, 0, 4)
                    else:
                        program.setAttributeBuffer(loc, GLC.GL_FLOAT, 0, 4)
                    self.m_vbo_color.release()
                    enabled_locs.append(loc)

            for loc in enabled_locs:
                program.enableAttributeArray(loc)

            program.bind()

            program.setUniformValue("u_mvp", mat_mvp)
            if (loc := program.uniformLocation("u_normal")) != -1:
                program.setUniformValue(loc, mat_normal)

            # load uniforms defined by the shader
            for name, data in self._shaderProgram.uniformData.items():
                if (loc := program.uniformLocation(name)) == -1:
                    warnings.warn(f'Could not find uniform variable "{name}"')
                    continue

                data = np.ascontiguousarray(data, dtype=np.float32)

                if QT_LIB.startswith('PySide') and QtVersionInfo < (6, 9):
                    # PYSIDE-3005
                    if self._glUniform1fv is None:
                        self._glUniform1fv = OpenGLHelpers.get_gl_uniform_1fv(view.context())
                    self._glUniform1fv(loc, data.size, data.ctypes.data)
                else:
                    # PySide6 and PyQt6 accept ndarray and list
                    # but PyQt5 accepts only list
                    glfn.glUniform1fv(loc, len(data), data.tolist())

            if (faces := self.faces) is None:
                glfn.glDrawArrays(GLC.GL_TRIANGLES, 0, np.prod(self.vertexes.shape[:-1]))
            else:
                self.m_ibo_faces.bind()
                glfn.glDrawElements(GLC.GL_TRIANGLES, faces.size, GLC.GL_UNSIGNED_INT, NULL)
                self.m_ibo_faces.release()

            program.release()

            for loc in enabled_locs:
                program.disableAttributeArray(loc)

        if self.opts['drawEdges']:
            shader_program = shaders.getShaderProgram("default")
            program = self.shaderProgram(view, shader_program)

            enabled_locs = []

            if (loc := program.attributeLocation("a_position")) != -1:
                self.m_vbo_edgeVerts.bind()
                program.setAttributeBuffer(loc, GLC.GL_FLOAT, 0, 3)
                self.m_vbo_edgeVerts.release()
                enabled_locs.append(loc)

            # edge colors are always just one single color
            if (loc := program.attributeLocation("a_color")) != -1:
                color = self.opts['edgeColor']
                if not isinstance(color, QtGui.QColor):
                    color = QtGui.QColor.fromRgbF(*color)
                program.setAttributeValue(loc, color)

            for loc in enabled_locs:
                program.enableAttributeArray(loc)

            program.bind()
            program.setUniformValue("u_mvp", mat_mvp)

            self.m_ibo_edges.bind()
            glfn.glDrawElements(GLC.GL_LINES, self.edges.size, GLC.GL_UNSIGNED_INT, NULL)
            self.m_ibo_edges.release()

            program.release()

            for loc in enabled_locs:
                program.disableAttributeArray(loc)

        if self.opts['polygonOffset']:
            glfn.glDisable(GLC.GL_POLYGON_OFFSET_FILL)
            glfn.glPolygonOffset(0.0, 0.0)
