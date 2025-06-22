import numpy as np

from ..MeshData import MeshData
from .GLMeshItem import GLMeshItem
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import OpenGL.GL as ogl
from pyqtgraph.Qt.QtGui import QVector3D

__all__ = ['GLSurfacePlotItem']

class GLSurfacePlotItem(GLMeshItem):
    """
    **Bases:** :class:`GLMeshItem <pyqtgraph.opengl.GLMeshItem>`
    
    Displays a surface plot on a regular x,y grid with optional wireframe overlay.
    """
    gl_options = {
        ogl.GL_DEPTH_TEST: True,
        ogl.GL_BLEND: True,
        ogl.GL_CULL_FACE: False,
        ogl.GL_LINE_SMOOTH: True,
        'glHint': (ogl.GL_LINE_SMOOTH_HINT, ogl.GL_NICEST),
        'glBlendFunc': (ogl.GL_SRC_ALPHA, ogl.GL_ONE_MINUS_SRC_ALPHA),
    }

    gl_surface_options = {
        **gl_options,
        ogl.GL_POLYGON_OFFSET_FILL: True,
        'glPolygonOffset': (1.0, 1.0),
    }

    gl_line_options = {
        **gl_options,
        ogl.GL_POLYGON_OFFSET_FILL: False,
    }

    mesh_keys = ('x', 'y', 'z', 'colors')
    grid_keys = ('showGrid', 'lineColor', 'lineWidth', 'lineAntialias')

    def __init__(self, parentItem=None, **kwds):
        """
        The x, y, z, colors, showGrid, lineColor, lineWidth and lineAntialias
        arguments are passed to setData().
        All other keyword arguments are passed to GLMeshItem.__init__().
        """
        self._x = None
        self._y = None
        self._z = None
        self._color = None
        self._showGrid = False
        self._lineColor = (0, 0, 0, 1)
        self._lineWidth = 1.0
        self._lineAntialias = True
        self._vertexes = None
        self._meshdata = MeshData()
        self._grid_lines = []

        # splitout GLSurfacePlotItem from kwds
        surface_keys = self.mesh_keys + self.grid_keys
        surface_kwds = {}
        for arg in surface_keys:
            if arg in kwds:
                surface_kwds[arg] = kwds.pop(arg)

        mesh_kwds = {**kwds, 'glOptions': self.gl_surface_options}
        super().__init__(parentItem=parentItem, meshdata=self._meshdata, **mesh_kwds)
        
        self.setData(**surface_kwds)
        
    def setData(self, **kwds):
        """
        Update the data in this surface plot. 
        
        ==============  =====================================================================
        **Arguments:**
        x,y             1D arrays of values specifying the x,y positions of vertexes in the
                        grid. If these are omitted, then the values will be assumed to be
                        integers.
        z               2D array of height values for each grid vertex.
        colors          (width, height, 4) array of vertex colors.
        showGrid        Show the grid lines.
        lineColor       Color of the grid lines.
        lineWidth       Width of the grid lines.
        lineAntialias   Enable antialiasing for the grid lines.
        ==============  =====================================================================
        
        All arguments are optional.
        
        Note that if vertex positions are updated, the normal vectors for each triangle must 
        be recomputed. This is somewhat expensive if the surface was initialized with smooth=False
        and very expensive if smooth=True. For faster performance, initialize with 
        computeNormals=False and use per-vertex colors or a normal-independent shader program.
        """

        for arg in self.grid_keys:
            if arg in kwds:
                setattr(self, '_' + arg, kwds[arg])

        x, y, z, colors = map(kwds.get, self.mesh_keys)

        if x is not None:
            if self._x is None or len(x) != len(self._x):
                self._vertexes = None
            self._x = x
        
        if y is not None:
            if self._y is None or len(y) != len(self._y):
                self._vertexes = None
            self._y = y
        
        if z is not None:
            if self._x is not None and z.shape[0] != len(self._x):
                raise Exception('Z values must have shape (len(x), len(y))')
            if self._y is not None and z.shape[1] != len(self._y):
                raise Exception('Z values must have shape (len(x), len(y))')
            self._z = z
            if self._vertexes is not None and self._z.shape != self._vertexes.shape[:2]:
                self._vertexes = None
        
        if colors is not None:
            self._colors = colors
            self._meshdata.setVertexColors(colors)
        
        if self._z is None:
            return
        
        updateMesh = False
        newVertexes = False
        
        ## Generate vertex and face array
        if self._vertexes is None:
            newVertexes = True
            self._vertexes = np.empty((self._z.shape[0], self._z.shape[1], 3), dtype=np.float32)
            self.generateFaces()
            self._meshdata.setFaces(self._faces)
            updateMesh = True
        
        ## Copy x, y, z data into vertex array
        if newVertexes or x is not None:
            if x is None:
                if self._x is None:
                    x = np.arange(self._z.shape[0])
                else:
                    x = self._x
            self._vertexes[:, :, 0] = x.reshape(len(x), 1)
            updateMesh = True
        
        if newVertexes or y is not None:
            if y is None:
                if self._y is None:
                    y = np.arange(self._z.shape[1])
                else:
                    y = self._y
            self._vertexes[:, :, 1] = y.reshape(1, len(y))
            updateMesh = True
        
        if newVertexes or z is not None:
            self._vertexes[...,2] = self._z
            updateMesh = True
        
        ## Update MeshData
        if updateMesh:
            self._meshdata.setVertexes(self._vertexes.reshape(self._vertexes.shape[0]*self._vertexes.shape[1], 3))
            self.meshDataChanged()

        # rebuild grid whenever mesh or parent changes
        self._update_grid()

    def generateFaces(self):
        cols = self._z.shape[1]-1
        rows = self._z.shape[0]-1
        faces = np.empty((cols*rows*2, 3), dtype=np.uint32)
        rowtemplate1 = np.arange(cols).reshape(cols, 1) + np.array([[0, 1, cols+1]])
        rowtemplate2 = np.arange(cols).reshape(cols, 1) + np.array([[cols+1, 1, cols+2]])
        for row in range(rows):
            start = row * cols * 2 
            faces[start:start+cols] = rowtemplate1 + row * (cols+1)
            faces[start+cols:start+(cols*2)] = rowtemplate2 + row * (cols+1)
        self._faces = faces

    def _update_grid(self):
        if not self._showGrid or self._z is None:
            return

        if not (view := self.view()):
            return

        for ln in self._grid_lines:
            view.removeItem(ln)
        self._grid_lines.clear()

        tr = self.viewTransform()
        opts = {
            'glOptions': self.gl_line_options,
            'antialias': self._lineAntialias,
            'color': self._lineColor,
            'width': self._lineWidth,
        }

        def map_pts(arr):
            return np.vstack([
                [pt.x(), pt.y(), pt.z()]
                for pt in (tr.map(QVector3D(x, y, z)) for x, y, z in arr)
            ]).astype(np.float32)

        rows, cols = self._z.shape
        for i in range(rows):
            pts = np.column_stack([
                np.full(cols, self._x[i] if self._x is not None else i),
                self._y if self._y is not None else np.arange(cols),
                self._z[i]
            ])
            ln = gl.GLLinePlotItem(pos=map_pts(pts), **opts)
            view.addItem(ln)
            self._grid_lines.append(ln)

        for j in range(cols):
            pts = np.column_stack([
                self._x if self._x is not None else np.arange(rows),
                np.full(rows, self._y[j] if self._y is not None else j),
                self._z[:, j]
            ])
            ln = gl.GLLinePlotItem(pos=map_pts(pts), **opts)
            view.addItem(ln)
            self._grid_lines.append(ln)

    def _setView(self, v):
        super()._setView(v)
        self._update_grid()
