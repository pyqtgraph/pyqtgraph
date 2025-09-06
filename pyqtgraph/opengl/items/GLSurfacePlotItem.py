import numpy as np

from ..MeshData import MeshData
from .GLMeshItem import GLMeshItem
from .GLLinePlotItem import GLLinePlotItem
from OpenGL import GL as ogl

__all__ = ['GLSurfacePlotItem']

class GLSurfacePlotItem(GLMeshItem):
    """
    **Bases:** :class:`GLMeshItem <pyqtgraph.opengl.GLMeshItem>`
    
    Displays a surface plot on a regular x,y grid with optional wireframe overlay.
    """

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
        self._lineAntialias = False
        self._vertexes = None
        self._meshdata = MeshData()

        # splitout GLSurfacePlotItem from kwds
        surface_keys = self.mesh_keys + self.grid_keys
        surface_kwds = {}
        for arg in surface_keys:
            if arg in kwds:
                surface_kwds[arg] = kwds.pop(arg)

        super().__init__(meshdata=self._meshdata, **kwds)
        
        self.lineplot = GLLinePlotItem(parentItem=self, mode='lines', glOptions='translucent')
        # in GLViewWidget.drawItemTree(), at the same depth value, child items
        # come before the parent. make it such that our grid lines get drawn
        # after the surface mesh.
        self.lineplot.setDepthValue(self.depthValue() + 1)
        self.setParentItem(parentItem)

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

    def paint(self):
        if self._showGrid:
            ogl.glEnable(ogl.GL_POLYGON_OFFSET_FILL)
            ogl.glPolygonOffset(1.0, 1.0)
        super().paint()
        if self._showGrid:
            ogl.glDisable(ogl.GL_POLYGON_OFFSET_FILL)
            ogl.glPolygonOffset(0.0, 0.0)

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

        opts = {
            'antialias':  self._lineAntialias,
            'color':      self._lineColor,
            'width':      self._lineWidth,
        }

        z = self._z.astype(np.float32)
        rows, cols = z.shape

        x = (self._x if self._x is not None else np.arange(rows, dtype=z.dtype))
        y = (self._y if self._y is not None else np.arange(cols, dtype=z.dtype))

        xvals, yvals = np.meshgrid(x, y, indexing='ij')  # shape (rows, cols)
        verts_flat = np.column_stack((xvals.ravel(), yvals.ravel(), z.ravel()))

        idx = np.arange(z.size, dtype=np.int32).reshape(rows, cols)
        h = np.column_stack((idx[:, :-1].ravel(), idx[:, 1:].ravel()))
        v = np.column_stack((idx[:-1, :].ravel(), idx[1:, :].ravel()))
        edges = np.vstack((h, v))

        pts = verts_flat[edges].reshape(-1, 3)

        self.lineplot.setData(pos=pts, **opts)
