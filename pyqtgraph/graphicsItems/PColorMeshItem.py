import enum
import importlib
import warnings

import numpy as np

from .. import Qt, colormap
from .. import functions as fn
from .. import getConfigOption
from ..Qt import compat
from ..Qt import OpenGLConstants as GLC
from ..Qt import QtCore, QtGui, QtWidgets, QT_LIB
from .GraphicsObject import GraphicsObject

__all__ = ['PColorMeshItem']


class DirtyFlag(enum.Flag):
    XY = enum.auto()
    Z = enum.auto()
    LUT = enum.auto()
    DIM = enum.auto()


class QuadInstances:
    def __init__(self):
        self.nrows = -1
        self.ncols = -1
        self.pointsarray = Qt.internals.PrimitiveArray(QtCore.QPointF, 2)
        self.resize(0, 0)

    def resize(self, nrows, ncols):
        if nrows == self.nrows and ncols == self.ncols:
            return

        self.nrows = nrows
        self.ncols = ncols

        # (nrows + 1) * (ncols + 1) vertices, (x, y)
        self.pointsarray.resize((nrows+1)*(ncols+1))
        points = self.pointsarray.instances()
        # points is a flattened list of a 2d array of
        # QPointF(s) of shape (nrows+1, ncols+1)

        # pre-create quads from those instances of QPointF(s).
        # store the quads as a flattened list of a 2d array
        # of polygons of shape (nrows, ncols)
        polys = []
        for r in range(nrows):
            for c in range(ncols):
                bl = points[(r+0)*(ncols+1)+(c+0)]
                tl = points[(r+0)*(ncols+1)+(c+1)]
                br = points[(r+1)*(ncols+1)+(c+0)]
                tr = points[(r+1)*(ncols+1)+(c+1)]
                poly = (bl, br, tr, tl)
                polys.append(poly)
        self.polys = polys

    def ndarray(self):
        return self.pointsarray.ndarray()

    def instances(self):
        return self.polys


class PColorMeshItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    """

    sigLevelsChanged = QtCore.Signal(object)  # emits tuple with levels (low,high) when color levels are changed.

    def __init__(self, *args, **kwargs):
        """
        Create a pseudocolor plot with convex polygons.

        Call signature:

        ``PColorMeshItem([x, y,] z, **kwargs)``

        x and y can be used to specify the corners of the quadrilaterals.
        z must be used to specified to color of the quadrilaterals.

        Parameters
        ----------
        x, y : np.ndarray, optional, default None
            2D array containing the coordinates of the polygons
        z : np.ndarray
            2D array containing the value which will be mapped into the polygons
            colors.
            If x and y is None, the polygons will be displaced on a grid
            otherwise x and y will be used as polygons vertices coordinates as::

                (x[i+1, j], y[i+1, j])           (x[i+1, j+1], y[i+1, j+1])
                                    +---------+
                                    | z[i, j] |
                                    +---------+
                    (x[i, j], y[i, j])           (x[i, j+1], y[i, j+1])

            "ASCII from: <https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.pcolormesh.html>".
        colorMap : pyqtgraph.ColorMap
            Colormap used to map the z value to colors.
            default ``pyqtgraph.colormap.get('viridis')``
        levels: tuple, optional, default None
            Sets the minimum and maximum values to be represented by the colormap (min, max). 
            Values outside this range will be clipped to the colors representing min or max.
            ``None`` disables the limits, meaning that the colormap will autoscale 
            the next time ``setData()`` is called with new data.
        enableAutoLevels: bool, optional, default True
            Causes the colormap levels to autoscale whenever ``setData()`` is called. 
            It is possible to override this value on a per-change-basis by using the
            ``autoLevels`` keyword argument when calling ``setData()``.
            If ``enableAutoLevels==False`` and ``levels==None``, autoscaling will be 
            performed once when the first z data is supplied. 
        edgecolors : dict, optional
            The color of the edges of the polygons.
            Default None means no edges.
            Only cosmetic pens are supported.
            The dict may contains any arguments accepted by :func:`mkColor() <pyqtgraph.mkColor>`.
            Example: ``mkPen(color='w', width=2)``
        antialiasing : bool, default False
            Whether to draw edgelines with antialiasing.
            Note that if edgecolors is None, antialiasing is always False.
        """

        GraphicsObject.__init__(self)

        self.qpicture = None  ## rendered picture for display
        self.x = None
        self.y = None
        self.z = None
        self._dataBounds = None
        self.glstate = None

        self.edgecolors = kwargs.get('edgecolors', None)
        if self.edgecolors is not None:
            self.edgecolors = fn.mkPen(self.edgecolors)
            # force the pen to be cosmetic. see discussion in
            # https://github.com/pyqtgraph/pyqtgraph/pull/2586
            self.edgecolors.setCosmetic(True)
        self.antialiasing = kwargs.get('antialiasing', False)
        self.levels = kwargs.get('levels', None)
        self._defaultAutoLevels = kwargs.get('enableAutoLevels', True)
        
        if 'colorMap' in kwargs:
            cmap = kwargs.get('colorMap')
            if not isinstance(cmap, colormap.ColorMap):
                raise ValueError('colorMap argument must be a ColorMap instance')
            self.cmap = cmap
        else:
            self.cmap = colormap.get('viridis')

        self.lut_qcolor = self.cmap.getLookupTable(nPts=256, mode=self.cmap.QCOLOR)

        self.quads = QuadInstances()

        # If some data have been sent we directly display it
        if len(args)>0:
            self.setData(*args)


    def _prepareData(self, args) -> DirtyFlag:
        """
        Check the shape of the data.
        Return a set of 2d array x, y, z ready to be used to draw the picture.
        """

        dirtyFlags = DirtyFlag.XY | DirtyFlag.Z | DirtyFlag.DIM

        # User didn't specified data
        if len(args)==0:

            self.x = None
            self.y = None
            self.z = None

            self._dataBounds = None
            
        # User only specified z
        elif len(args)==1:
            # If x and y is None, the polygons will be displaced on a grid
            x = np.arange(0, args[0].shape[0]+1, 1)
            y = np.arange(0, args[0].shape[1]+1, 1)
            self.x, self.y = np.meshgrid(x, y, indexing='ij')
            self.z = args[0]

            self._dataBounds = ((x[0], x[-1]), (y[0], y[-1]))

        # User specified x, y, z
        elif len(args)==3:
            # specifying None explicitly means to retain the existing value
            if (x := args[0]) is None:
                x = self.x
            if (y := args[1]) is None:
                y = self.y
            if (z := args[2]) is None:
                z = self.z

            if args[0] is None and args[1] is None:
                dirtyFlags &= ~DirtyFlag.XY
            if args[2] is None:
                dirtyFlags &= ~DirtyFlag.Z

            if self.z is not None and z.shape == self.z.shape:
                dirtyFlags &= ~DirtyFlag.DIM

            # Shape checking
            xy_shape = (z.shape[0]+1, z.shape[1]+1)
            if x.shape != xy_shape:
                raise ValueError('The dimension of x should be one greater than the one of z')
            if y.shape != xy_shape:
                raise ValueError('The dimension of y should be one greater than the one of z')
        
            self.x = x
            self.y = y
            self.z = z

            xmn, xmx = np.min(self.x), np.max(self.x)
            ymn, ymx = np.min(self.y), np.max(self.y)
            self._dataBounds = ((xmn, xmx), (ymn, ymx))

        else:
            raise ValueError('Data must been sent as (z) or (x, y, z)')

        return dirtyFlags

    def setData(self, *args, **kwargs):
        """
        Set the data to be drawn.

        Parameters
        ----------
        x, y : np.ndarray, optional, default None
            2D array containing the coordinates of the polygons
        z : np.ndarray
            2D array containing the value which will be mapped into the polygons
            colors.
            If x and y is None, the polygons will be displaced on a grid
            otherwise x and y will be used as polygons vertices coordinates as::
                
                (x[i+1, j], y[i+1, j])           (x[i+1, j+1], y[i+1, j+1])
                                    +---------+
                                    | z[i, j] |
                                    +---------+
                    (x[i, j], y[i, j])           (x[i, j+1], y[i, j+1])

            "ASCII from: <https://matplotlib.org/3.2.1/api/_as_gen/
                         matplotlib.pyplot.pcolormesh.html>".
        autoLevels: bool, optional
            If set, overrides the value of ``enableAutoLevels``
        """
        old_bounds = self._dataBounds
        dirtyFlags = self._prepareData(args)
        boundsChanged = old_bounds != self._dataBounds

        self._rerender(
            autoLevels=kwargs.get('autoLevels', self._defaultAutoLevels)
        )

        if boundsChanged:
            self.prepareGeometryChange()
            self.informViewBoundsChanged()

        if self.glstate is not None:
            self.glstate.dataChange(dirtyFlags)

        self.update()

    def _rerender(self, *, autoLevels):
        self.qpicture = None
        if self.z is not None:
            if (self.levels is None) or autoLevels:
                # Autoscale colormap
                z_min = self.z.min()
                z_max = self.z.max()
                self.setLevels( (z_min, z_max), update=False)

    def _drawPicture(self) -> QtGui.QPicture:
        # on entry, the following members are all valid: x, y, z, levels
        # this function does not alter any state (besides using self.quads)

        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        # We set the pen of all polygons once
        if self.edgecolors is None:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
        else:
            painter.setPen(self.edgecolors)
            if self.antialiasing:
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        ## Prepare colormap
        # First we get the LookupTable
        lut = self.lut_qcolor
        # Second we associate each z value, that we normalize, to the lut
        scale = len(lut) - 1
        lo, hi = self.levels[0], self.levels[1]
        rng = hi - lo
        if rng == 0:
            rng = 1
        norm = fn.rescaleData(self.z, scale / rng, lo, dtype=int, clip=(0, len(lut)-1))

        if Qt.QT_LIB.startswith('PyQt'):
            drawConvexPolygon = lambda x : painter.drawConvexPolygon(*x)
        else:
            drawConvexPolygon = painter.drawConvexPolygon

        self.quads.resize(self.z.shape[0], self.z.shape[1])
        memory = self.quads.ndarray()
        memory[..., 0] = self.x.ravel()
        memory[..., 1] = self.y.ravel()
        polys = self.quads.instances()

        # group indices of same coloridx together
        color_indices, counts = np.unique(norm, return_counts=True)
        sorted_indices = np.argsort(norm, axis=None)

        offset = 0
        for coloridx, cnt in zip(color_indices, counts):
            indices = sorted_indices[offset:offset+cnt]
            offset += cnt
            painter.setBrush(lut[coloridx])
            for idx in indices:
                drawConvexPolygon(polys[idx])

        painter.end()
        return picture


    def setLevels(self, levels, update=True):
        """
        Sets color-scaling levels for the mesh. 
        
        Parameters
        ----------
            levels: tuple
                ``(low, high)`` 
                sets the range for which values can be represented in the colormap.
            update: bool, optional
                Controls if mesh immediately updates to reflect the new color levels.
        """
        self.levels = levels
        self.sigLevelsChanged.emit(levels)
        if update:
            self._rerender(autoLevels=False)
            self.update()

    def getLevels(self):
        """
        Returns a tuple containing the current level settings. See :func:`~setLevels`.
        The format is ``(low, high)``.
        """
        return self.levels


    
    def setLookupTable(self, lut, update=True):
        self.cmap = None    # invalidate since no longer consistent with lut
        self.lut_qcolor = lut[:]
        if self.glstate is not None:
            self.glstate.dataChange(DirtyFlag.LUT)
        if update:
            self._rerender(autoLevels=False)
            self.update()

    def getColorMap(self):
        return self.cmap

    def setColorMap(self, cmap):
        self.setLookupTable(cmap.getLookupTable(nPts=256, mode=cmap.QCOLOR), update=True)
        self.cmap = cmap

    def enableAutoLevels(self):
        self._defaultAutoLevels = True

    def disableAutoLevels(self):
        self._defaultAutoLevels = False

    def paint(self, painter, opt, widget):
        if self.z is None:
            return

        if (
            getConfigOption('enableExperimental')
            and isinstance(widget, QtWidgets.QOpenGLWidget)
            and self.cmap is not None   # don't support setting colormap by setLookupTable
        ):
            if self.glstate is None:
                self.glstate = OpenGLState()
            painter.beginNativePainting()
            try:
                self.paintGL(widget)
            finally:
                painter.endNativePainting()

            if (
                self.edgecolors is not None
                and self.edgecolors.style() != QtCore.Qt.PenStyle.NoPen
            ):
                painter.setPen(self.edgecolors)
                if self.antialiasing:
                    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                for idx in range(self.x.shape[0]):
                    painter.drawPolyline(fn.arrayToQPolygonF(self.x[idx, :], self.y[idx, :]))
                for idx in range(self.x.shape[1]):
                    painter.drawPolyline(fn.arrayToQPolygonF(self.x[:, idx], self.y[:, idx]))

            return

        if self.qpicture is None:
            self.qpicture = self._drawPicture()
        painter.drawPicture(0, 0, self.qpicture)

    def width(self):
        if self._dataBounds is None:
            return 0
        bounds = self._dataBounds[0]
        return bounds[1]-bounds[0]

    def height(self):
        if self._dataBounds is None:
            return 0
        bounds = self._dataBounds[1]
        return bounds[1]-bounds[0]

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if self._dataBounds is None:
            return (None, None)
        return self._dataBounds[ax]

    def pixelPadding(self):
        # pen is known to be cosmetic
        pen = self.edgecolors
        no_pen = (pen is None) or (pen.style() == QtCore.Qt.PenStyle.NoPen)
        return 0 if no_pen else (pen.widthF() or 1) * 0.5

    def boundingRect(self):
        xmn, xmx = self.dataBounds(ax=0)
        if xmn is None or xmx is None:
            return QtCore.QRectF()
        ymn, ymx = self.dataBounds(ax=1)
        if ymn is None or ymx is None:
            return QtCore.QRectF()

        px = py = 0
        pxPad = self.pixelPadding()
        if pxPad > 0:
            # determine length of pixel in local x, y directions
            px, py = self.pixelVectors()
            px = 0 if px is None else px.length()
            py = 0 if py is None else py.length()
            # return bounds expanded by pixel size
            px *= pxPad
            py *= pxPad

        return QtCore.QRectF(xmn-px, ymn-py, (2*px)+xmx-xmn, (2*py)+ymx-ymn)

    def paintGL(self, widget):
        if (view := self.getViewBox()) is None:
            return

        num_vtx_stencil = 4
        X, Y, Z = self.x, self.y, self.z

        glstate = self.glstate
        glf = glstate.setup(widget.context())

        proj = QtGui.QMatrix4x4()
        proj.ortho(0, widget.width(), widget.height(), 0, -999999, 999999)

        tr = self.sceneTransform()
        # OpenGL only sees the float32 version of our data, and this may cause
        # precision issues. To mitigate this, we shift the origin of our data
        # to the center of its bounds.
        # Note that xc, yc are double precision Python floats. Subtracting them
        # from the x, y ndarrays will automatically upcast the latter to double
        # precision.
        if glstate.render_cache is None:
            origin = None
            dirty_bits = DirtyFlag.XY | DirtyFlag.Z | DirtyFlag.LUT | DirtyFlag.DIM
        else:
            origin, dirty_bits = glstate.render_cache
        if origin is None or DirtyFlag.XY in dirty_bits:
            # the origin point is calculated once per data change.
            # once the data is uploaded, the origin point is fixed.
            center = self.boundingRect().center()
            origin = center.x(), center.y()

        xc, yc = origin
        tr.translate(xc, yc)
        mvp_curve = proj * QtGui.QMatrix4x4(tr)

        mvp_stencil = proj
        rect = view.mapRectToScene(view.boundingRect())
        x0, y0, x1, y1 = rect.getCoords()
        stencil_vtx = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float32)
        stencil_lum = np.zeros((num_vtx_stencil, 1), dtype=np.float32)

        if glstate.use_ibo:
            vtx_array_shape = X.shape
            num_ind_mesh = np.prod(Z.shape) * 6
        else:
            vtx_array_shape = Z.shape + (6,)
            num_ind_mesh = 0
        num_vtx_mesh = np.prod(vtx_array_shape)

        # resize (and invalidate) gpu buffers if needed.
        # a reallocation can only occur together with a change in data.
        # i.e. reallocation ==> change in data (render_cache is None)
        if DirtyFlag.DIM in dirty_bits:
            vbo_num_vtx = num_vtx_stencil + num_vtx_mesh
            glstate.m_vbo_pos.bind()
            glstate.m_vbo_pos.allocate(vbo_num_vtx * 2 * 4)
            glstate.m_vbo_pos.release()
            glstate.m_vbo_lum.bind()
            glstate.m_vbo_lum.allocate(vbo_num_vtx * 1 * 4)
            glstate.m_vbo_lum.release()

            if glstate.use_ibo:
                # let the bottom-left of each quad be its "anchor".
                # then each quad is made up of 2 triangles
                #   (TR, TL, BL); (BR, TR, BL)
                # that have indices
                #   (stride + 1, stride + 0, 0); (1, stride + 1, 0)
                # where "0" is the relative index of BR
                # and "stride" advances to the next row
                # note that both triangles are created such that their 3rd vertex is at "BL"
                stride = Z.shape[1] + 1
                dim0 = np.arange(0, Z.shape[0]*stride, stride, dtype=np.uint32)[:, np.newaxis, np.newaxis]
                dim1 = np.arange(Z.shape[1], dtype=np.uint32)[np.newaxis, :, np.newaxis]
                dim2 = np.array([stride + 1, stride + 0, 0, 1, stride + 1, 0], dtype=np.uint32)[np.newaxis, np.newaxis, :] + num_vtx_stencil
                buf_ind = dim0 + dim1 + dim2

                glstate.m_vbo_ind.bind()
                glstate.m_vbo_ind.allocate(buf_ind, buf_ind.nbytes)
                glstate.m_vbo_ind.release()

            dirty_bits &= ~DirtyFlag.DIM

        buf_pos = stencil_vtx
        buf_lum = stencil_lum

        if DirtyFlag.LUT in dirty_bits:
            lut = self.cmap.getLookupTable(nPts=256, alpha=True)
            glstate.setTextureLut(lut)
            dirty_bits &= ~DirtyFlag.LUT

        if DirtyFlag.XY in dirty_bits:
            buf_pos = np.empty((num_vtx_stencil + num_vtx_mesh, 2), dtype=np.float32)
            buf_pos[:num_vtx_stencil, :] = stencil_vtx
            pos = buf_pos[num_vtx_stencil:, :].reshape(vtx_array_shape + (2,))

            if glstate.use_ibo:
                pos[..., 0] = X - xc
                pos[..., 1] = Y - yc
            else:
                XY = np.dstack((X - xc, Y - yc)).astype(np.float32)
                pos[..., 0, :] = XY[:-1, :-1, :] # BL
                pos[..., 1, :] = XY[1:, :-1, :]  # BR
                pos[..., 2, :] = XY[:-1, 1:, :]  # TL
                pos[..., 3, :] = XY[1:, :-1, :]  # BR
                pos[..., 4, :] = XY[1:, 1:, :]   # TR
                pos[..., 5, :] = XY[:-1, 1:, :]  # TL

            dirty_bits &= ~DirtyFlag.XY

        if DirtyFlag.Z in dirty_bits:
            buf_lum = np.empty((num_vtx_stencil + num_vtx_mesh, 1), dtype=np.float32)
            buf_lum[:num_vtx_stencil, :] = stencil_lum
            lum = buf_lum[num_vtx_stencil:, :].reshape(vtx_array_shape)

            if glstate.use_ibo:
                lum[:-1, :-1] = Z
            else:
                lum[..., :] = np.expand_dims(Z, axis=2)

            dirty_bits &= ~DirtyFlag.Z

        # upload VBO, minimally for the stencil
        glstate.m_vbo_pos.bind()
        glstate.m_vbo_pos.write(0, buf_pos, buf_pos.nbytes)
        glstate.m_vbo_pos.release()
        glstate.m_vbo_lum.bind()
        glstate.m_vbo_lum.write(0, buf_lum, buf_lum.nbytes)
        glstate.m_vbo_lum.release()

        glstate.render_cache = [origin, dirty_bits]

        glstate.m_texture.bind()
        glstate.m_program.bind()
        glstate.m_vao.bind()

        glstate.setUniformValue("u_mvp", mvp_stencil)

        lo, hi = self.levels
        rng = hi - lo
        if rng == 0:
            rng = 1
        glstate.setUniformValue("u_rescale", QtGui.QVector2D(1/rng, lo))

        # set clipping viewport
        glf.glEnable(GLC.GL_STENCIL_TEST)
        glf.glColorMask(False, False, False, False) # disable drawing to frame buffer
        glf.glDepthMask(False)  # disable drawing to depth buffer
        glf.glStencilFunc(GLC.GL_NEVER, 1, 0xFF)
        glf.glStencilOp(GLC.GL_REPLACE, GLC.GL_KEEP, GLC.GL_KEEP)

        ## draw stencil pattern
        glf.glStencilMask(0xFF)
        glf.glClear(GLC.GL_STENCIL_BUFFER_BIT)
        glf.glDrawArrays(GLC.GL_TRIANGLE_STRIP, 0, num_vtx_stencil)

        glf.glColorMask(True, True, True, True)
        glf.glDepthMask(True)
        glf.glStencilMask(0x00)
        glf.glStencilFunc(GLC.GL_EQUAL, 1, 0xFF)

        glstate.setUniformValue("u_mvp", mvp_curve)

        if glstate.use_ibo:
            NULL = compat.voidptr(0) if QT_LIB.startswith("PySide") else None
            glf.glDrawElements(GLC.GL_TRIANGLES, num_ind_mesh, GLC.GL_UNSIGNED_INT, NULL)
        else:
            glf.glDrawArrays(GLC.GL_TRIANGLES, num_vtx_stencil, num_vtx_mesh)
        glstate.m_vao.release()

        # destroy the texture to avoid the following warning:
        # "QOpenGLTexturePrivate::destroy() called without a current context."
        # this is inefficient... but better slow than to have a leak.
        glstate.m_texture.destroy()
        glstate.render_cache[1] |= DirtyFlag.LUT


class OpenGLState:
    VERT_SRC_COMPAT = """
        attribute vec4 a_position;
        attribute float a_luminance;
        varying float v_luminance;
        uniform mat4 u_mvp;
        uniform vec2 u_rescale;
        void main() {
            v_luminance = clamp(u_rescale.x * (a_luminance - u_rescale.y), 0.0, 1.0);
            gl_Position = u_mvp * a_position;
        }
    """
    FRAG_SRC_COMPAT = """
        varying mediump float v_luminance;
        uniform mediump sampler2D u_texture;
        void main() {
            gl_FragColor = texture2D(u_texture, vec2(v_luminance, 0));
        }
    """

    VERT_SRC = """
        #version 140
        in vec4 a_position;
        in float a_luminance;
        flat out float v_luminance;
        uniform mat4 u_mvp;
        uniform vec2 u_rescale;
        void main() {
            v_luminance = clamp(u_rescale.x * (a_luminance - u_rescale.y), 0.0, 1.0);
            gl_Position = u_mvp * a_position;
        }
    """
    FRAG_SRC = """
        #version 140
        flat in float v_luminance;
        out vec4 FragColor;
        uniform sampler2D u_texture;
        void main() {
            FragColor = texture(u_texture, vec2(v_luminance, 0));
        }
    """

    def __init__(self):
        self.context = None
        self.functions = None
        self.render_cache = None
        self.m_vao = None
        self.m_vbo_pos = None
        self.m_vbo_lum = None
        self.m_vbo_ind = None
        self.m_texture = None
        self.m_program = None

        self.use_ibo = True

    def setup(self, context):
        if self.context is context:
            return self.functions

        if self.context is not None:
            self.context.aboutToBeDestroyed.disconnect(self.cleanup)
            self.cleanup()

        self.context = context
        self.context.aboutToBeDestroyed.connect(self.cleanup)

        self.functions = self.getFunctions(context)

        if QT_LIB in ["PyQt5", "PySide2"]:
            QtOpenGL = QtGui
        else:
            QtOpenGL = importlib.import_module(f'{QT_LIB}.QtOpenGL')

        if self.use_ibo is not False and self.context.format().version() >= (3, 1):
            VERT_SRC, FRAG_SRC = OpenGLState.VERT_SRC, OpenGLState.FRAG_SRC
            self.use_ibo = True
        else:
            VERT_SRC, FRAG_SRC = OpenGLState.VERT_SRC_COMPAT, OpenGLState.FRAG_SRC_COMPAT
            self.use_ibo = False
        self.m_program = QtOpenGL.QOpenGLShaderProgram(self.context)
        self.m_program.addShaderFromSourceCode(QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex, VERT_SRC)
        self.m_program.addShaderFromSourceCode(QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment, FRAG_SRC)
        self.m_program.link()

        self.m_vao = QtOpenGL.QOpenGLVertexArrayObject(self.context)
        self.m_vao.create()
        self.m_vbo_pos = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.m_vbo_pos.create()
        self.m_vbo_lum = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.m_vbo_lum.create()
        self.m_vbo_ind = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.IndexBuffer)
        self.m_vbo_ind.create()

        self.m_vao.bind()
        self.m_vbo_ind.bind()
        self.m_vbo_pos.bind()
        loc_pos = self.m_program.attributeLocation("a_position")
        self.m_program.enableAttributeArray(loc_pos)
        self.m_program.setAttributeBuffer(loc_pos, GLC.GL_FLOAT, 0, 2)
        self.m_vbo_pos.release()
        loc_lum = self.m_program.attributeLocation("a_luminance")
        self.m_vbo_lum.bind()
        self.m_program.enableAttributeArray(loc_lum)
        self.m_program.setAttributeBuffer(loc_lum, GLC.GL_FLOAT, 0, 1)
        self.m_vbo_lum.release()
        self.m_vao.release()
        self.m_vbo_ind.release()

        self.m_texture = QtOpenGL.QOpenGLTexture(QtOpenGL.QOpenGLTexture.Target.Target2D)

        return self.functions

    def getFunctions(self, context):
        if QT_LIB == 'PyQt5':
            # it would have been cleaner to call context.versionFunctions().
            # however, when there are multiple GraphicsItems, the following bug occurs:
            # all except one of the C++ objects of the returned versionFunctions() get
            # deleted.
            import PyQt5._QOpenGLFunctions_2_0 as QtOpenGLFunctions
            glf = QtOpenGLFunctions.QOpenGLFunctions_2_0()
            glf.initializeOpenGLFunctions()
        elif QT_LIB == 'PySide2':
            import PySide2.QtOpenGLFunctions as QtOpenGLFunctions
            glf = QtOpenGLFunctions.QOpenGLFunctions_2_0()
            glf.initializeOpenGLFunctions()
        else:
            QtOpenGL = importlib.import_module(f'{QT_LIB}.QtOpenGL')
            profile = QtOpenGL.QOpenGLVersionProfile()
            profile.setVersion(2, 0)
            glf = QtOpenGL.QOpenGLVersionFunctionsFactory.get(profile, context)

        return glf

    def setUniformValue(self, key, value):
        # convenience function to mask the warnings
        with warnings.catch_warnings():
            # PySide2 : RuntimeWarning: SbkConverter: Unimplemented C++ array type.
            warnings.simplefilter("ignore")
            self.m_program.setUniformValue(key, value)

    def cleanup(self):
        # this method should restore the state back to __init__

        if self.m_program is not None:
            self.m_program.setParent(None)
            self.m_program = None
        for name in ['m_texture', 'm_vbo_pos', 'm_vbo_lum', 'm_vbo_ind', 'm_vao']:
            obj = getattr(self, name)
            if obj is not None:
                obj.destroy()
                setattr(self, name, None)

        self.context = None
        self.functions = None
        self.render_cache = None

    def setTextureLut(self, lut):
        tex = self.m_texture
        if not tex.isCreated():
            tex.setFormat(tex.TextureFormat.RGBAFormat)
            tex.setSize(256, 1)
            tex.allocateStorage()
            tex.setMinMagFilters(tex.Filter.Nearest, tex.Filter.Nearest)
            tex.setWrapMode(tex.WrapMode.ClampToEdge)
        tex.setData(tex.PixelFormat.RGBA, tex.PixelType.UInt8, lut)

    def dataChange(self, dirtyFlags : DirtyFlag):
        if self.render_cache is None:
            return

        if DirtyFlag.XY in dirtyFlags:
            self.render_cache[0] = None
            self.render_cache[1] |= DirtyFlag.XY

        if DirtyFlag.Z in dirtyFlags:
            self.render_cache[1] |= DirtyFlag.Z

        if DirtyFlag.DIM in dirtyFlags:
            self.render_cache[1] |= DirtyFlag.DIM

        if DirtyFlag.LUT in dirtyFlags:
            self.render_cache[1] |= DirtyFlag.LUT
