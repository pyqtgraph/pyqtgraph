import enum
import importlib

import numpy as np

from .. import Qt, colormap
from .. import functions as fn
from .. import getConfigOption
from ..Qt import compat
from ..Qt import OpenGLConstants as GLC
from ..Qt import OpenGLHelpers
from ..Qt import QtCore, QtGui, QT_LIB
from .GraphicsObject import GraphicsObject

if QT_LIB in ["PyQt5", "PySide2"]:
    QtOpenGL = QtGui
else:
    QtOpenGL = importlib.import_module(f'{QT_LIB}.QtOpenGL')

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
        polys = np.ndarray(nrows*ncols, dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                bl = points[(r+0)*(ncols+1)+(c+0)]
                tl = points[(r+0)*(ncols+1)+(c+1)]
                br = points[(r+1)*(ncols+1)+(c+0)]
                tr = points[(r+1)*(ncols+1)+(c+1)]
                poly = (bl, br, tr, tl)
                polys[r*ncols+c] = poly
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
        if self.z is not None and np.any(np.isfinite(self.z)):
            if (self.levels is None) or autoLevels:
                # Autoscale colormap
                z_min = np.nanmin(self.z)
                z_max = np.nanmax(self.z)
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

        z_invalid = np.isnan(self.z)
        skip_nans = np.any(z_invalid)
        if skip_nans:
            # note: flattens array
            valid_z = self.z[~z_invalid]
            if len(valid_z) == 0:
                # nothing to draw => return
                painter.end()
                return picture
        else:
            valid_z = self.z

        ## Prepare colormap
        # First we get the LookupTable
        lut = self.lut_qcolor
        # Second we associate each z value, that we normalize, to the lut
        scale = len(lut) - 1
        lo, hi = self.levels[0], self.levels[1]
        rng = hi - lo
        if rng == 0:
            rng = 1
        norm = fn.rescaleData(valid_z, scale / rng, lo, dtype=int, clip=(0, len(lut)-1))

        if Qt.QT_LIB.startswith('PyQt'):
            drawConvexPolygon = lambda x : painter.drawConvexPolygon(*x)
        else:
            drawConvexPolygon = painter.drawConvexPolygon

        self.quads.resize(self.z.shape[0], self.z.shape[1])
        memory = self.quads.ndarray()
        memory[..., 0] = self.x.ravel()
        memory[..., 1] = self.y.ravel()
        polys = self.quads.instances()

        if skip_nans:
            polys = polys[(~z_invalid).flat]

        # group indices of same coloridx together
        color_indices, counts = np.unique(norm, return_counts=True)
        # note: returns flattened array
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
            isinstance(widget, OpenGLHelpers.GraphicsViewGLWidget)
            and self.cmap is not None   # don't support setting colormap by setLookupTable
        ):
            if self.glstate is None:
                self.glstate = OpenGLState(widget)
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

        X, Y, Z = self.x, self.y, self.z

        glstate = self.glstate
        glstate.setup(widget.context())
        glfn = widget.getFunctions()
        program = widget.retrieveProgram("PColorMeshItem")

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

        proj = QtGui.QMatrix4x4()
        proj.ortho(widget.rect())
        tr = self.sceneTransform()
        xc, yc = origin
        tr.translate(xc, yc)
        mvp = proj * QtGui.QMatrix4x4(tr)

        if glstate.flat_shading:
            vtx_array_shape = X.shape
        else:
            vtx_array_shape = Z.shape + (4,)
        num_vtx_mesh = np.prod(vtx_array_shape)
        num_ind_mesh = np.prod(Z.shape) * 6

        # resize (and invalidate) gpu buffers if needed.
        # a reallocation can only occur together with a change in data.
        # i.e. reallocation ==> change in data (render_cache is None)
        if DirtyFlag.DIM in dirty_bits:
            glstate.m_vbo_pos.bind()
            glstate.m_vbo_pos.allocate(num_vtx_mesh * 2 * 4)
            glstate.m_vbo_pos.release()
            glstate.m_vbo_lum.bind()
            glstate.m_vbo_lum.allocate(num_vtx_mesh * 1 * 4)
            glstate.m_vbo_lum.release()

            if glstate.flat_shading:
                # let the bottom-left of each quad be its "anchor".
                # then each quad is made up of 2 triangles
                #   (TR, TL, BL); (BR, TR, BL)
                # that have indices
                #   (stride + 1, stride + 0, 0); (1, stride + 1, 0)
                # where "0" is the relative index of BL
                # and "stride" advances to the next row
                # note that both triangles are created such that their 3rd vertex is at "BL"
                stride = Z.shape[1] + 1
                dim0 = np.arange(0, Z.shape[0]*stride, stride, dtype=np.uint32)[:, np.newaxis, np.newaxis]
                dim1 = np.arange(Z.shape[1], dtype=np.uint32)[np.newaxis, :, np.newaxis]
                dim2 = np.array([stride + 1, stride + 0, 0, 1, stride + 1, 0], dtype=np.uint32)[np.newaxis, np.newaxis, :]
                buf_ind = dim0 + dim1 + dim2
            else:
                # for each quad, we store 4 vertices contiguously (BL, BR, TL, TR)
                # then each quad is made up of 2 triangles
                #   (TR, TL, BL); (BR, TR, BL)
                # that have indices
                #   (3, 2, 0); (1, 3, 0)
                strides = np.cumprod(vtx_array_shape[::-1])[::-1]
                dim0 = np.arange(0, strides[0], strides[1], dtype=np.uint32)[:, np.newaxis, np.newaxis]
                dim1 = np.arange(0, strides[1], strides[2], dtype=np.uint32)[np.newaxis, :, np.newaxis]
                dim2 = np.array([3, 2, 0, 1, 3, 0], dtype=np.uint32)[np.newaxis, np.newaxis, :]
                buf_ind = dim0 + dim1 + dim2

            glstate.m_vbo_ind.bind()
            glstate.m_vbo_ind.allocate(buf_ind, buf_ind.nbytes)
            glstate.m_vbo_ind.release()

            dirty_bits &= ~DirtyFlag.DIM

        if DirtyFlag.LUT in dirty_bits:
            lut = self.cmap.getLookupTable(nPts=256, alpha=True)
            glstate.setTextureLut(lut)
            dirty_bits &= ~DirtyFlag.LUT

        if DirtyFlag.XY in dirty_bits:
            pos = np.empty(vtx_array_shape + (2,), dtype=np.float32)

            if glstate.flat_shading:
                pos[..., 0] = X - xc
                pos[..., 1] = Y - yc
            else:
                XY = np.dstack((X - xc, Y - yc)).astype(np.float32)
                pos[..., 0, :] = XY[:-1, :-1, :] # BL
                pos[..., 1, :] = XY[1:, :-1, :]  # BR
                pos[..., 2, :] = XY[:-1, 1:, :]  # TL
                pos[..., 3, :] = XY[1:, 1:, :]   # TR

            glstate.m_vbo_pos.bind()
            glstate.m_vbo_pos.write(0, pos, pos.nbytes)
            glstate.m_vbo_pos.release()

            dirty_bits &= ~DirtyFlag.XY

        if DirtyFlag.Z in dirty_bits:
            lum = np.empty(vtx_array_shape, dtype=np.float32)

            if glstate.flat_shading:
                lum[:-1, :-1] = Z
            else:
                lum[..., :] = np.expand_dims(Z, axis=2)

            glstate.m_vbo_lum.bind()
            glstate.m_vbo_lum.write(0, lum, lum.nbytes)
            glstate.m_vbo_lum.release()

            dirty_bits &= ~DirtyFlag.Z

        glstate.render_cache = [origin, dirty_bits]

        widget.setViewboxClip(view)

        glstate.m_vao.bind()
        glstate.m_texture.bind()
        program.bind()

        lo, hi = self.levels
        rng = hi - lo
        if rng == 0:
            rng = 1
        OpenGLHelpers.setUniformValue(program, "u_rescale", QtGui.QVector2D(1/rng, lo))

        OpenGLHelpers.setUniformValue(program, "u_mvp", mvp)

        NULL = compat.voidptr(0) if QT_LIB.startswith("PySide") else None
        glfn.glDrawElements(GLC.GL_TRIANGLES, num_ind_mesh, GLC.GL_UNSIGNED_INT, NULL)

        glstate.m_vao.release()


class OpenGLState(QtCore.QObject):
    VERT_SRC_COMPAT = """
        attribute vec4 a_position;
        attribute float a_luminance;
        varying float v_luminance;
        uniform mat4 u_mvp;
        uniform vec2 u_rescale;
        void main() {
            v_luminance = u_rescale.x * (a_luminance - u_rescale.y);
            gl_Position = u_mvp * a_position;
        }
    """
    FRAG_SRC_COMPAT = """
        #ifdef GL_ES
        precision mediump float;
        #endif
        varying float v_luminance;
        uniform sampler2D u_texture;
        void main() {
            if (!(v_luminance == v_luminance)) discard;
            float s = clamp(v_luminance, 0.0, 1.0);
            gl_FragColor = texture2D(u_texture, vec2(s, 0));
        }
    """

    VERT_SRC = """
        in vec4 a_position;
        in float a_luminance;
        flat out float v_luminance;
        uniform mat4 u_mvp;
        uniform vec2 u_rescale;
        void main() {
            v_luminance = u_rescale.x * (a_luminance - u_rescale.y);
            gl_Position = u_mvp * a_position;
        }
    """
    FRAG_SRC = """
        #ifdef GL_ES
        precision mediump float;
        #endif
        flat in float v_luminance;
        out vec4 FragColor;
        uniform sampler2D u_texture;
        void main() {
            if (isnan(v_luminance)) discard;
            float s = clamp(v_luminance, 0.0, 1.0);
            FragColor = texture(u_texture, vec2(s, 0));
        }
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.context = None
        self.render_cache = None
        self.m_vao = QtOpenGL.QOpenGLVertexArrayObject(self)
        self.m_vbo_pos = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.m_vbo_lum = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)
        self.m_vbo_ind = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.IndexBuffer)
        self.m_texture = QtOpenGL.QOpenGLTexture(QtOpenGL.QOpenGLTexture.Target.Target2D)

    def setup(self, context):
        if self.context is context:
            return

        if self.context is not None:
            self.context.aboutToBeDestroyed.disconnect(self.cleanup)
            self.cleanup()

        self.context = context
        self.context.aboutToBeDestroyed.connect(self.cleanup)

        is_opengles = self.context.isOpenGLES()
        gl_version = self.context.format().version()
        if not is_opengles and gl_version >= (3, 1):
            moderngl = True
        elif is_opengles and gl_version >= (3, 0):
            moderngl = True
        else:
            moderngl = False

        if moderngl:
            self.flat_shading = True
            if not is_opengles:
                glsl_version = "#version 140"
            else:
                glsl_version = "#version 300 es"
            VERT_SRC = "\n".join([glsl_version, OpenGLState.VERT_SRC])
            FRAG_SRC = "\n".join([glsl_version, OpenGLState.FRAG_SRC])
        else:
            self.flat_shading = False
            VERT_SRC = OpenGLState.VERT_SRC_COMPAT
            FRAG_SRC = OpenGLState.FRAG_SRC_COMPAT

        glwidget = self.parent()
        program = glwidget.retrieveProgram("PColorMeshItem")
        if program is None:
            program = QtOpenGL.QOpenGLShaderProgram()
            if not program.addShaderFromSourceCode(QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex, VERT_SRC):
                raise RuntimeError(program.log())
            if not program.addShaderFromSourceCode(QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment, FRAG_SRC):
                raise RuntimeError(program.log())
            program.bindAttributeLocation("a_position", 0)
            program.bindAttributeLocation("a_luminance", 1)
            if not program.link():
                raise RuntimeError(program.log())
        glwidget.storeProgram("PColorMeshItem", program)

        self.m_vao.create()
        self.m_vbo_pos.create()
        self.m_vbo_lum.create()
        self.m_vbo_ind.create()


        loc_pos, loc_lum = 0, 1
        self.m_vao.bind()
        self.m_vbo_ind.bind()
        self.m_vbo_pos.bind()
        program.enableAttributeArray(loc_pos)
        program.setAttributeBuffer(loc_pos, GLC.GL_FLOAT, 0, 2)
        self.m_vbo_pos.release()
        self.m_vbo_lum.bind()
        program.enableAttributeArray(loc_lum)
        program.setAttributeBuffer(loc_lum, GLC.GL_FLOAT, 0, 1)
        self.m_vbo_lum.release()
        self.m_vao.release()
        self.m_vbo_ind.release()

    def cleanup(self):
        # this method should restore the state back to __init__
        glwidget = self.parent()
        glwidget.makeCurrent()

        for name in ['m_texture', 'm_vbo_pos', 'm_vbo_lum', 'm_vbo_ind', 'm_vao']:
            obj = getattr(self, name)
            obj.destroy()

        self.context = None
        self.render_cache = None

        glwidget.doneCurrent()

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
