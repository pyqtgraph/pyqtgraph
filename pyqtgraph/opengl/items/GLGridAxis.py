import numpy as np

from ... import getConfigOption
from ...Qt import QtGui, QtCore
from ...Vector import Vector
from ..GLGraphicsItem import GLGraphicsItem
from ...opengl import GLLinePlotItem, GLMeshItem, GLTextItem
from OpenGL import GL


def check_visibility(azimuth_range, azimuth, elevation_range=None, elevation=None):
    """Check if item should be visible based on camera angles.

    Args:
        azimuth_range: tuple (min, max) or None
        elevation_range: tuple (min, max) or None
        azimuth: current azimuth angle
        elevation: current elevation angle

    Returns:
        bool: True if visible, False otherwise

    """
    if azimuth_range:
        min_az, max_az = azimuth_range
        az = azimuth + 360 if max_az > 360 and azimuth < min_az % 360 else azimuth
        if not (min_az <= az < max_az):
            return False

    if elevation_range:
        min_el, max_el = elevation_range
        if not (min_el <= elevation <= max_el):
            return False

    return True


def other_axes(axis):
    """Return the two axes other than the given one."""
    return [i for i in range(3) if i != axis]


class GLPolygonOffsetMeshItem(GLMeshItem):
    """GLMeshItem with modified painter."""

    def paint(self):
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)
        super().paint()
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(0.0, 0.0)


class GLGridPlane(GLGraphicsItem):
    """Grid plane in 3D space."""

    def __init__(self, parentItem=None, **kwargs):
        super().__init__(parentItem=parentItem)
        self.axis = 0
        self.offset = 0.0
        self.coords = (0, 1), (0, 1)
        self.limits = (-0.05, 1.05), (-0.05, 1.05)
        white_bg = getConfigOption('background') == 'w'
        self.face_color = (0.95, 0.95, 0.95, 1) if white_bg else (0.05, 0.05, 0.05, 1)
        self.line_color = (0.7, 0.7, 0.7, 1) if white_bg else (0.3, 0.3, 0.3, 1)
        self.line_antialias = False
        self.line_width = 1.0
        self.azimuth_range: tuple | None = None
        self.elevation_range: tuple | None = None

        self._mesh = GLPolygonOffsetMeshItem(parentItem=self, computeNormals=False)

        self._lineplot = GLLinePlotItem(parentItem=self, mode='lines', glOptions='translucent')
        self._lineplot.setDepthValue(self.depthValue() + 1)

        self.setParentItem(parentItem)
        self.setData(**kwargs)

    def setData(self, **kwargs):
        """Update the grid plane

        ====================  ==================================================
        **Arguments:**
        ------------------------------------------------------------------------
        axis                  int indicating the grid plan axis 0: x, 1: y, 2: z
        offset                the offset along the axis orthogonal to the plane
        coords                tuples with the coordinates for the first and
                              second axis and the grid
        limits                tuples with the limits for the first and
                              second axis and the grid
        face_color            RGBA tuple with floats (0.0-1.0) for the grid
                              plane surface color
        line_color            RGBA tuple with floats (0.0-1.0) for the grid
                              lines color
        line_antialias        boolean indicating if grid lines are antialiased
        line_width            float indicating the grid lines width
        azimuth_range         tuple (min, max) or list of tuples for visibility
        elevation_range       tuple (min, max) or list of tuples for visibility
        ====================  ==================================================
        """
        args = ('axis', 'offset', 'coords', 'limits', 'face_color',
                'line_color', 'line_antialias', 'line_width', 'azimuth_range',
                'elevation_range')
        for arg in args:
            if arg in kwargs:
                setattr(self, arg, kwargs[arg])
        self._mesh.setMeshData(
            **dict(self._backplane_face()), color=self.face_color
        )
        self._lineplot.setData(
            pos=self._line_segments_positions(),
            color=self.line_color,
            antialias=self.line_antialias,
            width=self.line_width,
        )
        self.update()

    def is_visible(self, azimuth, elevation):
        """Check if plane should be visible based on camera angles."""
        return check_visibility(
            self.azimuth_range, azimuth, self.elevation_range, elevation
        )

    def _grid_positions(self):
        """Create grid positions in the specified plane."""
        axes = other_axes(self.axis)

        for idx, coord in enumerate(self.coords):
            for c in coord:
                p1, p2 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                p1[self.axis] = p2[self.axis] = self.offset
                p1[axes[idx]] = p2[axes[idx]] = c
                p1[axes[1 - idx]], p2[axes[1 - idx]] = self.limits[1 - idx]
                yield np.array([p1, p2])

    def _line_segments_positions(self):
        if lines := list(self._grid_positions()):
            return np.vstack(lines).astype(np.float32)
        return np.empty((0, 3), dtype=np.float32)

    def _backplane_face(self):
        """Create a backplane face in the specified plane."""
        axes = other_axes(self.axis)
        lim1, lim2 = self.limits

        vertices = np.zeros((4, 3))
        vertices[:, self.axis] = self.offset
        vertices[:, axes[0]] = [lim1[0], lim1[1], lim1[1], lim1[0]]
        vertices[:, axes[1]] = [lim2[0], lim2[0], lim2[1], lim2[1]]

        yield 'vertexes', vertices
        yield 'faces', np.array([[0, 1, 2], [0, 2, 3]])


class InconsistentCoordsError(Exception):
    """Raised when coords and coords labels do not have the same length."""


class GLAxis(GLGraphicsItem):
    """Axis with ticks and labels in 3D space."""

    sides = (
        QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
        QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
    )

    def __init__(self, parentItem=None, **kwargs):
        super().__init__(parentItem=parentItem)
        self.coords = (0, 1)
        self.coords_labels = tuple(f'{coord:.1f}' for coord in self.coords)
        self.limits = (-0.05, 1.05)
        self.other_limits = (-0.05, 1.05), (-0.05, 1.05)
        self.axis = 0
        self.faces = (-1, -1)
        self.tick_axis = 1
        self.label_side = 0
        self.tick_offset_factor = 0.02
        self.font = QtGui.QFont('Helvetica', 10)
        black_fg = getConfigOption('foreground') == 'k'
        self.label_color = (0, 0, 0, 1) if black_fg else (0.86, 0.86, 0.86, 1)
        self.line_color = (0, 0, 0, 1) if black_fg else (1, 1, 1, 1)
        self.line_antialias = False
        self.line_width = 1.0
        self.azimuth_range: tuple | None = None
        self.elevates: bool = True

        self._is_bottom = True
        self._labels = []
        self._lineplot = GLLinePlotItem(parentItem=self, mode='lines', glOptions='translucent')
        self._lineplot.setDepthValue(self.depthValue() + 1)

        self.setData(**kwargs)

    def setData(self, **kwargs):
        """Update the axis labels

        ====================  ==================================================
        **Arguments:**
        ------------------------------------------------------------------------
        coords                tuple with the coordinates for the axis ticks
        coords_labels         tuple with the axis tick labels
        limits                tuple with the axis limits
        other_limits          tuples with the limits for the other two axes
        axis                  int indicating the axis 0: x, 1: y, 2
        faces                 tuple with the faces for the other two axes
        tick_axis             int indicating which axis the ticks point to
        label_side            int indicating the side of the labels 0: low, 1:
                              high
        tick_offset_factor    float for tick label offset scaling factor
        font                  QFont for the labels
        label_color           RGBA tuple with floats (0.0-1.0) for the labels
                              and axis line color
        line_color            RGBA tuple with floats (0.0-1.0) for the axis line
                              color
        line_antialias        boolean indicating if axis line is antialiased
        line_width            float indicating the axis line width
        azimuth_range         tuple (min, max) or list of tuples for visibility
        elevates              boolean indicating if axis moves up/down with
                              elevation
        ====================  ==================================================
        """
        args = ('coords', 'coords_labels', 'limits', 'other_limits', 'axis',
                'faces', 'tick_axis', 'label_side', 'tick_offset_factor',
                'font', 'label_color', 'line_color', 'line_antialias',
                'line_width', 'azimuth_range', 'elevates')
        for arg in args:
            if arg in kwargs:
                setattr(self, arg, kwargs[arg])
        self.update_labels()
        self._lineplot.setData(
            pos=self._build_line_segments(),
            color=self.line_color,
            antialias=self.line_antialias,
            width=self.line_width,
        )
        self.update()

    def is_visible(self, azimuth, elevation):
        """Check if axis should be visible based on camera angles."""
        return check_visibility(self.azimuth_range, azimuth)

    def move_up(self):
        if self._is_bottom:
            self._move_axis_z(self.other_limits[1][1])
            self._is_bottom = False

    def move_down(self):
        if not self._is_bottom:
            self._move_axis_z(self.other_limits[1][0])
            self._is_bottom = True

    def elevate(self, elevation):
        if not self.elevates:
            return
        (self.move_up if elevation < 0 else self.move_down)()

    def tick_offset(self):
        a0, a1 = self.limits
        return self.tick_offset_factor * (
            (a1 - a0) + sum(hi - lo for lo, hi in self.other_limits)
        )

    def axis_coordinates(self, coord):
        pos = np.zeros(3, dtype=float)
        pos[self.axis] = coord
        for fixed_axis, face, (lo, hi) in zip(
            other_axes(self.axis), self.faces, self.other_limits
        ):
            pos[fixed_axis] = lo if face < 0 else hi
        return pos

    def tick_delta(self):
        delta = np.zeros(3)
        idx = other_axes(self.axis).index(self.tick_axis)
        delta[self.tick_axis] = self.faces[idx]
        return delta

    def tick_coordinates(self, coord):
        base = self.axis_coordinates(coord)
        return np.vstack([base, base + self.tick_offset()*self.tick_delta()])

    def update_labels(self):
        """Update existing labels or create new ones as needed."""
        if len(self.coords) != len(self.coords_labels):
            raise InconsistentCoordsError("coords and coords_labels must have the same length.")
        alignment = self.sides[self.label_side]
        color = tuple(round(c * 255) for c in self.label_color)
        for index, (coord, label) in enumerate(zip(self.coords, self.coords_labels)):
            pos = self.tick_coordinates(coord)[1]
            if index < len(self._labels):
                self._labels[index].setData(
                    pos=pos,
                    text=label,
                    color=color,
                    alignment=alignment
                )
            else:
                self._labels.append(GLTextItem(
                    parentItem=self,
                    pos=pos,
                    text=label,
                    color=color,
                    font=self.font,
                    alignment=alignment,
                ))

        num_needed = len(self.coords)
        for label in self._labels[num_needed:]:
            label.setParentItem(None)
        self._labels = self._labels[:num_needed]

    def _build_line_segments(self, z=None):
        if segments := list(self._yield_line_segments(z)):
            return np.vstack(segments).astype(np.float32)
        return np.empty((0, 3), np.float32)

    def _yield_line_segments(self, z=None):
        for coord in self.coords:
            segment = self.tick_coordinates(coord)
            segment[1] = 0.5 * (segment[0] + segment[1])
            if z is not None:
                segment[:, 2] = z
            yield segment

        axis = np.vstack([self.axis_coordinates(x) for x in self.limits])
        if z is not None:
            axis[:, 2] = z
        yield axis

    def _move_axis_z(self, z):
        for label in self._labels:
            x, y, _ = label.pos
            label.setData(pos=(x, y, z))
        self._lineplot.setData(pos=self._build_line_segments(z))


class GLGridAxis(GLGraphicsItem):
    """Draw a grid with axes, ticks and labels in 3D space."""

    grid_plane_config = (
        (0, 0, (270.0, 450.0), None),
        (0, 1, (90.0, 270.0), None),
        (1, 0, (0.0, 180.0), None),
        (1, 1, (180.0, 360.0), None),
        (2, 0, None, (0.0, 90.0)),
        (2, 1, None, (-90.0, 0.0)),
    )
    axis_config = (
        (0, (-1, -1), 1, 0, (180.0, 270.0), True),
        (0, (-1, -1), 1, 1, (270.0, 360.0), True),
        (0, (+1, -1), 1, 1, (90.0, 180.0), True),
        (0, (+1, -1), 1, 0, (0.0, 90.0), True),
        (1, (-1, -1), 0, 1, (180.0, 270.0), True),
        (1, (-1, -1), 0, 0, (90.0, 180.0), True),
        (1, (+1, -1), 0, 0, (270.0, 360.0), True),
        (1, (+1, -1), 0, 1, (0.0, 90.0), True),
        (2, (-1, -1), 1, 0, (135.0, 180.0), False),
        (2, (-1, +1), 0, 0, (45.0, 90.0), False),
        (2, (+1, -1), 0, 0, (215.0, 270.0), False),
        (2, (+1, +1), 1, 0, (315.0, 360.0), False),
        (2, (+1, +1), 0, 1, (90.0, 135.0), False),
        (2, (+1, -1), 1, 1, (0.0, 45.0), False),
        (2, (-1, +1), 1, 1, (180.0, 215.0), False),
        (2, (-1, -1), 0, 1, (270.0, 315.0), False),
    )

    def __init__(self, parentItem=None, **kwargs):
        super().__init__(parentItem=parentItem)
        self.coords = {axis: [-1.0, 0.0, 1.0] for axis in 'xyz'}
        self.coords_labels = {key: [f'{x:.1f}' for x in value] for key, value in self.coords.items()}
        self.limits = {axis: [-1.05, 1.05] for axis in 'xyz'}
        self._last_view = [0.0, 0.0]
        self._grid = [
            GLGridPlane(
                parentItem=self,
                axis=axis,
                azimuth_range=azimuth_range,
                elevation_range=elevation_range,
            )
            for axis, _, azimuth_range, elevation_range in self.grid_plane_config
        ]
        self._axes = [
            GLAxis(
                parentItem=self,
                axis=axis,
                faces=faces,
                tick_axis=tick_axis,
                label_side=label_side,
                azimuth_range=azimuth_range,
                elevates=elevates,
            )
            for axis, faces, tick_axis, label_side, azimuth_range, elevates in self.axis_config
        ]
        self.setData(**kwargs)

    def setData(self, **kwargs):
        """Update the grid axis

        ====================  ==================================================
        **Arguments:**
        ------------------------------------------------------------------------
        coords                dict with the coordinates for the 'x', 'y', 'z'
        coords_labels         dict with the axis tick labels for 'x', 'y', 'z'
        limits                dict with the limits for the 'x', 'y', 'z'
        ====================  ==================================================
        """
        args = ('coords', 'coords_labels', 'limits')
        for arg in args:
            if arg in kwargs:
                setattr(self, arg, kwargs.pop(arg))
        for grid, config in zip(self._grid, self.grid_plane_config):
            axis, offset_side = config[:2]
            coord1, coord2 = ('xyz'[i] for i in other_axes(axis))
            grid.setData(
                offset=self.limits['xyz'[axis]][offset_side],
                coords=[self.coords[coord1], self.coords[coord2]],
                limits=[self.limits[coord1], self.limits[coord2]],
                **kwargs
            )
        for axis, config in zip(self._axes, self.axis_config):
            axis_int = config[0]
            coord1, coord2 = ('xyz'[i] for i in other_axes(axis_int))
            axis.setData(
                coords=self.coords['xyz'[axis_int]],
                coords_labels=self.coords_labels['xyz'[axis_int]],
                limits=self.limits['xyz'[axis_int]],
                other_limits=[self.limits[coord1], self.limits[coord2]],
                **kwargs
            )
        self.update()

    def bounding_box_corners(self):
        xlim, ylim, zlim = self.limits['x'], self.limits['y'], self.limits['z']
        return (
            np.array([xlim[0], ylim[0], zlim[0]]),
            np.array([xlim[1], ylim[1], zlim[1]]),
        )

    def best_camera(self, distance_factor=1.5, method='perspective'):
        field_of_view = 60.0
        if method == 'orthographic':
            field_of_view = 1.0
            distance_factor = 1.4
        bbox_min, bbox_max = self.bounding_box_corners()
        center = (bbox_min + bbox_max) / 2.0
        new_pos = Vector(*center)
        bounding_box_diagonal = np.linalg.norm(bbox_max - bbox_min)
        fov_rad = np.radians(field_of_view)
        camera_distance = (bounding_box_diagonal / 2.0) / np.tan(fov_rad / 2.0) * distance_factor
        return {'pos': new_pos, 'distance': camera_distance}

    def view_angle(self):
        """Get the current view angle."""
        if not self.view():
            return 0.0, 0.0
        camera_params = self.view().cameraParams()
        azimuth, elevation = camera_params['azimuth'], camera_params['elevation']
        azimuth = np.mod(azimuth, 360.0)
        return azimuth, elevation

    def paint(self):
        super().paint()

        azimuth, elevation = self.view_angle()
        if self._last_view == [azimuth, elevation]:
            return
        self._last_view = [azimuth, elevation]

        for grid in self._grid:
            grid.hide()
            if grid.is_visible(azimuth, elevation):
                grid.show()

        for axis in self._axes:
            axis.hide()
            if axis.is_visible(azimuth, elevation):
                axis.show()
                axis.elevate(elevation)
