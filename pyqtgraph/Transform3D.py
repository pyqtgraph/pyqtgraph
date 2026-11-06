# pyright: reportImportCycles=false, reportUnnecessaryTypeIgnoreComment=false
from collections.abc import Sequence
from typing import cast, overload

import numpy as np

from . import functions as fn
from .Qt import QtCore, QtGui
from .Vector import Vector

type _QtPoint = QtCore.QPoint | QtCore.QPointF | QtGui.QVector3D | QtGui.QVector4D


class Transform3D(QtGui.QMatrix4x4):
    """
    Extension of QMatrix4x4 with some helpful methods added.
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        m11: float,
        m12: float,
        m13: float,
        m14: float,
        m21: float,
        m22: float,
        m23: float,
        m24: float,
        m31: float,
        m32: float,
        m33: float,
        m34: float,
        m41: float,
        m42: float,
        m43: float,
        m44: float,
        /,
    ) -> None: ...
    @overload
    def __init__(self, matrix: QtGui.QMatrix4x4 | QtGui.QTransform, /) -> None: ...
    @overload
    def __init__[RowT: Sequence[float]](
        self,
        rows: (
            list[RowT]
            | tuple[RowT, ...]
            | np.ndarray[tuple[int, int], np.dtype[np.integer | np.floating]]
        ),
        /,
    ) -> None: ...
    def __init__(
        self,
        *args: (
            float
            | QtGui.QMatrix4x4
            | QtGui.QTransform
            | Sequence[Sequence[float]]
            | np.ndarray[tuple[int, int], np.dtype[np.integer | np.floating]]
        ),
    ) -> None:
        values = args
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, (list, tuple, np.ndarray)):
                rows = cast("Sequence[Sequence[float]]", arg)
                values = [x for y in rows for x in y]
                if len(values) != 16:
                    raise TypeError("Single argument to Transform3D must have 16 elements.")
            elif isinstance(arg, QtGui.QMatrix4x4):
                # same values as copyDataTo(), which PySide6 leaves untyped
                values = arg.transposed().data()

        # PySide6, and PyQt6 before 6.12, leave part of __init__() untyped
        QtGui.QMatrix4x4.__init__(self, *values)  # pyright: ignore[reportUnknownMemberType]

    def matrix(self, nd: int = 3) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        if nd == 3:
            return np.array(self.transposed().data()).reshape(4,4)
        elif nd == 2:
            m = np.array(self.transposed().data()).reshape(4,4)
            m[2] = m[3]
            m[:,2] = m[:,3]
            return m[:3,:3]
        else:
            raise Exception("Argument 'nd' must be 2 or 3")

    @overload
    def map[QtPointT: _QtPoint](self, obj: QtPointT) -> QtPointT: ...
    @overload
    def map(self, obj: list[int] | list[float]) -> list[float]: ...
    @overload
    def map(self, obj: tuple[float, float]) -> tuple[float, float]: ...
    @overload
    def map(self, obj: tuple[float, float, float]) -> tuple[float, float, float]: ...
    @overload
    def map(
        self, obj: np.ndarray[tuple[int, ...], np.dtype[np.integer | np.floating]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]: ...
    # Qt names the argument point, and PySide6 also maps QPainterPath.Element
    def map(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        obj: (
            _QtPoint
            | Sequence[float]
            | np.ndarray[tuple[int, ...], np.dtype[np.integer | np.floating]]
        ),
    ) -> (
        QtCore.QPoint
        | QtCore.QPointF
        | QtGui.QVector3D
        | QtGui.QVector4D
        | list[float]
        | tuple[float, ...]
        | np.ndarray[tuple[int, ...], np.dtype[np.floating]]
    ):
        """
        Extends QMatrix4x4.map() to allow mapping (3, ...) arrays of coordinates
        """
        if isinstance(obj, np.ndarray) and obj.shape[0] in (2,3):
            if obj.ndim >= 2:
                return fn.transformCoordinates(self, obj)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            else:
                point = cast("np.ndarray[tuple[int], np.dtype[np.number]]", obj)
                v = QtGui.QMatrix4x4.map(self, Vector(point))
                return np.array([v.x(), v.y(), v.z()])[:obj.shape[0]]
        elif isinstance(obj, (list, tuple)):
            v = QtGui.QMatrix4x4.map(self, Vector(obj))
            return type(obj)([v.x(), v.y(), v.z()])[:len(obj)]
        else:
            # other arrays and sequences also get here, and Qt rejects them
            qt_obj = cast("_QtPoint", obj)
            retval = QtGui.QMatrix4x4.map(self, qt_obj)
            if not isinstance(retval, type(qt_obj)):
                return type(qt_obj)(retval)  # pyright: ignore[reportCallIssue, reportArgumentType]
            return retval

    def inverted(self) -> tuple["Transform3D", bool]:
        inv, b = QtGui.QMatrix4x4.inverted(self)
        return Transform3D(inv), b
