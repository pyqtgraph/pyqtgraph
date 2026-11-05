"""
Vector.py -  Extension of QVector3D which adds a few missing methods.
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.
"""
from collections.abc import Iterator, Sequence
from math import acos, degrees
from typing import cast, overload

import numpy as np

from .Qt import QtCore, QtGui

type vector_like = (
    QtGui.QVector3D
    | QtCore.QPointF
    | QtCore.QPoint
    | QtCore.QSizeF
    | Sequence[float]  # (x, y) or (x, y, z)
    | np.ndarray[tuple[int], np.dtype[np.number]]  # [x, y] or [x, y, z]
)
"""Single argument accepted by :class:`Vector() <pyqtgraph.Vector>`"""


class Vector(QtGui.QVector3D):
    """Extension of QVector3D which adds a few helpful methods."""

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, x: float, y: float, /) -> None: ...
    @overload
    def __init__(self, x: float, y: float, z: float, /) -> None: ...
    @overload
    def __init__(self, xyz: vector_like, /) -> None: ...
    def __init__(self, *args: float | vector_like) -> None:
        """
        Handle additional constructions of a Vector.

        Parameters
        ----------
        *args : float or vector_like
            Any of:

            * 3 numbers (x, y and z)
            * 2 numbers (x and y, z is 0)
            * a sequence of 2 or 3 numbers (z is 0 if missing)
            * a QSizeF, QPoint or QPointF (z is 0)
            * a QVector3D
        """
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, QtCore.QSizeF):
                super().__init__(float(arg.width()), float(arg.height()), 0)
                return
            elif isinstance(arg, (QtCore.QPoint, QtCore.QPointF)):
                super().__init__(float(arg.x()), float(arg.y()), 0)
                return
            elif hasattr(arg, '__getitem__') and not isinstance(arg, QtGui.QVector3D):
                vals = list(cast("Sequence[float]", arg))
                if len(vals) == 2:
                    vals.append(0)
                if len(vals) != 3:
                    raise Exception('Cannot init Vector with sequence of length %d' % len(vals))
                super().__init__(vals[0], vals[1], vals[2])
                return
            elif isinstance(arg, QtGui.QVector3D):
                # PySide6 6.1 does not accept initialization from QVector3D
                super().__init__(arg.x(), arg.y(), arg.z())
                return
        elif len(args) == 2:
            x, y = cast("tuple[float, float]", args)
            super().__init__(x, y, 0)
            return
        super().__init__(*args)

    def __len__(self) -> int:
        return 3

    def __getitem__(self, i: int) -> float:
        if i == 0:
            return self.x()
        elif i == 1:
            return self.y()
        elif i == 2:
            return self.z()
        else:
            raise IndexError("Vector has no index %s" % str(i))

    def __setitem__(self, i: int, x: float) -> None:
        if i == 0:
            return self.setX(x)
        elif i == 1:
            return self.setY(x)
        elif i == 2:
            return self.setZ(x)
        else:
            raise IndexError("Vector has no index %s" % str(i))

    def __iter__(self) -> Iterator[float]:
        yield(self.x())
        yield(self.y())
        yield(self.z())

    def angle(self, a: vector_like) -> float | None:
        """
        Return the angle in degrees between this vector and the vector a.

        Parameters
        ----------
        a : vector_like
            The other vector.

        Returns
        -------
        float or None
            The angle in degrees, or None if one of the vectors has zero length.
        """
        if not isinstance(a, QtGui.QVector3D):
            a = Vector(a)
        n1 = self.length()
        n2 = a.length()
        if n1 == 0. or n2 == 0.:
            return None
        ## Probably this should be done with arctan2 instead..
        cos = QtGui.QVector3D.dotProduct(self, a) / (n1 * n2)
        rads = acos(min(max(cos, -1.0), 1.0)) ### in radians
#        c = self.crossProduct(a)
#        if c > 0:
#            ang *= -1.
        return degrees(rads)

    def __abs__(self) -> "Vector":
        return Vector(abs(self.x()), abs(self.y()), abs(self.z()))
