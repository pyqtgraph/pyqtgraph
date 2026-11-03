"""
Point.py -  Extension of QPointF which adds a few missing methods.
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from math import atan2, degrees, hypot
from typing import Any, overload

import numpy as np

from .Qt import QtCore

type point_like = (
    QtCore.QPointF
    | QtCore.QPoint
    | QtCore.QSize
    | QtCore.QSizeF
    | Sequence[float]  # (x, y)
    | np.ndarray[Any, Any]  # [x, y]
    | float  # same value for x and y
)
"""Single argument accepted by :class:`Point() <pyqtgraph.Point>`"""


class Point(QtCore.QPointF):
    """Extension of QPointF which adds a few missing methods."""

    __slots__ = ()

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, x: float, y: float, /) -> None: ...
    @overload
    def __init__(self, xy: point_like, /) -> None: ...
    def __init__(self, *args: Any) -> None:
        if len(args) == 1:
            if isinstance(args[0], (QtCore.QSize, QtCore.QSizeF)):
                super().__init__(float(args[0].width()), float(args[0].height()))
                return
            elif isinstance(args[0], (int, float)):
                super().__init__(float(args[0]), float(args[0]))
                return
            elif hasattr(args[0], '__getitem__'):
                super().__init__(float(args[0][0]), float(args[0][1]))
                return
        elif len(args) == 2:
            super().__init__(args[0], args[1])
            return
        super().__init__(*args)

    def __len__(self) -> int:
        return 2

    # PySide6 stubs declare QPointF.__reduce__ as returning str
    def __reduce__(self) -> tuple[type[Point], tuple[float, float]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return (Point, (self.x(), self.y()))

    def __getitem__(self, i: int) -> float:
        if i == 0:
            return self.x()
        elif i == 1:
            return self.y()
        else:
            raise IndexError("Point has no index %s" % str(i))

    def __iter__(self) -> Iterator[float]:
        yield(self.x())
        yield(self.y())

    def __setitem__(self, i: int, x: float) -> None:
        if i == 0:
            return self.setX(x)
        elif i == 1:
            return self.setY(x)
        else:
            raise IndexError("Point has no index %s" % str(i))

    def __radd__(self, a: point_like) -> Point:
        return self._math_('__radd__', a)

    def __add__(self, a: point_like) -> Point:
        return self._math_('__add__', a)

    def __rsub__(self, a: point_like) -> Point:
        return self._math_('__rsub__', a)

    def __sub__(self, a: point_like) -> Point:
        return self._math_('__sub__', a)

    def __rmul__(self, a: point_like) -> Point:
        return self._math_('__rmul__', a)

    def __mul__(self, a: point_like) -> Point:
        return self._math_('__mul__', a)

    def __rdiv__(self, a: point_like) -> Point:
        return self._math_('__rdiv__', a)

    def __div__(self, a: point_like) -> Point:
        return self._math_('__div__', a)

    def __truediv__(self, a: point_like) -> Point:
        return self._math_('__truediv__', a)

    def __rtruediv__(self, a: point_like) -> Point:
        return self._math_('__rtruediv__', a)

    def __rpow__(self, a: point_like) -> Point:
        return self._math_('__rpow__', a)

    def __pow__(self, a: point_like) -> Point:
        return self._math_('__pow__', a)

    def _math_(self, op: str, x: point_like) -> Point:
        if not isinstance(x, QtCore.QPointF):
            x = Point(x)
        return Point(getattr(self.x(), op)(x.x()), getattr(self.y(), op)(x.y()))

    def length(self) -> float:
        """Returns the vector length of this Point."""
        return hypot(self.x(), self.y())  # length

    def norm(self) -> Point:
        """Returns a vector in the same direction with unit length."""
        return self / self.length()

    def angle(self, a: QtCore.QPointF | QtCore.QPoint, units: str = "degrees") -> float:
        """
        Returns the angle in degrees from the vector a to self.

        Parameters
        ----------
        a : Point, QPointF or QPoint
            The Point to return the angle with
        units : str, optional
            The units with which to compute the angle with, "degrees" or "radians",
            default "degrees"

        Returns
        -------
        float
            The angle between two vectors
        """
        rads = atan2(self.y(), self.x()) - atan2(a.y(), a.x())
        if units == "radians":
            return rads
        return degrees(rads)

    def dot(self, a: point_like) -> float:
        """Returns the dot product of a and this Point."""
        if not isinstance(a, QtCore.QPointF):
            a = Point(a)
        return Point.dotProduct(self, a)

    def cross(self, a: point_like) -> float:
        """Returns the cross product of a and this Point"""
        if not isinstance(a, QtCore.QPointF):
            a = Point(a)
        return self.x() * a.y() - self.y() * a.x()

    def proj(self, b: Point) -> Point:
        """Return the projection of this vector onto the vector b"""
        b1 = b.norm()
        return self.dot(b1) * b1

    def __repr__(self) -> str:
        return "Point(%f, %f)" % (self.x(), self.y())

    def min(self) -> float:
        return min(self.x(), self.y())

    def max(self) -> float:
        return max(self.x(), self.y())

    def copy(self) -> Point:
        return Point(self)

    def toQPoint(self) -> QtCore.QPoint:
        return self.toPoint()
