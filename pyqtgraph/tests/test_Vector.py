# -*- coding: utf-8 -*-
import pytest
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


def test_Vector_init():
    """Test construction of Vector objects from a variety of source types."""
    # separate values without z
    v = pg.Vector(0, 1)
    assert v.z() == 0

    v = pg.Vector(0.0, 1.0)
    assert v.z() == 0

    # separate values with 3 args
    v = pg.Vector(0, 1, 2)
    v = pg.Vector(0.0, 1.0, 2.0)

    # all in a list
    v = pg.Vector([0, 1])
    assert v.z() == 0
    v = pg.Vector([0, 1, 2])

    # QSizeF
    v = pg.Vector(QtCore.QSizeF(1, 2))
    assert v.x() == 1
    assert v.z() == 0

    # QPoint
    v = pg.Vector(QtCore.QPoint(0, 1))
    assert v.z() == 0
    v = pg.Vector(QtCore.QPointF(0, 1))
    assert v.z() == 0

    # QVector3D
    qv = QtGui.QVector3D(1, 2, 3)
    v = pg.Vector(qv)
    assert v == qv

    with pytest.raises(Exception):
        v = pg.Vector(1, 2, 3, 4)


def test_Vector_interface():
    """Test various aspects of the Vector API."""
    v = pg.Vector(-1, 2)

    # len
    assert len(v) == 3

    # indexing
    assert v[0] == -1
    assert v[2] == 0
    with pytest.raises(IndexError):
        x = v[4]

    assert v[1] == 2
    v[1] = 5
    assert v[1] == 5

    # iteration
    v2 = pg.Vector(*v)
    assert v2 == v

    assert abs(v).x() == 1

    # angle
    v1 = pg.Vector(1, 0)
    v2 = pg.Vector(1, 1)
    assert abs(v1.angle(v2) - 45) < 0.001
