import numpy as np
import pytest
from numpy.testing import assert_array_equal

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

ROWS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]
VALUES = [float(x) for row in ROWS for x in row]


def translation():
    tr = pg.Transform3D()
    tr.translate(1, 2, 3)
    return tr


def test_Transform3D_init_identity():
    assert pg.Transform3D().isIdentity()


def test_Transform3D_init_values():
    assert_array_equal(pg.Transform3D(*VALUES).matrix(), ROWS)


@pytest.mark.parametrize("rows", [
    ROWS,
    tuple(tuple(row) for row in ROWS),
    np.array(ROWS),
    np.array(ROWS, dtype=float),
])
def test_Transform3D_init_rows(rows):
    assert_array_equal(pg.Transform3D(rows).matrix(), ROWS)


def test_Transform3D_init_flat_sequence():
    assert_array_equal(pg.Transform3D(range(16)).matrix(), np.arange(16).reshape(4, 4))


def test_Transform3D_init_wrong_size():
    with pytest.raises(TypeError):
        pg.Transform3D([[1, 2], [3, 4]])


@pytest.mark.parametrize("matrix", [
    QtGui.QMatrix4x4(*VALUES),
    pg.Transform3D(ROWS),
])
def test_Transform3D_init_matrix(matrix):
    assert_array_equal(pg.Transform3D(matrix).matrix(), ROWS)


def test_Transform3D_init_qtransform():
    tr = pg.Transform3D(QtGui.QTransform.fromTranslate(1, 2))
    assert tr.map(QtGui.QVector3D(0, 0, 0)) == QtGui.QVector3D(1, 2, 0)


def test_Transform3D_matrix_2d():
    tr = pg.Transform3D(ROWS)
    assert_array_equal(tr.matrix(nd=2), [[1, 2, 4], [5, 6, 8], [13, 14, 16]])
    with pytest.raises(Exception):
        tr.matrix(nd=4)


@pytest.mark.parametrize("obj, expected", [
    (QtCore.QPoint(1, 1), QtCore.QPoint(2, 3)),
    (QtCore.QPointF(1, 1), QtCore.QPointF(2, 3)),
    (pg.Point(1, 1), pg.Point(2, 3)),
    (QtGui.QVector3D(1, 1, 1), QtGui.QVector3D(2, 3, 4)),
    (pg.Vector(1, 1, 1), pg.Vector(2, 3, 4)),
    (QtGui.QVector4D(1, 1, 1, 1), QtGui.QVector4D(2, 3, 4, 1)),
    ([1, 1], [2, 3]),
    ([1, 1, 1], [2, 3, 4]),
    ((1, 1), (2, 3)),
    ((1, 1, 1), (2, 3, 4)),
])
def test_Transform3D_map(obj, expected):
    mapped = translation().map(obj)
    assert type(mapped) is type(obj)
    assert mapped == expected


@pytest.mark.parametrize("coords, expected", [
    (np.array([1, 1]), [2, 3]),
    (np.array([1.0, 1.0, 1.0]), [2, 3, 4]),
    (np.array([[0, 1], [0, 1]]), [[1, 2], [2, 3]]),
    (np.array([[0, 1], [0, 1], [0, 1]]), [[1, 2], [2, 3], [3, 4]]),
])
def test_Transform3D_map_array(coords, expected):
    assert_array_equal(translation().map(coords), expected)


def test_Transform3D_map_array_wrong_shape():
    with pytest.raises(TypeError):
        translation().map(np.zeros((4, 2)))


def test_Transform3D_inverted():
    tr = translation()
    inv, invertible = tr.inverted()
    assert invertible
    assert isinstance(inv, pg.Transform3D)
    assert (tr * inv).isIdentity()


def test_Transform3D_inverted_singular():
    inv, invertible = pg.Transform3D(np.zeros((4, 4))).inverted()
    assert not invertible
    assert isinstance(inv, pg.Transform3D)
    assert inv.isIdentity()
