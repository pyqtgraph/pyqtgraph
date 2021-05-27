# -*- coding: utf-8 -*-
from collections import OrderedDict
from copy import deepcopy
from contextlib import suppress
from pyqtgraph.functions import arrayToQPath, eq
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pyqtgraph as pg

np.random.seed(12345)


def testSolve3D():
    p1 = np.array([[0,0,0,1],
                   [1,0,0,1],
                   [0,1,0,1],
                   [0,0,1,1]], dtype=float)
    
    # transform points through random matrix
    tr = np.random.normal(size=(4, 4))
    tr[3] = (0,0,0,1)
    p2 = np.dot(tr, p1.T).T[:,:3]
    
    # solve to see if we can recover the transformation matrix.
    tr2 = pg.solve3DTransform(p1, p2)
    
    assert_array_almost_equal(tr[:3], tr2[:3])


def test_interpolateArray_order0():
    check_interpolateArray(order=0)


def test_interpolateArray_order1():
    check_interpolateArray(order=1)


def check_interpolateArray(order):
    def interpolateArray(data, x):
        result = pg.interpolateArray(data, x, order=order)
        assert result.shape == x.shape[:-1] + data.shape[x.shape[-1]:]
        return result
    
    data = np.array([[ 1.,   2.,   4.  ],
                     [ 10.,  20.,  40. ],
                     [ 100., 200., 400.]])
    
    # test various x shapes
    interpolateArray(data, np.ones((1,)))
    interpolateArray(data, np.ones((2,)))
    interpolateArray(data, np.ones((1, 1)))
    interpolateArray(data, np.ones((1, 2)))
    interpolateArray(data, np.ones((5, 1)))
    interpolateArray(data, np.ones((5, 2)))
    interpolateArray(data, np.ones((5, 5, 1)))
    interpolateArray(data, np.ones((5, 5, 2)))
    with pytest.raises(TypeError):
        interpolateArray(data, np.ones((3,)))
    with pytest.raises(TypeError):
        interpolateArray(data, np.ones((1, 3,)))
    with pytest.raises(TypeError):
        interpolateArray(data, np.ones((5, 5, 3,)))
    
    x = np.array([[  0.3,   0.6],
                  [  1. ,   1. ],
                  [  0.501,   1. ],   # NOTE: testing at exactly 0.5 can yield different results from map_coordinates
                  [  0.501,   2.501],  # due to differences in rounding
                  [ 10. ,  10. ]])
    
    result = interpolateArray(data, x)
    # make sure results match ndimage.map_coordinates
    import scipy.ndimage
    spresult = scipy.ndimage.map_coordinates(data, x.T, order=order)
    #spresult = np.array([  5.92,  20.  ,  11.  ,   0.  ,   0.  ])  # generated with the above line
    
    assert_array_almost_equal(result, spresult)
    
    # test mapping when x.shape[-1] < data.ndim
    x = np.array([[  0.3,   0],
                  [  0.3,   1],
                  [  0.3,   2]])
    r1 = interpolateArray(data, x)
    x = np.array([0.3])  # should broadcast across axis 1
    r2 = interpolateArray(data, x)
    
    assert_array_almost_equal(r1, r2)
    
    
    # test mapping 2D array of locations
    x = np.array([[[0.501, 0.501], [0.501, 1.0], [0.501, 1.501]],
                  [[1.501, 0.501], [1.501, 1.0], [1.501, 1.501]]])
    
    r1 = interpolateArray(data, x)
    r2 = scipy.ndimage.map_coordinates(data, x.transpose(2,0,1), order=order)
    #r2 = np.array([[   8.25,   11.  ,   16.5 ],  # generated with the above line
                   #[  82.5 ,  110.  ,  165.  ]])

    assert_array_almost_equal(r1, r2)


def test_subArray():
    a = np.array([0, 0, 111, 112, 113, 0, 121, 122, 123, 0, 0, 0, 211, 212, 213, 0, 221, 222, 223, 0, 0, 0, 0])
    b = pg.subArray(a, offset=2, shape=(2,2,3), stride=(10,4,1))
    c = np.array([[[111,112,113], [121,122,123]], [[211,212,213], [221,222,223]]])
    assert np.all(b == c)
    
    # operate over first axis; broadcast over the rest
    aa = np.vstack([a, a/100.]).T
    cc = np.empty(c.shape + (2,))
    cc[..., 0] = c
    cc[..., 1] = c / 100.
    bb = pg.subArray(aa, offset=2, shape=(2,2,3), stride=(10,4,1))
    assert np.all(bb == cc)
    
    
def test_rescaleData():
    dtypes = map(np.dtype, ('ubyte', 'uint16', 'byte', 'int16', 'int', 'float'))
    for dtype1 in dtypes:
        for dtype2 in dtypes:
            data = (np.random.random(size=10) * 2**32 - 2**31).astype(dtype1)
            for scale, offset in [(10, 0), (10., 0.), (1, -50), (0.2, 0.5), (0.001, 0)]:
                if dtype2.kind in 'iu':
                    lim = np.iinfo(dtype2)
                    lim = lim.min, lim.max
                else:
                    lim = (-np.inf, np.inf)
                s1 = np.clip(float(scale) * (data-float(offset)), *lim).astype(dtype2)
                s2 = pg.rescaleData(data, scale, offset, dtype2)
                assert s1.dtype == s2.dtype
                if dtype2.kind in 'iu':
                    assert np.all(s1 == s2)
                else:
                    assert np.allclose(s1, s2)


def test_eq():
    eq = pg.functions.eq
    
    zeros = [0, 0.0, np.float64(0), np.float32(0), np.int32(0), np.int64(0)]
    for i,x in enumerate(zeros):
        for y in zeros[i:]:
            assert eq(x, y)
            assert eq(y, x)
    
    assert eq(np.nan, np.nan)
    
    # test 
    class NotEq(object):
        def __eq__(self, x):
            return False
        
    noteq = NotEq()
    assert eq(noteq, noteq) # passes because they are the same object
    assert not eq(noteq, NotEq())


    # Should be able to test for equivalence even if the test raises certain
    # exceptions
    class NoEq(object):
        def __init__(self, err):
            self.err = err
        def __eq__(self, x):
            raise self.err
        
    noeq1 = NoEq(AttributeError())
    noeq2 = NoEq(ValueError())
    noeq3 = NoEq(Exception())
    
    assert eq(noeq1, noeq1)
    assert not eq(noeq1, noeq2)
    assert not eq(noeq2, noeq1)
    with pytest.raises(Exception):
        eq(noeq3, noeq2)

    # test array equivalence
    # note that numpy has a weird behavior here--np.all() always returns True
    # if one of the arrays has size=0; eq() will only return True if both arrays
    # have the same shape.
    a1 = np.zeros((10, 20)).astype('float')
    a2 = a1 + 1
    a3 = a2.astype('int')
    a4 = np.empty((0, 20))
    assert not eq(a1, a2)  # same shape/dtype, different values
    assert not eq(a1, a3)  # same shape, different dtype and values
    assert not eq(a1, a4)  # different shape (note: np.all gives True if one array has size 0)

    assert not eq(a2, a3)  # same values, but different dtype
    assert not eq(a2, a4)  # different shape
    
    assert not eq(a3, a4)  # different shape and dtype
    
    assert eq(a4, a4.copy())
    assert not eq(a4, a4.T)

    # test containers

    assert not eq({'a': 1}, {'a': 1, 'b': 2})
    assert not eq({'a': 1}, {'a': 2})
    d1 = {'x': 1, 'y': np.nan, 3: ['a', np.nan, a3, 7, 2.3], 4: a4}
    d2 = deepcopy(d1)
    assert eq(d1, d2)
    d1_ordered = OrderedDict(d1)
    d2_ordered = deepcopy(d1_ordered)
    assert eq(d1_ordered, d2_ordered)
    assert not eq(d1_ordered, d2)
    items = list(d1.items())
    assert not eq(OrderedDict(items), OrderedDict(reversed(items)))
    
    assert not eq([1,2,3], [1,2,3,4])
    l1 = [d1, np.inf, -np.inf, np.nan]
    l2 = deepcopy(l1)
    t1 = tuple(l1)
    t2 = tuple(l2)
    assert eq(l1, l2)
    assert eq(t1, t2)

    assert eq(set(range(10)), set(range(10)))
    assert not eq(set(range(10)), set(range(9)))


@pytest.mark.parametrize("s,suffix,expected", [
    # usual cases
    ("100 uV", "V", ("100", "u", "V")),
    ("100 µV", "V", ("100", "µ", "V")),
    ("4.2 nV", None, ("4.2", "n", "V")),
    ("1.2 m", "m", ("1.2", "", "m")),
    ("1.2 m", None, ("1.2", "", "m")),
    ("5.0e9", None, ("5.0e9", "", "")),
    ("2 units", "units", ("2", "", "units")),
    # siPrefix with explicit empty suffix
    ("1.2 m", "", ("1.2", "m", "")),
    ("5.0e-9 M", "", ("5.0e-9", "M", "")),
    # weirder cases that should return the reasonable thing
    ("4.2 nV", "nV", ("4.2", "", "nV")),
    ("4.2 nV", "", ("4.2", "n", "")),
    ("1.2 j", "", ("1.2", "", "")),
    ("1.2 j", None, ("1.2", "", "j")),
    # expected error cases
    ("100 uV", "v", ValueError),
])
def test_siParse(s, suffix, expected):
    if isinstance(expected, tuple):
        assert pg.siParse(s, suffix=suffix) == expected
    else:
        with pytest.raises(expected):
            pg.siParse(s, suffix=suffix)


QT_LIB = pg.Qt.QT_LIB
MoveToElement = pg.QtGui.QPainterPath.ElementType.MoveToElement
LineToElement = pg.QtGui.QPainterPath.ElementType.LineToElement
@pytest.mark.parametrize(
    "x, y, connect, expected", [
        (
            np.arange(6), np.arange(0, -6, step=-1), 'all', (
                (MoveToElement, 0.0, 0.0),
                (LineToElement, 1.0, -1.0),
                (LineToElement, 2.0, -2.0),
                (LineToElement, 3.0, -3.0),
                (LineToElement, 4.0, -4.0),
                (LineToElement, 5.0, -5.0),
            )
        ), 
        (
            np.arange(6), np.arange(0, -6, step=-1), 'pairs', (
                (MoveToElement, 0.0, 0.0),
                (LineToElement, 1.0, -1.0),
                (MoveToElement, 2.0, -2.0),
                (LineToElement, 3.0, -3.0),
                (MoveToElement, 4.0, -4.0),
                (LineToElement, 5.0, -5.0),
            )
        ),
        (
            np.arange(5), np.arange(0, -5, step=-1), 'pairs', (
                (MoveToElement, 0.0, 0.0),
                (LineToElement, 1.0, -1.0),
                (MoveToElement, 2.0, -2.0),
                (LineToElement, 3.0, -3.0),
                (MoveToElement, 4.0, -4.0)
            ) 
        ),
        (
            np.arange(5), np.array([0, -1, np.NaN, -3, -4]), 'finite', (
                (MoveToElement, 0.0, 0.0),
                (LineToElement, 1.0, -1.0),
                (LineToElement, 2.0, np.nan) if qt6 else (LineToElement, 1.0, -1.0),
                (MoveToElement, 3.0, -3.0),
                (LineToElement, 4.0, -4.0)
            ) 
        ),
        (
            np.array([0, 1, np.NaN, 3, 4]), np.arange(0, -5, step=-1), 'finite', (
                (MoveToElement, 0.0, 0.0),
                (LineToElement, 1.0, -1.0),
                (LineToElement, np.nan, -2.0) if qt6 else (LineToElement, 1.0, -1.0),
                (MoveToElement, 3.0, -3.0),
                (LineToElement, 4.0, -4.0)
            )
        ),
        (
            np.arange(5), np.arange(0, -5, step=-1), np.array([0, 1, 0, 1, 0]), (
                (MoveToElement, 0.0, 0.0),
                (MoveToElement, 1.0, -1.0),
                (LineToElement, 2.0, -2.0),
                (MoveToElement, 3.0, -3.0),
                (LineToElement, 4.0, -4.0)
            )
        )
    ]
)
def test_arrayToQPath(xs, ys, connect, expected):
    path = arrayToQPath(xs, ys, connect=connect)
    for i in range(path.elementCount()):
        with suppress(NameError):
            # nan elements add two line-segments, for simplicity of test config
            # we can ignore the second segment
            if (eq(element.x, np.nan) or eq(element.y, np.nan)):
                continue
        element = path.elementAt(i)
        assert expected[i] == (element.type, element.x, element.y)

