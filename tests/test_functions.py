from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pyqtgraph as pg
from pyqtgraph.functions import arrayToQPath, eq, SignalBlock
from pyqtgraph.Qt import QtCore, QtGui

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
    pytest.importorskip("scipy")

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


@pytest.mark.parametrize("order", [0, 1])
def test_interpolateArray_flat(order: int):
    # Inputing an array with a dimension of length 1 should still
    # produce a non-zero result
    data = np.ones((3, 1, 5))
    x = np.asarray([
        [[0.,  0.5], [0.,  1.5], [0.,  2.5]],
        [[1.,  0.5], [1.,  1.5],[1.,  2.5]],
        [[2.,  0.5],[2.,  1.5],[2.,  2.5]],
    ])
    result = pg.interpolateArray(data, x, order=order)
    assert np.any(result)


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
    rng = np.random.default_rng(12345)
    dtypes = map(np.dtype, ('ubyte', 'uint16', 'byte', 'int16', 'int', 'float'))
    for dtype1 in dtypes:
        for dtype2 in dtypes:
            if dtype1.kind in 'iu':
                lim = np.iinfo(dtype1)
                data = rng.integers(lim.min, lim.max, size=10, dtype=dtype1, endpoint=True)
            else:
                data = (rng.random(size=10) * 2**32 - 2**31).astype(dtype1)
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
    assert eq(noteq, noteq)  # passes because they are the same object
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

@pytest.mark.parametrize("s,suffix,power,expected", [
    # usual cases
    ("100 uV", "V", 1, 1e-4),
    ("100 µV", "V", 1, 1e-4),
    ("4.2 nV", None, 1, 4.2e-9),
    ("1.2 m", "m", 1, 1.2),
    # siPrefix with explicit empty suffix
    ("1.2 m", "", 1, 1.2e-3),
    ("5.0e-9 M", "", 1, 5.0e-3),
    # weirder cases that should return the reasonable thing    
    ("4.2 nV", "", 1, 4.2e-9),
    ("1.2 j", "", 1, 1.2),
    ("1.2 j", None, 1, 1.2),
    # cases with power != 1
    ("100 uV^2", "V^2", 2, 1e-10),
    ("4.2 nV^2", None, 3, 4.2e-27),
    ("100.2 um^(1/2)", "m^(1/2)", 0.5, 0.1002),
    ("100 km^2", "m^2", 2, 1e+8),
])
def test_siEval(s, suffix, power, expected):
    result = pg.siEval(s, suffix=suffix, unitPower=power)
    assert np.isclose(result, expected)

@pytest.mark.parametrize("s,suffix,expected", [
    ("1,2 j", "", ("1,2", "", "")),
    ("1,2 j", None, ("1,2", "", "j")),
    (",2 j", None, (",2", "", "j")),])
def test_siParse_with_comma_as_decimal_separator(s, suffix, expected):
    assert pg.siParse(s, suffix=suffix, regex=pg.functions.FLOAT_REGEX_COMMA) == expected

def test_CIELab_reconversion():
    color_list = [ pg.Qt.QtGui.QColor('#100235') ] # known problematic values
    for _ in range(20):
        qcol = pg.Qt.QtGui.QColor()
        qcol.setRgbF( *np.random.random((3)) )
        color_list.append(qcol)
    
    for qcol1 in color_list:
        vec_Lab  = pg.functions.colorCIELab( qcol1 )
        qcol2 = pg.functions.CIELabColor(*vec_Lab)
        for val1, val2 in zip( qcol1.getRgb(), qcol2.getRgb() ):
            assert abs(val1-val2)<=1, f'Excess CIELab reconversion error ({qcol1.name() } > {vec_Lab } > {qcol2.name()})'

MoveToElement = pg.QtGui.QPainterPath.ElementType.MoveToElement
LineToElement = pg.QtGui.QPainterPath.ElementType.LineToElement
_dtypes = []
for bits in 32, 64:
    for base in 'int', 'float', 'uint':
        _dtypes.append(f'{base}{bits}')
_dtypes.extend(['uint8', 'uint16'])

def _handle_underflow(dtype, *elements):
    """Wrapper around path description which converts underflow into proper points"""
    out = []
    dtype = np.dtype(dtype)
    # get the signed integer type of the same width
    dtype_int = np.dtype(f'i{dtype.itemsize}')
    for el in elements:
        newElement = [el[0]]
        for ii in range(1, 3):
            coord = el[ii]
            if dtype.kind == 'u' and coord < 0:
                # coord is a float with a negative integral value.
                # for unsigned integer types, we want negative values to
                # wrap-around. to get consistent wrap-around behavior
                # across different numpy versions and machine platforms,
                # we first convert coord to a signed integer.
                coord = np.array(coord, dtype=dtype_int).astype(dtype)
            newElement.append(float(coord))
        out.append(tuple(newElement))
    return out

@pytest.mark.parametrize(
    "xs, ys, connect, expected", [
        *(
            (
                np.arange(6, dtype=dtype), np.arange(0, -6, step=-1).astype(dtype), 'all',
                _handle_underflow(dtype,
                                  (MoveToElement, 0.0, 0.0),
                                  (LineToElement, 1.0, -1.0),
                                  (LineToElement, 2.0, -2.0),
                                  (LineToElement, 3.0, -3.0),
                                  (LineToElement, 4.0, -4.0),
                                  (LineToElement, 5.0, -5.0)
                                  )
            ) for dtype in _dtypes
        ),
        *(
            (
                np.arange(6, dtype=dtype), np.arange(0, -6, step=-1).astype(dtype), 'pairs',
                _handle_underflow(dtype,
                                  (MoveToElement, 0.0, 0.0),
                                  (LineToElement, 1.0, -1.0),
                                  (MoveToElement, 2.0, -2.0),
                                  (LineToElement, 3.0, -3.0),
                                  (MoveToElement, 4.0, -4.0),
                                  (LineToElement, 5.0, -5.0),
                                  )
            ) for dtype in _dtypes
        ),
        *(
            (
                np.arange(5, dtype=dtype), np.arange(0, -5, step=-1).astype(dtype), 'pairs',
                _handle_underflow(dtype,
                                  (MoveToElement, 0.0, 0.0),
                                  (LineToElement, 1.0, -1.0),
                                  (MoveToElement, 2.0, -2.0),
                                  (LineToElement, 3.0, -3.0),
                                  (MoveToElement, 4.0, -4.0)
                                  )
            ) for dtype in _dtypes
        ),
        # NaN types don't coerce to integers, don't test for all types since that doesn't make sense
        (
            np.arange(5), np.array([0, -1, np.nan, -3, -4]), 'finite', (
                (MoveToElement, 0.0, 0.0),
                (LineToElement, 1.0, -1.0),
                (LineToElement, 1.0, -1.0),
                (MoveToElement, 3.0, -3.0),
                (LineToElement, 4.0, -4.0)
            ) 
        ),
        (
            np.array([0, 1, np.nan, 3, 4]), np.arange(0, -5, step=-1), 'finite', (
                (MoveToElement, 0.0, 0.0),
                (LineToElement, 1.0, -1.0),
                (LineToElement, 1.0, -1.0),
                (MoveToElement, 3.0, -3.0),
                (LineToElement, 4.0, -4.0)
            )
        ),
        *(
            (
                np.arange(5, dtype=dtype), np.arange(0, -5, step=-1).astype(dtype), np.array([0, 1, 0, 1, 0]),
                _handle_underflow(dtype,
                                  (MoveToElement, 0.0, 0.0),
                                  (MoveToElement, 1.0, -1.0),
                                  (LineToElement, 2.0, -2.0),
                                  (MoveToElement, 3.0, -3.0),
                                  (LineToElement, 4.0, -4.0)
                                  )
            ) for dtype in _dtypes
        ),
        # Empty path with all types of connection
        *(
            (
                np.arange(0), np.arange(0, dtype=dtype), conn, ()
            ) for conn in ['all', 'pairs', 'finite', np.array([])] for dtype in _dtypes
        ),
    ]
)
def test_arrayToQPath(xs, ys, connect, expected):
    path = arrayToQPath(xs, ys, connect=connect)
    element = None
    for i in range(path.elementCount()):
        # nan elements add two line-segments, for simplicity of test config
        # we can ignore the second segment
        if element is not None and (eq(element.x, np.nan) or eq(element.y, np.nan)):
            continue
        element = path.elementAt(i)
        assert eq(expected[i], (element.type, element.x, element.y))


def test_ndarray_from_qpolygonf():
    # test that we get an empty ndarray from an empty QPolygonF
    poly = pg.functions.create_qpolygonf(0)
    arr = pg.functions.ndarray_from_qpolygonf(poly)
    assert isinstance(arr, np.ndarray)


def test_ndarray_from_qimage():
    # for QImages created w/o specifying bytesPerLine, Qt will pad
    # each line to a multiple of 4-bytes.
    # test that we can handle such QImages.
    h = 10

    fmt = QtGui.QImage.Format.Format_RGB888
    for w in [5, 6, 7, 8]:
        qimg = QtGui.QImage(w, h, fmt)
        qimg.fill(0)
        arr = pg.functions.ndarray_from_qimage(qimg)
        assert arr.shape == (h, w, 3)

    fmt = QtGui.QImage.Format.Format_Grayscale8
    for w in [5, 6, 7, 8]:
        qimg = QtGui.QImage(w, h, fmt)
        qimg.fill(0)
        arr = pg.functions.ndarray_from_qimage(qimg)
        assert arr.shape == (h, w)

def test_colorDistance():
    pg.colorDistance([pg.Qt.QtGui.QColor(0,0,0), pg.Qt.QtGui.QColor(255,0,0)])
    pg.colorDistance([])


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (["r"], [255, 0, 0, 255]),
        (["g"], [0, 255, 0, 255]),
        (["b"], [0, 0, 255, 255]),
        (["c"], [0, 255, 255, 255]),
        (["m"], [255, 0, 255, 255]),
        (["y"], [255, 255, 0, 255]),
        (["k"], [0, 0, 0, 255]),
        (["w"], [255, 255, 255, 255]),
        (["d"], [150, 150, 150, 255]),
        (["l"], [200, 200, 200, 255]),
        (["s"], [100, 100, 150, 255]),
        ([0.75], [191, 191, 191, 255]),
        ([11, 22, 33], [11, 22, 33, 255]),
        ([11, 22, 33, 44], [11, 22, 33, 44]),
        ([(11, 22, 33)], [11, 22, 33, 255]),
        ([(11, 22, 33, 44)], [11, 22, 33, 44]),
        ([0], [255, 0, 0, 255]),
        ([1], [255, 170, 0, 255]),
        ([2], [170, 255, 0, 255]),
        ([3], [0, 255, 0, 255]),
        ([4], [0, 255, 170, 255]),
        ([5], [0, 170, 255, 255]),
        ([9], [255, 0, 0, 255]),
        ([(0, 2)], [255, 0, 0, 255]),
        ([(1, 2)], [0, 255, 255, 255]),
        ([(2, 2)], [255, 0, 0, 255]),
        (["#89a"], [136, 153, 170, 255]),
        (["#89ab"], [136, 153, 170, 187]),
        (["#4488cc"], [68, 136, 204, 255]),
        (["#4488cc00"], [68, 136, 204, 0]),
        ([QtGui.QColor(1, 2, 3, 4)], [1, 2, 3, 4]),
        (["steelblue"], [70, 130, 180, 255]),
        (["lawngreen"], [124, 252, 0, 255]),
    ],
)
def test_mkColor(test_input, expected):
    qcol: QtGui.QColor = pg.functions.mkColor(*test_input)
    assert list(qcol.getRgb()) == expected

def test_signal_block_unconnected():
    """Test that SignalBlock does not end up connecting an unconnected slot"""
    class Sender(QtCore.QObject):
        signal = QtCore.Signal()

    class Receiver:
        def __init__(self):
            self.counter = 0

        def slot(self):
            self.counter += 1

    sender = Sender()
    receiver = Receiver()
    with SignalBlock(sender.signal, receiver.slot):
        pass
    sender.signal.emit()
    assert receiver.counter == 0

@pytest.mark.parametrize("x,precision,suffix,power,expected", [
    # usual cases
    (0, 3, 'V', 1, "0 V"),
    (1, 3, 'V', 1, "1 V"),
    (1.2, 3, 'V', 1, "1.2 V"),
    (1.23456, 3, 'V', 1, "1.23 V"),
    (1.23456, 4, 'V', 1, "1.235 V"),
    (12.3456, 3, 'V', 1, "12.3 V"),
    (123.456, 3, 'V', 1, "123 V"),
    (1234.56, 3, 'V', 1, "1.23 kV"),
    (12345.6, 3, 'V', 1, "12.3 kV"),
    (123456., 3, 'V', 1, "123 kV"),
    (1234567., 3, 'V', 1, "1.23 MV"),
    (12345678., 3, 'V', 1, "12.3 MV"),
    (123456789., 3, 'V', 1, "123 MV"),
    (1234567890., 3, 'V', 1, "1.23 GV"),
    (12345678900., 3, 'V', 1, "12.3 GV"),
    (123456789000., 3, 'V', 1, "123 GV"),
    (0.123456789, 3, 'V', 1, "123 mV"),
    (0.0123456789, 3, 'V', 1, "12.3 mV"),
    (0.00123456789, 3, 'V', 1, "1.23 mV"),
    # Different power
    (0, 3, 'V²', 2, "0 V²"),
    (123.456, 3, 'V²', 2, "123 V²"),
    (1234.56, 4, 'V²', 2, "1235 V²"),
    (1234567.8, 3, 'V²', 2, "1.23 kV²"),
    (0.00000123, 3, 'V²', 2, "1.23 mV²"),
    (1, 3, 'V^-1', -1, "1 V^-1"),
    (0.1, 3, 'V^-1', -1, "100 kV^-1"),
    (0.001, 3, 'V^-1', -1, "1 kV^-1"),
    (123.456, 3, 'V^-1', -1, "123 V^-1"),
    (123456.7, 3, 'V^-1', -1, "123 mV^-1"),
    (12345.6, 3, 'V^-1', -1, "12.3 mV^-1"),
    (12345.6, 3, 'V^(1/2)', 0.5, "12.3 MV^(1/2)"),

])
def test_siFormat(x, precision, suffix, power, expected):
    result = pg.siFormat(x, precision=precision, suffix=suffix, power=power)
    assert result == expected