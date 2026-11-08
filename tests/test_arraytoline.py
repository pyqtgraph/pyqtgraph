import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.arraytoline import arrayToLineSegments, _compact_moveto

def check_path_elements(path, expected_elements):
    assert path.elementCount() == len(expected_elements)
    for idx, item in enumerate(expected_elements):
        element = path.elementAt(idx)
        assert item == (element.type, element.x, element.y)

def build_expected_elements(segment):
    MoveTo = QtGui.QPainterPath.ElementType.MoveToElement
    LineTo = QtGui.QPainterPath.ElementType.LineToElement

    expected_elements = []
    for idx, coords in enumerate(segment):
        command = MoveTo if idx == 0 else LineTo
        expected_elements.append((command,) + coords)
    return expected_elements

def test_arraytoline():
    ycoords = np.array([1,4,2,3,np.inf,5,7,6,-np.inf,8,10,9,np.nan,-1,-2,0])
    xcoords = np.arange(len(ycoords))

    # connect = 'all'
    qpath = pg.arraytoline.arrayToQPath(xcoords, ycoords, connect='all')
    expected_elements = build_expected_elements([
        (0.0, 1.0), (1.0, 4.0), (2.0, 2.0), (3.0, 3.0), (5.0, 5.0), (6.0, 7.0), (7.0, 6.0),
        (9.0, 8.0), (10.0, 10.0), (11.0, 9.0), (13.0, -1.0), (14.0, -2.0), (15.0, 0.0),
    ])
    check_path_elements(qpath, expected_elements)

    # connect = 'finite'
    qpath = pg.arraytoline.arrayToQPath(xcoords, ycoords, connect='finite')

    expected_segments = [
        [(0.0, 1.0), (1.0, 4.0), (2.0, 2.0), (3.0, 3.0)],
        [(5.0, 5.0), (6.0, 7.0), (7.0, 6.0)],
        [(9.0, 8.0), (10.0, 10.0), (11.0, 9.0)],
        [(13.0, -1.0), (14.0, -2.0), (15.0, 0.0)],
    ]

    expected_elements = [e for seg in expected_segments for e in build_expected_elements(seg)]
    check_path_elements(qpath, expected_elements)

    qpath_finite = pg.arraytoline._arraytoqpath_finite
    path1 = qpath_finite(xcoords, ycoords, method='per_segment')
    path2 = qpath_finite(xcoords, ycoords, method='bulk')
    assert path1 == path2

    # connect = 'pairs'
    qpath = pg.arraytoline.arrayToQPath(xcoords, ycoords, connect='pairs')

    expected_segments = [
        [(0.0, 1.0), (1.0, 4.0)],
        [(2.0, 2.0), (3.0, 3.0)],
        [(6.0, 7.0), (7.0, 6.0)],
        [(10.0, 10.0), (11.0, 9.0)],
        [(14.0, -2.0), (15.0, 0.0)],
    ]

    expected_elements = [e for seg in expected_segments for e in build_expected_elements(seg)]
    check_path_elements(qpath, expected_elements)

    # connect = 'array'
    connect_array = np.array([1,1,1,0,1,1,0,0,1,0,0,0,1,1,0,0])
    qpath = pg.arraytoline.arrayToQPath(xcoords, ycoords, connect=connect_array)

    # x, y are the values after applying _compute_backfill_indices.
    # m is the mask of finite points, used as a visual aid only.
    # note that this test is only checking existing behavior, not necessarily the correct behavior.
    # in particular, runs of MoveTo(s) are not considered valid in QPainterPath, but this is what
    # the current implementation produces.
    c = np.concatenate([[0], connect_array[:-1]]).tolist()
    x = [0, 1, 2, 3, 3, 5, 6, 7, 7, 9,10,11,11,13,14,15]
    y = [1, 4, 2, 3, 3, 5, 7, 6, 6, 8,10, 9, 9,-1,-2, 0]
    m = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]

    compact_mask = _compact_moveto(np.array(c, dtype=bool))
    c = np.array(c)[compact_mask]
    x = np.array(x)[compact_mask]
    y = np.array(y)[compact_mask]
    m = np.array(m)[compact_mask]

    Command = [QtGui.QPainterPath.ElementType.MoveToElement,
               QtGui.QPainterPath.ElementType.LineToElement]

    expected_elements = [(Command[c[i]], x[i], y[i]) for i in range(len(m))]
    check_path_elements(qpath, expected_elements)

def test_arrayToLineSegments():
    # test the boundary case where the dataset consists of a single point
    xy = np.array([0.])
    parray = arrayToLineSegments(xy, xy, connect='all', finiteCheck=True)
    segs = parray.drawargs()
    assert isinstance(segs, tuple) and len(segs) in [1, 2]
    if len(segs) == 1:
        assert len(segs[0]) == 0
    elif len(segs) == 2:
        assert segs[1] == 0

@pytest.mark.parametrize(
    "c, expected", [
        ([], []),
        ([0], [0]),
        ([0, 0, 0, 1, 1, 1, 0, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1]),
        ([0, 0, 0, 0], [0, 0, 0, 0]),
        ([0, 1, 0, 1, 0], [1, 1, 1, 1, 0]),
        ([0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0]),
    ]
)
def test_compact_moveto(c, expected):
    c = np.array(c, dtype=bool)
    original = c.copy()
    result = _compact_moveto(c)
    assert result.dtype == bool
    assert np.array_equal(result, np.array(expected, dtype=bool))
    # the input must not be modified
    assert np.array_equal(c, original)
    # every LineTo is preserved
    assert np.all(result[c])


@pytest.mark.parametrize(
    "c", [
        [1],
        [1, 1],
        [1, 0],
        [1, 1, 1, 0, 0, 0],
    ]
)
def test_compact_moveto_illegal_first_element(c):
    # a LineTo cannot be the first element of a path
    with pytest.raises(ValueError):
        _compact_moveto(np.array(c, dtype=bool))


@pytest.mark.parametrize(
    "c", [
        [0, 1, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1],
        [0],
    ]
)
def test_compact_moveto_strips_moveto_without_lineto(c):
    c = np.array(c, dtype=bool)
    mask = _compact_moveto(c)
    # keep every LineTo, and a MoveTo only where a LineTo follows it
    assert np.array_equal(mask, c | np.concatenate([c[1:], [False]]))
    # nothing which draws is stripped
    assert not c[~mask].any()
    # each retained MoveTo is immediately followed by a LineTo
    kept_moveto = np.nonzero(mask & ~c)[0]
    assert kept_moveto.size == 0 or c[kept_moveto + 1].all()


def test_compact_moveto_selects_elements():
    # the mask selects the elements to keep; their element types and
    # coordinates are unchanged, so the path drawn is the same
    c = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    xy = np.arange(18).reshape(9, 2)
    mask = _compact_moveto(c)
    # the first two MoveTo elements are overwritten by the third
    assert np.array_equal(c[mask], np.array([0, 1, 1, 1, 0, 1, 1]))
    assert np.array_equal(xy[mask], xy[2:])
