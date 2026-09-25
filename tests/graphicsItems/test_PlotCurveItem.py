import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph.arraytoline import arrayToLineSegments
from tests.image_testing import assertImageApproved


def test_PlotCurveItem():
    p = pg.GraphicsLayoutWidget()
    p.resize(200, 150)
    p.ci.setContentsMargins(4, 4, 4, 4)  # default margins vary by platform
    p.show()
    v = p.addViewBox()
    data = np.array([1,4,2,3,np.inf,5,7,6,-np.inf,8,10,9,np.nan,-1,-2,0])
    c = pg.PlotCurveItem(data)
    c.setSegmentedLineMode('off')   # test images assume non-segmented-line-mode
    v.addItem(c)
    v.autoRange()
    
    # Check auto-range works. Some platform differences may be expected.
    checkRange = np.array([[-1.1457564053237301, 16.145756405323731], [-3.076811473165955, 11.076811473165955]])
    assert np.allclose(v.viewRange(), checkRange)
    
    assertImageApproved(p, 'plotcurveitem/connectall', "Plot curve with all points connected.")
    
    c.setData(data, connect='pairs')
    assertImageApproved(p, 'plotcurveitem/connectpairs', "Plot curve with pairs connected.")

    MoveTo = QtGui.QPainterPath.ElementType.MoveToElement
    LineTo = QtGui.QPainterPath.ElementType.LineToElement

    c.setData(data, connect='finite')
    path = c.getPath()

    expected_segments = [
        [(0.0, 1.0), (1.0, 4.0), (2.0, 2.0), (3.0, 3.0)],
        [(5.0, 5.0), (6.0, 7.0), (7.0, 6.0)],
        [(9.0, 8.0), (10.0, 10.0), (11.0, 9.0)],
        [(13.0, -1.0), (14.0, -2.0), (15.0, 0.0)],
    ]

    expected_elements = []
    for segment in expected_segments:
        for idx, coords in enumerate(segment):
            command = MoveTo if idx == 0 else LineTo
            expected_elements.append((command,) + coords)

    assert path.elementCount() == len(expected_elements)
    for idx, item in enumerate(expected_elements):
        element = path.elementAt(idx)
        assert item == (element.type, element.x, element.y)

    qpath_finite = pg.arraytoline._arraytoqpath_finite
    xcoords = np.arange(len(data))
    path1 = qpath_finite(xcoords, data, method='per_segment')
    path2 = qpath_finite(xcoords, data, method='bulk')
    assert path1 == path2

    c.setData(data, connect=np.array([1,1,1,0,1,1,0,0,1,0,0,0,1,1,0,0]))
    assertImageApproved(p, 'plotcurveitem/connectarray', "Plot curve with connection array.")

    p.close()


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
