# -*- coding: utf-8 -*-
import pyqtgraph as pg


def setup_module(module):
    try:
        app = pg.QtGui.QApplication()
    except RuntimeError:
        pass  # QApplication instance already exists

def test_zoom_normal():
    vb = pg.ViewBox()
    testRange = pg.QtCore.QRect(0, 0, 10, 20)
    vb.setRange(testRange, padding=0)
    vbViewRange = vb.getState()['viewRange']
    assert vbViewRange == [[testRange.left(), testRange.right()],
                           [testRange.top(), testRange.bottom()]]

def test_zoom_limit():
    """Test zooming with X and Y limits set"""
    vb = pg.ViewBox()
    vb.setLimits(xMin=0, xMax=10, yMin=0, yMax=10)

    # Try zooming within limits. Should return unmodified
    testRange = pg.QtCore.QRect(0, 0, 9, 9)
    vb.setRange(testRange, padding=0)
    vbViewRange = vb.getState()['viewRange']
    assert vbViewRange == [[testRange.left(), testRange.right()],
                           [testRange.top(), testRange.bottom()]]

    # And outside limits. both view range and targetRange should be set to limits
    testRange = pg.QtCore.QRect(-5, -5, 16, 20)
    vb.setRange(testRange, padding=0)

    expected = [[0, 10], [0, 10]]
    vbState = vb.getState()

    assert vbState['targetRange'] == expected
    assert vbState['viewRange'] == expected

def test_zoom_range_limit():
    """Test zooming with XRange and YRange limits set, but no X and Y limits"""
    vb = pg.ViewBox()
    vb.setLimits(minXRange=5, maxXRange=10, minYRange=5, maxYRange=10)

    # Try something within limits
    testRange = pg.QtCore.QRect(-15, -15, 7, 7)
    vb.setRange(testRange, padding=0)

    expected = [[testRange.left(), testRange.right()],
               [testRange.top(), testRange.bottom()]]

    vbViewRange = vb.getState()['viewRange']
    assert vbViewRange == expected

    # and outside limits
    testRange = pg.QtCore.QRect(-15, -15, 17, 17)

    # Code should center the required width reduction, so move each side by 3
    expected = [[testRange.left() + 3, testRange.right() - 3],
               [testRange.top() + 3, testRange.bottom() - 3]]

    vb.setRange(testRange, padding=0)
    vbViewRange = vb.getState()['viewRange']
    vbTargetRange = vb.getState()['targetRange']

    assert vbViewRange == expected
    assert vbTargetRange == expected

def test_zoom_ratio():
    """Test zooming with a fixed aspect ratio set"""
    vb = pg.ViewBox(lockAspect=1)

    # Give the viewbox a size of the proper aspect ratio to keep things easy
    vb.setFixedHeight(10)
    vb.setFixedWidth(10)

    # request a range with a good ratio
    testRange = pg.QtCore.QRect(0, 0, 10, 10)
    vb.setRange(testRange, padding=0)
    expected = [[testRange.left(), testRange.right()],
                [testRange.top(), testRange.bottom()]]

    viewRange = vb.getState()['viewRange']
    viewWidth = viewRange[0][1] - viewRange[0][0]
    viewHeight = viewRange[1][1] - viewRange[1][0]

    # Assert that the width and height are equal, since we locked the aspect ratio at 1
    assert viewWidth == viewHeight

    # and for good measure, that it is the same as the test range
    assert viewRange == expected

    # Now try to set to something with a different aspect ratio
    testRange = pg.QtCore.QRect(0, 0, 10, 20)
    vb.setRange(testRange, padding=0)

    viewRange = vb.getState()['viewRange']
    viewWidth = viewRange[0][1] - viewRange[0][0]
    viewHeight = viewRange[1][1] - viewRange[1][0]

    # Don't really care what we got here, as long as the width and height are the same
    assert viewWidth == viewHeight

def test_zoom_ratio2():
    """Slightly more complicated zoom ratio test, where the view box shape does not match the ratio"""
    vb = pg.ViewBox(lockAspect=1)

    # twice as wide as tall
    vb.setFixedHeight(10)
    vb.setFixedWidth(20)

    # more or less random requested range
    testRange = pg.QtCore.QRect(0, 0, 10, 15)
    vb.setRange(testRange, padding=0)

    viewRange = vb.getState()['viewRange']
    viewWidth = viewRange[0][1] - viewRange[0][0]
    viewHeight = viewRange[1][1] - viewRange[1][0]

    # View width should be twice as wide as the height,
    # since the viewbox is twice as wide as it is tall.
    assert viewWidth == 2 * viewHeight

def test_zoom_ratio_with_limits1():
    """Test zoom with both ratio and limits set"""
    vb = pg.ViewBox(lockAspect=1)

    # twice as wide as tall
    vb.setFixedHeight(10)
    vb.setFixedWidth(20)

    # set some limits
    vb.setLimits(xMin=-5, xMax=5, yMin=-5, yMax=5)

    # Try to zoom too tall
    testRange = pg.QtCore.QRect(0, 0, 6, 10)
    vb.setRange(testRange, padding=0)

    viewRange = vb.getState()['viewRange']
    viewWidth = viewRange[0][1] - viewRange[0][0]
    viewHeight = viewRange[1][1] - viewRange[1][0]

    # Make sure our view is within limits and the proper aspect ratio
    assert viewRange[0][0] >= -5
    assert viewRange[0][1] <= 5
    assert viewRange[1][0] >= -5
    assert viewRange[1][1] <= 5
    assert viewWidth == 2 * viewHeight

def test_zoom_ratio_with_limits2():
    vb = pg.ViewBox(lockAspect=1)

    # twice as wide as tall
    vb.setFixedHeight(10)
    vb.setFixedWidth(20)

    # set some limits
    vb.setLimits(xMin=-5, xMax=5, yMin=-5, yMax=5)

    # Same thing, but out-of-range the other way
    testRange = pg.QtCore.QRect(0, 0, 16, 6)
    vb.setRange(testRange, padding=0)

    viewRange = vb.getState()['viewRange']
    viewWidth = viewRange[0][1] - viewRange[0][0]
    viewHeight = viewRange[1][1] - viewRange[1][0]

    # Make sure our view is within limits and the proper aspect ratio
    assert viewRange[0][0] >= -5
    assert viewRange[0][1] <= 5
    assert viewRange[1][0] >= -5
    assert viewRange[1][1] <= 5
    assert viewWidth == 2 * viewHeight

def test_zoom_ratio_with_limits_out_of_range():
    vb = pg.ViewBox(lockAspect=1)

    # twice as wide as tall
    vb.setFixedHeight(10)
    vb.setFixedWidth(20)

    # set some limits
    vb.setLimits(xMin=-5, xMax=5, yMin=-5, yMax=5)

    # Request something completely out-of-range and out-of-aspect
    testRange = pg.QtCore.QRect(10, 10, 25, 100)
    vb.setRange(testRange, padding=0)

    viewRange = vb.getState()['viewRange']
    viewWidth = viewRange[0][1] - viewRange[0][0]
    viewHeight = viewRange[1][1] - viewRange[1][0]

    # Make sure our view is within limits and the proper aspect ratio
    assert viewRange[0][0] >= -5
    assert viewRange[0][1] <= 5
    assert viewRange[1][0] >= -5
    assert viewRange[1][1] <= 5
    assert viewWidth == 2 * viewHeight


if __name__ == "__main__":
    setup_module(None)
    test_zoom_ratio()
