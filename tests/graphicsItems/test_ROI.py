import math
import platform

import numpy as np
import pytest
from packaging.version import Version, parse

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtTest
from tests.image_testing import assertImageApproved
from tests.ui_testing import mouseClick, mouseDrag, mouseMove

app = pg.mkQApp()
pg.setConfigOption("mouseRateLimit", 0)


@pytest.mark.parametrize("roi, name", [
        (pg.ROI([1, 1],        [27, 28], pen='y', antialias=False), 'baseroi'),
        (pg.RectROI([1, 1],    [27, 28], pen='y', antialias=False), 'rectroi'),
        (pg.EllipseROI([1, 1], [27, 28], pen='y', antialias=False), 'ellipseroi'),
        (
            pg.PolyLineROI(
                [[0, 0], [27, 0], [0, 28]],
                closed=True,
                pos=(1, 1),
                antialias=False
            ),'polylineroi'
        ),
    ]
)
@pytest.mark.parametrize("transpose", [True, False])
def test_getArrayRegion(roi, name, transpose):    
    # For some ROIs, resize should not be used.
    testResize = not isinstance(roi, pg.PolyLineROI)

    origMode = pg.getConfigOption('imageAxisOrder')
    try:
        pg.setConfigOptions(
            imageAxisOrder='row-major' if transpose else 'col-major'
        )
        check_getArrayRegion(
            roi, f"roi/{name}",
            testResize,
            transpose=transpose
        )
    finally:
        pg.setConfigOptions(imageAxisOrder=origMode)


def check_getArrayRegion(roi, name, testResize=True, transpose=False):
    # edges corner pixels seem to be slightly different on windows
    if (
        isinstance(roi, (pg.ROI, pg.RectROI))
        and platform.system() == "Windows"
    ):
        pxCount = 1
    else:
        pxCount = -1

    initState = roi.getState()
    win = pg.GraphicsView()
    win.resize(200, 400)
    win.show()
    # Don't use Qts' layouts for testing--these generate unpredictable results.
    # Instead, manually place the ViewBoxes
    vb1 = pg.ViewBox()
    win.scene().addItem(vb1)
    vb1.setPos(6, 6)
    vb1.resize(188, 191)

    vb2 = pg.ViewBox()
    win.scene().addItem(vb2)
    vb2.setPos(6, 203)
    vb2.resize(188, 191)

    img1 = pg.ImageItem(border='w')
    img2 = pg.ImageItem(border='w')

    vb1.addItem(img1)
    vb2.addItem(img2)

    np.random.seed(0)
    data = np.random.normal(size=(7, 30, 31, 5))
    data[0, :, :, :] += 10
    data[:, 1, :, :] += 10
    data[:, :, 2, :] += 10
    data[:, :, :, 3] += 10

    if transpose:
        data = data.transpose(0, 2, 1, 3)

    img1.setImage(data[0, ..., 0])
    vb1.setAspectLocked()
    vb1.enableAutoRange(True, True)

    roi.setZValue(10)
    vb1.addItem(roi)

    if isinstance(roi, pg.RectROI):
        first_arg = [28.0, 27.0] if transpose else [27.0, 28.0]
        assert roi.getAffineSliceParams(
            data,
            img1,
            axes=(1, 2)
        ) == (first_arg, ((1.0, 0.0), (0.0, 1.0)), (1.0, 1.0))


    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    # assert np.all((rgn == data[:, 1:-2, 1:-2, :]) | (rgn == 0))
    img2.setImage(rgn[0, ..., 0])
    vb2.setAspectLocked()
    vb2.enableAutoRange(True, True)

    app.processEvents()
    assertImageApproved(
        win,
        f'{name}/roi_getarrayregion',
        'Simple ROI region selection.',
        pxCount=pxCount
    )

    with pytest.raises(TypeError):
        roi.setPos(0, False)

    roi.setPos([0.5, 1.5])
    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    img2.setImage(rgn[0, ..., 0])
    app.processEvents()
    assertImageApproved(
        win, 
        f'{name}/roi_getarrayregion_halfpx',
        'Simple ROI region selection, 0.5 pixel shift.',
        pxCount=pxCount
    )

    roi.setAngle(45)
    roi.setPos([3, 0])
    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    img2.setImage(rgn[0, ..., 0])
    app.processEvents()
    assertImageApproved(
        win, 
        f'{name}/roi_getarrayregion_rotate',
        'Simple ROI region selection, rotation.',
        pxCount=pxCount
    )

    if testResize:
        roi.setSize([60, 60])
        rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
        img2.setImage(rgn[0, ..., 0])
        app.processEvents()
        assertImageApproved(
            win, 
            f'{name}/roi_getarrayregion_resize',
            'Simple ROI region selection, resized.',
            pxCount=pxCount
        )

    img1.setPos(0, img1.height())
    img1.setTransform(QtGui.QTransform().scale(1, -1).rotate(20), True)
    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    img2.setImage(rgn[0, ..., 0])
    app.processEvents()
    assertImageApproved(
        win,
        f'{name}/roi_getarrayregion_img_trans',
        'Simple ROI region selection, image transformed.',
        pxCount=pxCount
    )

    vb1.invertY()
    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    img2.setImage(rgn[0, ..., 0])
    app.processEvents()
    assertImageApproved(
        win, 
        f'{name}/roi_getarrayregion_inverty',
        'Simple ROI region selection, view inverted.',
        pxCount=pxCount
    )

    roi.setState(initState)
    img1.setPos(0, 0)
    img1.setTransform(QtGui.QTransform.fromScale(1, 0.5))
    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    img2.setImage(rgn[0, ..., 0])
    app.processEvents()
    assertImageApproved(
        win, 
        f'{name}/roi_getarrayregion_anisotropic',
        'Simple ROI region selection, image scaled anisotropically.',
        pxCount=pxCount
    )
    # allow the roi to be re-used
    roi.scene().removeItem(roi)
    win.hide()


def test_mouseClickEvent():
    plt = pg.GraphicsView()
    plt.resize(200, 200)
    plt.show()
    vb = pg.ViewBox()
    plt.scene().addItem(vb)
    vb.resize(200, 200)
    QtTest.QTest.qWaitForWindowExposed(plt)
    QtTest.QTest.qWait(100)

    roi = pg.RectROI((0, 0), (10, 20), removable=True)
    vb.addItem(roi)
    app.processEvents()

    mouseClick(
        plt,
        roi.mapToScene(pg.Point(2, 2)),
        QtCore.Qt.MouseButton.LeftButton
    )


def test_mouseDragEventSnap():
    pg.setConfigOptions(antialias=False)
    plt = pg.GraphicsView()
    plt.resize(200, 200)
    plt.show()
    vb = pg.ViewBox()
    plt.scene().addItem(vb)
    vb.resize(200, 200)
    QtTest.QTest.qWaitForWindowExposed(plt)
    QtTest.QTest.qWait(100)

    # A Rectangular roi with scaleSnap enabled
    initial_x = 20
    initial_y = 20
    roi = pg.RectROI(
        (initial_x, initial_y),
        (20, 20),
        scaleSnap=True,
        translateSnap=True,
        snapSize=1.0,
        movable=True
    )
    vb.addItem(roi)
    app.processEvents()

    # Snap size roundtrip
    assert roi.snapSize == 1.0
    roi.snapSize = 0.2
    assert roi.snapSize == 0.2
    roi.snapSize = 1.0
    assert roi.snapSize == 1.0

    # Snap position check
    snapped = roi.getSnapPosition(pg.Point(2.5, 3.5), snap=True)
    assert snapped == pg.Point(2.0, 4.0)

    # Only drag in y direction
    roi_position = roi.mapToView(pg.Point(initial_x, initial_y))
    mouseDrag(
        plt,
        roi_position,
        roi_position + pg.Point(0, 10),
        QtCore.Qt.MouseButton.LeftButton
    )
    assert roi.pos() == pg.Point(initial_x, 19)

    mouseDrag(
        plt,
        roi_position,
        roi_position + pg.Point(0, 10),
        QtCore.Qt.MouseButton.LeftButton
    )
    assert roi.pos() == pg.Point(initial_x, 18)

    # Only drag in x direction
    mouseDrag(
        plt,
        roi_position,
        roi_position + pg.Point(10, 0),
        QtCore.Qt.MouseButton.LeftButton
    )
    assert roi.pos() == pg.Point(21, 18)

@pytest.mark.parametrize("roi, name", [
        (
            pg.PolyLineROI(
                [[0, 0], [10, 0], [0, 15]],
                closed=True,
                pen=0.3,
                antialias=False
            ),
            'closed'
        ),
        (
            pg.PolyLineROI(
                [[0, 0], [10, 0], [0, 15]],
                closed=False,
                pen=0.3,
                antialias=False 
            ),
            'open'
        )
    ]
)
def test_PolyLineROI(roi, name):
    plt = pg.GraphicsView()
    plt.resize(200, 200)
    plt.show()
    vb = pg.ViewBox()
    plt.scene().addItem(vb)
    vb.resize(200, 200)
    # plt.plotItem = pg.PlotItem()
    # plt.scene().addItem(plt.plotItem)
    # plt.plotItem.resize(200, 200)

    plt.scene().minDragTime = 0  # let us simulate mouse drags very quickly.

    # seemingly arbitrary requirements; might need longer wait time for some platforms.
    QtTest.QTest.qWaitForWindowExposed(plt)
    QtTest.QTest.qWait(100)

    vb.clear()
    vb.addItem(roi)
    vb.autoRange()
    app.processEvents()

    assertImageApproved(
        plt,
        f'roi/polylineroi/{name}_init',
        f'Init {name} polyline.'
    )
    initState = roi.getState()
    assert len(roi.getState()['points']) == 3

    # hover over center
    center = roi.mapToScene(pg.Point(3, 3))
    mouseMove(plt, center)
    assertImageApproved(
        plt,
        f'roi/polylineroi/{name}_hover_roi',
        'Hover mouse over center of ROI.'
    )

    # drag ROI
    mouseDrag(
        plt,
        center, center + pg.Point(10, -10),
        QtCore.Qt.MouseButton.LeftButton
    )
    assertImageApproved(
        plt,
        f'roi/polylineroi/{name}_drag_roi',
        'Drag mouse over center of ROI.'
    )

    # hover over handle
    pt = roi.mapToScene(
        pg.Point(roi.getState()['points'][2])
    )
    mouseMove(plt, pt)
    assertImageApproved(
        plt,
        f'roi/polylineroi/{name}_hover_handle',
        'Hover mouse over handle.'
    )

    # drag handle
    mouseDrag(plt,
        pt,
        pt + pg.Point(5, 20),
        QtCore.Qt.MouseButton.LeftButton
    )
    
    assertImageApproved(
        plt,
        f'roi/polylineroi/{name}_drag_handle',
        'Drag mouse over handle.'
    )

    # hover over segment
    pt = roi.mapToScene(
        (
            pg.Point(roi.getState()['points'][2]) + 
            pg.Point(roi.getState()['points'][1])
        ) * 0.5
    )

    mouseMove(plt, pt + pg.Point(0, 2))
    assertImageApproved(
        plt,
        f'roi/polylineroi/{name}_hover_segment',
        'Hover mouse over diagonal segment.'
    )

    # click segment
    mouseClick(plt, pt, QtCore.Qt.MouseButton.LeftButton)
    assertImageApproved(
        plt,
        f'roi/polylineroi/{name}_click_segment',
        'Click mouse over segment.',
        pxCount=3
    )

    # drag new handle
    mouseMove(plt, pt + pg.Point(10, -10))
    # pg bug: have to move the mouse off/on again to register hover
    mouseDrag(
        plt,
        pt, pt + pg.Point(10, -10),
        QtCore.Qt.MouseButton.LeftButton
    )

    assertImageApproved(
        plt,
        f'roi/polylineroi/{name}_drag_new_handle',
        'Drag mouse over created handle.',
        pxCount=2
    )

    # clear all points
    roi.clearPoints()
    assertImageApproved(
        plt,
        f'roi/polylineroi/{name}_clear',
        'All points cleared.'
    )
    assert len(roi.getState()['points']) == 0

    # call setPoints
    roi.setPoints(initState['points'])
    assertImageApproved(
        plt,
        f'roi/polylineroi/{name}_setpoints',
        'Reset points to initial state.',
    )
    assert len(roi.getState()['points']) == 3

    # call setState
    roi.setState(initState)
    assertImageApproved(
        plt,
        f'roi/polylineroi/{name}_setstate',
        'Reset ROI to initial state.'
    )
    assert len(roi.getState()['points']) == 3

    plt.hide()


@pytest.mark.parametrize("p1,p2", [
    ((1, 1), (2, 5)),
    ((0.1, 0.1), (-1, 5)),
    ((3, -1), (5, -6)),
    ((-2, 1), (-4, -8)),
])
def test_LineROI_coords(p1, p2):
    pg.setConfigOptions(antialias=False)
    pw = pg.PlotWidget()
    pw.show()

    lineroi = pg.LineROI(p1, p2, width=0.5, pen="r")
    pw.addItem(lineroi)

    # first two handles are the scale-rotate handles positioned by pos1, pos2
    for expected, (_, scenepos) in zip(
        [p1, p2],
        lineroi.getSceneHandlePositions()
    ):
        got = lineroi.mapSceneToParent(scenepos)
        assert math.isclose(got.x(), expected[0])
        assert math.isclose(got.y(), expected[1])
