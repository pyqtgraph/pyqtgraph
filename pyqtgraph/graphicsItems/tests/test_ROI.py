import numpy as np
import pytest
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtTest
from pyqtgraph.tests import assertImageApproved, mouseMove, mouseDrag, mouseClick, TransposedImageItem


app = pg.mkQApp()


def test_getArrayRegion():
    pr = pg.PolyLineROI([[0, 0], [27, 0], [0, 28]], closed=True)
    pr.setPos(1, 1)
    rois = [
        (pg.ROI([1, 1], [27, 28], pen='y'), 'baseroi'),
        (pg.RectROI([1, 1], [27, 28], pen='y'), 'rectroi'),
        (pg.EllipseROI([1, 1], [27, 28], pen='y'), 'ellipseroi'),
        (pr, 'polylineroi'),
    ]
    for roi, name in rois:
        # For some ROIs, resize should not be used.
        testResize = not isinstance(roi, pg.PolyLineROI)
        
        origMode = pg.getConfigOption('imageAxisOrder')
        try:
            pg.setConfigOptions(imageAxisOrder='col-major')
            check_getArrayRegion(roi, 'roi/'+name, testResize)
            #pg.setConfigOptions(imageAxisOrder='row-major')
            #check_getArrayRegion(roi, 'roi/'+name, testResize, transpose=True)
        finally:
            pg.setConfigOptions(imageAxisOrder=origMode)
    
    
def check_getArrayRegion(roi, name, testResize=True, transpose=False):
    initState = roi.getState()
    
    #win = pg.GraphicsLayoutWidget()
    win = pg.GraphicsView()
    win.show()
    win.resize(200, 400)
    
    # Don't use Qt's layouts for testing--these generate unpredictable results.
    #vb1 = win.addViewBox()
    #win.nextRow()
    #vb2 = win.addViewBox()
    
    # Instead, place the viewboxes manually 
    vb1 = pg.ViewBox()
    win.scene().addItem(vb1)
    vb1.setPos(6, 6)
    vb1.resize(188, 191)

    vb2 = pg.ViewBox()
    win.scene().addItem(vb2)
    vb2.setPos(6, 203)
    vb2.resize(188, 191)
    
    img1 = TransposedImageItem(border='w', transpose=transpose)
    img2 = TransposedImageItem(border='w', transpose=transpose)

    vb1.addItem(img1)
    vb2.addItem(img2)
    
    np.random.seed(0)
    data = np.random.normal(size=(7, 30, 31, 5))
    data[0, :, :, :] += 10
    data[:, 1, :, :] += 10
    data[:, :, 2, :] += 10
    data[:, :, :, 3] += 10
    
    img1.setImage(data[0, ..., 0])
    vb1.setAspectLocked()
    vb1.enableAutoRange(True, True)
    
    roi.setZValue(10)
    vb1.addItem(roi)

    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    #assert np.all((rgn == data[:, 1:-2, 1:-2, :]) | (rgn == 0))
    img2.setImage(rgn[0, ..., 0])
    vb2.setAspectLocked()
    vb2.enableAutoRange(True, True)
    
    app.processEvents()
    
    assertImageApproved(win, name+'/roi_getarrayregion', 'Simple ROI region selection.')

    with pytest.raises(TypeError):
        roi.setPos(0, False)

    roi.setPos([0.5, 1.5])
    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    img2.setImage(rgn[0, ..., 0])
    app.processEvents()
    assertImageApproved(win, name+'/roi_getarrayregion_halfpx', 'Simple ROI region selection, 0.5 pixel shift.')

    roi.setAngle(45)
    roi.setPos([3, 0])
    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    img2.setImage(rgn[0, ..., 0])
    app.processEvents()
    assertImageApproved(win, name+'/roi_getarrayregion_rotate', 'Simple ROI region selection, rotation.')

    if testResize:
        roi.setSize([60, 60])
        rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
        img2.setImage(rgn[0, ..., 0])
        app.processEvents()
        assertImageApproved(win, name+'/roi_getarrayregion_resize', 'Simple ROI region selection, resized.')

    img1.scale(1, -1)
    img1.setPos(0, img1.height())
    img1.rotate(20)
    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    img2.setImage(rgn[0, ..., 0])
    app.processEvents()
    assertImageApproved(win, name+'/roi_getarrayregion_img_trans', 'Simple ROI region selection, image transformed.')

    vb1.invertY()
    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    img2.setImage(rgn[0, ..., 0])
    app.processEvents()
    assertImageApproved(win, name+'/roi_getarrayregion_inverty', 'Simple ROI region selection, view inverted.')

    roi.setState(initState)
    img1.resetTransform()
    img1.setPos(0, 0)
    img1.scale(1, 0.5)
    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    img2.setImage(rgn[0, ..., 0])
    app.processEvents()
    assertImageApproved(win, name+'/roi_getarrayregion_anisotropic', 'Simple ROI region selection, image scaled anisotropically.')
    
    # allow the roi to be re-used
    roi.scene().removeItem(roi)


def test_PolyLineROI():
    rois = [
        (pg.PolyLineROI([[0, 0], [10, 0], [0, 15]], closed=True, pen=0.3), 'closed'),
        (pg.PolyLineROI([[0, 0], [10, 0], [0, 15]], closed=False, pen=0.3), 'open')
    ]
    
    #plt = pg.plot()
    plt = pg.GraphicsView()
    plt.show()
    plt.resize(200, 200)
    vb = pg.ViewBox()
    plt.scene().addItem(vb)
    vb.resize(200, 200)
    #plt.plotItem = pg.PlotItem()
    #plt.scene().addItem(plt.plotItem)
    #plt.plotItem.resize(200, 200)
    

    plt.scene().minDragTime = 0  # let us simulate mouse drags very quickly.

    # seemingly arbitrary requirements; might need longer wait time for some platforms..
    QtTest.QTest.qWaitForWindowShown(plt)
    QtTest.QTest.qWait(100)
    
    for r, name in rois:
        vb.clear()
        vb.addItem(r)
        vb.autoRange()
        app.processEvents()
        
        assertImageApproved(plt, 'roi/polylineroi/'+name+'_init', 'Init %s polyline.' % name)
        initState = r.getState()
        assert len(r.getState()['points']) == 3
        
        # hover over center
        center = r.mapToScene(pg.Point(3, 3))
        mouseMove(plt, center)
        assertImageApproved(plt, 'roi/polylineroi/'+name+'_hover_roi', 'Hover mouse over center of ROI.')
        
        # drag ROI
        mouseDrag(plt, center, center + pg.Point(10, -10), QtCore.Qt.LeftButton)
        assertImageApproved(plt, 'roi/polylineroi/'+name+'_drag_roi', 'Drag mouse over center of ROI.')
        
        # hover over handle
        pt = r.mapToScene(pg.Point(r.getState()['points'][2]))
        mouseMove(plt, pt)
        assertImageApproved(plt, 'roi/polylineroi/'+name+'_hover_handle', 'Hover mouse over handle.')
        
        # drag handle
        mouseDrag(plt, pt, pt + pg.Point(5, 20), QtCore.Qt.LeftButton)
        assertImageApproved(plt, 'roi/polylineroi/'+name+'_drag_handle', 'Drag mouse over handle.')
        
        # hover over segment 
        pt = r.mapToScene((pg.Point(r.getState()['points'][2]) + pg.Point(r.getState()['points'][1])) * 0.5)
        mouseMove(plt, pt+pg.Point(0, 2))
        assertImageApproved(plt, 'roi/polylineroi/'+name+'_hover_segment', 'Hover mouse over diagonal segment.')
        
        # click segment
        mouseClick(plt, pt, QtCore.Qt.LeftButton)
        assertImageApproved(plt, 'roi/polylineroi/'+name+'_click_segment', 'Click mouse over segment.')
        
        r.clearPoints()
        assertImageApproved(plt, 'roi/polylineroi/'+name+'_clear', 'All points cleared.')
        assert len(r.getState()['points']) == 0
        
        r.setPoints(initState['points'])
        assertImageApproved(plt, 'roi/polylineroi/'+name+'_setpoints', 'Reset points to initial state.')
        assert len(r.getState()['points']) == 3
        
        r.setState(initState)
        assertImageApproved(plt, 'roi/polylineroi/'+name+'_setstate', 'Reset ROI to initial state.')
        assert len(r.getState()['points']) == 3
        
    