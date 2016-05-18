import numpy as np
import pytest
import pyqtgraph as pg
from pyqtgraph.tests import assertImageApproved


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
        
        check_getArrayRegion(roi, 'roi/'+name, testResize)
    
    
def check_getArrayRegion(roi, name, testResize=True):
    initState = roi.getState()
    
    win = pg.GraphicsLayoutWidget()
    win.show()
    win.resize(200, 400)
    
    vb1 = win.addViewBox()
    win.nextRow()
    vb2 = win.addViewBox()
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
    
    img1.setImage(data[0, ..., 0])
    vb1.setAspectLocked()
    vb1.enableAutoRange(True, True)
    
    roi.setZValue(10)
    vb1.addItem(roi)

    rgn = roi.getArrayRegion(data, img1, axes=(1, 2))
    assert np.all((rgn == data[:, 1:-2, 1:-2, :]) | (rgn == 0))
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

    # test features:
    #   pen / hoverpen
    #   handle pen / hoverpen
    #   handle types + mouse interaction
    #   getstate
    #   savestate
    #   restore state
    #   getarrayregion
    #   getarrayslice
    #   returnMappedCoords
    #   getAffineSliceParams
    #   getGlobalTransform
    #   
    # test conditions:
    #   y inverted
    #   extra array axes
    #   imageAxisOrder
    #   roi classes
    #   image transforms--rotation, scaling, flip
    #   view transforms--anisotropic scaling
    #   ROI transforms
    #   ROI parent transforms


    
    