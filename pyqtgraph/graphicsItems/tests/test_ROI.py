import pyqtgraph as pg
pg.mkQApp()

vb = pg.ViewBox()
data = pg.np.ones((7, 100, 110, 5))
image_tx = pg.ImageItem(data[:, :, 0, 0])
image_xy = pg.ImageItem(data[0, :, :, 0])
image_yz = pg.ImageItem(data[0, 0, :, :])
vb.addItem(image_tx)
vb.addItem(image_xy)
vb.addItem(image_yz)

size = (10, 15)
pos = (0, 0)
rois = [
    pg.ROI(pos, size),
    pg.RectROI(pos, size),
    pg.EllipseROI(pos, size),
    pg.CircleROI(pos, size),
    pg.PolyLineROI([pos, size]),
]

for roi in rois:
    vb.addItem(roi)


def test_getArrayRegion():
    global vb, image, rois, data, size
    
    # Test we can call getArrayRegion without errors 
    # (not checking for data validity)
    for roi in rois:
        arr = roi.getArrayRegion(data, image_tx)
        assert arr.shape == size + data.shape[2:]
        
        arr = roi.getArrayRegion(data, image_tx, axes=(0, 1))
        assert arr.shape == size + data.shape[2:]
        
        arr = roi.getArrayRegion(data.transpose(1, 0, 2, 3), image_tx, axes=(1, 0))
        assert arr.shape == size + data.shape[2:]
        
        arr = roi.getArrayRegion(data, image_xy, axes=(1, 2))
        assert arr.shape == data.shape[:1] + size + data.shape[3:]
        
        arr = roi.getArrayRegion(data.transpose(0, 2, 1, 3), image_xy, axes=(2, 1))
        assert arr.shape == data.shape[:1] + size + data.shape[3:]
        
        arr, coords = roi.getArrayRegion(data, image_xy, axes=(1, 2), returnMappedCoords=True)
        assert arr.shape == data.shape[:1] + size + data.shape[3:]
        assert coords.shape == (2,) + size
        
        
    
    