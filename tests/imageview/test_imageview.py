import numpy as np

import pyqtgraph as pg

app = pg.mkQApp()

def test_nan_image():
    img = np.ones((10,10))
    img[0,0] = np.nan
    v = pg.image(img)
    v.imageItem.getHistogram()
    app.processEvents()
    v.window().close()

def test_init_with_mode_and_imageitem():
    data = np.random.randint(256, size=(256, 256, 3))
    imgitem = pg.ImageItem(data)
    pg.ImageView(imageItem=imgitem, levelMode="rgba")
    assert(pg.image is not None)
