import pyqtgraph as pg
import numpy as np

app = pg.mkQApp()

def test_nan_image():
    img = np.ones((10,10))
    img[0,0] = np.nan
    v = pg.image(img)
    app.processEvents()
    v.window().close()
