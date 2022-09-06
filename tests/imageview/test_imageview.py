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


def test_timeslide_snap():
    count = 30
    frames = np.ones((count, 10, 10))
    iv = pg.ImageView(discreteTimeLine=True)
    assert iv.nframes == 0
    iv.setImage(frames, xvals=np.linspace(0., 1., count))
    iv.show()
    assert iv.nframes == count
    iv.play(count / 2)
    iv.timeLine.setPos(0.49999)
    ind, val = iv.timeIndex(iv.timeLine)
    # :TODO: this val isn't what I expect, nor did I expect the `- 1` in the next statement.
    assert ind == (count / 2) - 1
    # assert val == 0.5


def test_init_with_mode_and_imageitem():
    data = np.random.randint(256, size=(256, 256, 3))
    imgitem = pg.ImageItem(data)
    pg.ImageView(imageItem=imgitem, levelMode="rgba")
    assert(pg.image is not None)
