import numpy as np

import pyqtgraph as pg

app = pg.mkQApp()


def test_nan_image():
    img = np.ones((10,10))
    img[0,0] = np.nan
    iv = pg.ImageView()
    iv.setImage(img)
    iv.show()
    iv.getImageItem().getHistogram()
    app.processEvents()
    iv.window().close()


def test_timeslide_snap():
    count = 31
    frames = np.ones((count, 10, 10))
    iv = pg.ImageView(discreteTimeLine=True)
    assert iv.nframes() == 0
    iv.setImage(frames, xvals=(np.linspace(0., 1., count)))
    iv.show()
    assert iv.nframes() == count
    speed = count / 2
    iv.play(speed)
    assert iv.playRate == speed
    iv.timeLine.setPos(0.51)  # side effect: also pauses playback
    assert iv.playRate == 0
    ind, val = iv.timeIndex(iv.timeLine)
    assert ind == count // 2
    assert val == 0.5
    iv.togglePause()  # restarts playback
    assert iv.playRate == speed
    iv.togglePause()  # pauses playback
    assert iv.playRate == 0
    iv.play()
    assert iv.playRate == speed


def test_init_with_mode_and_imageitem():
    data = np.random.randint(256, size=(256, 256, 3))
    imgitem = pg.ImageItem(data)
    pg.ImageView(imageItem=imgitem, levelMode="rgba")
    assert(pg.image is not None)


def test_set_image_disables_histogram_auto_range_across_frames():
    data = np.stack((np.arange(100).reshape(10, 10),
                     np.arange(1000, 1100).reshape(10, 10)))
    iv = pg.ImageView()
    iv.setImage(data, autoHistogramRange=False)
    iv.setHistogramRange(-10, 10, padding=0)

    iv.setCurrentIndex(1)

    assert iv.ui.histogram.getHistogramRange() == [-10, 10]
    iv.close()


def test_set_histogram_auto_range_controls_updates_across_frames():
    data = np.stack((np.arange(100).reshape(10, 10),
                     np.arange(1000, 1100).reshape(10, 10)))
    iv = pg.ImageView()
    iv.setImage(data)
    iv.setHistogramRange(-10, 10, padding=0)

    iv.setCurrentIndex(1)

    assert iv.ui.histogram.getHistogramRange() != [-10, 10]

    iv.setHistogramAutoRange(False)
    iv.setHistogramRange(-10, 10, padding=0)
    iv.setCurrentIndex(0)

    assert iv.ui.histogram.getHistogramRange() == [-10, 10]
    iv.close()
