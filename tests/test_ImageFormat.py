import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


rng = np.random.default_rng()


def check_format(shape, dtype, levels, lut, expected_format, *, transparentLocations=None):
    data = np.zeros(shape, dtype=dtype)
    qimage = pg.functions_qimage.try_make_qimage(data, levels=levels, lut=lut, transparentLocations=transparentLocations)
    assert qimage is not None and qimage.format() == expected_format


def test_uint8():
    Format = QtGui.QImage.Format
    dtype = np.uint8
    w, h = 192, 108
    lo, hi = 50, 200
    lut_none = None
    lut_mono1 = rng.integers(256, size=256, dtype=np.uint8)
    lut_mono2 = rng.integers(256, size=(256, 1), dtype=np.uint8)
    lut_rgb = rng.integers(256, size=(256, 3), dtype=np.uint8)
    lut_rgba = rng.integers(256, size=(256, 4), dtype=np.uint8)

    # lut with less than 256 entries
    lut_mono1_s = rng.integers(256, size=255, dtype=np.uint8)
    lut_mono2_s = rng.integers(256, size=(255, 1), dtype=np.uint8)
    lut_rgb_s = rng.integers(256, size=(255, 3), dtype=np.uint8)
    lut_rgba_s = rng.integers(256, size=(255, 4), dtype=np.uint8)

    # lut with more than 256 entries
    lut_mono1_l = rng.integers(256, size=257, dtype=np.uint8)
    lut_mono2_l = rng.integers(256, size=(257, 1), dtype=np.uint8)
    lut_rgb_l = rng.integers(256, size=(257, 3), dtype=np.uint8)
    lut_rgba_l = rng.integers(256, size=(257, 4), dtype=np.uint8)

    levels = None
    check_format((h, w), dtype, levels, lut_none, Format.Format_Grayscale8)
    check_format((h, w, 3), dtype, levels, lut_none, Format.Format_RGB888)
    check_format((h, w, 4), dtype, levels, lut_none, Format.Format_RGBA8888)

    levels = [lo, hi]
    check_format((h, w), dtype, levels, lut_none, Format.Format_Indexed8)
    levels = None
    check_format((h, w), dtype, levels, lut_mono1, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_mono2, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgb, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgba, Format.Format_Indexed8)
    levels = [lo, hi]
    check_format((h, w), dtype, levels, lut_mono1, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_mono2, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgb, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgba, Format.Format_Indexed8)

    check_format((h, w), dtype, levels, lut_mono1_s, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_mono2_s, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgb_s, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgba_s, Format.Format_Indexed8)

    check_format((h, w), dtype, levels, lut_mono1_l, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_mono2_l, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgb_l, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgba_l, Format.Format_Indexed8)

    levels = [lo, hi]
    check_format((h, w, 3), dtype, levels, lut_none, Format.Format_RGB888)


def test_uint16():
    Format = QtGui.QImage.Format
    dtype = np.uint16
    w, h = 192, 108
    lo, hi = 100, 10000
    lut_none = None

    lut_mono1 = rng.integers(256, size=256, dtype=np.uint8)
    lut_mono2 = rng.integers(256, size=(256, 1), dtype=np.uint8)
    lut_rgb = rng.integers(256, size=(256, 3), dtype=np.uint8)
    lut_rgba = rng.integers(256, size=(256, 4), dtype=np.uint8)

    # lut with less than 256 entries
    lut_mono1_s = rng.integers(256, size=255, dtype=np.uint8)
    lut_mono2_s = rng.integers(256, size=(255, 1), dtype=np.uint8)
    lut_rgb_s = rng.integers(256, size=(255, 3), dtype=np.uint8)
    lut_rgba_s = rng.integers(256, size=(255, 4), dtype=np.uint8)

    # lut with more than 256 entries
    lut_mono1_l = rng.integers(256, size=257, dtype=np.uint8)
    lut_mono2_l = rng.integers(256, size=(257, 1), dtype=np.uint8)
    lut_rgb_l = rng.integers(256, size=(257, 3), dtype=np.uint8)
    lut_rgba_l = rng.integers(256, size=(257, 4), dtype=np.uint8)

    levels = None
    check_format((h, w), dtype, levels, lut_none, Format.Format_Grayscale16)
    check_format((h, w, 3), dtype, levels, lut_none, Format.Format_RGBX64)
    check_format((h, w, 4), dtype, levels, lut_none, Format.Format_RGBA64)

    levels = [lo, hi]
    check_format((h, w), dtype, levels, lut_none, Format.Format_Grayscale8)
    levels = None
    check_format((h, w), dtype, levels, lut_mono1, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_mono2, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgb, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgba, Format.Format_Indexed8)

    check_format((h, w), dtype, levels, lut_mono1_s, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_mono2_s, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgb_s, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgba_s, Format.Format_Indexed8)

    check_format((h, w), dtype, levels, lut_mono1_l, Format.Format_Grayscale8)
    check_format((h, w), dtype, levels, lut_mono2_l, Format.Format_Grayscale8)
    check_format((h, w), dtype, levels, lut_rgb_l, Format.Format_RGBX8888)
    check_format((h, w), dtype, levels, lut_rgba_l, Format.Format_RGBA8888)

    levels = [lo, hi]
    check_format((h, w, 3), dtype, levels, lut_none, Format.Format_RGB888)


def test_float32():
    Format = QtGui.QImage.Format
    dtype = np.float32
    w, h = 192, 108
    lo, hi = -1, 1
    lut_none = None

    lut_mono1 = rng.integers(256, size=256, dtype=np.uint8)
    lut_mono2 = rng.integers(256, size=(256, 1), dtype=np.uint8)
    lut_rgb = rng.integers(256, size=(256, 3), dtype=np.uint8)
    lut_rgba = rng.integers(256, size=(256, 4), dtype=np.uint8)

    # lut with less than 256 entries
    lut_mono1_s = rng.integers(256, size=255, dtype=np.uint8)
    lut_mono2_s = rng.integers(256, size=(255, 1), dtype=np.uint8)
    lut_rgb_s = rng.integers(256, size=(255, 3), dtype=np.uint8)
    lut_rgba_s = rng.integers(256, size=(255, 4), dtype=np.uint8)

    # lut with more than 256 entries
    lut_mono1_l = rng.integers(256, size=257, dtype=np.uint8)
    lut_mono2_l = rng.integers(256, size=(257, 1), dtype=np.uint8)
    lut_rgb_l = rng.integers(256, size=(257, 3), dtype=np.uint8)
    lut_rgba_l = rng.integers(256, size=(257, 4), dtype=np.uint8)

    levels = [lo, hi]

    check_format((h, w), dtype, levels, lut_none, Format.Format_Grayscale8)
    check_format((h, w, 3), dtype, levels, lut_none, Format.Format_RGB888)

    check_format((h, w), dtype, levels, lut_mono1, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_mono2, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgb, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgba, Format.Format_Indexed8)

    check_format((h, w), dtype, levels, lut_mono1_s, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_mono2_s, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgb_s, Format.Format_Indexed8)
    check_format((h, w), dtype, levels, lut_rgba_s, Format.Format_Indexed8)

    check_format((h, w), dtype, levels, lut_mono1_l, Format.Format_Grayscale8)
    check_format((h, w), dtype, levels, lut_mono2_l, Format.Format_Grayscale8)
    check_format((h, w), dtype, levels, lut_rgb_l, Format.Format_RGBX8888)
    check_format((h, w), dtype, levels, lut_rgba_l, Format.Format_RGBA8888)

    all_lut_types = [
        lut_none,
        lut_mono1, lut_mono2, lut_rgb, lut_rgba,
        lut_mono1_s, lut_mono2_s, lut_rgb_s, lut_rgba_s,
        lut_mono1_l, lut_mono2_l, lut_rgb_l, lut_rgba_l,
    ]

    center = (np.array([h//2]), np.array([w//2]))

    for lut in all_lut_types:
        check_format((h, w), dtype, levels, lut, Format.Format_RGBA8888, transparentLocations=center)

    check_format((h, w, 3), dtype, levels, lut_none, Format.Format_RGBA8888, transparentLocations=center)
