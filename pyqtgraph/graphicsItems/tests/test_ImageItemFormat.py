import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


def check_format(shape, dtype, levels, lut, expected_format):
    data = np.zeros(shape, dtype=dtype)
    item = pg.ImageItem(data, autoLevels=False)
    item.setLevels(levels)
    item.setLookupTable(lut)
    item.render()
    assert item.qimage.format() == expected_format


def test_uint8():
    Format = QtGui.QImage.Format
    dtype = np.uint8
    w, h = 192, 108
    lo, hi = 50, 200
    lut_none = None
    lut_mono1 = np.random.randint(256, size=256, dtype=np.uint8)
    lut_mono2 = np.random.randint(256, size=(256, 1), dtype=np.uint8)
    lut_rgb = np.random.randint(256, size=(256, 3), dtype=np.uint8)
    lut_rgba = np.random.randint(256, size=(256, 4), dtype=np.uint8)

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

    levels = [lo, hi]
    check_format((h, w, 3), dtype, levels, lut_none, Format.Format_RGB888)


def test_uint16():
    Format = QtGui.QImage.Format
    dtype = np.uint16
    w, h = 192, 108
    lo, hi = 100, 10000
    lut_none = None
    lut_mono1 = np.random.randint(256, size=256, dtype=np.uint8)
    lut_mono2 = np.random.randint(256, size=(256, 1), dtype=np.uint8)
    lut_rgb = np.random.randint(256, size=(256, 3), dtype=np.uint8)
    lut_rgba = np.random.randint(256, size=(256, 4), dtype=np.uint8)

    levels = None
    try:
        fmt_gray16 = Format.Format_Grayscale16
    except AttributeError:
        fmt_gray16 = Format.Format_ARGB32
    check_format((h, w), dtype, levels, lut_none, fmt_gray16)
    check_format((h, w, 3), dtype, levels, lut_none, Format.Format_RGB888)
    check_format((h, w, 4), dtype, levels, lut_none, Format.Format_RGBA64)

    levels = [lo, hi]
    check_format((h, w), dtype, levels, lut_none, Format.Format_Grayscale8)
    levels = None
    check_format((h, w), dtype, levels, lut_mono1, Format.Format_Grayscale8)
    check_format((h, w), dtype, levels, lut_mono2, Format.Format_Grayscale8)
    check_format((h, w), dtype, levels, lut_rgb, Format.Format_RGBX8888)
    check_format((h, w), dtype, levels, lut_rgba, Format.Format_RGBA8888)

    levels = [lo, hi]
    check_format((h, w, 3), dtype, levels, lut_none, Format.Format_RGB888)
