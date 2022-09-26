import numpy as np

import pyqtgraph as pg


def test_qimage_writethrough():
    w, h = 256, 256
    backstore = np.ones((h, w), dtype=np.uint8)
    ptr0 = backstore.ctypes.data
    fmt = pg.Qt.QtGui.QImage.Format.Format_Grayscale8
    qimg = pg.functions.ndarray_to_qimage(backstore, fmt)

    def get_pointer(obj):
        if hasattr(obj, 'setsize'):
            return int(obj)
        else:
            return np.frombuffer(obj, dtype=np.uint8).ctypes.data

    # test that QImage is using the provided buffer (i.e. zero-copy)
    ptr1 = get_pointer(qimg.constBits())
    assert ptr0 == ptr1

    # test that QImage is not const (i.e. no COW)
    # if QImage is const, then bits() returns a copy
    ptr2 = get_pointer(qimg.bits())
    assert ptr1 == ptr2

    # test that data gets written through to provided buffer
    qimg.fill(0)
    assert np.all(backstore == 0)
