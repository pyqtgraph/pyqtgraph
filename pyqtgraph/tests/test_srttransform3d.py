import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

testPoints = np.array([
                       [0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [-1, -1, 0],
                       [0, -1, -1]])


def testMatrix():
    """
    SRTTransform3D => Transform3D => SRTTransform3D
    """
    tr = pg.SRTTransform3D()
    tr.setRotate(45, (0, 0, 1))
    tr.setScale(0.2, 0.4, 1)
    tr.setTranslate(10, 20, 40)
    assert tr.getRotation() == (45, QtGui.QVector3D(0, 0, 1))
    assert tr.getScale() == QtGui.QVector3D(0.2, 0.4, 1)
    assert tr.getTranslation() == QtGui.QVector3D(10, 20, 40)

    tr2 = pg.Transform3D(tr)
    assert np.all(tr.matrix() == tr2.matrix())

    # This is the most important test:
    # The transition from Transform3D to SRTTransform3D is a tricky one.
    tr3 = pg.SRTTransform3D(tr2)
    assert_array_almost_equal(tr.matrix(), tr3.matrix())
    assert_almost_equal(tr3.getRotation()[0], tr.getRotation()[0])
    assert_array_almost_equal(tr3.getRotation()[1], tr.getRotation()[1])
    assert_array_almost_equal(tr3.getScale(), tr.getScale())
    assert_array_almost_equal(tr3.getTranslation(), tr.getTranslation())
