import pytest

import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget
from pyqtgraph.Qt import QtGui

pg.mkQApp()


@pytest.mark.parametrize("rotationMethod", ["euler", "quaternion"])
def test_cameraPosition_matches_viewMatrix(rotationMethod):
    view = GLViewWidget(rotationMethod=rotationMethod)
    view.setCameraPosition(pos=QtGui.QVector3D(1, 2, 3), distance=10, elevation=30, azimuth=45)
    view.orbit(20, -10)
    # in eye space, the camera is at the origin
    expected = view.viewMatrix().inverted()[0].map(QtGui.QVector3D())
    assert (view.cameraPosition() - expected).length() == pytest.approx(0, abs=1e-4)
