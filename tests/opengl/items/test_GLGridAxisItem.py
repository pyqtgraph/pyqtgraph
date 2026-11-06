import pytest

import pyqtgraph as pg
from pyqtgraph.opengl import GLGridAxisItem, GLViewWidget

pg.mkQApp()


@pytest.mark.parametrize("elevation, azimuth", [(30, 45), (-20, 200), (80, -100)])
def test_view_angle_rotation_methods(elevation, azimuth):
    # view_angle() raised KeyError with rotationMethod='quaternion' (#3629)
    angles = {}
    for method in ["euler", "quaternion"]:
        view = GLViewWidget(rotationMethod=method)
        item = GLGridAxisItem()
        view.addItem(item)
        view.setCameraPosition(elevation=elevation, azimuth=azimuth)
        angles[method] = item.view_angle()
    assert angles["quaternion"] == pytest.approx(angles["euler"], abs=1e-3)
