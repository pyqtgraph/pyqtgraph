import pytest
import pyqtgraph as pg
from numpy.testing import assert_almost_equal, assert_raises

app = pg.mkQApp()


@pytest.mark.parametrize("lock_aspect", [True, False])
def test_TextItem_setAngle(lock_aspect):
    plt = pg.plot()
    plt.setXRange(-1, 1)
    plt.setYRange(-10, 10)
    plt.setAspectLocked(lock_aspect)
    item = pg.TextItem(text="test")
    plt.addItem(item)

    assert item.transformAngle() == 0.0

    for angle in [10, -30, 90]:
        item.setAngle(angle)
        # negative because transformAngle measures from item to parent
        assert_almost_equal(-item.transformAngle() % 360, angle % 360)
