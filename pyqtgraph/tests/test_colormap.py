import pyqtgraph as pg
from numpy.testing import assert_equal

POS = [0.0, 1.0]
COLORS = [[255, 255, 255], [0, 0, 0]]


def test_colormap():
    cmap = pg.ColorMap(POS, COLORS)

    nPts = 4
    lutB = cmap.getLookupTable(nPts=nPts, alpha=False, mode='byte')
    lutC = cmap.getLookupTable(nPts=nPts, alpha=False, mode='qcolor')

    for i in range(nPts):
        b = lutB[i]
        c = lutC[i]
        assert_equal(b[0], c.red())
        assert_equal(b[1], c.green())
        assert_equal(b[2], c.blue())


if __name__ == '__main__':
    test_colormap()
