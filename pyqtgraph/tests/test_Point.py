import pytest
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import math

angles = [
	((1, 0), (0, 1), 90),
	((0, 1), (1, 0), -90),
	((-1, 0), (-1, 0), 0),
	((0, -1), (0, 1), 180),
]
@pytest.mark.parametrize("p1, p2, angle", angles)
def test_Point_angle(p1, p2, angle):
	p1 = pg.Point(*p1)
	p2 = pg.Point(*p2)
	assert p2.angle(p1) == angle


inits = [
	(QtCore.QSizeF(1, 0), (1.0, 0.0)),
	((0, -1), (0.0, -1.0)),
	([1, 1], (1.0, 1.0)),
]
@pytest.mark.parametrize("initArgs, positions", inits)
def test_Point_init(initArgs, positions):
	if isinstance(initArgs, QtCore.QSizeF):
		point = pg.Point(initArgs)
	else:
		point = pg.Point(*initArgs)
	assert (point.x(), point.y()) == positions

lengths = [
	((0, 1), 1),
	((1, 0), 1),
	((0, 0), 0),
	((1, 1), math.sqrt(2)),
	((-1, -1), math.sqrt(2))
]
@pytest.mark.parametrize("initArgs, length", lengths)
def test_Point_length(initArgs, length):
	point = pg.Point(initArgs)
	assert point.length() == length

min_max = [
	((0, 1), 0, 1),
	((1, 0), 0, 1),
	((-math.inf, 0), -math.inf, 0),
	((0, math.inf), 0, math.inf)
]
@pytest.mark.parametrize("initArgs, min_, max_", min_max)
def test_Point_min_max(initArgs, min_, max_):
	point = pg.Point(initArgs)
	assert min(point) == min_
	assert max(point) == max_

projections = [
	((0, 1), (1, 0), (1, 1))
]
@pytest.mark.parametrize("p1_arg, p2_arg, projection", projections)
def test_Point_projection(p1_arg, p2_arg, projection):
	p1 = pg.Point(p1_arg)
	p2 = pg.Point(p2_arg)
	p1.proj(p2) == projection