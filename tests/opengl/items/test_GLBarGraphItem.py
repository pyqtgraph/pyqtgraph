import pytest
pytest.importorskip('OpenGL')

import numpy as np

from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLBarGraphItem

from common import ensure_parentItem


def test_parentItem():
    parent = GLGraphicsItem()
    child = GLBarGraphItem(np.ndarray([0,0,0]), np.ndarray([0,0,0]), parentItem=parent)
    ensure_parentItem(parent, child)
