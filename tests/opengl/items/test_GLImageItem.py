import pytest
pytest.importorskip('OpenGL')

from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLAxisItem

from common import ensure_parentItem


def test_parentItem():
    parent = GLGraphicsItem()
    child = GLAxisItem(parentItem=parent)
    ensure_parentItem(parent, child)
