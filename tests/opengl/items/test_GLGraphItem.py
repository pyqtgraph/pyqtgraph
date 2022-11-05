import pytest
pytest.importorskip('OpenGL')

from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLGraphItem

from common import ensure_parentItem


def test_parentItem():
    parent = GLGraphicsItem()
    child = GLGraphItem(parentItem=parent)
    ensure_parentItem(parent, child)
