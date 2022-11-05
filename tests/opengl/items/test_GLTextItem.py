import pytest
pytest.importorskip('OpenGL')

from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLTextItem

from common import ensure_parentItem


def test_parentItem():
    parent = GLGraphicsItem()
    child = GLTextItem(parentItem=parent)
    ensure_parentItem(parent, child)
