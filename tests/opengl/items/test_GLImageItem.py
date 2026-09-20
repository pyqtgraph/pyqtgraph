from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLImageItem

from common import ensure_parentItem


def test_parentItem():
    parent = GLGraphicsItem()
    child = GLImageItem(None, parentItem=parent)
    ensure_parentItem(parent, child)
