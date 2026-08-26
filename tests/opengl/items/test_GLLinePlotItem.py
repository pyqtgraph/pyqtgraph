from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLLinePlotItem

from common import ensure_parentItem


def test_parentItem():
    parent = GLGraphicsItem()
    child = GLLinePlotItem(parentItem=parent)
    ensure_parentItem(parent, child)
