from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLGradientLegendItem

from common import ensure_parentItem


def test_parentItem():
    parent = GLGraphicsItem()
    child = GLGradientLegendItem(parentItem=parent)
    ensure_parentItem(parent, child)
