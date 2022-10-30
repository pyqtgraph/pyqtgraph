from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLMeshItem

from common import ensure_parentItem


def test_parentItem():
    parent = GLGraphicsItem()
    child = GLMeshItem(parentItem=parent)
    ensure_parentItem(parent, child)
