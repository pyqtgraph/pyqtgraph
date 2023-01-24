from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem


def ensure_parentItem(parent: GLGraphicsItem, child: GLGraphicsItem):
    assert child in parent.childItems()
    assert parent is child.parentItem()
