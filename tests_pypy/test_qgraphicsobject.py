from pyqtgraph.Qt import QtWidgets

class GraphicsObject(QtWidgets.QGraphicsObject):
    def itemChange(self, change, value):
        return QtWidgets.QGraphicsObject.itemChange(self, change, value)

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication([])

def test_qgraphicsobject():
    scene = QtWidgets.QGraphicsScene()

    parent = GraphicsObject()
    child = GraphicsObject(parent)
    scene.addItem(parent)
    items = scene.items()

    assert len(items) == 2

    for item in items:
        print(item)

    # on PyPy, we get QGraphicsItem instead of GraphicsObject
    assert child in items
