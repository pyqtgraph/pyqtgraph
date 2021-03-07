from pyqtgraph.Qt import QtCore
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg


def test_basics_graphics_view():
    app = pg.mkQApp()
    view = pg.GraphicsView()
    background_role = view.backgroundRole()
    assert background_role == QtGui.QPalette.Window

    assert view.backgroundBrush().color() == QtGui.QColor(0, 0, 0, 255)

    assert view.focusPolicy() == QtCore.Qt.StrongFocus
    assert view.transformationAnchor() == QtGui.QGraphicsView.NoAnchor
    minimal_update = QtGui.QGraphicsView.MinimalViewportUpdate
    assert view.viewportUpdateMode() == minimal_update
    assert view.frameShape() == QtGui.QFrame.NoFrame
    assert view.hasMouseTracking() is True

    # Default properties
    # --------------------------------------

    assert view.mouseEnabled is False
    assert view.aspectLocked is False
    assert view.autoPixelRange is True
    assert view.scaleCenter is False
    assert view.clickAccepted is False
    assert view.centralWidget is not None
    # assert view._background == "default"
    assert view._background == "gr_bg"

    # Set background color
    # --------------------------------------
    view.setBackground("w")
    assert view._background == "w"
    # assert view.backgroundBrush().color() == QtCore.Qt.white
    assert view.backgroundBrush().color().name() == '#ffffff' #QtCore.Qt.white

    # Set anti aliasing
    # --------------------------------------
    aliasing = QtGui.QPainter.Antialiasing
    # Default is set to `False`
    assert not view.renderHints() & aliasing == aliasing
    view.setAntialiasing(True)
    assert view.renderHints() & aliasing == aliasing
    view.setAntialiasing(False)
    assert not view.renderHints() & aliasing == aliasing

    # Enable mouse
    # --------------------------------------
    view.enableMouse(True)
    assert view.mouseEnabled is True
    assert view.autoPixelRange is False
    view.enableMouse(False)
    assert view.mouseEnabled is False
    assert view.autoPixelRange is True

    # Add and remove item
    # --------------------------------------
    central_item = QtGui.QGraphicsWidget()
    view.setCentralItem(central_item)
    assert view.centralWidget is central_item
    # XXX: Removal of central item is not clear in code
    scene = view.sceneObj
    assert isinstance(scene, pg.GraphicsScene)
    assert central_item in scene.items()

    item = QtGui.QGraphicsWidget()
    assert item not in scene.items()
    view.addItem(item)
    assert item in scene.items()
    view.removeItem(item)
    assert item not in scene.items()

    # Close the graphics view
    # --------------------------------------

    view.close()
    assert view.centralWidget is None
    assert view.currentItem is None
    assert view.sceneObj is None
    assert view.closed is True
