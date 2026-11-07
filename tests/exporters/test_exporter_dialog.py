from types import SimpleNamespace

import pyqtgraph as pg

app = pg.mkQApp()


def test_export_dialog():
    plt = pg.PlotWidget()
    y1 = [1, 3, 2, 3, 1, 6, 9, 8, 4, 2]
    plt.plot(y=y1)
    plt.show()

    scene = plt.scene()
    assert scene.exportDialog is None

    # Exercise the export action without opening a native popup, which needs
    # a compositor input serial on Wayland rather than a synthetic mouse event.
    event = SimpleNamespace(acceptedItem=plt.getViewBox())
    export_action = scene.getContextMenus(event)[0]
    export_action.trigger()

    dialog = scene.exportDialog
    assert dialog.isVisible()
    assert dialog.ui.itemTree.currentItem().gitem is plt.plotItem

    dialog.close()
    assert not dialog.isVisible()
    export_action.trigger()
    assert scene.exportDialog is dialog
    assert dialog.isVisible()
    dialog.close()
    app.processEvents()
    plt.close()
