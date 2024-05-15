import pyqtgraph as pg

from tests.ui_testing import mouseClick


app = pg.mkQApp()


def test_export_dialog():
    plt = pg.PlotWidget()
    y1 = [1,3,2,3,1,6,9,8,4,2]
    plt.plot(y=y1)
    plt.show()

    # # export dialog doesn't exist
    assert plt.scene().exportDialog is None      
    mouseClick(
        plt,
        pos=pg.Qt.QtCore.QPointF(plt.mapFromGlobal(plt.geometry().center())),
        button=pg.Qt.QtCore.Qt.MouseButton.RightButton
    )
    plt.scene().contextMenu[0].trigger() # show show dialog
    plt.scene().showExportDialog()
    assert plt.scene().exportDialog.isVisible()
    plt.scene().exportDialog.close()
    app.processEvents()
    plt.close()
