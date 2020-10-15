import pyqtgraph as pg
import numpy as np

app = pg.mkQApp()


def test_ErrorBarItem_defer_data():
    plot = pg.PlotWidget()
    plot.show()

    # plot some data away from the origin to set the view rect
    x = np.arange(5) + 10
    curve = pg.PlotCurveItem(x=x, y=x)
    plot.addItem(curve)
    app.processEvents()
    r_no_ebi = plot.viewRect()

    # ErrorBarItem with no data shouldn't affect the view rect
    err = pg.ErrorBarItem()
    plot.addItem(err)
    app.processEvents()
    r_empty_ebi = plot.viewRect()

    assert r_no_ebi.height() == r_empty_ebi.height()

    err.setData(x=x, y=x, bottom=x, top=x)
    app.processEvents()
    r_ebi = plot.viewRect()

    assert r_ebi.height() > r_empty_ebi.height()

    # unset data, ErrorBarItem disappears and view rect goes back to original
    err.setData(x=None, y=None)
    app.processEvents()
    r_clear_ebi = plot.viewRect()

    assert r_clear_ebi.height() == r_empty_ebi.height()

    plot.close()
