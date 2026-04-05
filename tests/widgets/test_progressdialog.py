import pyqtgraph as pg

pg.mkQApp()

def test_progress_dialog():
    with pg.ProgressDialog("test", 0, 1) as dlg:
        dlg += 1
