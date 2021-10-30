from pyqtgraph import mkQApp, ProgressDialog

mkQApp()


def test_progress_dialog():
    with ProgressDialog("test", 0, 1) as dlg:
        dlg += 1
