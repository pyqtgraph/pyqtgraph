import pytest
import sys

import pyqtgraph as pg

pg.mkQApp()

@pytest.mark.skipif(
    sys.platform.startswith("linux")
    and pg.Qt.QT_LIB == "PySide6"
    and pg.Qt.PySide6.__version_info__ > (6, 3),
    reason="taking gui thread causes segfault"
)
def test_progress_dialog():
    with pg.ProgressDialog("test", 0, 1) as dlg:
        dlg += 1
