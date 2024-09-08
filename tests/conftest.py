import os
import sys

import pytest

from pyqtgraph.Qt import QtCore

try:
    QtCore.QLoggingCategory.setFilterRules("qt.pyside.libpyside.warning=true")
except AttributeError:
    pass  # PySide2 does not expose `QLoggingCategory` although Qt5 has it


@pytest.fixture
def tmp_module(tmp_path):
    module_path = os.fsdecode(tmp_path)
    sys.path.insert(0, module_path)
    yield module_path
    sys.path.remove(module_path)
