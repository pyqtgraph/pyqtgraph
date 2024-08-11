import os
import sys

import pytest
from pyqtgraph.Qt import QtCore

QtCore.QLoggingCategory.setFilterRules("qt.pyside.libpyside.warning=true")


@pytest.fixture
def tmp_module(tmp_path):
    module_path = os.fsdecode(tmp_path)
    sys.path.insert(0, module_path)
    yield module_path
    sys.path.remove(module_path)
