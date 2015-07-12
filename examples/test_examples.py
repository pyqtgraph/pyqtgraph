from __future__ import print_function, division, absolute_import
from pyqtgraph import Qt
from . import utils

files = utils.buildFileList(utils.examples)

import pytest


@pytest.mark.parametrize("f", files)
def test_examples(f):
    # Test the examples with whatever the current QT_LIB front
    # end is
    utils.testFile(f[0], f[1], utils.sys.executable, Qt.QT_LIB)
