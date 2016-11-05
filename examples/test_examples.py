from __future__ import print_function, division, absolute_import
from pyqtgraph import Qt
from . import utils
import itertools
import pytest

# apparently importlib does not exist in python 2.6...
try:
    import importlib
except ImportError:
    # we are on python 2.6
    print("If you want to test the examples, please install importlib from "
          "pypi\n\npip install importlib\n\n")
    pass

files = utils.buildFileList(utils.examples)
frontends = {Qt.PYQT4: False, Qt.PYSIDE: False}
# sort out which of the front ends are available
for frontend in frontends.keys():
    try:
        importlib.import_module(frontend)
        frontends[frontend] = True
    except ImportError:
        pass


@pytest.mark.parametrize(
    "frontend, f", itertools.product(sorted(list(frontends.keys())), files))
def test_examples(frontend, f):
    # Test the examples with all available front-ends
    print('frontend = %s. f = %s' % (frontend, f))
    if not frontends[frontend]:
        pytest.skip('%s is not installed. Skipping tests' % frontend)
    utils.testFile(f[0], f[1], utils.sys.executable, frontend)

if __name__ == "__main__":
    pytest.cmdline.main()
