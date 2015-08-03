from __future__ import print_function, division, absolute_import
from pyqtgraph.util import six
from pyqtgraph import Qt
from examples import utils
import importlib
import itertools
import pytest

files = utils.buildFileList(utils.examples)

frontends = {Qt.PYQT4: False, Qt.PYSIDE: False}
# frontends = {Qt.PYQT4: False, Qt.PYQT5: False, Qt.PYSIDE: False}

# sort out which of the front ends are available
for frontend in frontends.keys():
    try:
        importlib.import_module(frontend)
        frontends[frontend] = True
    except ImportError:
        pass


@pytest.mark.parametrize(
    "frontend, f", itertools.product(sorted(list(six.iterkeys(frontends))), files))
def test_examples(frontend, f):
    # Test the examples with all available front-ends
    print('frontend = %s. f = %s' % (frontend, f))
    if not frontends[frontend]:
        pytest.skip('{} is not installed. Skipping tests'.format(frontend))
    utils.testFile(f[0], f[1], utils.sys.executable, frontend)

if __name__ == "__main__":
    pytest.cmdline.main()
