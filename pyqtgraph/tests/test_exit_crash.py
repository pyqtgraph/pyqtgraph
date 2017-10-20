import os, sys, subprocess, tempfile
import pyqtgraph as pg
import pytest
from pyqtgraph.util import six

code = """
import sys
sys.path.insert(0, '{path}')
import pyqtgraph as pg
app = pg.mkQApp()
w = pg.{classname}({args})
"""

skipmessage = ('unclear why this test is failing. skipping until someone has'
               ' time to fix it')

@pytest.mark.skipif(True, reason=skipmessage)
def test_exit_crash():
    # For each Widget subclass, run a simple python script that creates an
    # instance and then shuts down. The intent is to check for segmentation
    # faults when each script exits.
    tmp = tempfile.mktemp(".py")
    path = os.path.dirname(pg.__file__)

    initArgs = {
        'CheckTable': "[]",
        'ProgressDialog': '"msg"',
        'VerticalLabel': '"msg"',
    }

    for name in dir(pg):
        obj = getattr(pg, name)
        if not isinstance(obj, type) or not issubclass(obj, pg.QtGui.QWidget):
            continue

        print(name)
        argstr = initArgs.get(name, "")
        open(tmp, 'w').write(code.format(path=path, classname=name, args=argstr))
        proc = subprocess.Popen([sys.executable, tmp])
        assert proc.wait() == 0

    os.remove(tmp)
