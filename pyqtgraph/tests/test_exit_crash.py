# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import tempfile
import pyqtgraph as pg
import six
import pytest
import textwrap
import time

code = """
import sys
sys.path.insert(0, '{path}')
import pyqtgraph as pg
app = pg.mkQApp()
w = pg.{classname}({args})
"""

skipmessage = ('unclear why this test is failing. skipping until someone has'
               ' time to fix it')


def call_with_timeout(*args, **kwargs):
    """Mimic subprocess.call with timeout for python < 3.3"""
    wait_per_poll = 0.1
    try:
        timeout = kwargs.pop('timeout')
    except KeyError:
        timeout = 10

    rc = None
    p = subprocess.Popen(*args, **kwargs)
    assert int(timeout/wait_per_poll) >= 1, (
        "{timeout}/{wait_per_poll}={timeout/wait_per_poll}"
    )
    for i in range(int(timeout/wait_per_poll)):
        rc = p.poll()
        if rc is not None:
            break
        time.sleep(wait_per_poll)
    return rc


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
        with open(tmp, 'w') as f:
            f.write(code.format(path=path, classname=name, args=argstr))
        proc = subprocess.Popen([sys.executable, tmp])
        assert proc.wait() == 0

    os.remove(tmp)


def test_pg_exit():
    # test the pg.exit() function
    code = textwrap.dedent("""
        import pyqtgraph as pg
        app = pg.mkQApp()
        pg.plot()
        pg.exit()
    """)
    rc = call_with_timeout([sys.executable, '-c', code], timeout=5, shell=False)
    assert rc == 0
