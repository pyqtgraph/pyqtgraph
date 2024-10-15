import contextlib
import errno
import importlib
import itertools
import os
import platform
import subprocess
import sys
import time
from argparse import Namespace
from collections import namedtuple

import pytest

from pyqtgraph import Qt

if __name__ == "__main__" and (__package__ is None or __package__==''):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import examples
    __package__ = "examples"

from . import utils


def buildFileList(examples, files=None):
    if files is None:
        files = []
    for key, val in examples.items():
        if isinstance(val, dict):
            buildFileList(val, files)
        elif isinstance(val, Namespace):
            files.append((key, val.filename))
        else:
            files.append((key, val))
    return files


path = os.path.abspath(os.path.dirname(__file__))
files = [("Example App", "RunExampleApp.py")]
for ex in [utils.examples_, utils.others]:
    files = buildFileList(ex, files)
files = sorted(set(files))
frontends = {
    Qt.PYQT5: False,
    Qt.PYQT6: False,
    Qt.PYSIDE2: False,
    Qt.PYSIDE6: False,
}
# sort out which of the front ends are available
for frontend in frontends.keys():
    with contextlib.suppress(ImportError):
        importlib.import_module(frontend)
        frontends[frontend] = True

installedFrontends = sorted([
    frontend for frontend, isPresent in frontends.items() if isPresent
])



exceptionCondition = namedtuple("exceptionCondition", ["condition", "reason"])
conditionalExamples = {
    "hdf5.py": exceptionCondition(
        False,
        reason="Example requires user interaction"
    ),
    "jupyter_console_example.py": exceptionCondition(
        importlib.util.find_spec("qtconsole") is not None,
        reason="No need to test with qtconsole not being installed"
    ),
    "RemoteSpeedTest.py": exceptionCondition(
        False,
        reason="Test is being problematic on CI machines"
    ),
}


@pytest.mark.parametrize("frontend, f", [
        pytest.param(
            frontend,
            f,
            marks=pytest.mark.skipif(
                conditionalExamples[f[1]].condition is False,
                reason=conditionalExamples[f[1]].reason
            ) if f[1] in conditionalExamples.keys() else (),
        )
        for frontend, f, in itertools.product(installedFrontends, files)
    ],
    ids=[
        f" {f[1]} - {frontend} " for frontend, f in itertools.product(
            installedFrontends,
            files
        )
    ]
)
def testExamples(frontend, f):
    name, file = f
    global path
    fn = os.path.join(path, file)
    os.chdir(path)
    sys.stdout.write(f"{name}")
    sys.stdout.flush()
    import1 = f"import {frontend}" if frontend != '' else ''
    import2 = os.path.splitext(os.path.split(fn)[1])[0]
    code = """
try:
    {0}
    import faulthandler
    faulthandler.enable()
    import pyqtgraph as pg
    import {1}
    import sys
    print("test complete")
    sys.stdout.flush()
    pg.Qt.QtCore.QTimer.singleShot(1000, pg.Qt.QtWidgets.QApplication.quit)
    pg.exec()
    names = [x for x in dir({1}) if not x.startswith('_')]
    for name in names:
        delattr({1}, name)
except:
    print("test failed")
    raise

""".format(import1, import2)
    env = dict(os.environ)
    example_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.dirname(os.path.dirname(example_dir))
    env['PYTHONPATH'] = f'{path}{os.pathsep}{example_dir}'
    process = subprocess.Popen([sys.executable],
                                stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                text=True,
                                env=env)
    process.stdin.write(code)
    process.stdin.close()

    output = ''
    fail = False
    while True:
        try:
            c = process.stdout.read(1)
        except IOError as err:
            if err.errno == errno.EINTR:
                # Interrupted system call; just try again.
                c = ''
            else:
                raise
        output += c
        if output.endswith('test complete'):
            break
        if output.endswith('test failed'):
            fail = True
            break
    start = time.time()
    killed = False
    while process.poll() is None:
        time.sleep(0.1)
        if time.time() - start > 2.0 and not killed:
            process.kill()
            killed = True

    stdout, stderr = (process.stdout.read(), process.stderr.read())
    process.stdout.close()
    process.stderr.close()

    if (fail or
        'Exception:' in stderr or
        'Error:' in stderr):
        if (not fail 
            and name == "RemoteGraphicsView" 
            and "pyqtgraph.multiprocess.remoteproxy.ClosedError" in stderr):
            # This test can intermittently fail when the subprocess is killed
            return None
        print(stdout)
        print(stderr)
        pytest.fail(
            f"{stdout}\n{stderr}\nFailed {name} Example Test Located in {file}",
            pytrace=False
        )

if __name__ == "__main__":
    pytest.cmdline.main()
