from collections import namedtuple
from pyqtgraph import Qt
import errno
import time
import importlib
import itertools
import pytest
import os, sys
import platform
import subprocess
from argparse import Namespace
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
for ex in [utils.examples, utils.others]:
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
    try:
        importlib.import_module(frontend)
        frontends[frontend] = True
    except ImportError:
        pass

installedFrontends = sorted([
    frontend for frontend, isPresent in frontends.items() if isPresent
])

darwin_opengl_broken = (platform.system() == "Darwin" and
            tuple(map(int, platform.mac_ver()[0].split("."))) >= (10, 16) and
            sys.version_info < (3, 9, 1))

darwin_opengl_reason = ("pyopenGL cannot find openGL library on big sur: "
                        "https://github.com/python/cpython/pull/21241")

exceptionCondition = namedtuple("exceptionCondition", ["condition", "reason"])
conditionalExamples = {
    "hdf5.py": exceptionCondition(
        False,
        reason="Example requires user interaction"
    ),
    "RemoteSpeedTest.py": exceptionCondition(
        False,
        reason="Test is being problematic on CI machines"
    ),
    'GLVolumeItem.py': exceptionCondition(
        not darwin_opengl_broken,
        reason=darwin_opengl_reason
    ),
    'GLIsosurface.py': exceptionCondition(
        not darwin_opengl_broken,
        reason=darwin_opengl_reason
    ),
    'GLSurfacePlot.py': exceptionCondition(
        not darwin_opengl_broken,
        reason=darwin_opengl_reason
    ),
    'GLScatterPlotItem.py': exceptionCondition(
        not darwin_opengl_broken,
        reason=darwin_opengl_reason
    ),
    'GLshaders.py': exceptionCondition(
        not darwin_opengl_broken,
        reason=darwin_opengl_reason
    ),
    'GLLinePlotItem.py': exceptionCondition(
        not darwin_opengl_broken,
        reason=darwin_opengl_reason
    ),
    'GLMeshItem.py': exceptionCondition(
        not darwin_opengl_broken,
        reason=darwin_opengl_reason
    ),
    'GLImageItem.py': exceptionCondition(
        not darwin_opengl_broken,
        reason=darwin_opengl_reason
    ),
    'GLBarGraphItem.py': exceptionCondition(
        not darwin_opengl_broken,
        reason=darwin_opengl_reason
    ),
    'GLViewWidget.py': exceptionCondition(
        not darwin_opengl_broken,
        reason=darwin_opengl_reason
    )
}

@pytest.mark.skipif(
    Qt.QT_LIB == "PySide2"
    and tuple(map(int, Qt.PySide2.__version__.split("."))) >= (5, 14) 
    and tuple(map(int, Qt.PySide2.__version__.split("."))) < (5, 14, 2, 2), 
    reason="new PySide2 doesn't have loadUi functionality"
)
@pytest.mark.parametrize(
    "frontend, f",
    [
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
    ids = [
        " {} - {} ".format(f[1], frontend)
        for frontend, f in itertools.product(
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
    import1 = "import %s" % frontend if frontend != '' else ''
    import2 = os.path.splitext(os.path.split(fn)[1])[0]
    code = """
try:
    {0}
    import initExample
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
    process = subprocess.Popen([sys.executable],
                                stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                text=True)
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
