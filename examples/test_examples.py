# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from pyqtgraph import Qt
from . import utils
from collections import namedtuple
import errno
import importlib
import itertools
import pytest
import os, sys
import subprocess
import time


path = os.path.abspath(os.path.dirname(__file__))

# printing on travis ci frequently leads to "interrupted system call" errors.
# as a workaround, we overwrite the built-in print function (bleh)
if os.getenv('TRAVIS') is not None:
    if sys.version_info[0] < 3:
        import __builtin__ as builtins
    else:
        import builtins

    def flaky_print(*args):
        """Wrapper for print that retries in case of IOError.
        """
        count = 0
        while count < 5:
            count += 1
            try:
                orig_print(*args)
                break
            except IOError:
                if count >= 5:
                    raise
                pass
    orig_print = builtins.print
    builtins.print = flaky_print
    print("Installed wrapper for flaky print.")


files = sorted(set(utils.buildFileList(utils.examples)))
frontends = {
    Qt.PYQT4: False,
    Qt.PYQT5: False,
    Qt.PYSIDE: False,
    Qt.PYSIDE2: False
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
    "optics_demos.py": exceptionCondition(
        not frontends[Qt.PYSIDE],
        reason=(
            "Test fails due to PySide bug: ",
            "https://bugreports.qt.io/browse/PYSIDE-671"
        )
    ),
    'GLVolumeItem.py': exceptionCondition(
        not(sys.platform == "darwin" and
            sys.version_info[0] == 2 and
            (frontends[Qt.PYQT4] or frontends[Qt.PYSIDE])),
        reason=(
            "glClear does not work on macOS + Python2.7 + Qt4: ",
            "https://github.com/pyqtgraph/pyqtgraph/issues/939"
        )
    ),
    'GLIsosurface.py': exceptionCondition(
        not(sys.platform == "darwin" and
            sys.version_info[0] == 2 and
            (frontends[Qt.PYQT4] or frontends[Qt.PYSIDE])),
        reason=(
            "glClear does not work on macOS + Python2.7 + Qt4: ",
            "https://github.com/pyqtgraph/pyqtgraph/issues/939"
        )
    ),
    'GLSurfacePlot.py': exceptionCondition(
        not(sys.platform == "darwin" and
            sys.version_info[0] == 2 and
            (frontends[Qt.PYQT4] or frontends[Qt.PYSIDE])),
        reason=(
            "glClear does not work on macOS + Python2.7 + Qt4: ",
            "https://github.com/pyqtgraph/pyqtgraph/issues/939"
        )
    ),
    'GLScatterPlotItem.py': exceptionCondition(
        not(sys.platform == "darwin" and
            sys.version_info[0] == 2 and
            (frontends[Qt.PYQT4] or frontends[Qt.PYSIDE])),
        reason=(
            "glClear does not work on macOS + Python2.7 + Qt4: ",
            "https://github.com/pyqtgraph/pyqtgraph/issues/939"
        )
    ),
    'GLshaders.py': exceptionCondition(
        not(sys.platform == "darwin" and
            sys.version_info[0] == 2 and
            (frontends[Qt.PYQT4] or frontends[Qt.PYSIDE])),
        reason=(
            "glClear does not work on macOS + Python2.7 + Qt4: ",
            "https://github.com/pyqtgraph/pyqtgraph/issues/939"
        )
    ),
    'GLLinePlotItem.py': exceptionCondition(
        not(sys.platform == "darwin" and
            sys.version_info[0] == 2 and
            (frontends[Qt.PYQT4] or frontends[Qt.PYSIDE])),
        reason=(
            "glClear does not work on macOS + Python2.7 + Qt4: ",
            "https://github.com/pyqtgraph/pyqtgraph/issues/939"
        )
    ),
    'GLMeshItem.py': exceptionCondition(
        not(sys.platform == "darwin" and
            sys.version_info[0] == 2 and
            (frontends[Qt.PYQT4] or frontends[Qt.PYSIDE])),
        reason=(
            "glClear does not work on macOS + Python2.7 + Qt4: ",
            "https://github.com/pyqtgraph/pyqtgraph/issues/939"
        )
    ),
    'GLImageItem.py': exceptionCondition(
        not(sys.platform == "darwin" and
            sys.version_info[0] == 2 and
            (frontends[Qt.PYQT4] or frontends[Qt.PYSIDE])),
        reason=(
            "glClear does not work on macOS + Python2.7 + Qt4: ",
            "https://github.com/pyqtgraph/pyqtgraph/issues/939"
        )
    )
}

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
def testExamples(frontend, f, graphicsSystem=None):
    # runExampleFile(f[0], f[1], sys.executable, frontend)

    name, file = f
    global path
    fn = os.path.join(path, file)
    os.chdir(path)
    sys.stdout.write("{} ".format(name))
    sys.stdout.flush()
    import1 = "import %s" % frontend if frontend != '' else ''
    import2 = os.path.splitext(os.path.split(fn)[1])[0]
    graphicsSystem = (
        '' if graphicsSystem is None else  "pg.QtGui.QApplication.setGraphicsSystem('%s')" % graphicsSystem
    )
    code = """
try:
    %s
    import initExample
    import pyqtgraph as pg
    %s
    import %s
    import sys
    print("test complete")
    sys.stdout.flush()
    import time
    while True:  ## run a little event loop
        pg.QtGui.QApplication.processEvents()
        time.sleep(0.01)
except:
    print("test failed")
    raise

""" % (import1, graphicsSystem, import2)
    if sys.platform.startswith('win'):
        process = subprocess.Popen([sys.executable],
                                    stdin=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    stdout=subprocess.PIPE)
    else:
        process = subprocess.Popen(['exec %s -i' % (sys.executable)],
                                   shell=True,
                                   stdin=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
    process.stdin.write(code.encode('UTF-8'))
    process.stdin.close()
    output = ''
    fail = False
    while True:
        try:
            c = process.stdout.read(1).decode()
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
    time.sleep(1)
    process.kill()
    #res = process.communicate()
    res = (process.stdout.read(), process.stderr.read())
    if (fail or
        'exception' in res[1].decode().lower() or
        'error' in res[1].decode().lower()):
        print(res[0].decode())
        print(res[1].decode())
        pytest.fail("{}\n{}\nFailed {} Example Test Located in {} "
            .format(res[0].decode(), res[1].decode(), name, file),
            pytrace=False)

if __name__ == "__main__":
    pytest.cmdline.main()
