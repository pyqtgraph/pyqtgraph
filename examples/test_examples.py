# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from collections import namedtuple
from pyqtgraph import Qt
from pyqtgraph.python2_3 import basestring
from pyqtgraph.pgcollections import OrderedDict
from .ExampleApp import examples

import errno
import importlib
import itertools
import pytest
import os, sys
import platform
import subprocess
import time
from argparse import Namespace
if __name__ == "__main__" and (__package__ is None or __package__==''):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import examples
    __package__ = "examples"


def buildFileList(examples, files=None):
    if files is None:
        files = [("Example App", "test_ExampleApp.py")]
    for key, val in examples.items():
        if isinstance(val, OrderedDict):
            buildFileList(val, files)
        elif isinstance(val, Namespace):
            files.append((key, val.filename))
        else:
            files.append((key,val))
    return files



path = os.path.abspath(os.path.dirname(__file__))
files = sorted(set(buildFileList(examples)))
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

exceptionCondition = namedtuple("exceptionCondition", ["condition", "reason"])
conditionalExamples = {
    "test_ExampleApp.py": exceptionCondition(
        not(platform.system() == "Linux" and frontends[Qt.PYSIDE2]),
        reason="Unexplained, intermittent segfault and subsequent timeout on CI"
    ),
    "hdf5.py": exceptionCondition(
        False,
        reason="Example requires user interaction"
    ),
    "RemoteSpeedTest.py": exceptionCondition(
        False,
        reason="Test is being problematic on CI machines"
    ),
    "RemoteGraphicsView.py": exceptionCondition(
        not(platform.system() == "Darwin"),
        reason="FileNotFoundError for pyqtgraph_shmem_* file"
    ),
    "ProgressDialog.py": exceptionCondition(
        not(platform.system() == "Linux"),
        reason="QXcbConnection: XCB error"
    ),
    'GLVolumeItem.py': exceptionCondition(
        not(platform.system() == "Darwin" and
            tuple(map(int, platform.mac_ver()[0].split("."))) >= (10, 16) and 
            (sys.version_info <= (3, 8, 7) or
            (sys.version_info >= (3, 9) and sys.version_info < (3, 9, 1)))),
        reason=(
            "pyopenGL cannot find openGL libray on big sur: "
            "https://github.com/python/cpython/pull/21241"
        )
    ),
    'GLIsosurface.py': exceptionCondition(
        not(platform.system() == "Darwin" and
            tuple(map(int, platform.mac_ver()[0].split("."))) >= (10, 16) and 
            (sys.version_info <= (3, 8, 7) or
            (sys.version_info >= (3, 9) and sys.version_info < (3, 9, 1)))),
        reason=(
            "pyopenGL cannot find openGL libray on big sur: "
            "https://github.com/python/cpython/pull/21241"
        )
    ),
    'GLSurfacePlot.py': exceptionCondition(
        not(platform.system() == "Darwin" and
            tuple(map(int, platform.mac_ver()[0].split("."))) >= (10, 16) and 
            (sys.version_info <= (3, 8, 7) or
            (sys.version_info >= (3, 9) and sys.version_info < (3, 9, 1)))),
        reason=(
            "pyopenGL cannot find openGL libray on big sur: "
            "https://github.com/python/cpython/pull/21241"
        )
    ),
    'GLScatterPlotItem.py': exceptionCondition(
        not(platform.system() == "Darwin" and
            tuple(map(int, platform.mac_ver()[0].split("."))) >= (10, 16) and 
            (sys.version_info <= (3, 8, 7) or
            (sys.version_info >= (3, 9) and sys.version_info < (3, 9, 1)))),
        reason=(
            "pyopenGL cannot find openGL libray on big sur: "
            "https://github.com/python/cpython/pull/21241"
        )
    ),
    'GLshaders.py': exceptionCondition(
        not(platform.system() == "Darwin" and
            tuple(map(int, platform.mac_ver()[0].split("."))) >= (10, 16) and 
            (sys.version_info <= (3, 8, 7) or
            (sys.version_info >= (3, 9) and sys.version_info < (3, 9, 1)))),
        reason=(
            "pyopenGL cannot find openGL libray on big sur: "
            "https://github.com/python/cpython/pull/21241"
        )
    ),
    'GLLinePlotItem.py': exceptionCondition(
        not(platform.system() == "Darwin" and
            tuple(map(int, platform.mac_ver()[0].split("."))) >= (10, 16) and 
            (sys.version_info <= (3, 8, 7) or
            (sys.version_info >= (3, 9) and sys.version_info < (3, 9, 1)))),
        reason=(
            "pyopenGL cannot find openGL libray on big sur: "
            "https://github.com/python/cpython/pull/21241"
        )
    ),
    'GLMeshItem.py': exceptionCondition(
        not(platform.system() == "Darwin" and
            tuple(map(int, platform.mac_ver()[0].split("."))) >= (10, 16) and 
            (sys.version_info <= (3, 8, 7) or
            (sys.version_info >= (3, 9) and sys.version_info < (3, 9, 1)))),
        reason=(
            "pyopenGL cannot find openGL libray on big sur: "
            "https://github.com/python/cpython/pull/21241"
        )
    ),
    'GLImageItem.py': exceptionCondition(
        not(platform.system() == "Darwin" and
            tuple(map(int, platform.mac_ver()[0].split("."))) >= (10, 16) and 
            (sys.version_info <= (3, 8, 7) or
            (sys.version_info >= (3, 9) and sys.version_info < (3, 9, 1)))),
        reason=(
            "pyopenGL cannot find openGL libray on big sur: "
            "https://github.com/python/cpython/pull/21241"
        )
    ),
    'GLBarGraphItem.py': exceptionCondition(
        not(platform.system() == "Darwin" and
            tuple(map(int, platform.mac_ver()[0].split("."))) >= (10, 16) and
            (sys.version_info <= (3, 8, 7) or
            (sys.version_info >= (3, 9) and sys.version_info < (3, 9, 1)))),
        reason=(
            "pyopenGL cannot find openGL libray on big sur: "
            "https://github.com/python/cpython/pull/21241"
        )
    ),
    'GLViewWidget.py': exceptionCondition(
        not(platform.system() == "Darwin" and
            tuple(map(int, platform.mac_ver()[0].split("."))) >= (10, 16) and
            (sys.version_info <= (3, 8, 7) or
            (sys.version_info >= (3, 9) and sys.version_info < (3, 9, 1)))),
        reason=(
            "pyopenGL cannot find openGL libray on big sur: "
            "https://github.com/python/cpython/pull/21241"
        )
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
