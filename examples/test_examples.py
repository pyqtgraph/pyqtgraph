from __future__ import print_function, division, absolute_import
from pyqtgraph import Qt
from . import utils
import errno
import importlib
import itertools
import pkgutil
import pytest
import os, sys
import subprocess
import time


path = os.path.abspath(os.path.dirname(__file__))


def runExampleFile(name, f, exe, lib, graphicsSystem=None):
    global path
    fn = os.path.join(path,f)
    os.chdir(path)
    sys.stdout.write("{} ".format(name))
    sys.stdout.flush()
    import1 = "import %s" % lib if lib != '' else ''
    import2 = os.path.splitext(os.path.split(fn)[1])[0]
    graphicsSystem = '' if graphicsSystem is None else "pg.QtGui.QApplication.setGraphicsSystem('%s')" % graphicsSystem
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
        process = subprocess.Popen([exe],
                                    stdin=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    stdout=subprocess.PIPE)
    else:
        process = subprocess.Popen(['exec %s -i' % (exe)],
                                   shell=True,
                                   stdin=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
    process.stdin.write(code.encode('UTF-8'))
    process.stdin.close() ##?
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
    if fail or 'exception' in res[1].decode().lower() or 'error' in res[1].decode().lower():
        print(res[0].decode())
        print(res[1].decode())
        return False
    return True


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


files = utils.buildFileList(utils.examples)
frontends = {Qt.PYQT4: False, Qt.PYQT5: False, Qt.PYSIDE: False, Qt.PYSIDE2: False}
# sort out which of the front ends are available
for frontend in frontends.keys():
    try:
        importlib.import_module(frontend)
        frontends[frontend] = True
    except ImportError:
        pass
    except ModuleNotFoundError:
        pass

installed = sorted([frontend for frontend, isPresent in frontends.items() if isPresent])

# keep a dictionary of example files and their non-standard dependencies
specialExamples = {
    "hdf5.py": ["h5py"]
}


@pytest.mark.parametrize(
	"frontend, f",
	[
		pytest.param(
			frontend,
            f,
			marks=pytest.mark.skipif(any(pkgutil.find_loader(pkg) is None for pkg in specialExamples[f[1]]),
                                     reason="Skipping Example for Missing Dependencies") if f[1] in specialExamples.keys() else (),
		)
		for frontend, f, in itertools.product(installed, files)
	],
    ids = [" {} - {} ".format(f[1], frontend) for frontend, f in itertools.product(installed, files)]
)
def testExamples(frontend, f):
    assert runExampleFile(f[0], f[1], sys.executable, frontend)

if __name__ == "__main__":
    pytest.cmdline.main()
