# Build with `python setup.py build_exe`
from cx_Freeze import setup, Executable
from pathlib import Path

import shutil
from glob import glob
# Remove the build folder
shutil.rmtree("build", ignore_errors=True)
shutil.rmtree("dist", ignore_errors=True)
import sys

includes = ['pyqtgraph.graphicsItems',
            'numpy', 'atexit']
excludes = ['cvxopt','_gtkagg', '_tkagg', 'bsddb', 'curses', 'email', 'pywin.debugger',
    'pywin.debugger.dbgcon', 'pywin.dialogs', 'tcl','tables',
    'Tkconstants', 'Tkinter', 'zmq','PySide','pysideuic','scipy','matplotlib']

# Workaround for making sure the templates are included in the frozen app package
include_files = []
import pyqtgraph
pg_folder = Path(pyqtgraph.__file__).parent
templates = pg_folder.rglob('*template*.py')
sources = [str(w) for w in templates]
destinations = ['lib' + w.replace(str(pg_folder.parent), '') for w in sources]
for a in zip(sources, destinations):
    include_files.append(a)

print(include_files)

if sys.version[0] == '2':
    # causes syntax error on py2
    excludes.append('PyQt4.uic.port_v3')

base = None
if sys.platform == "win32":
    base = "Win32GUI"

build_exe_options = {'excludes': excludes,
    'includes':includes, 'include_msvcr':True,
    'optimize':1, "include_files": include_files,}

setup(name = "cx_freeze plot test",
      version = "0.2",
      description = "cx_freeze plot test",
      options = {"build_exe": build_exe_options},
      executables = [Executable("plotTest.py", base=base)])


