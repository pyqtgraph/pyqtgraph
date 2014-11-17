# Build with `python setup.py build_exe`
from cx_Freeze import setup, Executable

import shutil
from glob import glob
# Remove the build folder
shutil.rmtree("build", ignore_errors=True)
shutil.rmtree("dist", ignore_errors=True)
import sys

includes = ['PyQt4.QtCore', 'PyQt4.QtGui', 'sip', 'pyqtgraph.graphicsItems',
            'numpy', 'atexit']
excludes = ['cvxopt','_gtkagg', '_tkagg', 'bsddb', 'curses', 'email', 'pywin.debugger',
    'pywin.debugger.dbgcon', 'pywin.dialogs', 'tcl','tables',
    'Tkconstants', 'Tkinter', 'zmq','PySide','pysideuic','scipy','matplotlib']

if sys.version[0] == '2':
    # causes syntax error on py2
    excludes.append('PyQt4.uic.port_v3')

base = None
if sys.platform == "win32":
    base = "Win32GUI"

build_exe_options = {'excludes': excludes,
    'includes':includes, 'include_msvcr':True,
    'compressed':True, 'copy_dependent_files':True, 'create_shared_zip':True,
    'include_in_shared_zip':True, 'optimize':2}

setup(name = "cx_freeze plot test",
      version = "0.1",
      description = "cx_freeze plot test",
      options = {"build_exe": build_exe_options},
      executables = [Executable("plotTest.py", base=base)])


