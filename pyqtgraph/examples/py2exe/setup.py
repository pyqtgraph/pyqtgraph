import shutil
from distutils.core import setup

# Remove the build folder
shutil.rmtree("build", ignore_errors=True)
shutil.rmtree("dist", ignore_errors=True)
import sys

includes = ['PyQt4', 'PyQt4.QtGui', 'PyQt4.QtSvg', 'sip', 'pyqtgraph.graphicsItems']
excludes = ['_gtkagg', '_tkagg', 'bsddb', 'curses', 'email', 'pywin.debugger',
            'pywin.debugger.dbgcon', 'pywin.dialogs', 'tcl',
            'Tkconstants', 'Tkinter', 'zmq']
if sys.version[0] == '2':
    # causes syntax error on py2
    excludes.append('PyQt4.uic.port_v3')

packages = []
dll_excludes = ['libgdk-win32-2.0-0.dll', 'libgobject-2.0-0.dll', 'tcl84.dll',
                'tk84.dll', 'MSVCP90.dll']
icon_resources = []
bitmap_resources = []
other_resources = []
data_files = []
setup(
  data_files=data_files,
  console=['plotTest.py'] ,
  options={"py2exe": {"excludes": excludes,
                      "includes": includes,
                      "dll_excludes": dll_excludes,
                      "optimize": 0,
                      "compressed": 2,
                      "bundle_files": 1}},
  zipfile=None,
) 
