## make this version of pyqtgraph importable before any others
## we do this to make sure that, when running examples, the correct library
## version is imported (if there are multiple versions present).
import sys, os

if not hasattr(sys, 'frozen'):
    if __file__ == '<stdin>':
        path = os.getcwd()
    else:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path.rstrip(os.path.sep)
    if 'pyqtgraph' in os.listdir(path):
        sys.path.insert(0, path) ## examples adjacent to pyqtgraph (as in source tree)
    else:
        for p in sys.path:
            if len(p) < 3:
                continue
            if path.startswith(p):  ## If the example is already in an importable location, promote that location
                sys.path.remove(p)
                sys.path.insert(0, p)

## should force example to use PySide instead of PyQt
if 'pyside2' in sys.argv:
    os.putenv('PYQTGRAPH_QT_LIB', 'PySide2')
elif 'pyside' in sys.argv:
    os.putenv('PYQTGRAPH_QT_LIB', 'PySide')
elif 'pyqt' in sys.argv:
    os.putenv('PYQTGRAPH_QT_LIB', 'PyQt4')
elif 'pyqt5' in sys.argv:
    os.putenv('PYQTGRAPH_QT_LIB', 'PyQt5')

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg    
    
## Force use of a specific graphics system
use_gs = 'default'
for gs in ['raster', 'native', 'opengl']:
    if gs in sys.argv:
        use_gs = gs
        QtGui.QApplication.setGraphicsSystem(gs)
        break

print("Using %s (%s graphics system)" % (pg.Qt.QT_LIB, use_gs))

## Enable fault handling to give more helpful error messages on crash. 
## Only available in python 3.3+
try:
    import faulthandler
    faulthandler.enable()
except ImportError:
    pass
