## make this version of pyqtgraph importable before any others
import sys, os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
path.rstrip(os.path.sep)
if path.endswith('pyqtgraph'):
    sys.path.insert(0, os.path.join(path, '..'))  ## examples installed inside pyqtgraph package
elif 'pyqtgraph' in os.listdir(path):
    sys.path.insert(0, path) ## examples adjacent to pyqtgraph (as in source)

## should force example to use PySide instead of PyQt
if 'pyside' in sys.argv:  
    from PySide import QtGui
elif 'pyqt' in sys.argv: 
    from PyQt4 import QtGui
else:
    from pyqtgraph.Qt import QtGui
    
## Force use of a specific graphics system
for gs in ['raster', 'native', 'opengl']:
    if gs in sys.argv:
        QtGui.QApplication.setGraphicsSystem(gs)
        break

