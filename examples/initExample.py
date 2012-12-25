## make this version of pyqtgraph importable before any others
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

if 'pyside' in sys.argv:  ## should force example to use PySide instead of PyQt
    from PySide import QtGui
elif 'pyqt' in sys.argv: 
    from PyQt4 import QtGui
else:
    from pyqtgraph.Qt import QtGui
    
for gs in ['raster', 'native', 'opengl']:
    if gs in sys.argv:
        QtGui.QApplication.setGraphicsSystem(gs)
        break

