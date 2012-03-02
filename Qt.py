## Do all Qt imports from here to allow easier PyQt / PySide compatibility

#from PySide import QtGui, QtCore, QtOpenGL, QtSvg
from PyQt4 import QtGui, QtCore, QtOpenGL, QtSvg
if not hasattr(QtCore, 'Signal'):
    QtCore.Signal = QtCore.pyqtSignal
