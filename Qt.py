## Do all Qt imports from here to allow easier PyQt / PySide compatibility

USE_PYSIDE = False   ## If False, import PyQt4. If True, import PySide
                     ## Note that when switching between PyQt and PySide, all template
                     ## files (*.ui) must be rebuilt for the target library.

if USE_PYSIDE:
    from PySide import QtGui, QtCore, QtOpenGL, QtSvg
    import PySide
    VERSION_INFO = 'PySide ' + PySide.__version__
else:
    from PyQt4 import QtGui, QtCore
    try:
        from PyQt4 import QtSvg
    except ImportError:
        pass
    try:
        from PyQt4 import QtOpenGL
    except ImportError:
        pass

    QtCore.Signal = QtCore.pyqtSignal
    VERSION_INFO = 'PyQt4 ' + QtCore.PYQT_VERSION_STR + ' Qt ' + QtCore.QT_VERSION_STR
