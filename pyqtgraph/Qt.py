"""
This module exists to smooth out some of the differences between PySide and PyQt4:

* Automatically import either PyQt4 or PySide depending on availability
* Allow to import QtCore/QtGui pyqtgraph.Qt without specifying which Qt wrapper
  you want to use.
* Declare QtCore.Signal, .Slot in PyQt4  
* Declare loadUiType function for Pyside

"""

import sys, re

PYSIDE = 0
PYQT4 = 1
PYQT5 = 2

USE_QT_PY = None

## Automatically determine whether to use PyQt or PySide. 
## This is done by first checking to see whether one of the libraries
## is already imported. If not, then attempt to import PyQt4, then PySide.
if 'PyQt4' in sys.modules:
    USE_QT_PY = PYQT4
if 'PyQt5' in sys.modules:
    USE_QT_PY = PYQT5
elif 'PySide' in sys.modules:
    USE_QT_PY = PYSIDE
else:
    try:
        import PyQt4
        USE_QT_PY = PYQT4
    except ImportError:
        try:
            import PyQt5
            USE_QT_PY = PYQT5
        except ImportError:
            try:
                import PySide
                USE_QT_PY = PYSIDE
            except:
                pass

if USE_QT_PY == None:
    raise Exception("PyQtGraph requires one of PyQt4, PyQt5 or PySide; none of these packages could be imported.")

if USE_QT_PY == PYSIDE:
    from PySide import QtGui, QtCore, QtOpenGL, QtSvg
    import PySide
    VERSION_INFO = 'PySide ' + PySide.__version__
    
    # Make a loadUiType function like PyQt has
    
    # Credit: 
    # http://stackoverflow.com/questions/4442286/python-code-genration-with-pyside-uic/14195313#14195313

    def loadUiType(uiFile):
        """
        Pyside "loadUiType" command like PyQt4 has one, so we have to convert the ui file to py code in-memory first    and then execute it in a special frame to retrieve the form_class.
        """
        import pysideuic
        import xml.etree.ElementTree as xml
        from io import StringIO
        
        parsed = xml.parse(uiFile)
        widget_class = parsed.find('widget').get('class')
        form_class = parsed.find('class').text

        with open(uiFile, 'r') as f:
            o = StringIO()
            frame = {}

            pysideuic.compileUi(f, o, indent=0)
            pyc = compile(o.getvalue(), '<string>', 'exec')
            exec(pyc, frame)

            #Fetch the base_class and form class based on their type in the xml from designer
            form_class = frame['Ui_%s'%form_class]
            base_class = eval('QtGui.%s'%widget_class)

        return form_class, base_class

elif USE_QT_PY == PYQT4:

    from PyQt4 import QtGui, QtCore, uic
    try:
        from PyQt4 import QtSvg
    except ImportError:
        pass
    try:
        from PyQt4 import QtOpenGL
    except ImportError:
        pass


    loadUiType = uic.loadUiType

    QtCore.Signal = QtCore.pyqtSignal
    VERSION_INFO = 'PyQt4 ' + QtCore.PYQT_VERSION_STR + ' Qt ' + QtCore.QT_VERSION_STR

elif USE_QT_PY == PYQT5:
    
    # We're using PyQt5 which has a different structure so we're going to use a shim to
    # recreate the Qt4 structure for Qt5
    from PyQt5 import QtGui, QtCore, QtWidgets, Qt, uic
    try:
        from PyQt5 import QtSvg
    except ImportError:
        pass
    try:
        from PyQt5 import QtOpenGL
    except ImportError:
        pass

    # Re-implement deprecated APIs
    def scale(self, sx, sy):
        self.setTransform(QtGui.QTransform.fromScale(sx, sy), True)
    QtWidgets.QGraphicsItem.scale = scale

    def rotate(self, angle):
        self.setRotation(self.rotation() + angle)
    QtWidgets.QGraphicsItem.rotate = rotate

    def translate(self, dx, dy):
        self.setTransform(QtGui.QTransform.fromTranslate(dx, dy), True)
    QtWidgets.QGraphicsItem.translate = translate

    def setMargin(self, i):
        self.setContentsMargins(i, i, i, i)
    QtWidgets.QGridLayout.setMargin = setMargin

    def setResizeMode(self, mode):
        self.setSectionResizeMode(mode)
    QtWidgets.QHeaderView.setResizeMode = setResizeMode

    
    QtGui.QApplication = QtWidgets.QApplication
    QtGui.QGraphicsScene = QtWidgets.QGraphicsScene
    QtGui.QGraphicsObject = QtWidgets.QGraphicsObject
    QtGui.QGraphicsWidget = QtWidgets.QGraphicsWidget

    QtGui.QApplication.setGraphicsSystem = None
    QtCore.Signal = Qt.pyqtSignal
    
    # Import all QtWidgets objects into QtGui
    for o in dir(QtWidgets):
        if o.startswith('Q'):
            setattr(QtGui, o, getattr(QtWidgets,o) )
    
## Make sure we have Qt >= 4.7
versionReq = [4, 7]
USE_PYSIDE = USE_QT_PY == PYSIDE # still needed internally elsewhere
QtVersion = PySide.QtCore.__version__ if USE_QT_PY ==  PYSIDE else QtCore.QT_VERSION_STR
m = re.match(r'(\d+)\.(\d+).*', QtVersion)
if m is not None and list(map(int, m.groups())) < versionReq:
    print(list(map(int, m.groups())))
    raise Exception('pyqtgraph requires Qt version >= %d.%d  (your version is %s)' % (versionReq[0], versionReq[1], QtVersion))

