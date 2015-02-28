"""
This module exists to smooth out some of the differences between PySide and PyQt4:

* Automatically import either PyQt4 or PySide depending on availability
* Allow to import QtCore/QtGui pyqtgraph.Qt without specifying which Qt wrapper
  you want to use.
* Declare QtCore.Signal, .Slot in PyQt4  
* Declare loadUiType function for Pyside

"""

import sys, re

PYSIDE = 'PySide'
PYQT4 = 'PyQt4'
PYQT5 = 'PyQt5'

QT_LIB = None

## Automatically determine whether to use PyQt or PySide. 
## This is done by first checking to see whether one of the libraries
## is already imported. If not, then attempt to import PyQt4, then PySide.
libOrder = [PYQT4, PYSIDE, PYQT5]

for lib in libOrder:
    if lib in sys.modules:
        QT_LIB = lib
        break

if QT_LIB is None:
    for lib in libOrder:
        try:
            __import__(lib)
            QT_LIB = lib
            break
        except ImportError:
            pass

if QT_LIB == None:
    raise Exception("PyQtGraph requires one of PyQt4, PyQt5 or PySide; none of these packages could be imported.")

if QT_LIB == PYSIDE:
    from PySide import QtGui, QtCore, QtOpenGL, QtSvg
    import PySide
    try:
        from PySide import shiboken
        isQObjectAlive = shiboken.isValid
    except ImportError:
        def isQObjectAlive(obj):
            try:
                if hasattr(obj, 'parent'):
                    obj.parent()
                elif hasattr(obj, 'parentItem'):
                    obj.parentItem()
                else:
                    raise Exception("Cannot determine whether Qt object %s is still alive." % obj)
            except RuntimeError:
                return False
            else:
                return True
    
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

elif QT_LIB == PYQT4:

    from PyQt4 import QtGui, QtCore, uic
    try:
        from PyQt4 import QtSvg
    except ImportError:
        pass
    try:
        from PyQt4 import QtOpenGL
    except ImportError:
        pass

    VERSION_INFO = 'PyQt4 ' + QtCore.PYQT_VERSION_STR + ' Qt ' + QtCore.QT_VERSION_STR

elif QT_LIB == PYQT5:
    
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
        tr = self.transform()
        tr.scale(sx, sy)
        self.setTransform(tr)
    QtWidgets.QGraphicsItem.scale = scale

    def rotate(self, angle):
        tr = self.transform()
        tr.rotate(angle)
        self.setTransform(tr)
    QtWidgets.QGraphicsItem.rotate = rotate

    def translate(self, dx, dy):
        tr = self.transform()
        tr.translate(dx, dy)
        self.setTransform(tr)
    QtWidgets.QGraphicsItem.translate = translate

    #def setMargin(self, i):
        #self.setContentsMargins(i, i, i, i)
    #QtWidgets.QGridLayout.setMargin = setMargin

    #def setResizeMode(self, mode):
        #self.setSectionResizeMode(mode)
    #QtWidgets.QHeaderView.setResizeMode = setResizeMode

    
    QtGui.QApplication = QtWidgets.QApplication
    QtGui.QGraphicsScene = QtWidgets.QGraphicsScene
    QtGui.QGraphicsObject = QtWidgets.QGraphicsObject
    QtGui.QGraphicsWidget = QtWidgets.QGraphicsWidget

    QtGui.QApplication.setGraphicsSystem = None
    
    # Import all QtWidgets objects into QtGui
    for o in dir(QtWidgets):
        if o.startswith('Q'):
            setattr(QtGui, o, getattr(QtWidgets,o) )
    
    VERSION_INFO = 'PyQt5 ' + QtCore.PYQT_VERSION_STR + ' Qt ' + QtCore.QT_VERSION_STR

# Common to PyQt4 and 5
if QT_LIB.startswith('PyQt'):
    import sip
    def isQObjectAlive(obj):
        return not sip.isdeleted(obj)
    loadUiType = uic.loadUiType

    QtCore.Signal = QtCore.pyqtSignal
    

    
## Make sure we have Qt >= 4.7
versionReq = [4, 7]
USE_PYSIDE = QT_LIB == PYSIDE
USE_PYQT4 = QT_LIB == PYQT4
USE_PYQT5 = QT_LIB == PYQT5
QtVersion = PySide.QtCore.__version__ if QT_LIB == PYSIDE else QtCore.QT_VERSION_STR
m = re.match(r'(\d+)\.(\d+).*', QtVersion)
if m is not None and list(map(int, m.groups())) < versionReq:
    print(list(map(int, m.groups())))
    raise Exception('pyqtgraph requires Qt version >= %d.%d  (your version is %s)' % (versionReq[0], versionReq[1], QtVersion))
