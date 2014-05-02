"""
This module exists to smooth out some of the differences between PySide and
PyQt4, as well as the more significant changes made with Qt5/PyQt5:

* Automatically import either PyQt4 or PySide depending on availability
* Allow to import QtCore/QtGui pyqtgraph.Qt without specifying which Qt wrapper
  you want to use.
* Declare QtCore.Signal, .Slot in PyQt4  
* Declare loadUiType function for Pyside
* Fake out the new Qt5/PyQt5 object locations so that they mirror the old Qt4
   locations.  A lot of things moved around (eg: all widgets moved from QtGui
   to QtWidgets!)

"""

import sys, re

from .python2_3 import asUnicode  #for some pyside patching


_QT_LIBS = (QT_PYQT5, QT_PYQT4, QT_PYSIDE) = ("PyQt5", "PyQt4", "PySide")
QT_LIB = None  #none found yet

## Automatically determine whether to use PyQt or PySide. 
## This is done by first checking to see whether one of the libraries
## is already imported. If not, then attempt to import PyQt4, then PySide.

#First see if the user imported a preferential module...
for module in _QT_LIBS:
    if module in sys.modules:
        QT_LIB = wrapperType
        break
    
#if no preferred module was imported, try importing in our order...
if not QT_LIB:
    for module in _QT_LIBS:
        try:
            __import__(module)
        except ImportError:
            pass
        else:
            QT_LIB = module
            break

if not QT_LIB:
    raise Exception(("PyQtGraph requires one of %r, but none could be "
                     "imported." % _WRAPPER_MODULE.values()))

print "PyQtGraph is using Qt wrapper: " + module

#Set global PySide awareness...
# - a lot of existing code uses the pre-existing USE_PYSIDE flag
USE_PYSIDE = (QT_LIB == QT_PYSIDE)

if USE_PYSIDE:
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
    
    # Make a loadUiType function like PyQt has
    
    # Credit: 
    # http://stackoverflow.com/questions/4442286/python-code-genration-with-pyside-uic/14195313#14195313

    class StringIO(object):
        """Alternative to built-in StringIO needed to circumvent unicode/ascii issues"""
        def __init__(self):
            self.data = []
        
        def write(self, data):
            self.data.append(data)
            
        def getvalue(self):
            return ''.join(map(asUnicode, self.data)).encode('utf8')
        
    def loadUiType(uiFile):
        """
        Pyside "loadUiType" command like PyQt4 has one, so we have to convert
        the ui file to py code in-memory first and then execute it in a
        special frame to retrieve the form_class.
        """
        import pysideuic
        import xml.etree.ElementTree as xml
        #from io import StringIO
        
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
    
elif QT_LIB == QT_PYQT4:
    from PyQt4 import QtGui, QtCore, uic
    try:
        from PyQt4 import QtSvg
    except ImportError:
        pass
    try:
        from PyQt4 import QtOpenGL
    except ImportError:
        pass

elif QT_LIB == QT_PYQT5:
    #Qt5 moved several things around. eg: all Widgets moved from QtGui to
    #QtWidgets. The strategy to make PyQtGraph code work is to make PyQt5
    #look exactly like PyQt4 (for the cases where pyqtgraph needs it).
    import PyQt5
    from PyQt5 import uic
    from PyQt5 import QtCore as _Qt5Core
    from PyQt5 import QtWidgets as _Qt5Widgets  #most widgets moved from QtGui
    from PyQt5 import QtGui as _Qt5Gui
    
    QtCore = _Qt5Core
    QtGui = _Qt5Widgets #this is the big switcheroo
    
    #not all of QtGui moved over to QtWidgets.  Copy some of them back...
    leftoverGuiAttrs = ["QPainterPath", "QTransform", "QVector3D", "QMatrix4x4",
                        "QColor", "QPalette", "QBrush", "QPixmap", "QImage",
                        "QPen", "QDoubleValidator"]
    for attr in leftoverGuiAttrs:
        setattr(QtGui, attr, getattr(_Qt5Gui, attr))
    
    QtGui.QApplication.setGraphicsSystem = None #mfitzpatrick did this
    
    #There are a number of files generated by pyuic4 that expect to be able
    #to import PyQt4, so we will make it look like we did...
    PyQt4 = PyQt5
    PyQt4.QtGui = QtGui
    sys.modules["PyQt4"] = PyQt4
    
    try:
        from PyQt5 import QtSvg
    except ImportError:
        pass
    try:
        from PyQt5 import QtOpenGL
    except ImportError:
        pass

if QT_LIB in (QT_PYQT4, QT_PYQT5):
    import sip
    def isQObjectAlive(obj):
        return not sip.isdeleted(obj)
    loadUiType = uic.loadUiType

    QtCore.Signal = QtCore.pyqtSignal
    VERSION_INFO = 'PyQt4 ' + QtCore.PYQT_VERSION_STR + ' Qt ' + QtCore.QT_VERSION_STR


def _GetQtVersionTuple():
    if QT_LIB in (QT_PYQT5, QT_PYQT4):
        return tuple(QtCore.PYQT_VERSION_STR.split("."))
    elif QT_LIB == QT_PYSIDE:
        return tuple(QtCore.__version__.split("."))


def _GetQtVersionStr():
    """Returns a string for exporting as QtVersion.
    
    Previous versions of this module had an available QtVersion variable and
    this is to generate that.
    
    """
    if QT_LIB in (QT_PYQT5, QT_PYQT4):
        return QtCore.QT_VERSION_STR
    elif QT_LIB == QT_PYSIDE:
        QtVersion = PySide.QtCore.__version__


def _GetVERSION_INFO():
    """Returns a string for exporting as VERSION_INFO.
    
    Strings returned are not consistent between PySide and PyQt. This is an
    artifact of previous versions of this module and it has been kept the
    same to avoid potential compatibility issues.
    
    """
    #Note that the strings returned between PyQt and Pyside do not follow the
    #same convention. This difference has been preserved for compatibility.
    if QT_LIB in (QT_PYQT5, QT_PYQT4):
        wrapperVer = QtCore.PYQT_VERSION_STR
        qtVer = QtCore.QT_VERSION_STR
        return "PyQt%s %s Qt %s" % (wrapperVer[0], wrapperVer, qtVer)
    elif QT_LIB == QT_PYSIDE:
        wrapperVer = PySide.__version__
        #qtVer = ??
        return "PySide %s" % wrapperVer


## Make sure we have Qt >= 4.7
versionReq = (4, 7)
QtVersion = _GetQtVersionStr()
if _GetQtVersionTuple() < versionReq:
    print(list(qtVersion[:2]))
    raise Exception(("pyqtgraph requires Qt version >= %d.%d  (your version "
                     "is %s)") % (versionReq[0], versionReq[1], QtVersion))

VERSION_INFO = _GetVERSION_INFO()
