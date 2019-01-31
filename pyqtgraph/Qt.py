"""
This module exists to smooth out some of the differences between PySide and PyQt4:

* Automatically import either PyQt4 or PySide depending on availability
* Allow to import QtCore/QtGui pyqtgraph.Qt without specifying which Qt wrapper
  you want to use.
* Declare QtCore.Signal, .Slot in PyQt4
* Declare loadUiType function for Pyside

"""

import os, sys, re, time
import importlib

from .python2_3 import asUnicode

PYSIDE = 'PySide'
PYSIDE2 = 'PySide2'
PYQT4 = 'PyQt4'
PYQT5 = 'PyQt5'
QT_LIBS = [PYQT4, PYSIDE, PYQT5, PYSIDE2]

sip = None
QT_LIB = os.getenv('PYQTGRAPH_QT_LIB')

## Automatically determine whether to use PyQt or PySide (unless specified by
## environment variable).
## This is done by first checking to see whether one of the libraries
## is already imported. If not, then attempt to import PyQt4, then PySide.
if QT_LIB is not None:
    for lib in QT_LIBS:
        if lib.lower() == str(QT_LIB).lower():
            QT_LIB = lib
            break

if QT_LIB is None:
    for lib in QT_LIBS:
        if lib in sys.modules:
            QT_LIB = lib
            break

if QT_LIB is None:
    for lib in QT_LIBS:
        try:
            __import__(lib)
            QT_LIB = lib
            break
        except ImportError:
            pass

if QT_LIB is None:
    raise Exception("PyQtGraph requires one of PyQt4, PyQt5 or PySide; none of these packages could be imported.")

if QT_LIB == PYSIDE:
    from PySide import QtGui, QtCore, QtOpenGL, QtSvg
    try:
        from PySide import QtTest
        if not hasattr(QtTest.QTest, 'qWait'):
            @staticmethod
            def qWait(msec):
                start = time.time()
                QtGui.QApplication.processEvents()
                while time.time() < start + msec * 0.001:
                    QtGui.QApplication.processEvents()
            QtTest.QTest.qWait = qWait
                
    except ImportError:
        pass
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

        from stackoverflow: http://stackoverflow.com/a/14195313/3781327

        seems like this might also be a legitimate solution, but I'm not sure
        how to make PyQt4 and pyside look the same...
            http://stackoverflow.com/a/8717832
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

elif QT_LIB == PYQT4:

    import sip
    from PyQt4 import QtGui, QtCore, uic
    try:
        from PyQt4 import QtSvg
    except ImportError:
        pass
    try:
        from PyQt4 import QtOpenGL
    except ImportError:
        pass
    try:
        from PyQt4 import QtTest
    except ImportError:
        pass

    VERSION_INFO = 'PyQt4 ' + QtCore.PYQT_VERSION_STR + ' Qt ' + QtCore.QT_VERSION_STR

elif QT_LIB == PYQT5:
    
    # We're using PyQt5 which has a different structure so we're going to use a shim to
    # recreate the Qt4 structure for Qt5
    import sip
    from PyQt5 import QtGui, QtCore, QtWidgets, uic
    try:
        from PyQt5 import QtSvg
    except ImportError:
        pass
    try:
        from PyQt5 import QtOpenGL
    except ImportError:
        pass
    try:
        from PyQt5 import QtTest
        QtTest.QTest.qWaitForWindowShown = QtTest.QTest.qWaitForWindowExposed
    except ImportError:
        pass

    # Re-implement deprecated APIs

    __QGraphicsItem_scale = QtWidgets.QGraphicsItem.scale

    def scale(self, *args):
        if args:
            sx, sy = args
            tr = self.transform()
            tr.scale(sx, sy)
            self.setTransform(tr)
        else:
            return __QGraphicsItem_scale(self)

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

    def setMargin(self, i):
        self.setContentsMargins(i, i, i, i)
    QtWidgets.QGridLayout.setMargin = setMargin

    def setResizeMode(self, *args):
        self.setSectionResizeMode(*args)
    QtWidgets.QHeaderView.setResizeMode = setResizeMode

    
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

elif QT_LIB == PYSIDE2:
    # We're using PySide2 which has a similar structure to PyQt5
    from PySide2 import QtGui, QtCore, QtWidgets, __version__
    try:
        from PySide2 import QtSvg
    except ImportError:
        pass
    try:
        from PySide2 import QtOpenGL
    except ImportError:
        pass
    try:
        from PySide2 import QtTest
        QtTest.QTest.qWaitForWindowShown = QtTest.QTest.qWaitForWindowExposed

        if not hasattr(QtTest.QTest, 'qWait'):
            @staticmethod
            def qWait(msec):
                start = time.time()
                QtGui.QApplication.processEvents()
                while time.time() < start + msec * 0.001:
                    QtGui.QApplication.processEvents()
            QtTest.QTest.qWait = qWait
    except ImportError:
        pass

    try:
        from shiboken2 import isValid as isQObjectAlive
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

    VERSION_INFO = 'PySide2 ' + __version__

    # Re-implement deprecated APIs

    __QGraphicsItem_scale = QtWidgets.QGraphicsItem.scale

    def scale(self, *args):
        if args:
            sx, sy = args
            tr = self.transform()
            tr.scale(sx, sy)
            self.setTransform(tr)
        else:
            return __QGraphicsItem_scale(self)

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

    def setMargin(self, i):
        self.setContentsMargins(i, i, i, i)
    QtWidgets.QGridLayout.setMargin = setMargin

    def setResizeMode(self, *args):
        self.setSectionResizeMode(*args)
    QtWidgets.QHeaderView.setResizeMode = setResizeMode

    QtGui.QApplication = QtWidgets.QApplication
    QtGui.QGraphicsScene = QtWidgets.QGraphicsScene
    QtGui.QGraphicsObject = QtWidgets.QGraphicsObject
    QtGui.QGraphicsWidget = QtWidgets.QGraphicsWidget

    QtGui.QApplication.setGraphicsSystem = None

    # Import all QtWidgets objects into QtGui
    for o in dir(QtWidgets):
        if o.startswith('Q'):
            setattr(QtGui, o, getattr(QtWidgets,o) )

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

        from stackoverflow: http://stackoverflow.com/a/14195313/3781327

        seems like this might also be a legitimate solution, but I'm not sure
        how to make PyQt4 and pyside look the same...
            http://stackoverflow.com/a/8717832
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

    # PySide2 QPoint - QPointF does not work.  This override still doesn't work.
    # I changed the code by converting QPoint to QPointF
    ____QPoint__sub__ = QtCore.QPoint.__sub__
    ____QPointF__sub__ = QtCore.QPointF.__sub__

    def qpoint_sub(self, other):
        if isinstance(other, QtCore.QPointF):
            return QtCore.QPointF(self.x() - other.x(), self.y() - other.y())
        return ____QPoint__sub__(self, other)
    QtCore.QPoint.__sub__ = qpoint_sub

    def qpointf_sub(self, other):
        return QtCore.QPointF(self.x() - other.x(), self.y() - other.y())
    QtCore.QPointF.__sub__ = qpointf_sub


    ____QMatrix4x4__init__ = QtGui.QMatrix4x4.__init__

    def qmatrix4x4_init(self, *args):
        if len(args) == 1 and isinstance(args[0], QtGui.QMatrix4x4):
            m = args[0]
            args = [m[r, c] for r in range(4) for c in range(4)]
        return ____QMatrix4x4__init__(self, *args)

    QtGui.QMatrix4x4.__init__ = qmatrix4x4_init

else:
    raise ValueError("Invalid Qt lib '%s'" % QT_LIB)


# Common to PyQt4 and 5
if QT_LIB.startswith('PyQt'):
    import sip

    def isQObjectAlive(obj):
        return not sip.isdeleted(obj)
    loadUiType = uic.loadUiType

    QtCore.Signal = QtCore.pyqtSignal


# Make sure we have Qt >= 4.7
versionReq = [4, 7]
USE_PYSIDE2 = QT_LIB == PYSIDE2
USE_PYSIDE = QT_LIB == PYSIDE
USE_PYQT4 = QT_LIB == PYQT4
USE_PYQT5 = QT_LIB == PYQT5

QtVersion = '0.0.0'
if hasattr(QtCore, '__version__'):
    QtVersion = QtCore.__version__
elif hasattr(QtCore, 'QT_VERSION_STR'):
    QtVersion = QtCore.QT_VERSION_STR

m = re.match(r'(\d+)\.(\d+).*', QtVersion)
if m is not None and list(map(int, m.groups())) < versionReq:
    print(list(map(int, m.groups())))
    raise Exception('pyqtgraph requires Qt version >= %d.%d  (your version is %s)' % (versionReq[0], versionReq[1], QtVersion))


def import_qt_file(filename):
    """Import the Qt file associated with the given filename.

    Example:
        >>> import_qt_file('pyqtgraph/canvas/TransformGuiTemplate.ui')
        <module 'pyqtgraph.canvas.TransformGuiTemplate_pyside2' from 'pyqtgraph\\canvas\\TransformGuiTemplate_pyside2.py'>
    """
    filename = os.path.splitext(filename)[0]
    filename = filename.replace('/', '.').replace('\\', '.')
    filename = ''.join((filename, '_', QT_LIB.lower()))  # Template_pyqt
    return importlib.import_module(filename)
