# -*- coding: utf-8 -*-
"""
This module exists to smooth out some of the differences between PySide and PyQt4:

* Automatically import either PyQt4 or PySide depending on availability
* Allow to import QtCore/QtGui pyqtgraph.Qt without specifying which Qt wrapper
  you want to use.
* Declare QtCore.Signal, .Slot in PyQt4
* Declare loadUiType function for Pyside

"""

import os, sys, re, time, subprocess, warnings
import enum

from .python2_3 import asUnicode

PYSIDE = 'PySide'
PYSIDE2 = 'PySide2'
PYSIDE6 = 'PySide6'
PYQT4 = 'PyQt4'
PYQT5 = 'PyQt5'
PYQT6 = 'PyQt6'

QT_LIB = os.getenv('PYQTGRAPH_QT_LIB')

## Automatically determine which Qt package to use (unless specified by
## environment variable).
## This is done by first checking to see whether one of the libraries
## is already imported. If not, then attempt to import in the order
## specified in libOrder.
if QT_LIB is None:
    libOrder = [PYQT5, PYSIDE2, PYSIDE6, PYQT6]

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

if QT_LIB is None:
    raise Exception("PyQtGraph requires one of PyQt5, PyQt6, PySide2 or PySide6; none of these packages could be imported.")


class FailedImport(object):
    """Used to defer ImportErrors until we are sure the module is needed.
    """
    def __init__(self, err):
        self.err = err
        
    def __getattr__(self, attr):
        raise self.err


# Make a loadUiType function like PyQt has

# Credit:
# http://stackoverflow.com/questions/4442286/python-code-genration-with-pyside-uic/14195313#14195313

class _StringIO(object):
    """Alternative to built-in StringIO needed to circumvent unicode/ascii issues"""
    def __init__(self):
        self.data = []
    
    def write(self, data):
        self.data.append(data)
        
    def getvalue(self):
        return ''.join(map(asUnicode, self.data)).encode('utf8')

    
def _loadUiType(uiFile):
    """
    PySide lacks a "loadUiType" command like PyQt4's, so we have to convert
    the ui file to py code in-memory first and then execute it in a
    special frame to retrieve the form_class.

    from stackoverflow: http://stackoverflow.com/a/14195313/3781327

    seems like this might also be a legitimate solution, but I'm not sure
    how to make PyQt4 and pyside look the same...
        http://stackoverflow.com/a/8717832
    """

    pyside2uic = None
    if QT_LIB == PYSIDE2:
        try:
            import pyside2uic
        except ImportError:
            # later versions of pyside2 have dropped pyside2uic; use the uic binary instead.
            pyside2uic = None

        if pyside2uic is None:
            pyside2version = tuple(map(int, PySide2.__version__.split(".")))
            if (5, 14) <= pyside2version < (5, 14, 2, 2):
                warnings.warn('For UI compilation, it is recommended to upgrade to PySide >= 5.15')

    # get class names from ui file
    import xml.etree.ElementTree as xml
    parsed = xml.parse(uiFile)
    widget_class = parsed.find('widget').get('class')
    form_class = parsed.find('class').text

    # convert ui file to python code
    if pyside2uic is None:
        uic_executable = QT_LIB.lower() + '-uic'
        uipy = subprocess.check_output([uic_executable, uiFile])
    else:
        o = _StringIO()
        with open(uiFile, 'r') as f:
            pyside2uic.compileUi(f, o, indent=0)
        uipy = o.getvalue()

    # execute python code
    pyc = compile(uipy, '<string>', 'exec')
    frame = {}
    exec(pyc, frame)

    # fetch the base_class and form class based on their type in the xml from designer
    form_class = frame['Ui_%s'%form_class]
    base_class = eval('QtGui.%s'%widget_class)

    return form_class, base_class


if QT_LIB == PYQT5:
    # We're using PyQt5 which has a different structure so we're going to use a shim to
    # recreate the Qt4 structure for Qt5
    from PyQt5 import QtGui, QtCore, QtWidgets, sip, uic
    
    try:
        from PyQt5 import QtSvg
    except ImportError as err:
        QtSvg = FailedImport(err)
    try:
        from PyQt5 import QtTest
    except ImportError as err:
        QtTest = FailedImport(err)

    VERSION_INFO = 'PyQt5 ' + QtCore.PYQT_VERSION_STR + ' Qt ' + QtCore.QT_VERSION_STR

elif QT_LIB == PYQT6:
    from PyQt6 import QtGui, QtCore, QtWidgets, sip, uic

    try:
        from PyQt6 import QtSvg
    except ImportError as err:
        QtSvg = FailedImport(err)
    try:
        from PyQt6 import QtOpenGLWidgets
    except ImportError as err:
        QtOpenGLWidgets = FailedImport(err)
    try:
        from PyQt6 import QtTest
    except ImportError as err:
        QtTest = FailedImport(err)

    VERSION_INFO = 'PyQt6 ' + QtCore.PYQT_VERSION_STR + ' Qt ' + QtCore.QT_VERSION_STR

elif QT_LIB == PYSIDE2:
    from PySide2 import QtGui, QtCore, QtWidgets
    
    try:
        from PySide2 import QtSvg
    except ImportError as err:
        QtSvg = FailedImport(err)
    try:
        from PySide2 import QtTest
    except ImportError as err:
        QtTest = FailedImport(err)

    import shiboken2
    isQObjectAlive = shiboken2.isValid
    import PySide2
    VERSION_INFO = 'PySide2 ' + PySide2.__version__ + ' Qt ' + QtCore.__version__

elif QT_LIB == PYSIDE6:
    from PySide6 import QtGui, QtCore, QtWidgets

    try:
        from PySide6 import QtSvg
    except ImportError as err:
        QtSvg = FailedImport(err)
    try:
        from PySide6 import QtOpenGLWidgets
    except ImportError as err:
        QtOpenGLWidgets = FailedImport(err)
    try:
        from PySide6 import QtTest
    except ImportError as err:
        QtTest = FailedImport(err)

    import shiboken6
    isQObjectAlive = shiboken6.isValid
    import PySide6
    VERSION_INFO = 'PySide6 ' + PySide6.__version__ + ' Qt ' + QtCore.__version__

else:
    raise ValueError("Invalid Qt lib '%s'" % QT_LIB)


# common to PyQt5, PyQt6, PySide2 and PySide6
if QT_LIB in [PYQT5, PYQT6, PYSIDE2, PYSIDE6]:
    # We're using Qt5 which has a different structure so we're going to use a shim to
    # recreate the Qt4 structure

    if QT_LIB in [PYQT5, PYSIDE2]:
        __QGraphicsItem_scale = QtWidgets.QGraphicsItem.scale	

        def scale(self, *args):
            warnings.warn(
                "Deprecated Qt API, will be removed in 0.13.0.",
                DeprecationWarning, stacklevel=2
            )
            if args:	
                sx, sy = args	
                tr = self.transform()	
                tr.scale(sx, sy)	
                self.setTransform(tr)	
            else:	
                return __QGraphicsItem_scale(self)
        QtWidgets.QGraphicsItem.scale = scale	

        def rotate(self, angle):
            warnings.warn(
                "Deprecated Qt API, will be removed in 0.13.0.",
                DeprecationWarning, stacklevel=2
            )
            tr = self.transform()	
            tr.rotate(angle)	
            self.setTransform(tr)	
        QtWidgets.QGraphicsItem.rotate = rotate	

        def translate(self, dx, dy):
            warnings.warn(
                "Deprecated Qt API, will be removed in 0.13.0.",
                DeprecationWarning, stacklevel=2
            )
            tr = self.transform()	
            tr.translate(dx, dy)	
            self.setTransform(tr)	
        QtWidgets.QGraphicsItem.translate = translate	

        def setMargin(self, i):
            warnings.warn(
                "Deprecated Qt API, will be removed in 0.13.0.",
                DeprecationWarning, stacklevel=2
            )
            self.setContentsMargins(i, i, i, i)	
        QtWidgets.QGridLayout.setMargin = setMargin	

        def setResizeMode(self, *args):
            warnings.warn(
                "Deprecated Qt API, will be removed in 0.13.0.",
                DeprecationWarning, stacklevel=2
            )
            self.setSectionResizeMode(*args)
        QtWidgets.QHeaderView.setResizeMode = setResizeMode	
    
    # Import all QtWidgets objects into QtGui
    for o in dir(QtWidgets):
        if o.startswith('Q'):
            setattr(QtGui, o, getattr(QtWidgets,o) )
    
    QtGui.QApplication.setGraphicsSystem = None


if QT_LIB in [PYQT6, PYSIDE6]:
    # We're using Qt6 which has a different structure so we're going to use a shim to
    # recreate the Qt5 structure

    if not isinstance(QtOpenGLWidgets, FailedImport):
        QtWidgets.QOpenGLWidget = QtOpenGLWidgets.QOpenGLWidget


# Common to PySide2 and PySide6
if QT_LIB in [PYSIDE2, PYSIDE6]:
    QtVersion = QtCore.__version__
    loadUiType = _loadUiType

    # PySide does not implement qWait
    if not isinstance(QtTest, FailedImport):
        if not hasattr(QtTest.QTest, 'qWait'):
            @staticmethod
            def qWait(msec):
                start = time.time()
                QtGui.QApplication.processEvents()
                while time.time() < start + msec * 0.001:
                    QtGui.QApplication.processEvents()
            QtTest.QTest.qWait = qWait


# Common to PyQt5 and PyQt6
if QT_LIB in [PYQT5, PYQT6]:
    QtVersion = QtCore.QT_VERSION_STR

    # PyQt, starting in v5.5, calls qAbort when an exception is raised inside
    # a slot. To maintain backward compatibility (and sanity for interactive
    # users), we install a global exception hook to override this behavior.
    if sys.excepthook == sys.__excepthook__:
        sys_excepthook = sys.excepthook
        def pyqt_qabort_override(*args, **kwds):
            return sys_excepthook(*args, **kwds)
        sys.excepthook = pyqt_qabort_override
    
    def isQObjectAlive(obj):
        return not sip.isdeleted(obj)
    
    loadUiType = uic.loadUiType

    QtCore.Signal = QtCore.pyqtSignal
    

if QT_LIB == PYSIDE6:
    # PySide6 6.0 has a missing binding
    if not hasattr(QtGui.QGradient, 'setStops'):
        def __setStops(self, stops):
            for pos, color in stops:
                self.setColorAt(pos, color)
        QtGui.QGradient.setStops = __setStops


if QT_LIB == PYQT6:
    # module.Class.EnumClass.Enum -> module.Class.Enum
    def promote_enums(module):
        class_names = [x for x in dir(module) if x.startswith('Q')]
        for class_name in class_names:
            klass = getattr(module, class_name)
            if not isinstance(klass, sip.wrappertype):
                continue
            attrib_names = [x for x in dir(klass) if x[0].isupper()]
            for attrib_name in attrib_names:
                attrib = getattr(klass, attrib_name)
                if not isinstance(attrib, enum.EnumMeta):
                    continue
                for e in attrib:
                    setattr(klass, e.name, e)

    promote_enums(QtCore)
    promote_enums(QtGui)
    promote_enums(QtWidgets)

    # QKeyEvent::key() returns an int
    # so comparison with a Key_* enum will always be False
    # here we convert the enum to its int value
    keys = ['Up', 'Down', 'Right', 'Left', 'Return', 'Enter', 'Delete', 'Backspace',
            'PageUp', 'PageDown', 'Home', 'End', 'Tab', 'Backtab', 'Escape', 'Space']
    for name in keys:
        e = getattr(QtCore.Qt.Key, 'Key_' + name)
        setattr(QtCore.Qt, e.name, e.value)

    # shim the old names for QPointF mouse coords
    QtGui.QSinglePointEvent.localPos = lambda o : o.position()
    QtGui.QSinglePointEvent.windowPos = lambda o : o.scenePosition()
    QtGui.QSinglePointEvent.screenPos = lambda o : o.globalPosition()

    QtWidgets.QApplication.exec_ = QtWidgets.QApplication.exec

    # PyQt6 6.0.0 has a bug where it can't handle certain Type values returned
    # by the Qt library.
    if QtCore.PYQT_VERSION == 0x60000:
        def new_method(self, old_method=QtCore.QEvent.type):
            try:
                typ = old_method(self)
            except ValueError:
                typ = QtCore.QEvent.Type.None_
            return typ
        QtCore.QEvent.type = new_method
        del new_method

    # PyQt6 6.1 renames some enums and flags to be in line with the other bindings.
    # "Alignment" and "Orientations" are PyQt6 6.0 and are used in the generated
    # ui files. Pending a regeneration of the template files, which would mean a
    # drop in support for PyQt6 6.0, provide the old names for PyQt6 6.1.
    # This is strictly a temporary private shim. Do not depend on it in your code.
    if hasattr(QtCore.Qt, 'AlignmentFlag') and not hasattr(QtCore.Qt, 'Alignment'):
        QtCore.Qt.Alignment = QtCore.Qt.AlignmentFlag
    if hasattr(QtCore.Qt, 'Orientation') and not hasattr(QtCore.Qt, 'Orientations'):
        QtCore.Qt.Orientations = QtCore.Qt.Orientation

# USE_XXX variables are deprecated
USE_PYSIDE = QT_LIB == PYSIDE
USE_PYQT4 = QT_LIB == PYQT4
USE_PYQT5 = QT_LIB == PYQT5

    
## Make sure we have Qt >= 5.12
versionReq = [5, 12]
m = re.match(r'(\d+)\.(\d+).*', QtVersion)
if m is not None and list(map(int, m.groups())) < versionReq:
    print(list(map(int, m.groups())))
    raise Exception('pyqtgraph requires Qt version >= %d.%d  (your version is %s)' % (versionReq[0], versionReq[1], QtVersion))

App = QtWidgets.QApplication
# subclassing QApplication causes segfaults on PySide{2, 6} / Python 3.8.7+

QAPP = None
def mkQApp(name=None):
    """
    Creates new QApplication or returns current instance if existing.
    
    ============== ========================================================
    **Arguments:**
    name           (str) Application name, passed to Qt
    ============== ========================================================
    """
    global QAPP
    
    def onPaletteChange(palette):
        color = palette.base().color().name()
        app = QtWidgets.QApplication.instance()
        app.setProperty('darkMode', color.lower() != "#ffffff")

    QAPP = QtGui.QApplication.instance()
    if QAPP is None:
        # hidpi handling
        qtVersionCompare = tuple(map(int, QtVersion.split(".")))
        if qtVersionCompare > (6, 0):
            # Qt6 seems to support hidpi without needing to do anything so continue
            pass
        elif qtVersionCompare > (5, 14):
            os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
            QtGui.QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        else:  # qt 5.12 and 5.13
            QtGui.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
            QtGui.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
        QAPP = QtGui.QApplication(sys.argv or ["pyqtgraph"])
        QAPP.paletteChanged.connect(onPaletteChange)
        QAPP.paletteChanged.emit(QAPP.palette())

    if name is not None:
        QAPP.setApplicationName(name)
    return QAPP


# exec() is used within _loadUiType, so we define as exec_() here and rename in pg namespace
def exec_():
    app = mkQApp()
    return app.exec() if hasattr(app, 'exec') else app.exec_()
