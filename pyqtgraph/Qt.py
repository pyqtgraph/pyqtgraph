"""
This module exists to smooth out some of the differences between PySide and
PyQt4, as well enable PyQt5 use via the qt_backport module.

* Automatically import either PyQt4 or PySide depending on availability
* Allow to import QtCore/QtGui pyqtgraph.Qt without specifying which Qt wrapper
  you want to use.
* Declare QtCore.Signal, .Slot in PyQt4  
* Declare loadUiType function for Pyside
* Make use of PyQt5 through qt_backport, when available.
* Provide visibility to the rest of pyqtgraph on what specific wrapper lib, api,
  or emulation is in use through various QT_ exports (for context-dependent
  implementations).

"""

import sys, re

from .python2_3 import asUnicode  #for some pyside patching

#Qt wrapper APIs, in order of preference...
# - Note that Qt5 can be used with the PyQt4 api trhough the use of qt_wrapper
QT_APIS = \
    (API_PYSIDE, API_PYQT4) = \
    ("PySide", "PyQt4")

QT_LIBS = \
    (LIB_PYQT4, LIB_PYSIDE, LIB_PYQT5) = \
    ("PyQt4", "PySide", "PyQt5")

#values below will be worked out as we go...
# - note that if QT_EMULATING, the resulting proxy objects can cause issues in
#    some cases.  If required, the root Qt object under the proxy can be accessed
#    using the '_qt_root_class' class property.  This is particulary useful when
#    looking for specific subclasses of Qt objects.
QT_EMULATING = False
QT_API = None   #EMULATOR
QT_WRAPPER = None
QT_LIB = QT_WRAPPER  #synonym

#First see if the user already imported a preferred wrapper module...
for api in QT_APIS:
    if api in sys.modules:
        QT_API = api
        break
    
#if no wrapper module was imported, try importing in preferred order...
if not QT_API:
    for api in QT_APIS:
        try:
            __import__(api)
        except ImportError:
            pass
        else:
            QT_API = api
            break

if QT_API is None:
    #no usable api available!
    msg = ("PyQtGraph requires one of %r, but none of these could be "
           "imported." % (QT_APIS, ))
    #Let's be helpful and see if they have PyQt5 but forgot the wrapper...
    try:
        import PyQt5
    except ImportError:
        pass
    else:
        msg += (" PyQt5 was detected, but to use it you you must also install "
                "qt_backport.")
    raise Exception(msg)

#if here, we have a usable API imported.

try:
    import qt_backport
except ImportError:
    #if we can't import it, we are definitely not emulating!
    QT_WRAPPER = QT_API
    QT_EMULATING = False
else:
    if qt_backport.CURRENT_EMULATOR is None:
        QT_WRAPPER = QT_API
        QT_EMULATING = False
    else:
        QT_WRAPPER = qt_backport.CURRENT_WRAPPER
        QT_EMULATING = True

#Set global PySide awareness...
# - a lot of existing code uses this flag
USE_PYSIDE = (QT_API == API_PYSIDE)

if QT_API == API_PYSIDE:
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
    
elif QT_API == API_PYQT4:
    from PyQt4 import QtGui, QtCore, uic
    try:
        from PyQt4 import QtSvg
    except ImportError:
        pass
    try:
        from PyQt4 import QtOpenGL
    except ImportError:
        pass
    
    import sip
    def isQObjectAlive(obj):
        return not sip.isdeleted(obj)
    loadUiType = uic.loadUiType
    QtCore.Signal = QtCore.pyqtSignal


def _GetQtVersionTuple():
    if QT_API == API_PYQT4:
        return tuple(QtCore.PYQT_VERSION_STR.split("."))
    elif QT_API == API_PYSIDE:
        return tuple(QtCore.__version__.split("."))


def _GetQtVersionStr():
    """Returns a string for exporting as QtVersion.
    
    Previous versions of this module had an available QtVersion variable and
    this is to generate that.
    
    """
    if QT_API == API_PYQT4:
        return QtCore.QT_VERSION_STR
    elif QT_API == API_PYSIDE:
        return PySide.QtCore.__version__


def _GetVERSION_INFO():
    """Returns a string for exporting as VERSION_INFO.
    
    Strings returned are not consistent between PySide and PyQt. This is an
    artifact of previous versions of this module and it has been kept the
    same to avoid potential compatibility issues.
    
    """
    #Note that the strings returned between PyQt and Pyside do not follow the
    #same convention. This difference has been preserved for compatibility.
    if QT_EMULATING == False:
        if QT_API == API_PYQT4:
            wrapperVer = QtCore.PYQT_VERSION_STR
            qtVer = QtCore.QT_VERSION_STR
            ver_str = "PyQt%s %s Qt %s" % (wrapperVer[0], wrapperVer, qtVer)
        elif QT_API == API_PYSIDE:
            wrapperVer = PySide.__version__
            #qtVer = ??
            return "PySide %s" % wrapperVer
    else:
        ver_str = ("{qt_lib} v{qt_lib_ver} accessed via "
                   "qt_backport v{qtb_ver} using the "
                   "{api} API.".format(qt_lib = QT_LIB,
                                       qt_lib_ver = QtVersion,
                                       qtb_ver = qt_backport.__version__,
                                       api = QT_API))
    return ver_str


## Make sure we have Qt >= 4.7
versionReq = (4, 7)
QtVersion = _GetQtVersionStr()
if _GetQtVersionTuple() < versionReq:
    raise Exception(("pyqtgraph requires Qt version >= %d.%d  (your version "
                     "is %s)") % (versionReq[0], versionReq[1], QtVersion))

VERSION_INFO = _GetVERSION_INFO()
QT_LIB = QT_WRAPPER  #other modules expect QT_LIB
QT_VERSION = QtVersion  #preserving legacy `QtVersion` var

def print_lib_info():
    print "PyQtGraph is using:"
    print "  Qt api: %s" % QT_API
    print "  Qt lib: %s" % QT_LIB
    print "  Qt ver: %s" % QtVersion
    print "  qt_backport emulation: %s" % QT_EMULATING

if __debug__:
    print_lib_info()
