"""
This module exists to smooth out some of the differences between PySide and PyQt4:

* Automatically import either PyQt4 or PySide depending on availability
* Allow to import QtCore/QtGui pyqtgraph.Qt without specifying which Qt wrapper
  you want to use.
* Declare QtCore.Signal, .Slot in PyQt4  
* Declare loadUiType function for Pyside

"""

import sys, re

from .python2_3 import asUnicode

## Automatically determine whether to use PyQt or PySide. 
## This is done by first checking to see whether one of the libraries
## is already imported. If not, then attempt to import PyQt4, then PySide.
if 'PyQt4' in sys.modules:
    USE_PYSIDE = False
elif 'PySide' in sys.modules:
    USE_PYSIDE = True
else:
    try:
        import PyQt4
        USE_PYSIDE = False
    except ImportError:
        try:
            import PySide
            USE_PYSIDE = True
        except ImportError:
            raise Exception("PyQtGraph requires either PyQt4 or PySide; neither package could be imported.")

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
        Pyside "loadUiType" command like PyQt4 has one, so we have to convert the ui file to py code in-memory first    and then execute it in a special frame to retrieve the form_class.
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
    
    
else:
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
    VERSION_INFO = 'PyQt4 ' + QtCore.PYQT_VERSION_STR + ' Qt ' + QtCore.QT_VERSION_STR


## Make sure we have Qt >= 4.7
versionReq = [4, 7]
QtVersion = PySide.QtCore.__version__ if USE_PYSIDE else QtCore.QT_VERSION_STR
m = re.match(r'(\d+)\.(\d+).*', QtVersion)
if m is not None and list(map(int, m.groups())) < versionReq:
    print(list(map(int, m.groups())))
    raise Exception('pyqtgraph requires Qt version >= %d.%d  (your version is %s)' % (versionReq[0], versionReq[1], QtVersion))

