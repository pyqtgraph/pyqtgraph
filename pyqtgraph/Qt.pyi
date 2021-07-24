from typing import Optional, Any

try:
    from PyQt5 import QtWidgets, QtCore, QtGui

    QtCore = QtCore
    QtGui = QtGui
    QtWidgets = QtWidgets
except ImportError:
    try:
        from PyQt6 import QtWidgets, QtCore, QtGui

        QtCore = QtCore
        QtGui = QtGui
        QtWidgets = QtWidgets
    except ImportError:
        try:
            from PySide2 import QtWidgets, QtCore, QtGui

            QtCore = QtCore
            QtGui = QtGui
            QtWidgets = QtWidgets
        except ImportError:
            try:
                from PySide6 import QtWidgets, QtCore, QtGui

                QtCore = QtCore
                QtGui = QtGui
                QtWidgets = QtWidgets
            except ImportError:
                raise Exception('No suitable Qt binding found')

def mkQApp(name=None) -> QtWidgets.QApplication: ...
def exec_() -> int: ...

QAPP: Optional[QtWidgets.QApplication]
QT_LIB: str

USE_PYSIDE: bool
USE_PYQT4: bool
USE_PYQT5: bool

PYSIDE: str
PYSIDE2: str
PYSIDE6: str
PYQT4: str
PYQT5: str
PYQT6: str

def __getattr__(name) -> Any: ...