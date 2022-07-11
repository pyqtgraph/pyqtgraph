from typing import Union

try:
    from PyQt5 import QtCore, QtGui, QtWidgets

    QtCore = QtCore
    QtGui = QtGui
    QtWidgets = QtWidgets
except ImportError:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets

        QtCore = QtCore
        QtGui = QtGui
        QtWidgets = QtWidgets
    except ImportError:
        try:
            from PySide2 import QtCore, QtGui, QtWidgets

            QtCore = QtCore
            QtGui = QtGui
            QtWidgets = QtWidgets
        except ImportError:
            try:
                from PySide6 import QtCore, QtGui, QtWidgets

                QtCore = QtCore
                QtGui = QtGui
                QtWidgets = QtWidgets
            except ImportError:
                raise Exception("No suitable qt binding found")


App: QtWidgets.QApplication
VERSION_INFO: str
QT_LIB: str
QtVersion: str
def exec_() -> QtWidgets.QApplication: ...
def mkQApp(name: Union[str, None] = None) -> QtWidgets.QApplication: ...
def isQObjectAlive(obj: QtCore.QObject) -> bool: ...
