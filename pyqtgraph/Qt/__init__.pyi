"""
This stub file is to aid in the PyCharm auto-completion of the Qt imports.
"""

from typing import Union

try:
    from PyQt5 import QtCore, QtGui, QtSvg, QtTest, QtWidgets

    QtCore = QtCore
    QtGui = QtGui
    QtWidgets = QtWidgets
    QtTest = QtTest
    QtSvg = QtSvg
except ImportError:
    try:
        from PyQt6 import QtCore, QtGui, QtSvg, QtTest, QtWidgets

        QtCore = QtCore
        QtGui = QtGui
        QtWidgets = QtWidgets
        QtTest = QtTest
        QtSvg = QtSvg
    except ImportError:
        try:
            from PySide2 import QtCore, QtGui, QtSvg, QtTest, QtWidgets

            QtCore = QtCore
            QtGui = QtGui
            QtWidgets = QtWidgets
            QtTest = QtTest
            QtSvg = QtSvg
        except ImportError:
            try:
                from PySide6 import QtCore, QtGui, QtSvg, QtTest, QtWidgets

                QtCore = QtCore
                QtGui = QtGui
                QtWidgets = QtWidgets
                QtTest = QtTest
                QtSvg = QtSvg
            except ImportError as e:
                raise ImportError("No suitable qt binding found") from e


App: QtWidgets.QApplication
VERSION_INFO: str
QT_LIB: str
QtVersion: str
def exec_() -> QtWidgets.QApplication: ...
def mkQApp(name: Union[str, None] = None) -> QtWidgets.QApplication: ...
def isQObjectAlive(obj: QtCore.QObject) -> bool: ...
